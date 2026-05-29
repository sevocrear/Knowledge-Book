#!/usr/bin/env python3
"""
Fetch a YouTube video transcript via youtube-transcript-api (primary) or yt-dlp (fallback).

Demonstrates: list caption tracks; save plain or timed transcript to stdout or a file.
Expected: exit 0 and a non-empty transcript file when captions exist and YouTube allows access.

Usage:
  uv sync --group tools
  uv run python scripts/youtube_fetch_transcript.py "https://www.youtube.com/watch?v=C_GG5g38vLU"
  uv run python scripts/youtube_fetch_transcript.py C_GG5g38vLU --list
  uv run python scripts/youtube_fetch_transcript.py C_GG5g38vLU -o outputs/transcripts/C_GG5g38vLU.txt

On Linux with Chrome, auto mode uses yt-dlp + browser cookies (needs secretstorage in tools group).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

_LOG = logging.getLogger("youtube_fetch_transcript")

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)
_VTT_TS_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    r"\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
)
_CHROME_COOKIE_DIRS = (
    Path.home() / ".config" / "google-chrome",
    Path.home() / ".config" / "chromium",
)


@dataclass(frozen=True)
class Snippet:
    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class TranscriptBundle:
    video_id: str
    language: str
    language_code: str
    is_generated: bool
    snippets: list[Snippet]
    backend: str


def parse_video_id(url_or_id: str) -> str:
    """Extract 11-char video id from URL or pass through a bare id."""
    raw = url_or_id.strip()
    match = _VIDEO_ID_RE.search(raw)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw
    msg = f"Cannot parse YouTube video id from: {url_or_id!r}"
    raise ValueError(msg)


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _vtt_timestamp_to_seconds(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 2:
        minutes, sec = parts
        hours = 0
    else:
        hours, minutes, sec = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(sec)


def parse_vtt(path: Path) -> list[Snippet]:
    """Parse WebVTT (yt-dlp subtitle export) into snippets."""
    text = path.read_text(encoding="utf-8", errors="replace")
    snippets: list[Snippet] = []
    block_lines: list[str] = []
    block_start: float | None = None
    block_end: float | None = None

    def flush() -> None:
        nonlocal block_lines, block_start, block_end
        if block_start is None or not block_lines:
            block_lines = []
            block_start = None
            block_end = None
            return
        body = " ".join(line.strip() for line in block_lines if line.strip())
        body = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        if body:
            duration = max((block_end or block_start) - block_start, 0.0)
            snippets.append(Snippet(text=body, start=block_start, duration=duration))
        block_lines = []
        block_start = None
        block_end = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "WEBVTT" or line.startswith("NOTE"):
            flush()
            continue
        if line.isdigit():
            continue
        ts_match = _VTT_TS_RE.match(line)
        if ts_match:
            flush()
            block_start = _vtt_timestamp_to_seconds(ts_match.group("start"))
            block_end = _vtt_timestamp_to_seconds(ts_match.group("end"))
            continue
        if block_start is not None:
            block_lines.append(line)
    flush()
    return snippets


def render_plain(snippets: list[Snippet]) -> str:
    return "\n".join(s.text.strip() for s in snippets if s.text.strip())


def render_timed(snippets: list[Snippet]) -> str:
    lines: list[str] = []
    for s in snippets:
        text = s.text.strip()
        if not text:
            continue
        lines.append(f"({format_timestamp(s.start)})  {text}")
    return "\n".join(lines)


def resolve_ytdlp_command() -> list[str]:
    venv_bin = Path(sys.executable).resolve().parent / "yt-dlp"
    if venv_bin.is_file():
        return [str(venv_bin)]
    found = shutil.which("yt-dlp")
    if found:
        return [found]
    return [sys.executable, "-m", "yt_dlp"]


def default_browser_for_cookies() -> str | None:
    """Pick a browser profile for yt-dlp cookie export when user did not specify one."""
    for path in _CHROME_COOKIE_DIRS:
        if path.is_dir():
            return "chrome" if "chromium" not in path.name else "chromium"
    return None


def resolve_cookies_browser(explicit: str | None, backend: str) -> str | None:
    if explicit:
        return explicit
    if backend in ("auto", "yt-dlp"):
        return default_browser_for_cookies()
    return None


def _run_ytdlp(cmd: list[str], verbose: bool) -> subprocess.CompletedProcess[str]:
    _LOG.debug("running: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if verbose:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return proc


def _ytdlp_base_cmd(cookies_from_browser: str | None) -> list[str]:
    cmd = resolve_ytdlp_command()
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    return cmd


def list_tracks_api(api: YouTubeTranscriptApi, video_id: str) -> None:
    transcript_list = api.list(video_id)
    print(f"Video: https://www.youtube.com/watch?v={video_id}\n")
    print("backend: youtube-transcript-api\n")
    for t in transcript_list:
        generated = "auto" if t.is_generated else "manual"
        translatable = "yes" if t.is_translatable else "no"
        print(
            f"  {t.language_code:8}  {t.language:24}  "
            f"type={generated:6}  translatable={translatable}"
        )


def list_tracks_ytdlp(
    video_id: str,
    cookies_from_browser: str | None,
    verbose: bool,
) -> None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = _ytdlp_base_cmd(cookies_from_browser)
    cmd.extend(["--list-subs", url])
    proc = _run_ytdlp(cmd, verbose)
    print(f"Video: {url}\n")
    print("backend: yt-dlp\n")
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "yt-dlp --list-subs failed").strip())
    print(proc.stdout)


def fetch_via_api(
    api: YouTubeTranscriptApi,
    video_id: str,
    languages: list[str],
) -> TranscriptBundle:
    fetched = api.fetch(video_id, languages=languages)
    snippets = [
        Snippet(text=s.text, start=s.start, duration=s.duration) for s in fetched
    ]
    return TranscriptBundle(
        video_id=fetched.video_id,
        language=fetched.language,
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
        snippets=snippets,
        backend="youtube-transcript-api",
    )


def _pick_vtt(tmp: Path, video_id: str, lang: str) -> Path | None:
    patterns = [
        f"{video_id}.{lang}.vtt",
        f"{video_id}.{lang}*.vtt",
        f"{video_id}*.vtt",
    ]
    for pattern in patterns:
        matches = sorted(tmp.glob(pattern))
        if matches:
            return matches[0]
    return None


def fetch_via_ytdlp(
    video_id: str,
    languages: list[str],
    cookies_from_browser: str | None,
    verbose: bool,
) -> TranscriptBundle:
    url = f"https://www.youtube.com/watch?v={video_id}"
    lang_chain = list(dict.fromkeys(languages + ["en", "en-orig", "ru"]))

    with tempfile.TemporaryDirectory(prefix="yt-transcript-") as tmp_dir:
        tmp = Path(tmp_dir)
        out_tpl = str(tmp / f"{video_id}.%(ext)s")
        last_err = ""

        for lang in lang_chain:
            for sub_mode, generated in (
                ("--write-auto-sub", True),
                ("--write-sub", False),
            ):
                cmd = _ytdlp_base_cmd(cookies_from_browser)
                cmd.extend(
                    [
                        sub_mode,
                        "--sub-lang",
                        lang,
                        "--skip-download",
                        "--ignore-no-formats-error",
                        "-o",
                        out_tpl,
                        url,
                    ],
                )
                proc = _run_ytdlp(cmd, verbose)
                if proc.returncode != 0:
                    last_err = (proc.stderr or proc.stdout or "").strip()
                    _LOG.warning("yt-dlp failed for lang=%s mode=%s", lang, sub_mode)
                    continue

                vtt_path = _pick_vtt(tmp, video_id, lang)
                if vtt_path is None:
                    continue
                snippets = parse_vtt(vtt_path)
                if not snippets:
                    continue
                return TranscriptBundle(
                    video_id=video_id,
                    language=lang,
                    language_code=lang,
                    is_generated=generated,
                    snippets=snippets,
                    backend="yt-dlp",
                )

    msg = (
        "yt-dlp could not download subtitles. "
        "Install tools group (secretstorage for Chrome cookies). "
        "Try: --cookies-from-browser chrome. "
        f"Last error: {last_err or 'no vtt produced'}"
    )
    raise RuntimeError(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube transcript (captions) for a video.",
    )
    parser.add_argument(
        "video",
        help="YouTube URL or 11-character video id",
    )
    parser.add_argument(
        "-l",
        "--lang",
        action="append",
        dest="languages",
        metavar="CODE",
        help="Preferred language codes in order (default: en, ru). Repeatable.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available transcript tracks and exit",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("plain", "timed", "json"),
        default="timed",
        help="Output format (default: timed)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write transcript to this file (creates parent dirs)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "api", "yt-dlp"),
        default="auto",
        help="Fetch backend: API first, yt-dlp fallback (default: auto)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Browser for yt-dlp cookies (default on Linux: chrome if profile exists)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print yt-dlp logs to stderr",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        video_id = parse_video_id(args.video)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    languages = args.languages if args.languages else ["en", "ru", "en-US", "en-GB"]
    cookies_browser = resolve_cookies_browser(
        args.cookies_from_browser,
        args.backend,
    )
    if cookies_browser:
        _LOG.info("yt-dlp cookies from browser: %s", cookies_browser)

    api = YouTubeTranscriptApi()

    if args.list:
        if args.backend in ("auto", "api"):
            try:
                list_tracks_api(api, video_id)
                return 0
            except CouldNotRetrieveTranscript as exc:
                _LOG.warning("API list failed: %s", exc)
                if args.backend == "api":
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
        try:
            list_tracks_ytdlp(video_id, cookies_browser, args.verbose)
            return 0
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    bundle: TranscriptBundle | None = None
    errors: list[str] = []

    if args.backend in ("auto", "api"):
        try:
            bundle = fetch_via_api(api, video_id, languages)
            _LOG.info("fetched via youtube-transcript-api (%d snippets)", len(bundle.snippets))
        except TranscriptsDisabled:
            errors.append("transcripts disabled on YouTube")
        except NoTranscriptFound:
            errors.append(f"no API transcript for languages {languages!r}")
        except VideoUnavailable:
            errors.append("video unavailable")
        except CouldNotRetrieveTranscript as exc:
            errors.append(f"API failed: {exc}")

    if bundle is None and args.backend in ("auto", "yt-dlp"):
        try:
            bundle = fetch_via_ytdlp(
                video_id,
                languages,
                cookies_browser,
                args.verbose,
            )
            _LOG.info(
                "fetched via yt-dlp (%d snippets, lang=%s)",
                len(bundle.snippets),
                bundle.language_code,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            errors.append(str(exc))

    if bundle is None:
        print("error: could not fetch transcript.", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nHints:\n"
            "  uv sync --group tools\n"
            "  uv run python scripts/youtube_fetch_transcript.py ID -v --list\n"
            "  uv run python scripts/youtube_fetch_transcript.py ID "
            "--backend yt-dlp --cookies-from-browser chrome",
            file=sys.stderr,
        )
        return 1

    if args.format == "plain":
        body = render_plain(bundle.snippets)
    elif args.format == "timed":
        body = render_timed(bundle.snippets)
    else:
        payload = {
            "video_id": bundle.video_id,
            "language": bundle.language,
            "language_code": bundle.language_code,
            "is_generated": bundle.is_generated,
            "backend": bundle.backend,
            "snippets": [asdict(s) for s in bundle.snippets],
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)

    header = (
        f"# video: https://www.youtube.com/watch?v={bundle.video_id}\n"
        f"# language: {bundle.language_code} ({bundle.language}), "
        f"generated={bundle.is_generated}, backend={bundle.backend}\n\n"
    )
    if args.format != "json":
        body = header + body

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
        print(
            f"Wrote {len(bundle.snippets)} snippets -> {args.output} "
            f"(backend={bundle.backend})",
            file=sys.stderr,
        )
    else:
        print(body)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
