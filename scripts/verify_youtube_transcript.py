#!/usr/bin/env python3
"""
Verify that a transcript file from youtube_fetch_transcript.py is usable.

Expected: exit 0 when the file exists, has enough timed/plain lines, and is not empty/error-only.

Usage:
  uv run python scripts/verify_youtube_transcript.py outputs/transcripts/VIDEO_ID.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_TIMED_LINE = re.compile(r"^\(\d{2}:\d{2}(?::\d{2})?\)\s+.+")
_MIN_BODY_LINES = 20
_MIN_BODY_CHARS = 400
_MIN_TIMED_LINES = 10


def verify_transcript(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"file not found: {path}"]

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return ["file is empty"]

    lines = text.splitlines()
    body_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    body_chars = sum(len(ln) for ln in body_lines)

    if len(body_lines) < _MIN_BODY_LINES:
        errors.append(
            f"too few body lines: {len(body_lines)} < {_MIN_BODY_LINES}",
        )
    if body_chars < _MIN_BODY_CHARS:
        errors.append(
            f"too little text: {body_chars} chars < {_MIN_BODY_CHARS}",
        )

    timed = [ln for ln in body_lines if _TIMED_LINE.match(ln)]
    if len(timed) < _MIN_TIMED_LINES:
        errors.append(
            f"too few timed transcript lines: {len(timed)} < {_MIN_TIMED_LINES}",
        )

    lowered = text.lower()
    if "error: could not fetch transcript" in lowered:
        errors.append("file looks like a failed fetch log, not a transcript")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify YouTube transcript output file.")
    parser.add_argument("path", type=Path, help="Path to .txt transcript")
    args = parser.parse_args(argv)

    errors = verify_transcript(args.path)
    if errors:
        print(f"FAIL: {args.path}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    text = args.path.read_text(encoding="utf-8")
    body_lines = [ln for ln in text.splitlines() if not ln.startswith("#") and ln.strip()]
    print(
        f"OK: {args.path} ({len(body_lines)} body lines, "
        f"{sum(len(ln) for ln in body_lines)} chars)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
