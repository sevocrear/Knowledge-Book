from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

_MD_LINK_RE = re.compile(r"(!)?\[([^\]]*)\]\(([^)]+)\)")
_TOC_HEADING_RE = re.compile(r"^## Table of Contents\s*$", re.MULTILINE)
_ROOT_TOPIC_README_LINK_RE = re.compile(r"\]\(\./topics/([^/]+)/README\.md(?:#[^)]*)?\)")


def _is_external_link(raw: str) -> bool:
    lower = raw.lower()
    return lower.startswith(
        (
            "http://",
            "https://",
            "mailto:",
            "tel:",
            "ftp://",
            "data:",
        )
    )


def _normalize_link_target(raw: str) -> str:
    # Strip title: [text](path "title")
    raw = raw.strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1].strip()

    # Split off optional title part (best-effort, common Markdown patterns).
    parts = raw.split()
    if len(parts) >= 2 and (parts[-1].startswith('"') or parts[-1].startswith("'")):
        raw = parts[0]

    return raw


@dataclass(frozen=True)
class LinkRef:
    md_file: Path
    raw_target: str
    target_no_anchor: str
    anchor: str | None
    is_image: bool


def iter_markdown_links(md_file: Path) -> Iterator[LinkRef]:
    text = md_file.read_text(encoding="utf-8", errors="replace")
    for match in _MD_LINK_RE.finditer(text):
        is_image = match.group(1) is not None
        raw = _normalize_link_target(match.group(3))
        if not raw or raw.startswith("#") or _is_external_link(raw):
            continue

        target, anchor = (raw.split("#", 1) + [None])[:2]
        target = target.strip()
        if not target:
            continue

        yield LinkRef(
            md_file=md_file,
            raw_target=raw,
            target_no_anchor=target,
            anchor=anchor,
            is_image=is_image,
        )


@dataclass(frozen=True)
class BrokenLink:
    md_file: Path
    raw_target: str
    resolved_path: Path
    reason: str


def _resolve_link(md_file: Path, target_no_anchor: str, repo_root: Path) -> Path:
    # Absolute paths are treated as repo-root-relative (avoid leaking host FS).
    if target_no_anchor.startswith("/"):
        return (repo_root / target_no_anchor.lstrip("/")).resolve()
    return (md_file.parent / target_no_anchor).resolve()


def validate_markdown_files(md_files: Iterable[Path], repo_root: Path) -> list[BrokenLink]:
    broken: list[BrokenLink] = []

    for md in md_files:
        for ref in iter_markdown_links(md):
            resolved = _resolve_link(md, ref.target_no_anchor, repo_root)
            if not resolved.exists():
                broken.append(
                    BrokenLink(
                        md_file=md,
                        raw_target=ref.raw_target,
                        resolved_path=resolved,
                        reason="target does not exist",
                    )
                )
                continue

            if resolved.is_dir():
                broken.append(
                    BrokenLink(
                        md_file=md,
                        raw_target=ref.raw_target,
                        resolved_path=resolved,
                        reason="target is a directory (link should point to a file)",
                    )
                )

    return broken


def iter_topic_readmes(repo_root: Path) -> list[Path]:
    topics = repo_root / "topics"
    if not topics.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(topics.iterdir()):
        if not p.is_dir():
            continue
        readme = p / "README.md"
        if readme.is_file():
            out.append(readme.resolve())
    return out


def topic_has_table_of_contents(topic_readme: Path) -> bool:
    text = topic_readme.read_text(encoding="utf-8", errors="replace")
    return _TOC_HEADING_RE.search(text) is not None


def topic_slugs_on_disk(repo_root: Path) -> set[str]:
    slugs: set[str] = set()
    topics = repo_root / "topics"
    if not topics.is_dir():
        return slugs
    for p in topics.iterdir():
        if p.is_dir() and (p / "README.md").is_file():
            slugs.add(p.name)
    return slugs


def slugs_linked_from_root_readme(readme_path: Path) -> set[str]:
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    return set(_ROOT_TOPIC_README_LINK_RE.findall(text))


def validate_root_readme_topic_index(repo_root: Path) -> tuple[set[str], set[str]]:
    """Return (missing_in_readme, unknown_slug_in_readme)."""
    root_readme = repo_root / "README.md"
    if not root_readme.is_file():
        return set(), set()
    on_disk = topic_slugs_on_disk(repo_root)
    linked = slugs_linked_from_root_readme(root_readme)
    return on_disk - linked, linked - on_disk


def _default_md_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    readme = repo_root / "README.md"
    if readme.exists():
        files.append(readme)
    files.extend(sorted((repo_root / "topics").rglob("*.md")))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate knowledge-book markdown: links, topic TOCs, root README index."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (defaults to current working directory).",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit markdown files to validate (defaults to README.md and topics/**/*.md).",
    )
    args = parser.parse_args(argv)

    repo_root = args.root.resolve()
    md_files = [p.resolve() for p in (args.files if args.files is not None else _default_md_files(repo_root))]
    md_files = [p for p in md_files if p.exists()]

    exit_code = 0

    broken = validate_markdown_files(md_files, repo_root=repo_root)
    if broken:
        exit_code = 2
        print(f"Broken links found: {len(broken)}")
        for b in broken:
            rel_md = b.md_file.relative_to(repo_root) if b.md_file.is_relative_to(repo_root) else b.md_file
            rel_target = (
                b.resolved_path.relative_to(repo_root) if b.resolved_path.is_relative_to(repo_root) else b.resolved_path
            )
            print(f"- {rel_md}: ({b.raw_target}) -> {rel_target} [{b.reason}]")
    else:
        print(f"OK: no broken relative links in {len(md_files)} markdown files.")

    topic_readmes = iter_topic_readmes(repo_root)
    missing_toc = [p for p in topic_readmes if not topic_has_table_of_contents(p)]
    if missing_toc:
        exit_code = 2
        print(f"Missing '## Table of Contents' in {len(missing_toc)} topic README(s):")
        for p in missing_toc:
            rel = p.relative_to(repo_root) if p.is_relative_to(repo_root) else p
            print(f"- {rel}")
    else:
        print(f"OK: Table of Contents present in all {len(topic_readmes)} topic README(s).")

    missing_index, unknown_slug = validate_root_readme_topic_index(repo_root)
    if missing_index:
        exit_code = 2
        print("Topics on disk missing from root README.md ## Contents (./topics/<slug>/README.md link):")
        for slug in sorted(missing_index):
            print(f"- {slug}")
    if unknown_slug:
        exit_code = 2
        print("Root README.md links to unknown topic slug (no topics/<slug>/README.md on disk):")
        for slug in sorted(unknown_slug):
            print(f"- {slug}")
    if not missing_index and not unknown_slug and (repo_root / "README.md").is_file():
        print("OK: root README.md topic index matches topics/*/README.md on disk.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
