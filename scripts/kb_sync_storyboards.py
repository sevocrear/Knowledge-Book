#!/usr/bin/env python3
"""Синхронизирует NOTE_METADATA в scripts/kb_topic_metadata.py со frontmatter сторибордов клипов.

Для каждого topics/<slug>/visualizations/hyperframes/storyboard.md читается YAML-frontmatter
(title, description, tags, aliases, related, status, lang, type) и записывается в авто-блок между
маркерами `# --- storyboards (auto) ---` и `# --- end storyboards (auto) ---` внутри NOTE_METADATA.
Запуск: uv run python scripts/kb_sync_storyboards.py, затем kb_apply_obsidian_frontmatter.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
META = REPO_ROOT / "scripts" / "kb_topic_metadata.py"
START, END = "    # --- storyboards (auto) ---\n", "    # --- end storyboards (auto) ---\n"
_FM = re.compile(r"^---\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict[str, object]:
    m = _FM.match(text)
    if not m:
        raise ValueError("нет frontmatter")
    data: dict[str, object] = {}
    key = None
    for line in m.group(1).split("\n"):
        if re.match(r"^\s+-\s+", line) and key:
            data.setdefault(key, [])
            assert isinstance(data[key], list)
            data[key].append(line.split("-", 1)[1].strip().strip('"'))
        elif ":" in line:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            data[key] = [] if val == "" else val.strip('"')
    return data


def render_entry(key: str, fm: dict[str, object]) -> str:
    q = lambda s: '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'  # noqa: E731
    tags = "".join(f"            {q(t)},\n" for t in fm.get("tags", []))
    aliases = ", ".join(q(a) for a in fm.get("aliases", []))
    related = "".join(f"            {q(r)},\n" for r in fm.get("related", []))
    return (
        f"    {q(key)}: {{\n"
        f"        \"title\": {q(fm['title'])},\n"
        f"        \"description\": (\n            {q(fm['description'])}\n        ),\n"
        f"        \"tags\": [\n{tags}        ],\n"
        f"        \"aliases\": [{aliases}],\n"
        f"        \"related\": [\n{related}        ],\n"
        f"        \"status\": {q(fm.get('status', 'notes'))},\n"
        f"        \"lang\": {q(fm.get('lang', 'ru'))},\n"
        f"        \"type\": {q(fm.get('type', 'note'))},\n"
        f"    }},\n"
    )


def main() -> int:
    entries = []
    for sb in sorted((REPO_ROOT / "topics").glob("*/visualizations/hyperframes/storyboard.md")):
        slug = sb.parents[2].name
        try:
            fm = parse_frontmatter(sb.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"ПРОПУСК {sb.relative_to(REPO_ROOT)}: {e}", file=sys.stderr)
            continue
        if not all(k in fm for k in ("title", "description")):
            print(f"ПРОПУСК {sb.relative_to(REPO_ROOT)}: нет title/description", file=sys.stderr)
            continue
        entries.append(render_entry(f"{slug}/visualizations/hyperframes/storyboard", fm))
    block = START + "".join(entries) + END
    src = META.read_text(encoding="utf-8")
    if START in src and END in src:
        i, j = src.index(START), src.index(END) + len(END)
        src = src[:i] + block + src[j:]
    else:
        # убрать ручные записи сторибордов и вставить авто-блок в конец NOTE_METADATA
        src = re.sub(r'    "[^"]+/visualizations/hyperframes/storyboard": \{.*?\n    \},\n', "", src, flags=re.DOTALL)
        i = src.index("NOTE_METADATA: dict[str, TopicMeta] = {")
        j = src.index("\n}\n", i) + 1
        src = src[:j] + block + src[j:]
    META.write_text(src, encoding="utf-8")
    print(f"storyboards: {len(entries)} записей синхронизировано в NOTE_METADATA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
