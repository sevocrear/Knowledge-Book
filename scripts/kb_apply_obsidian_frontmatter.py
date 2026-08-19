#!/usr/bin/env python3
"""Apply Obsidian-style YAML frontmatter to knowledge-book markdown notes.

Also:
- ensures a top-level `# Title` heading when missing;
- regenerates docs/ indexes (MOCs, tags, topic catalog) for Obsidian/RAG.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from kb_topic_metadata import MOCS, NOTE_METADATA, TOPIC_METADATA  # noqa: E402

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n?", re.DOTALL)


def _yaml_escape(value: str) -> str:
    if any(ch in value for ch in (":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "'", '"', "%", "@", "`")):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def render_frontmatter(
    *,
    title: str,
    description: str,
    tags: list[str],
    aliases: list[str],
    related: list[str],
    status: str,
    lang: str,
    note_type: str,
    slug: str,
    updated: str,
) -> str:
    lines = [
        "---",
        f"title: {_yaml_escape(title)}",
        f"description: {_yaml_escape(description)}",
        "tags:",
    ]
    for tag in tags:
        lines.append(f"  - {tag}")
    lines.append("aliases:")
    for alias in aliases:
        lines.append(f"  - {_yaml_escape(alias)}")
    lines.append("related:")
    for rel in related:
        lines.append(f"  - {rel}")
    lines.extend(
        [
            f"status: {status}",
            f"lang: {lang}",
            f"type: {note_type}",
            f"slug: {slug}",
            f"updated: {updated}",
            "---",
            "",
        ]
    )
    return "\n".join(lines)


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        match = _FRONTMATTER_RE.match(text)
        if match:
            return text[match.end() :]
    return text


def ensure_h1(body: str, title: str) -> str:
    lines = body.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return f"# {title}\n"

    first = lines[i]
    stripped = first.lstrip()

    # Normalize indented ATX H1 (legacy inconsistency).
    if stripped.startswith("# ") and not stripped.startswith("## "):
        lines[i] = stripped
        return "\n".join(lines) + ("\n" if body.endswith("\n") else "")

    # Promote a leading title-like ## heading (common inconsistency in this repo).
    # Do not scan the whole body: fenced code often contains `# comments`.
    if stripped.startswith("## ") and "Оглавление" not in stripped and "Table of Contents" not in stripped:
        lines[i] = "# " + stripped[3:]
        return "\n".join(lines) + ("\n" if body.endswith("\n") else "")

    return f"# {title}\n\n" + body.lstrip("\n")


def _topic_link_from(docs_rel_dir: str, slug: str) -> str:
    """Relative markdown link from a docs/ subpath to topics/<slug>/README.md."""
    if docs_rel_dir in {"", "."}:
        prefix = "../topics"
    else:
        depth = len(Path(docs_rel_dir).parts)
        prefix = "/".join([".."] * (depth + 1)) + "/topics"
    return f"{prefix}/{slug}/README.md"


def _note_link_from(docs_rel_dir: str, key: str) -> str:
    parent, name = key.split("/", 1)
    if docs_rel_dir in {"", "."}:
        prefix = "../topics"
    else:
        depth = len(Path(docs_rel_dir).parts)
        prefix = "/".join([".."] * (depth + 1)) + "/topics"
    return f"{prefix}/{parent}/{name}.md"


def apply_to_file(path: Path, meta: dict, updated: str) -> bool:
    original = path.read_text(encoding="utf-8")
    body = strip_frontmatter(original)
    body = ensure_h1(body, meta["title"])
    slug = meta.get("slug") or path.parent.name
    if path.name != "README.md":
        slug = f"{path.parent.name}/{path.stem}"

    fm = render_frontmatter(
        title=meta["title"],
        description=meta["description"],
        tags=list(meta["tags"]),
        aliases=list(meta["aliases"]),
        related=list(meta["related"]),
        status=meta["status"],
        lang=meta["lang"],
        note_type=meta["type"],
        slug=slug,
        updated=updated,
    )
    new_text = fm + body.lstrip("\n")
    if not new_text.endswith("\n"):
        new_text += "\n"
    if new_text == original:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True




def write_docs_indexes(repo_root: Path, updated: str) -> None:
    docs = repo_root / "docs"
    (docs / "mocs").mkdir(parents=True, exist_ok=True)
    (docs / "tags").mkdir(parents=True, exist_ok=True)

    # docs/README.md
    moc_lines = "\n".join(
        f"- [{cfg['title']}](./mocs/{name}.md) — {cfg['description']}" for name, cfg in MOCS.items()
    )
    readme = f"""---
title: Документация Knowledge Book (слой Obsidian / RAG)
description: Индексы, таксономия тегов и Maps of Content над топиками в topics/.
tags:
  - kb/index
  - kb/docs
aliases:
  - docs home
  - knowledge book vault
  - оглавление книги
status: canonical
lang: ru
type: index
updated: {updated}
---

# Документация Knowledge Book

Слой `docs/` — это **индекс Obsidian / RAG** над каноническими конспектами в `topics/`.

## Как объяснить 5-летнему ребёнку

Большие уроки лежат в папках тем. Папка `docs/` — это **оглавление с наклейками (тегами)**, чтобы поисковый робот или Obsidian быстрее находил нужный урок.

## Как устроено

| Путь | Роль |
|------|------|
| `topics/<slug>/README.md` | Канонические конспекты (теория, формулы, примеры) |
| `docs/index.md` | Плоский каталог всех тем с описаниями |
| `docs/mocs/` | Maps of Content (тематические входы) |
| `docs/tags/` | Таксономия тегов и списки тем по тегу |
| `docs/SCHEMA.md` | Схема frontmatter для RAG / Obsidian |

## Maps of Content

{moc_lines}

## Быстрые ссылки

- [Полный каталог тем](./index.md)
- [Таксономия тегов](./tags/README.md)
- [Схема frontmatter](./SCHEMA.md)
- [Корневой README книги](../README.md)

## Соглашения

Каждый markdown-файл топика/заметки начинается с YAML frontmatter:

- `title`, `description` — основные поля для RAG
- `tags` — иерархические теги (`domain/*`, `concept/*`, `kb/*`)
- `aliases` — альтернативные имена / синонимы для поиска
- `related` — slug соседних тем
- `status`, `lang`, `type`, `slug`, `updated`

После правок метаданных пересобрать индексы:

```bash
uv run python scripts/kb_apply_obsidian_frontmatter.py
uv run python scripts/kb_validate_links.py
```
"""
    (docs / "README.md").write_text(readme, encoding="utf-8")

    # SCHEMA
    schema = f"""---
title: Схема Obsidian frontmatter
description: Обязательные поля YAML frontmatter для заметок книги, которые используют Obsidian и RAG.
tags:
  - kb/schema
  - kb/docs
status: canonical
lang: ru
type: schema
updated: {updated}
---

# Схема Obsidian frontmatter

## Обязательные поля

```yaml
---
title: Человекочитаемый заголовок
description: Одно-два предложения для RAG / сниппетов поиска
tags:
  - kb/topic          # или kb/note, kb/moc, kb/index
  - domain/cv         # широкая область
  - concept/attention # конкретная идея
aliases:
  - Альтернативное имя
related:
  - sibling-topic-slug
status: canonical     # canonical | notes | draft
lang: ru              # ru | en | mixed
type: topic           # topic | note | moc | index | schema
slug: topic-slug
updated: YYYY-MM-DD
---
```

## Пространства имён тегов

| Префикс | Смысл | Примеры |
|--------|---------|----------|
| `kb/` | Структура книги | `kb/topic`, `kb/note`, `kb/moc`, `kb/index` |
| `domain/` | Широкая область | `domain/cv`, `domain/llm`, `domain/robotics` |
| `concept/` | Конкретная идея | `concept/rag`, `concept/lora`, `concept/nms` |
| `source/` | Происхождение заметки | `source/youtube` |

## Зачем это RAG

1. **`description`** — плотное описание для поиска, независимое от длины заметки.
2. **`tags` + `aliases`** — расширяют recall по синонимам.
3. **`related`** — позволяют идти по графу соседних тем после попадания.
4. **`slug`** — стабильный ID для цитирования и метаданных чанков.

## Проверка

`scripts/kb_validate_links.py` проверяет, что у каждого `topics/*/README.md` и вложенной заметки есть корректный frontmatter с обязательными ключами.
"""
    (docs / "SCHEMA.md").write_text(schema, encoding="utf-8")

    # index.md
    index_lines = [
        "---",
        "title: Каталог тем",
        "description: Аннотированный каталог всех тем книги с тегами и описаниями для RAG.",
        "tags:",
        "  - kb/index",
        "status: canonical",
        "lang: ru",
        "type: index",
        f"updated: {updated}",
        "---",
        "",
        "# Каталог тем",
        "",
        "Канонические конспекты лежат в `topics/<slug>/README.md`. Описания ниже — те же строки, что хранятся во frontmatter для поиска.",
        "",
    ]
    for slug, meta in sorted(TOPIC_METADATA.items()):
        tag_str = ", ".join(f"`{t}`" for t in meta["tags"] if not t.startswith("kb/"))
        href = _topic_link_from(".", slug)
        index_lines.extend(
            [
                f"## [{meta['title']}]({href})",
                "",
                f"- **slug:** `{slug}`",
                f"- **description:** {meta['description']}",
                f"- **tags:** {tag_str}",
                f"- **aliases:** {', '.join(meta['aliases'][:6])}",
                "",
            ]
        )
        index_lines.extend(["## Вложенные заметки", ""])
    for key, meta in sorted(NOTE_METADATA.items()):
        href = _note_link_from(".", key)
        index_lines.extend(
            [
                f"### [{meta['title']}]({href})",
                "",
                f"- **description:** {meta['description']}",
                f"- **parent:** `{key.split('/', 1)[0]}`",
                "",
            ]
        )
    (docs / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    # MOCs
    for name, cfg in MOCS.items():
        lines = [
            "---",
            f"title: {_yaml_escape(str(cfg['title']))}",
            f"description: {_yaml_escape(str(cfg['description']))}",
            "tags:",
        ]
        for tag in cfg["tags"]:  # type: ignore[union-attr]
            lines.append(f"  - {tag}")
        lines.extend(
            [
                "type: moc",
                "status: canonical",
                f"updated: {updated}",
                "---",
                "",
                f"# {cfg['title']}",
                "",
                str(cfg["description"]),
                "",
                "## Темы",
                "",
            ]
        )
        for slug in cfg["topics"]:  # type: ignore[union-attr]
            meta = TOPIC_METADATA[slug]
            href = _topic_link_from("mocs", slug)
            lines.append(f"- [{meta['title']}]({href}) — {meta['description']}")
        lines.extend(["", "## См. также", "", "- [Все темы](../index.md)", "- [Теги](../tags/README.md)", ""])
        (docs / "mocs" / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")

    # Tag taxonomy + pages
    tag_to_slugs: dict[str, list[str]] = defaultdict(list)
    for slug, meta in TOPIC_METADATA.items():
        for tag in meta["tags"]:
            tag_to_slugs[tag].append(slug)
    for key, meta in NOTE_METADATA.items():
        for tag in meta["tags"]:
            tag_to_slugs[tag].append(key)

    tag_readme = [
        "---",
        "title: Таксономия тегов",
        "description: Иерархические теги заметок книги для фильтрации в Obsidian и RAG.",
        "tags:",
        "  - kb/index",
        "status: canonical",
        "lang: ru",
        "type: index",
        f"updated: {updated}",
        "---",
        "",
        "# Таксономия тегов",
        "",
        "Теги имеют вид `namespace/value`. Переходите на страницу тега, чтобы увидеть заметки с ним.",
        "",
    ]
    by_ns: dict[str, list[str]] = defaultdict(list)
    for tag in sorted(tag_to_slugs):
        ns = tag.split("/", 1)[0] if "/" in tag else "other"
        by_ns[ns].append(tag)

    for ns in sorted(by_ns):
        tag_readme.append(f"## `{ns}/`")
        tag_readme.append("")
        for tag in by_ns[ns]:
            safe = tag.replace("/", "-")
            n = len(tag_to_slugs[tag])
            n10, n100 = n % 10, n % 100
            if n10 == 1 and n100 != 11:
                note_word = "заметка"
            elif n10 in {2, 3, 4} and n100 not in {12, 13, 14}:
                note_word = "заметки"
            else:
                note_word = "заметок"
            tag_readme.append(f"- [`{tag}`](./{safe}.md) — {n} {note_word}")
        tag_readme.append("")
    (docs / "tags" / "README.md").write_text("\n".join(tag_readme) + "\n", encoding="utf-8")

    for tag, slugs in sorted(tag_to_slugs.items()):
        safe = tag.replace("/", "-")
        lines = [
            "---",
            f"title: {_yaml_escape('Тег: ' + tag)}",
            f"description: {_yaml_escape('Заметки с тегом ' + tag + ' в книге знаний.')}",
            "tags:",
            f"  - {tag}",
            "  - kb/tag-page",
            "type: index",
            "status: canonical",
            f"updated: {updated}",
            "---",
            "",
            f"# Тег `{tag}`",
            "",
            "## Заметки",
            "",
        ]
        for slug in sorted(set(slugs)):
            if "/" in slug and slug in NOTE_METADATA:
                meta = NOTE_METADATA[slug]
                href = _note_link_from("tags", slug)
            else:
                meta = TOPIC_METADATA[slug]
                href = _topic_link_from("tags", slug)
            lines.append(f"- [{meta['title']}]({href}) — {meta['description']}")
        lines.append("")
        (docs / "tags" / f"{safe}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--updated", default=date.today().isoformat())
    parser.add_argument("--skip-docs", action="store_true")
    args = parser.parse_args(argv)

    repo = args.root.resolve()
    topics = repo / "topics"
    changed = 0

    for slug, meta in TOPIC_METADATA.items():
        path = topics / slug / "README.md"
        if not path.is_file():
            raise SystemExit(f"Missing topic README: {path}")
        payload = dict(meta)
        payload["slug"] = slug
        if apply_to_file(path, payload, args.updated):
            changed += 1
            print(f"updated: topics/{slug}/README.md")

    for key, meta in NOTE_METADATA.items():
        path = topics / f"{key}.md"
        if not path.is_file():
            raise SystemExit(f"Missing note: {path}")
        payload = dict(meta)
        payload["slug"] = key
        if apply_to_file(path, payload, args.updated):
            changed += 1
            print(f"updated: topics/{key}.md")

    if not args.skip_docs:
        write_docs_indexes(repo, args.updated)
        print("regenerated: docs/")

    print(f"done: {changed} markdown file(s) changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
