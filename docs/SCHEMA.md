---
title: Obsidian Frontmatter Schema
description: Required YAML frontmatter fields for knowledge-book notes used by Obsidian and RAG.
tags:
  - kb/schema
  - kb/docs
status: canonical
lang: en
type: schema
updated: 2026-08-17
---

# Obsidian Frontmatter Schema

## Required fields

```yaml
---
title: Human-readable title
description: One or two sentences for RAG / search snippets
tags:
  - kb/topic          # or kb/note, kb/moc, kb/index
  - domain/cv         # coarse domain
  - concept/attention # fine-grained concepts
aliases:
  - Alternate Name
related:
  - sibling-topic-slug
status: canonical     # canonical | notes | draft
lang: ru              # ru | en | mixed
type: topic           # topic | note | moc | index | schema
slug: topic-slug
updated: YYYY-MM-DD
---
```

## Tag namespaces

| Prefix | Meaning | Examples |
|--------|---------|----------|
| `kb/` | Book structure | `kb/topic`, `kb/note`, `kb/moc`, `kb/index` |
| `domain/` | Broad field | `domain/cv`, `domain/llm`, `domain/robotics` |
| `concept/` | Concrete idea | `concept/rag`, `concept/lora`, `concept/nms` |
| `source/` | Provenance for notes | `source/youtube` |

## Why this helps RAG

1. **`description`** is a dense retrieval summary independent of note length.
2. **`tags` + `aliases`** expand recall for synonym queries.
3. **`related`** supports graph-style expansion after a hit.
4. **`slug`** is a stable ID for citations and chunk metadata.

## Validation

`scripts/kb_validate_links.py` checks that every `topics/*/README.md` and topic note has valid frontmatter with required keys.
