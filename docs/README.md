---
title: Knowledge Book Docs (Obsidian / RAG layer)
description: Obsidian-compatible indexes, tag taxonomy and Maps of Content over topics/.
tags:
  - kb/index
  - kb/docs
aliases:
  - docs home
  - knowledge book vault
status: canonical
lang: en
type: index
updated: 2026-08-10
---

# Knowledge Book Docs

This `docs/` layer is the **Obsidian / RAG index** over canonical topic notes in `topics/`.

## How would I describe it to a person who is 5 years old

The big lessons live in topic folders. This `docs/` folder is the **table of contents with stickers (tags)** so a search robot (or Obsidian) can find the right lesson fast.

## Layout

| Path | Role |
|------|------|
| `topics/<slug>/README.md` | Canonical deep notes (theory, formulas, examples) |
| `docs/index.md` | Flat catalog of all topics with descriptions |
| `docs/mocs/` | Maps of Content (thematic entry points) |
| `docs/tags/` | Tag taxonomy + per-tag topic lists |
| `docs/SCHEMA.md` | Frontmatter schema for RAG / Obsidian |

## Maps of Content

- [MOC: Mathematical & ML Foundations](./mocs/foundations.md) — Вероятность, метрики, классический ML и базовые строительные блоки.
- [MOC: Generative Models](./mocs/generative-models.md) — VAE, GAN и diffusion — три основных семейства генеративных моделей.
- [MOC: NLP, LLM & RAG](./mocs/nlp-llm.md) — Токенизация, эмбеддинги, transformers, LoRA, RAG и code agents.
- [MOC: Computer Vision](./mocs/computer-vision.md) — CNN, detection/segmentation, metric learning, SSL и tracking metrics.
- [MOC: Robotics & Embodied AI](./mocs/robotics-embodied.md) — Deep RL, VLA и vision-based обучение роботов.

## Quick links

- [Full topic catalog](./index.md)
- [Tag taxonomy](./tags/README.md)
- [Frontmatter schema](./SCHEMA.md)
- [Root knowledge-book README](../README.md)

## Conventions

Every topic/note markdown file starts with YAML frontmatter:

- `title`, `description` — primary RAG retrieval fields
- `tags` — hierarchical tags (`domain/*`, `concept/*`, `kb/*`)
- `aliases` — alternate names / search synonyms
- `related` — sibling topic slugs
- `status`, `lang`, `type`, `slug`, `updated`

Regenerate indexes after metadata edits:

```bash
uv run python scripts/kb_apply_obsidian_frontmatter.py
uv run python scripts/kb_validate_links.py
```
