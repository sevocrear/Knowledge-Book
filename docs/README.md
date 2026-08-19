---
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
updated: 2026-08-19
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

- [MOC: математика и основы ML](./mocs/foundations.md) — Вероятность, метрики, классический ML и базовые строительные блоки.
- [MOC: генеративные модели](./mocs/generative-models.md) — VAE, GAN и diffusion — три основных семейства генеративных моделей.
- [MOC: NLP, LLM и RAG](./mocs/nlp-llm.md) — Токенизация, эмбеддинги, transformers, LoRA, RAG и code agents.
- [MOC: компьютерное зрение](./mocs/computer-vision.md) — CNN, detection/segmentation, metric learning, SSL и tracking metrics.
- [MOC: робототехника и Embodied AI](./mocs/robotics-embodied.md) — Deep RL, VLA и vision-based обучение роботов.

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
