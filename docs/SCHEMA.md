---
title: Схема Obsidian frontmatter
description: Обязательные поля YAML frontmatter для заметок книги, которые используют Obsidian и RAG.
tags:
  - kb/schema
  - kb/docs
status: canonical
lang: ru
type: schema
updated: 2026-08-19
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
