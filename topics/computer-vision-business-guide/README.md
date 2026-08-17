---
title: "Компьютерное зрение: вводное руководство для бизнеса"
description: "Бизнес-книга (~30+ стр.) на понятном русском: что такое CV, как работает, задачи, отрасли (промышленность, офис, город, retail), внедрение, этика. EPUB + HTML."
tags:
  - kb/topic
  - domain/cv
  - domain/mlops
  - concept/classification
  - concept/object-detection
  - concept/segmentation
aliases:
  - Computer Vision Business Guide
  - CV для бизнеса
  - вводное руководство по компьютерному зрению
related:
  - convolutions-and-parameters-in-cnn
  - non-maximum-suppression-nms
  - sota-metrics-for-detection-segmentation-multiclass-classification
  - few-shot-anomaly-detection-anomalydino
  - action-recognition-and-object-tracking-metrics
status: canonical
lang: ru
type: topic
slug: computer-vision-business-guide
updated: 2026-08-17
---
# Компьютерное зрение: вводное руководство для бизнеса

> **Как объяснить пятилетнему ребёнку:** представь, что у завода или магазина есть очень внимательные глаза-камеры. Они не устают и запоминают, как выглядит «хорошо» и «плохо», — и сразу зовут взрослых, если что-то не так.

## Table of Contents

- [О книге](#о-книге)
- [Скачать](#скачать)
- [Содержание руководства (20 глав)](#содержание-руководства-20-глав)
- [Как собрать локально](#как-собрать-локально)
- [References](#references)

## О книге

Это **бизнес-издание** Knowledge Book — ~30+ страниц понятного русского текста без формул и кода. Руководство предназначено для менеджеров, HR, операций, закупок, безопасности и всех, кто принимает решения о проектах с камерами и ИИ.

**Что внутри:**

- определение компьютерного зрения и принцип «от пикселей к решению»;
- основные задачи: классификация, детекция, сегментация, OCR, трекинг;
- архитектура системы CV и роль данных;
- ограничения и реалистичные ожидания;
- сценарии: **промышленность**, **офис**, **улица/город**, **retail**, **медицина**, **логистика**, **безопасность**, **AgriTech**;
- дорожная карта внедрения, этика, глоссарий, чек-лист для руководителя;
- **16 сгенерированных иллюстраций** в едином стиле.

## Скачать

Готовые артефакты лежат в каталоге `dist/`:

| Формат | Файл | Назначение |
| --- | --- | --- |
| **E-book (EPUB)** | [`computer-vision-business-guide.epub`](./dist/computer-vision-business-guide.epub) | Читалки (Apple Books, Calibre, PocketBook) |
| **HTML + assets (ZIP)** | [`computer-vision-business-guide-html.zip`](./dist/computer-vision-business-guide-html.zip) | Верстка + все изображения для офлайн-просмотра |
| **HTML (онлайн)** | [`index.html`](./dist/index.html) | Открыть в браузере (рядом папка `assets/`) |

> После клонирования репозитория используйте прямые пути выше. В GitHub-PR ссылки на raw-файлы будут в описании pull request.

## Содержание руководства (20 глав)

1. Введение: зачем бизнесу компьютерное зрение  
2. Что такое компьютерное зрение  
3. Как машина «видит»  
4. Основные задачи CV  
5. Архитектура системы  
6. Данные, обучение и точность  
7. Ограничения и риски  
8. Промышленность  
9. Офисы  
10. Улица и город  
11. Розница  
12. Медицина  
13. Логистика  
14. Безопасность  
15. Сельское хозяйство  
16. Внедрение: дорожная карта  
17. Этика и приватность  
18. Будущее CV  
19. Глоссарий  
20. Чек-лист и итоги  

Исходники глав: `book/chapters/` (01–20, markdown).

## Как собрать локально

```bash
uv sync --group book
uv run python topics/computer-vision-business-guide/scripts/01_generate_illustrations.py
uv run python topics/computer-vision-business-guide/scripts/02_build_book.py
uv run pytest topics/computer-vision-business-guide/tests/
```

## References

### Технические топики Knowledge Book (углубление)

- [Свёртки и CNN](./../convolutions-and-parameters-in-cnn/README.md)
- [Non-Maximum Suppression (NMS)](./../non-maximum-suppression-nms/README.md)
- [Метрики detection / segmentation](./../sota-metrics-for-detection-segmentation-multiclass-classification/README.md)
- [Few-shot anomaly detection](./../few-shot-anomaly-detection-anomalydino/README.md)
- [Transformers и ViT](./../transformers-attention-and-vision-transformers-vit/README.md)
- [MOC: Computer Vision](../../docs/mocs/computer-vision.md)

### Внешние материалы

- [OpenCV — обзор компьютерного зрения](https://opencv.org/about/)
- [EU AI Act — официальный текст](https://artificialintelligenceact.eu/)
