---
title: "Сториборд клипа: NMS → Soft-NMS → NMS-free детекторы"
description: "Сториборд и команды сборки HyperFrames-клипа nms-soft-nms-nms-free: greedy NMS на примере из README (сортировка по score, IoU > τ → удалить), Soft-NMS с линейным и гауссовым понижением score и NMS-free детекторы (DETR one-to-one matching, dual-head YOLOv10 → YOLO26)."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/nms
  - concept/object-detection
aliases:
  - NMS storyboard
  - nms-soft-nms-nms-free
related:
  - non-maximum-suppression-nms
  - sota-metrics-for-detection-segmentation-multiclass-classification
status: notes
lang: ru
type: note
slug: non-maximum-suppression-nms/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# NMS → Soft-NMS → NMS-free детекторы — сториборд

Клип `assets/visualizations/nms-soft-nms-nms-free.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Что такое NMS?», «Алгоритм NMS», «Проблемы NMS в Production»,
«End-to-End Детекция без NMS», «YOLO26: Удаление NMS», «Transformer-based Детекторы: DETR и его потомки»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Зачем детектору нужен NMS? | «Кадр» с двумя объектами и четырьмя боксами из примера README (box 1 · 0.9, box 2 · 0.7, box 4 · 0.6 вокруг объекта 1; box 3 · 0.8 на объекте 2); карточка IoU; список, отсортированный по confidence; box 1 → зелёный «оставляем», подсветка пересечения IoU(1, 2) = 0.68 и IoU(1, 4) = 0.82 → box 2 и box 4 гаснут; box 3 → «оставляем»; карточка «Результат: box 1 и box 3» | на один объект приходит несколько боксов со своим confidence · greedy NMS: сортируем по score, берём лучший, удаляем соседей с IoU > τ<sub>nms</sub> · повторяем, пока список не пуст; цена — O(N²) и отдельный шаг вне сети |
| 2 | 14–29 с | Что Soft-NMS делает иначе? | Два объекта вплотную, боксы A · 0.9 и B · 0.8, пересечение IoU ≈ 0.56 > 0.5; карточка «Hard NMS: s<sub>B</sub> = 0, B удалён» — бокс B гаснет; карточка Soft-NMS (линейно s·(1 − IoU), гаусс s·exp(−IoU²/σ)) — B возвращается с подписью «0.8 → 0.35»; столбики score B: 0.8 → hard 0 / linear 0.35 / gauss 0.43 | жёсткий NMS удалит бокс B — настоящий объект потерян · Soft-NMS понижает score соседа тем сильнее, чем больше IoU · B выживает (0.35 / 0.43), но порог, σ и O(N²) остаются — постобработка вне сети |
| 3 | 29–43 с | Как детекторы обходятся без NMS? | Верх: DETR — 6 object queries, два объекта и ∅; зелёные линии one-to-one matching рисуются, остальные queries → ∅. Низ: dual-head YOLOv10 → YOLO26 — слева one-to-many голова (по три бокса на объект, нужен NMS), справа one-to-one голова (один зелёный бокс на объект, без NMS); подпись про ProgLoss. Карточки DETR (2020), YOLOv10 (2024) → YOLO26 (2026), «Что даёт NMS-free» | DETR: 100 object queries, Hungarian matching один-к-одному, остальные → ∅ · YOLOv10 → YOLO26: one-to-many голова — плотный supervision, one-to-one — до 300 финальных боксов · **ключевая идея**: модель сама выдаёт один бокс на объект → детерминированное время и экспорт без постобработки (YOLO26: на 43% быстрее на CPU) |

## Числа и откуда они

- Сцена 1 — пример из README (раздел «Реализация NMS»): боксы `[100,100,200,200]` (0.9), `[110,110,210,210]` (0.7),
  `[300,300,400,400]` (0.8), `[105,105,205,205]` (0.6), `iou_threshold = 0.5`, результат `[0, 2]`.
  IoU посчитаны по формуле README: IoU(1, 2) = 8100 / 11900 = 0.68; IoU(1, 4) = 9025 / 10975 = 0.82; IoU(1, 3) = 0.
  Сложность O(N²) и «шаг вне сети» — раздел «Проблемы NMS в Production».
- Сцена 2 — геометрия иллюстративная (помечено на экране): A = 240×350, B = 240×320, пересечение 180×320
  → IoU = 57600 / 103200 ≈ 0.56. Формулы Soft-NMS — из статьи Bodla et al., 2017 (помечено «в статье»):
  линейно 0.8 · (1 − 0.56) = 0.35; гаусс при σ = 0.5: 0.8 · exp(−0.56² / 0.5) = 0.43. В README Soft-NMS упомянут только в описании темы.
- Сцена 3 — README: DETR (2020) с 100 object queries, Hungarian matching и set prediction («Transformer-based Детекторы»);
  dual label assignment из YOLOv10 (2024), one-to-one голова YOLO26 выдаёт до 300 детекций без NMS, ProgLoss смещает вес
  от one-to-many к one-to-one, 43% ускорение на CPU относительно YOLO11 («YOLO26: Удаление NMS», «Результаты»).

## Сборка

```bash
cd topics/non-maximum-suppression-nms/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.2,11.5,17,21,27,32,37,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/nms-soft-nms-nms-free.mp4
ffmpeg -i renders/nms-soft-nms-nms-free.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/nms-soft-nms-nms-free.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/nms-soft-nms-nms-free.mp4 -o ../../assets/visualizations/nms-soft-nms-nms-free.gif
```
