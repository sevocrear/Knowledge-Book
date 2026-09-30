---
title: "Сториборд клипа: COCO AP, mIoU, Top-1 и F1"
description: "Сториборд и команды сборки HyperFrames-клипа coco-ap-miou-top1-f1: IoU-матчинг, PR-кривая и усреднение COCO AP по порогам IoU 0.50:0.95; per-class IoU и mIoU против pixel accuracy; матрица ошибок, Top-1/Top-5 и macro vs micro F1."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/metrics
  - concept/object-detection
aliases:
  - coco-ap-miou-top1-f1
related:
  - sota-metrics-for-detection-segmentation-multiclass-classification
  - roc-curve-and-roc-auc
status: notes
lang: ru
type: note
slug: sota-metrics-for-detection-segmentation-multiclass-classification/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# SOTA-метрики: COCO AP, mIoU, Top-1 и F1 — сториборд

Клип `assets/visualizations/coco-ap-miou-top1-f1.{mp4,gif}`, 44 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Два слоя “метрики”», «COCO-style AP», «Семантическая сегментация: mIoU»,
«Top-1 / Top-k accuracy», «Macro/micro/weighted F1 и balanced accuracy», «Практический cheat sheet»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–15 с | Как считают COCO AP в детекции? | «Картинка» с GT-боксом (зелёный) и предсказанием (синий), пересечение подсвечено, «IoU ≈ 0.62 ≥ τ = 0.5 → match»; кривая precision–recall с закрашенной площадью AP_τ; 10 столбиков AP_τ для τ = 0.50…0.95 и пунктир среднего AP@[.50:.95] = 0.50; карточки: IoU и матчинг, AP_τ = ∫₀¹ p(r) dr, COCO AP = 1/10 · Σ AP_τ | бокс совпал с GT, если IoU ≥ τ · ранжируем по confidence → PR-кривая, AP_τ — площадь · COCO AP — среднее по 10 порогам 0.50…0.95; AP50 мягче, AP@[.50:.95] — стандарт SOTA |
| 2 | 15–29 с | Почему mIoU, а не pixel accuracy? | Маска 20 × 10 = 200 пикселей: фон 188, объект 12 (оранжевая заливка); предсказание сдвинуто (пунктир) → клетки TP (6, зелёные), FP (6, синие), FN (6, красные); строки pixel accuracy = 188/200 = 0.94, IoU_объект = 6/18 = 0.33, IoU_фон = 182/194 = 0.94, mIoU = 0.64; карточки IoU_c = TP_c/(TP_c+FP_c+FN_c) и mIoU = 1/C · Σ IoU_c; столбики 0.94 / 0.94 / 0.33 / 0.64 | класс на каждый пиксель; фон 188 из 200 · TP = 6, FP = 6, FN = 6 → IoU_объект = 0.33 · pixel accuracy = 0.94 обманчиво высока, mIoU = 0.64 показывает промах |
| 3 | 29–44 с | Чем мерить мультиклассовую классификацию? | Ранжированный softmax для одной картинки (GT «собака» на 3-м месте): Top-1 ✗, Top-5 ✓; матрица ошибок 3×3 (A/B/C: 100/20/10 примеров, диагональ 90/14/5), per-class F1 0.91/0.68/0.45; micro-F1 = 0.84 vs macro-F1 = 0.68; карточки Top-1/Top-5 и micro/macro-F1 | Top-1 — класс на первом месте, Top-5 — в первой пятёрке · матрица ошибок: диагональ — верные ответы; long-tail · micro-F1 = 0.84 тянет частый класс, macro-F1 = 0.68 видит слабый · **ключевая идея**: метрика + протокол — COCO AP@[.50:.95], mIoU, Top-1, при дисбалансе macro-F1 и матрица ошибок |

Числа из README: COCO AP = (1/10) · Σ AP_τ по τ ∈ {0.50, 0.55, …, 0.95}; AP_τ = ∫₀¹ p(r) dr; IoU = |B ∩ B̂| / |B ∪ B̂|; IoU_c = TP_c / (TP_c + FP_c + FN_c); mIoU = (1/C) · Σ IoU_c; Top-1 / Top-5 (ImageNet); macro-F1 — средний F1 по классам, micro-F1 — TP/FP/FN глобально (доминируют частые классы); рекомендация «macro-F1 + confusion matrix» при long-tail.

Иллюстративные величины (помечены на экране «иллюстративно»): боксы с IoU ≈ 0.62 (пересечение 19 575 / объединение 31 625), PR-кривая и AP_τ = 0.72, 0.70, 0.67, 0.63, 0.58, 0.52, 0.45, 0.36, 0.25, 0.12 (среднее 0.50); маска 20 × 10 с объектом 4 × 3 и сдвигом предсказания на 2 клетки (TP/FP/FN = 6/6/6 → IoU_объект = 0.333, IoU_фон = 182/194 = 0.938, pixel accuracy = 188/200 = 0.94, mIoU = 0.636); softmax 0.41/0.33/0.15/0.07/0.04; матрица ошибок [[90, 5, 5], [4, 14, 2], [3, 2, 5]] → F1 по классам 0.914 / 0.683 / 0.455, macro-F1 = 0.684, micro-F1 = 109/130 = 0.838.

## Сборка

```bash
cd topics/sota-metrics-for-detection-segmentation-multiclass-classification/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,8,13,18,23,27.5,32,36.5,41,43.5
npx -y hyperframes@0.8.81 render --quality looks --output renders/coco-ap-miou-top1-f1.mp4
ffmpeg -i renders/coco-ap-miou-top1-f1.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/coco-ap-miou-top1-f1.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/coco-ap-miou-top1-f1.mp4 -o ../../assets/visualizations/coco-ap-miou-top1-f1.gif
```
