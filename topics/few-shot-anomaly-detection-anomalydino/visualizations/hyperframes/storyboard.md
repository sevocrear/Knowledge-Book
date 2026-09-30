---
title: "Сториборд клипа: AnomalyDINO — патч-фичи DINOv2 и memory bank"
description: "Сториборд и команды сборки HyperFrames-клипа anomalydino-patch-nn-memory-bank: memory bank из патч-фич DINOv2 без обучения, nearest-neighbour косинусное расстояние и anomaly map с маской, image score как среднее top-1 % расстояний и порог OK / defect."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/anomaly-detection
  - concept/few-shot
aliases:
  - anomalydino-patch-nn-memory-bank
  - AnomalyDINO storyboard
related:
  - few-shot-anomaly-detection-anomalydino
  - dinov3-self-supervised-vision-transformer-and-2d-rope
status: notes
lang: ru
type: note
slug: few-shot-anomaly-detection-anomalydino/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# AnomalyDINO: патч-фичи DINOv2 и memory bank без обучения — сториборд

Клип `assets/visualizations/anomalydino-patch-nn-memory-bank.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы 1 «Краткий абстракт», 2 «Постановка задачи», 3 «Идея AnomalyDINO в одном абзаце»,
4 «Pipeline: memory bank, masking, сравнение», 5 «Как строится маска аномалии», 6 «Метрики и бенчмарки»).
Все картинки — собственные сетки клеток и фигуры, без фотографий.

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Как запомнить «норму» без обучения? | Два эталонных кадра как сетки 6×6 патчей → блок «DINOv2 ViT-S/14 (frozen)» → бирюзовые патч-векторы летят в пунктирный «memory bank M»; карточки «Экстрактор f: f(x) = (p₁, …, pₙ)» и «Memory bank M = ⋃ᵢ {pⱼ⁽ⁱ⁾}» | k = 1…16 эталонов, никакого обучения · DINOv2 ViT-S/14 режет кадр на патчи 14 × 14 px и выдаёт вектор pⱼ · патч-векторы всех эталонов (плюс повороты-аугментации) складываем в memory bank M |
| 2 | 13–29 с | Как найти дефект на новом кадре? | Тестовый кадр 16×16 патчей с «пятном»; патч A (норма) и патч B (дефект) тянут линии к банку: d_NN = 0.05 и 0.58 (иллюстративно); клетки решётки d_NN красно-оранжевые → размытое пятно (anomaly map) → красный контур маски; карточки «d(x, y) = 1 − ⟨x, y⟩ / (‖x‖·‖y‖)», «d_NN(p; M) = min_{p_ref ∈ M} d(p, p_ref)», «Anomaly map: bilinear upsampling → Gaussian blur σ = 4.0 → порог» | для каждого патча ищем ближайший вектор в M по косинусному расстоянию · A: сосед близко, d_NN мало; B: ни на что не похож, d_NN велико · решётка d_NN → bilinear upsampling до H × W → Gaussian blur (σ = 4.0) · порог с валидации (F1-max / PRO) → бинарная маска дефекта |
| 3 | 29–42 с | Кадр целиком: OK или defect? | График отсортированных d_NN (первые 64 из 1024 патчей, иллюстративно) для кадра с дефектом и нормального; полоса top-1 % = 10 патчей; пунктир s = 0.40 / 0.13, порог τ; плашки «s > τ → defect», «s < τ → OK»; карточки «s(x) = mean of top-1 % d_NN» и «MVTec-AD, 1-shot: WinCLIP+ ~93.1 % → AnomalyDINO-S (672 px) 96.6 %; ~60 ms / кадр (ViT-S, 448 px)» | image score s — среднее по 1 % самых больших d_NN, хвост устойчивее max · s выше порога → defect, ниже → OK; порог калибруют на валидации · **ключевая идея**: без обучения — патчи DINOv2 + ближайший сосед в банке нормы; MVTec-AD 1-shot 96.6 % AUROC, ~60 мс на кадр |

## Числа и формулы (из README темы)

- Патч 14 × 14 px, backbone DINOv2 ViT-S/14, вход 448 или 672 px (сторона кратна 14) — разделы 3, 4.1.
- k ∈ {1, 2, 4, 8, 16} эталонных изображений; аугментации (повороты) расширяют M в few-shot режиме — разделы 2, 4.1.
- M = ⋃ᵢ {pⱼ⁽ⁱ⁾ : j ∈ [n]}; d(x, y) = 1 − ⟨x, y⟩ / (‖x‖‖y‖); d_NN(p; M) = min_{p_ref ∈ M} d(p, p_ref) — разделы 3, 4.1, 4.3.
- Image score s = mean of top-1 % d_NN (хвост на 99 %-квантили, устойчивее max) — разделы 3, 4.3.
- Anomaly map: bilinear upsampling до H × W → Gaussian smoothing σ = 4.0 → порог по val (F1-max / PRO) — раздел 5.
- MVTec-AD, 1-shot, image AUROC: WinCLIP+ ~93.1 % → AnomalyDINO-S (672) 96.6 %; inference ~60 ms / image (ViT-S, 448) — раздел 6.
- 1024 патчей = 32 × 32 при 448 px (448 / 14 = 32) → top-1 % = 10 патчей (арифметика из чисел README).
- Иллюстративные (помечены на экране): d_NN = 0.05 / 0.58 для патчей A / B; профили d_NN: norm(i) = 0.05 + 0.09·e^(−i/40),
  defect(i) = norm(i) + 0.48·e^(−i/7), их средние по top-10 — 0.13 и 0.40; порог τ = 0.25; тестовый кадр 16 × 16 патчей.

## Сборка

```bash
cd topics/few-shot-anomaly-detection-anomalydino/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.2,11,15,20,24.5,27.5,32,36,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/anomalydino-patch-nn-memory-bank.mp4
ffmpeg -i renders/anomalydino-patch-nn-memory-bank.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/anomalydino-patch-nn-memory-bank.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/anomalydino-patch-nn-memory-bank.mp4 -o ../../assets/visualizations/anomalydino-patch-nn-memory-bank.gif
```
