---
title: "Сториборд клипа: serving — закон очередей, балансировщик, dynamic batching"
description: "Сториборд и команды сборки HyperFrames-клипа serving-load-balancer-batching: ρ = λ/μ на 100 и 1000 клиентов, реплики за балансировщиком с автоскейлом по очереди, dynamic batching на GPU-ноде."
tags:
  - kb/note
  - kb/visualization
  - domain/mlops
  - concept/system-design
  - concept/model-serving
aliases:
  - serving storyboard
  - serving-load-balancer-batching
related:
  - ml-system-design-for-cv-and-nlp
  - triton-inference-server-and-gpu-model-serving
status: notes
lang: ru
type: note
slug: ml-system-design-for-cv-and-nlp/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Serving: закон очередей, балансировщик, dynamic batching — сториборд

Клип `assets/visualizations/serving-load-balancer-batching.{mp4,gif}`, 44 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Утилизация и закон Литтла», «Балансировка нагрузки», «Как сервить модель на 100 и на 1000 клиентов»: «Считаем на пальцах», таблица симуляции, «Dynamic batching»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Сколько GPU нужно на 1000 клиентов? | Слева: кластер клиентов со счётчиком 100 → 1000, очередь из 10 слотов, одна GPU (μ = 50/с); шкала утилизации ρ = λ/μ со счётчиком 0.8 (зелёная) → 8.0 (красная, «×10 сверх шкалы»); очередь заливается красным, теги «drop ≈ 87 %» и «p99 → таймаут 500 мс»; ряд из 8 GPU «нужно ≈ 400/50 = 8 GPU + запас ×1.3–2». Справа: карточка «Закон очередей» ρ = λ/μ, ρ ≥ 1 → очередь растёт без предела; таблица симуляции (100 кл./1 GPU: 0.80, 0, ≈165 мс · 1000/1: 8.00, ≈0.87, таймаут · 1000/10: 0.80, 0, ≈39 мс), строки подсвечиваются | μ = 50/с, 100 × 0.4 = λ 40/с, ρ = 0.8 — успеваем, p99 ≈ 165 мс · 1000 клиентов → λ = 400/с, ρ = 8: очередь без предела, ≈ 87 % не укладываются в 500 мс · нужно ≈ 8 GPU + запас на пик — сначала закон очередей, потом фреймворк |
| 2 | 14–29 с | Что делает балансировщик? | 1000 клиентов (λ = 400/с) → «балансировщик» (L7 · round-robin) → шина → 10 реплик GPU (2 × 5), у каждой зелёная полоска ρ 0.8; точки-запросы бегут клиент → балансировщик → шина → реплика по кругу (4 конечных цикла); пунктирный блок «автоскейл смотрит на: ✓ глубину очереди, ✓ GPU util, ✗ ~~CPU util~~»; зелёная плашка «ρ = 0.8 < 1: зелёная зона — 0 дропов, p99 ≈ 39 мс». Справа: карточки «Реплики» (ρ на ноду = 400 / (10 · 50) = 0.8), «Балансировка» (round-robin / least-loaded; health-check: GPU отвечает за SLO; backpressure + таймаут), «Результат симуляции» | реплики + балансировщик: λ делится на N нод, ρ каждой = 0.8 · автоскейл по глубине очереди и GPU util, а не по CPU · итог: 1000 клиентов на 10 GPU — 0 дропов, p99 ≈ 39 мс, снова ρ < 1 |
| 3 | 29–44 с | Зачем dynamic batching? | Одна GPU-нода: запросы (λ = 250/с) → «сборщик пачек · ждать ≤ 8 мс» с 8 слотами → «GPU kernel», один запуск на B = 8 (6 конечных циклов, ядро вспыхивает); столбики items/s: без пачки ≈ 100 (оранжевый, «~59 % таймаутов») vs dynamic batching ≈ 250 (зелёный, «0 дропов»); график T(B) почти линейно (оранжевый) и B / T(B) растёт (зелёный), рисуются штрихом. Справа: «Время ядра» T(B) = T₀ + t_item · B, «Пропускная способность» B / T(B) растёт с B, «Так делают»: Triton Inference Server · TensorRT · batched ONNX Runtime | GPU любит пачки: T(B) = T₀ + t_item·B · λ = 250/с: без пачки ≈ 100 items/s и ~59 % таймаутов; с батчингом (B ≤ 8, ≤ 8 мс) ≈ 250 items/s и 0 дропов · **ключевая идея**: закон очередей ρ < 1 → реплики + балансировщик + автоскейл по очереди → dynamic batching на ноде (Triton, TensorRT, ONNX Runtime) |

Числа: μ = 50 запросов/с на GPU; 100 × 0.4/с → λ = 40/с, ρ = 0.8; 1000 × 0.4/с → λ = 400/с, ρ = 8 → ≈ 400/50 = 8 GPU + запас ×1.3–2; таблица симуляции (`scripts/01_capacity_planning.py`, таймаут 500 мс); батчинг (`scripts/02_dynamic_batching.py`, λ = 250/с): без пачки ≈ 100 items/s и ~59 % таймаутов, с max B = 8 и ожиданием до 8 мс ≈ 250 items/s и 0 дропов. График T(B) и B / T(B) — качественный (T₀ = 4, t_item = 0.5 усл. ед.).

## Сборка

```bash
cd topics/ml-system-design-for-cv-and-nlp/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,9,12,17,23,27,31,38,43
npx -y hyperframes@0.8.81 render --quality looks --output renders/serving-load-balancer-batching.mp4
ffmpeg -i renders/serving-load-balancer-batching.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/serving-load-balancer-batching.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/serving-load-balancer-batching.mp4 -o ../../assets/visualizations/serving-load-balancer-batching.gif
```
