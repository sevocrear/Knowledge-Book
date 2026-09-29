---
title: "Сториборд клипа: Cross Entropy vs Focal Loss"
description: "Сториборд и команды сборки HyperFrames-клипа cross-entropy-vs-focal-loss: логиты → softmax → −log p, кривые Focal Loss для γ = 0, 1, 2, 5 и α, когда выбирать CE, а когда Focal (RetinaNet, dense prediction)."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/loss
  - concept/classification
aliases:
  - cross-entropy-vs-focal-loss
related:
  - classification-losses-cross-entropy-focal-loss
  - detection-segmentation-3d-losses
status: notes
lang: ru
type: note
slug: classification-losses-cross-entropy-focal-loss/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Cross Entropy vs Focal Loss — сториборд

Клип `assets/visualizations/cross-entropy-vs-focal-loss.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «2. Cross Entropy», «3. Focal Loss», «4. Cross Entropy vs Focal Loss: когда что использовать»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как считается cross-entropy? | Пайплайн «логиты z_k → softmax → p_k, Σ = 1»; карточки Softmax (p_k = exp(z_k)/Σ exp(z_j), пример z = (2.0, 0.5, −1.0) → p = (0.786, 0.175, 0.039), иллюстративно) и Cross-entropy (L = −Σ y_k log p_k = −log p_y*); кривая −log p по p ∈ (0, 1]; точка едет от p = 0.9 к p = 0.1 с отметками −log 0.9 = 0.105, −log 0.5 = 0.693, −log 0.1 = 2.303, «p → 0: потеря → ∞» | логиты → softmax → вероятности, Σ = 1 · CE = −log p_y*: смотрим только на правильный класс · p → 1: потеря → 0; уверенная ошибка штрафуется без предела |
| 2 | 14–30 с | Что меняет Focal Loss? | Кривые (1 − p)^γ·(−log p) для γ = 0 (= CE), 1, 2, 5 при α = 1; карточка L_FL = −α(1 − p)^γ log p; таблица модулятора: p = 0.9 → 1 / 0.1 / 0.01 / 10⁻⁵, p = 0.1 → 1 / 0.9 / 0.81 / 0.59; маркеры «лёгкий p = 0.9: CE 0.105 → γ=2: 0.001» и «сложный p = 0.1: CE 2.303 → γ=2: 1.865» с точками γ = 0/2/5; карточка α | FL = CE × (1 − p)^γ, γ = 0 — обычная CE · лёгкий (p = 0.9): при γ = 2 вклад в 100 раз меньше · сложный (p = 0.1): множитель 0.81, почти как CE · α ∈ (0, 1) — вес позитивов/негативов при дисбалансе |
| 3 | 30–43 с | Когда CE, а когда Focal Loss? | Сетка 14 × 6 anchor'ов, 5 объектов (оранжевые) среди фона (иллюстративно); столбики «вклад в градиент»: CE — фон 82 % / объекты 18 %, Focal — 20 % / 80 % (иллюстративно); карточки «Cross Entropy — когда» (сбалансированные классы, дешевле, nn.CrossEntropyLoss / nn.BCEWithLogitsLoss) и «Focal Loss — когда» (дисбаланс, RetinaNet, dense prediction, редкие пиксели, fraud, medical; боксы — L1 / Smooth L1 / IoU) | почти все anchor'ы — фон, лёгкие негативы · с CE их сумма доминирует в градиенте; Focal гасит лёгкие — RetinaNet · **ключевая идея**: CE — стандарт для сбалансированной классификации; Focal — при сильном дисбалансе и dense prediction; боксы — L1 / IoU |

Числа: −log 0.9 = 0.105, −log 0.5 = 0.693, −log 0.1 = 2.303; модулятор (1 − p)^γ при p = 0.9: 0.1, 0.01, 10⁻⁵ (γ = 1, 2, 5), при p = 0.1: 0.9, 0.81, 0.59;
FL при p = 0.1 и γ = 2: 0.81 · 2.303 = 1.865, при γ = 5: 0.59 · 2.303 = 1.360; при p = 0.9 и γ = 2: 0.01 · 0.105 = 0.001.
Формулы (softmax, CE, FL), «γ ≥ 0, обычно 1–5», «α ∈ (0, 1)», «γ = 0 → обычная CE», RetinaNet, список задач и PyTorch-классы — из README темы.
Пример логитов z = (2.0, 0.5, −1.0) и доли вклада в градиент 82/18 и 20/80 — иллюстративные (помечены на экране).

## Сборка

```bash
cd topics/classification-losses-cross-entropy-focal-loss/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,8,12.5,17,22,28,33,37.5,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/cross-entropy-vs-focal-loss.mp4
ffmpeg -i renders/cross-entropy-vs-focal-loss.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/cross-entropy-vs-focal-loss.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/cross-entropy-vs-focal-loss.mp4 -o ../../assets/visualizations/cross-entropy-vs-focal-loss.gif
```
