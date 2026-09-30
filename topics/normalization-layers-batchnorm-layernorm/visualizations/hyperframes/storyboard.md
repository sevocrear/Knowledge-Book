---
title: "Сториборд клипа: BatchNorm vs LayerNorm — по какой оси нормализуем"
description: "Сториборд и команды сборки HyperFrames-клипа batchnorm-vs-layernorm-axes: тензор N × C как сетка и статистики BatchNorm по столбцу канала, LayerNorm по строке объекта независимо от батча, сравнение и RMSNorm в LLM."
tags:
  - kb/note
  - kb/visualization
  - domain/dl-foundations
  - concept/normalization
  - concept/batchnorm
aliases:
  - batchnorm-vs-layernorm-axes
related:
  - normalization-layers-batchnorm-layernorm
  - transformers-attention-and-vision-transformers-vit
status: notes
lang: ru
type: note
slug: normalization-layers-batchnorm-layernorm/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# BatchNorm vs LayerNorm: по какой оси нормализуем — сториборд

Клип `assets/visualizations/batchnorm-vs-layernorm-axes.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 4 подписи внизу (в последней — 3), в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «2. Batch Normalization», «3. Layer Normalization»,
«4. BatchNorm vs LayerNorm: сравнение», «5. Связь с обучением»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–15 с | По какой оси считает статистики BatchNorm? | Сетка N × C входа X ∈ ℝ^{5×3×H×W}: строки n = 1…5, столбцы c = 1…3 (R, G, B), в клетке точки = H × W значений; оранжевая рамка обводит столбец канала, под ним μ_B, σ²_B, затем переезжает на c = 2 и c = 3; карточки «Статистики канала c» (μ_B, σ²_B, m = N·H·W), «Нормализация и γ, β» (x̂, y = γ_c·x̂ + β_c, 2·C параметров), «Обучение vs инференс» (running mean/var) | вход (N, C, H, W) = (5, 3, H, W) · канал c и все его m = N·H·W значений → μ_B, σ²_B · x̂ = (x − μ_B)/√(σ²_B + ε), y = γ_c·x̂ + β_c, всего 2·C · running mean/var → инференс при batch size = 1 |
| 2 | 15–30 с | Почему LayerNorm не зависит от батча? | Сетка N × d: объекты n = 1…4, признаки h₁…h₆ (d = 6, иллюстративно). Сначала синяя рамка по столбцу (BN), строки 3–4 гаснут (N = 2) и красная подпись «статистики шумные»; затем оранжевая рамка по строке объекта с μ_L, σ²_L справа, едет по всем 4 строкам; карточки «Статистики объекта» (μ_L, σ²_L по d), «Нормализация и γ, β» (ĥ_j, y_j = γ_j·ĥ_j + β_j), «Где используется» | маленький батч → μ_B, σ²_B шумные · LayerNorm берёт один объект h ∈ ℝ^d, μ_L, σ²_L по строке · ĥ_j, y_j = γ_j·ĥ_j + β_j — γ, β на каждую из d компонент · не зависит от размера и состава батча, batch size = 1 работает так же |
| 3 | 30–42 с | Что выбрать: BatchNorm или LayerNorm? | Слева две мини-сетки: BatchNorm (рамка по столбцу, чипы Conv → BN → ReLU) и LayerNorm (рамка по строке, чипы x → LN → Attention / MLP); справа таблица «ось / batch size / где / γ, β / бонус» и карточка «В современных LLM — RMSNorm» | CNN (ResNet, EfficientNet): Conv → BN → ReLU, шум батча ≈ регуляризация · Transformer (BERT, GPT, ViT): LayerNorm, в LLM чаще RMSNorm · **ключевая идея**: оба слоя приводят активации к нулевому среднему и единичной дисперсии, затем γ·x̂ + β; разница — ось: BN по батчу, LN по признакам |

Числа и формулы (все из README): N = 5, C = 3 (RGB), m = N·H·W, μ_B, σ²_B, x̂ = (x − μ_B)/√(σ²_B + ε), y = γ·x̂ + β,
2·C параметров BatchNorm; μ_L, σ²_L по d признакам, γ_j, β_j размерности d; running mean/var на инференсе,
batch size = 1; RMSNorm в LLaMA, Mistral, Qwen, Gemma — без вычитания среднего, деление на RMS и масштаб γ.
Иллюстративные величины помечены на экране: d = 6 признаков и N = 2 «маленький батч» в сцене 2; яркость клеток сетки — условная.

## Сборка

```bash
cd topics/normalization-layers-batchnorm-layernorm/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.5,10.5,14,18,22,25.5,29,33,37,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/batchnorm-vs-layernorm-axes.mp4
ffmpeg -i renders/batchnorm-vs-layernorm-axes.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/batchnorm-vs-layernorm-axes.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/batchnorm-vs-layernorm-axes.mp4 -o ../../assets/visualizations/batchnorm-vs-layernorm-axes.gif
```
