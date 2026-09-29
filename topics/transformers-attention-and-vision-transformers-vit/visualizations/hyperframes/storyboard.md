---
title: "Сториборд клипа: ViT — патчи, self-attention и CLS"
description: "Сториборд и команды сборки HyperFrames-клипа vit_patches_attention: картинка → патчи и токены с CLS, scaled dot-product attention для строки CLS, энкодер из L блоков и голова классификации."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/attention
  - concept/vit
aliases:
  - ViT storyboard
  - vit_patches_attention
related:
  - transformers-attention-and-vision-transformers-vit
  - embeddings-and-embedding-matrix
status: notes
lang: ru
type: note
slug: transformers-attention-and-vision-transformers-vit/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# ViT: патчи, self-attention и CLS-токен — сториборд

Клип `assets/visualizations/vit_patches_attention.{mp4,gif}`, 44 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «2. Scaled Dot-Product Attention и Q/K/V», «Multi-Head Attention»,
«6. Архитектура Vision Transformer (ViT)», «7.1 Классификация»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как картинка становится последовательностью токенов? | Стилизованная «картинка» 4×4 (небо / трава / оранжевый круг), прорастает сетка P×P, патчи разъезжаются, затем каждый сжимается и улетает в ряд из 16 токенов; на позицию 0 въезжает оранжевый CLS; под рядом появляются индексы 0…16 и «+ E_pos». Карточки «Патчи» N = (H/P)·(W/P) (224×224, P = 16 → 196), «Линейная проекция» патч P×P×C → вектор P²·C, xᵢ = E·pᵢ ∈ ℝᴰ, E ∈ ℝ^(P²C×D), «CLS + позиции» [x_CLS; x₁; …; x_N] + E_pos | режем H×W на патчи P×P → N = (H/P)(W/P) токенов · патч → вектор P²·C → проекция в D · CLS-токен + позиционные эмбеддинги (1D по индексу или 2D по координатам) |
| 2 | 14–30 с | Как токены смотрят друг на друга (self-attention)? | X → Q (синий), K (фиолетовый), V (зелёный) через X·W_Q, X·W_K, X·W_V; ряд из 17 токенов (CLS + 16), запрос CLS подсвечен; столбики весов A_CLS,j (сумма = 1.00, максимум на 4 патчах «оранжевого круга» 6, 7, 10, 11); стрелки от трёх самых больших столбиков (7, 11, 6) в блок Y_CLS = Σⱼ A_CLS,j · Vⱼ. Карточки «Scaled dot-product attention» (Q = XW_Q, K = XW_K, V = XW_V; S = QKᵀ/√d_k; A = softmax(S), Σⱼ Aᵢⱼ = 1; Y = AV) и «Multi-head» (h голов меньшей размерности → concat → W_O) | Q «что я ищу», K «что у меня есть», V «что сообщу, если меня выберут» · Sᵢⱼ = QᵢKⱼ/√d_k, softmax по строке, сумма = 1 · Yᵢ = Σⱼ Aᵢⱼ Vⱼ — взвешенная сумма значений; multi-head |
| 3 | 30–44 с | Откуда берётся класс? | Вход: ряд из 17 токенов снизу; блок энкодера строится снизу вверх: LayerNorm → Multi-Head Self-Attention → ⊕ (residual), LayerNorm → MLP → ⊕ (residual), скобка «× L блоков»; выход z_CLS, z₁ … z_N сверху; копия CLS вытягивается вправо в «Linear W_cls → softmax» → столбики «кот 0.86 / собака 0.09 / машина 0.05 (иллюстративно)». Карточки «Блок энкодера» и «Голова классификации» ŷ = softmax(W_cls · z_CLS), альтернатива без CLS — mean-pooling | энкодер: L одинаковых блоков — MHSA, MLP, residual, LayerNorm · для классификации берём обновлённый CLS: ŷ = softmax(W_cls z_CLS) · **ключевая идея**: ViT = патчи как токены + self-attention между всеми патчами; CLS собирает глобальную информацию и идёт в голову классификации |

Числа: N = 196 при 224×224 и P = 16 (14·14). Веса внимания A_CLS,j в сцене 2 — иллюстративный, детерминированный набор
[0.06, 0.02, 0.03, 0.03, 0.02, 0.03, 0.15, 0.17, 0.03, 0.03, 0.14, 0.16, 0.03, 0.03, 0.02, 0.03, 0.02] (сумма 1.00, j = 0 — CLS).
Вероятности классов в сцене 3 (0.86 / 0.09 / 0.05) — иллюстративные, помечены на экране.

## Сборка

```bash
cd topics/transformers-attention-and-vision-transformers-vit/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,6.5,12,16,21,27,32,37,43
npx -y hyperframes@0.8.81 render --quality looks --output renders/vit_patches_attention.mp4
ffmpeg -i renders/vit_patches_attention.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/vit_patches_attention.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/vit_patches_attention.mp4 -o ../../assets/visualizations/vit_patches_attention.gif
```
