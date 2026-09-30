---
title: "Сториборд клипа: VAE — encoder, репараметризация и ELBO"
description: "Сториборд и команды сборки HyperFrames-клипа vae-encoder-decoder-reparameterization: encoder выдаёт распределение (μ, σ), reparameterization trick z = μ + σ·ε, ELBO и генерация из prior N(0, I) с интерполяцией."
tags:
  - kb/note
  - kb/visualization
  - domain/generative
  - concept/vae
  - concept/elbo
aliases:
  - VAE storyboard
  - vae-encoder-decoder-reparameterization
related:
  - variational-autoencoders-vaes
  - diffusion-models
status: notes
lang: ru
type: note
slug: variational-autoencoders-vaes/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# VAE: encoder → репараметризация → ELBO — сториборд

Клип `assets/visualizations/vae-encoder-decoder-reparameterization.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Основная идея и интуиция», «Математические основы», «Архитектура и компоненты», «Процесс обучения», «Ключевые выводы»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–12 с | Что выдаёт encoder: точку или распределение? | Вход x (пиксельный тайл) → Encoder → плоскость латентного пространства (z₁, z₂); сначала одна оранжевая точка «autoencoder», затем она превращается в эллипс q(z\|x) с центром μ и полуосями σ₁, σ₂; зелёные сэмплы z внутри; карточки «Encoder» (μ, log σ²) и «Латентное распределение» q(z\|x) = N(μ, σ²·I), prior N(0, I) | autoencoder кодирует x в одну точку, VAE — в распределение · encoder выдаёт μ и log σ² · сэмплы z ~ N(μ, σ²) заполняют облако, пространство непрерывно |
| 2 | 12–27 с | Как пропустить градиент через сэмплирование? | Схема x → Encoder → μ, σ; блок ε ~ N(0, I); узел z (z = μ + σ ⊙ ε) → Decoder → x̂; затем красные стрелки обратного прохода (x̂ → decoder → z → μ, σ → encoder), ε без градиента; карточки: трюк репараметризации, Decoder, Backpropagation | напрямую сэмплировать нельзя · выносим случайность в ε, z = μ + σ ⊙ ε · decoder строит x̂ · градиенты идут через детерминированные μ_φ, σ_φ |
| 3 | 27–43 с | Чему учим VAE и как из него генерировать? | Латентная плоскость с пятью облаками q(z\|x) для разных x и пунктирным prior N(0, I): облака сначала далеко друг от друга («дыра между кодами»), KL стягивает их к prior; затем z ~ N(0, I) → decoder, интерполяция z между двумя кодами и пять плиток x̂ (круг → квадрат, иллюстративно); карточки: L = L_recon + β·L_KL, KL-член, генерация, интерполяция | учим reconstruction + KL · KL стягивает облака к N(0, I) · генерация из prior, интерполяция плавная · **ключевая идея**: encoder даёт распределение + репараметризация + ELBO; KL держит латент около N(0, I), поэтому z ~ N(0, I) декодируется в новые данные |

Формулы и числа (из README): z = μ_φ(x) + σ_φ(x) ⊙ ε, ε ~ N(0, I); σ_φ = exp(½ log σ²_φ); q_φ(z\|x) = N(μ_φ(x), σ²_φ(x)I); prior p(z) = N(0, I); L = L_recon + β·L_KL; L_KL = ½ Σ [σ² + μ² − 1 − log σ²]; β = 1 — обычный VAE; L_recon — BCE или MSE. Размерность латента 2D, положения облаков, тайлы x / x̂ и плитки интерполяции (круг → квадрат) — иллюстративные, помечены на экране.

## Сборка

```bash
cd topics/variational-autoencoders-vaes/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3.5,7,10.5,16,21,25,29,32,35,38,42
npx -y hyperframes@0.8.81 render --quality looks --output renders/vae-encoder-decoder-reparameterization.mp4
ffmpeg -i renders/vae-encoder-decoder-reparameterization.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/vae-encoder-decoder-reparameterization.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/vae-encoder-decoder-reparameterization.mp4 -o ../../assets/visualizations/vae-encoder-decoder-reparameterization.gif
```
