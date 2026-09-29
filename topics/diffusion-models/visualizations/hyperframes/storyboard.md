---
title: "Сториборд клипа: Diffusion — прямой и обратный процесс"
description: "Сториборд и команды сборки HyperFrames-клипа diffusion-forward-reverse-process: прямой процесс зашумления x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε, обратный процесс с сетью ε_θ и loss DDPM, ускорение через DDIM и latent diffusion."
tags:
  - kb/note
  - kb/visualization
  - domain/generative
  - concept/diffusion
  - concept/latent-diffusion
aliases:
  - diffusion-forward-reverse-process
related:
  - diffusion-models
  - variational-autoencoders-vaes
status: notes
lang: ru
type: note
slug: diffusion-models/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Diffusion: прямой и обратный процесс — сториборд

Клип `assets/visualizations/diffusion-forward-reverse-process.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Математические основы» → «Прямой процесс диффузии», «Обратный процесс диффузии»,
«Целевая функция обучения»; «Процесс обучения» → «Алгоритм обучения», «Ключевые решения в дизайне»; «Сэмплирование и генерация»;
«Ключевые варианты и расширения» → «DDIM», «Latent Diffusion Models»; «Текущее состояние» → «Улучшения качества и скорости»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как картинка превращается в шум? | «Картинка» 8×8 (иллюстративно) зашумляется по замкнутой форме при t = 0, 100, 250, 500, 1000 (ᾱ_t = 1.00, 0.90, 0.52, 0.08, ≈0); маркер на оси t; столбики плотности q(x_t) по одной координате: два «горба» → колокол N(0, I); карточки «Один шаг прямого процесса» q(x_t \| x_{t−1}) = N(x_t; √(1−β_t)·x_{t−1}, β_t·I) и «Замкнутая форма» x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε, α_t = 1 − β_t, ᾱ_t = ∏ α_s | прямой процесс: к x₀ шаг за шагом добавляем гауссовский шум по расписанию β_t · замкнутая форма: любой шаг t из x₀ одним сэмплом ε · при t → T ᾱ_t → 0, x_T ≈ ε ~ N(0, I) |
| 2 | 14–29 с | Как научиться убирать шум? | Пайплайн x_t (t = 500) → ε_θ(x_t, t) (U-Net + time embedding) → предсказанный шум; ниже истинный ε; MSE 0.97 → 0.08 (иллюстративно), предсказание сходится к ε; карточки «Упрощённый loss DDPM» L_simple = E_{t,x₀,ε}‖ε − ε_θ(x_t, t)‖² и «Обратный процесс» p_θ(x_{t−1} \| x_t) = N(x_{t−1}; μ_θ, Σ_θ), x̂₀ = (x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t; цепочка сэмплирования x_T ~ N(0, I) → x_500 → x_250 → x_100 → x₀ | сеть ε_θ получает x_t и t, предсказывает добавленный шум · учим MSE между ε и ε_θ — истинный ε известен · зная ε_θ, восстанавливаем x̂₀ и делаем шаг назад p_θ(x_{t−1} \| x_t) · генерация: из x_T ~ N(0, I) за T = 1000 шагов DDPM до x₀ |
| 3 | 29–42 с | Как ускорить: DDIM и latent diffusion? | Ось t: DDPM — все 1000 шагов (плотная линия), DDIM — 50 точек (step = T/50 = 20); карточка DDIM x_{t−1} = √ᾱ_{t−1}·x̂₀ + √(1−ᾱ_{t−1})·ε_θ(x_t, t), η = 0; пайплайн latent diffusion: изображение → 1·VAE encoder → 2·diffusion в латенте z 4×4 (шумим/расшумляем) → 3·VAE decoder → пиксели; карточка Latent Diffusion (Rombach et al., 2022) | DDPM идёт все T = 1000 шагов, DDIM — подпоследовательность 20–50 шагов, при η = 0 детерминированно · latent diffusion: VAE encoder → диффузия в латенте → VAE decoder · **ключевая идея**: diffusion = учимся убирать шум; forward добавляет шум по β_t, ε_θ его предсказывает; DDIM и латент делают генерацию быстрой |

Числа и формулы (все из README темы): q(x_t | x_{t−1}) = N(x_t; √(1−β_t)·x_{t−1}, β_t·I); q(x_t | x₀) = N(√ᾱ_t·x₀, (1−ᾱ_t)·I), x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε;
α_t = 1 − β_t, ᾱ_t = ∏_{s≤t} α_s; linear-расписание β_t от 0.0001 до 0.02; T = 1000 (DDPM); p_θ(x_{t−1} | x_t) = N(μ_θ, Σ_θ); L_simple = E‖ε − ε_θ(x_t, t)‖², t ~ Uniform(1, T);
x̂₀ = (x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t; DDIM: x_{t−1} = √ᾱ_{t−1}·x̂₀ + √(1−ᾱ_{t−1})·ε_θ(x_t, t), η = 0, 20–50 шагов (в коде README 50 шагов вместо 1000, step = 1000 // 50 = 20);
latent diffusion: VAE encoder → diffusion в латенте → VAE decoder (быстрее, меньше памяти). Значения ᾱ_t при t = 100/250/500/1000 (0.90/0.52/0.08/≈0) посчитаны по linear-расписанию README;
«картинка» 8×8, латент 4×4, фиксированный сэмпл ε и плотность по одной координате — иллюстративны (константы в скриптах сцен, без Math.random).

## Сборка

```bash
cd topics/diffusion-models/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 2.5,7,12,17,22,27.5,32,36,40.5
npx -y hyperframes@0.8.81 render --quality looks --output renders/diffusion-forward-reverse-process.mp4
ffmpeg -i renders/diffusion-forward-reverse-process.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/diffusion-forward-reverse-process.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/diffusion-forward-reverse-process.mp4 -o ../../assets/visualizations/diffusion-forward-reverse-process.gif
```
