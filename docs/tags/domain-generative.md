---
title: "Тег: domain/generative"
description: Заметки с тегом domain/generative в книге знаний.
tags:
  - domain/generative
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `domain/generative`

## Заметки

- [Diffusion Models (диффузионные модели)](../../topics/diffusion-models/README.md) — Прямой и обратный процесс диффузии, DDPM/DDIM, latent diffusion (Stable Diffusion), Consistency Models, Flow Matching и DiT.
- [Сториборд клипа: Diffusion — прямой и обратный процесс](../../topics/diffusion-models/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа diffusion-forward-reverse-process: прямой процесс зашумления x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε, обратный процесс с сетью ε_θ и loss DDPM, ускорение через DDIM и latent diffusion.
- [Generative Adversarial Networks (GAN)](../../topics/generative-adversarial-networks-gans/README.md) — Состязательное обучение generator/discriminator, mode collapse, современные варианты GAN и сравнение с VAE и diffusion.
- [Сториборд клипа: GAN — игра генератора и дискриминатора](../../topics/generative-adversarial-networks-gans/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа gan-generator-discriminator-game: пайплайн z → G → G(z) и x → D → вероятность, минимаксный критерий с чередованием шагов D и G (non-saturating loss), движение p_g к p_data, mode collapse и стабилизация обучения (WGAN).
- [Variational Autoencoders (VAE)](../../topics/variational-autoencoders-vaes/README.md) — ELBO, encoder/decoder, reparameterization trick, латентное пространство и роль VAE в современных generative pipelines.
- [Сториборд клипа: VAE — encoder, репараметризация и ELBO](../../topics/variational-autoencoders-vaes/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа vae-encoder-decoder-reparameterization: encoder выдаёт распределение (μ, σ), reparameterization trick z = μ + σ·ε, ELBO и генерация из prior N(0, I) с интерполяцией.

