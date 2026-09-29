---
title: "Сториборд клипа: GAN — игра генератора и дискриминатора"
description: "Сториборд и команды сборки HyperFrames-клипа gan-generator-discriminator-game: пайплайн z → G → G(z) и x → D → вероятность, минимаксный критерий с чередованием шагов D и G (non-saturating loss), движение p_g к p_data, mode collapse и стабилизация обучения (WGAN)."
tags:
  - kb/note
  - kb/visualization
  - domain/generative
  - concept/gan
  - concept/adversarial-training
aliases:
  - GAN storyboard
  - gan-generator-discriminator-game
related:
  - generative-adversarial-networks-gans
  - variational-autoencoders-vaes
status: notes
lang: ru
type: note
slug: generative-adversarial-networks-gans/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# GAN: игра генератора и дискриминатора — сториборд

Клип `assets/visualizations/gan-generator-discriminator-game.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Основная идея и интуиция», «Математические основы»,
«Архитектура и обучение», «Проблемы и решения»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Кто с кем играет в GAN? | Пайплайн слева направо: блок «z ~ N(0, I)» (шум, 100–512 измерений) → блок G (транспонированные свёртки, upsampling) → клетчатый сэмпл G(z); снизу настоящий сэмпл x ~ p_data; оба → блок D (CNN с downsampling) → карточка «D(·) ∈ (0, 1)» с двумя шкалами D(x) → 1 и D(G(z)) → 0 (иллюстративно); в конце подписи блоков меняются на «фальшивомонетчик» / «детектив» и блоки «спорят» (пульс) | G получает шум z ~ N(0, I) и выдаёт подделку G(z) · D видит настоящие x ~ p_data и подделки и выдаёт вероятность, что вход настоящий · «фальшивомонетчик» против «детектива»: чем сильнее D, тем лучше учится G |
| 2 | 13–27 с | Как чередуются шаги D и G? | Слева: цикл «Шаг D ⇄ Шаг G» («повторять до сходимости; D часто обновляют чаще G, например 5:1»), две шкалы 0…1 с живыми числами D(x) и D(G(z)) (иллюстративно): шаг D — 0.55 → 0.85 и 0.45 → 0.15; шаг G — D(G(z)) 0.15 → 0.45; второй круг — 0.80 / 0.25 → 0.40. Справа карточки: минимаксный критерий; loss_D = −[log D(x) + log(1 − D(G(z)))]; loss_G = −log D(G(z)) (non-saturating) | min_G max_D V(D, G): D максимизирует, G минимизирует · шаг D: максимизировать log D(x) + log(1 − D(G(z))) — D(x) → 1, D(G(z)) → 0 · шаг G: минимизировать log(1 − D(G(z))) или, эквивалентно, максимизировать log D(G(z)) — non-saturating · чередуем до сходимости — «гонка вооружений» |
| 3 | 27–42 с | Куда движется p_g при обучении? | 2-D облако: настоящие x (синие, две моды, фиксированные координаты) и подделки G(z) (оранжевые) — стартуют одним сгустком, разъезжаются по обеим модам; карточка D*(x) = p_data(x) / (p_data(x) + p_g(x)), при p_g = p_data D* = ½; затем подделки схлопываются в одну моду — красное кольцо «мода потеряна: mode collapse», карточка «Проблемы и решения»; разнообразие возвращается | в начале обучения два облака далеки · шаги G сдвигают p_g к p_data; оптимум p_g = p_data, D*(x) = ½ всюду · mode collapse: лечат mini-batch discrimination, feature matching, unrolled GANs · **ключевая идея**: GAN = минимаксная игра G и D с равновесием при p_g = p_data; сэмплы резкие, обучение нестабильно → WGAN / WGAN-GP, spectral norm |

Формулы — из раздела «Математические основы» README (минимаксный критерий, цель дискриминатора, non-saturating loss
генератора, оптимальный дискриминатор D*, глобальный оптимум p_g = p_data); loss_D и loss_G — из блока «Алгоритм обучения».
Размерность шума 100–512, архитектуры G/D, отношение обновлений 5:1 — раздел «Архитектура и обучение». Средства против
mode collapse и нестабильности (WGAN: Wasserstein вместо JS-дивергенции, WGAN-GP, spectral normalization) — раздел
«Проблемы и решения». Значения D(·) на шкалах и координаты точек — иллюстративные (помечены на экране).

## Сборка

```bash
cd topics/generative-adversarial-networks-gans/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,8,11,17,20,26,30,33,36,40
npx -y hyperframes@0.8.81 render --quality looks --output renders/gan-generator-discriminator-game.mp4
ffmpeg -i renders/gan-generator-discriminator-game.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/gan-generator-discriminator-game.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/gan-generator-discriminator-game.mp4 -o ../../assets/visualizations/gan-generator-discriminator-game.gif
```
