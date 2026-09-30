---
title: "Сториборд клипа: UKF — sigma points, predict и update"
description: "Сториборд и команды сборки HyperFrames-клипа ukf-sigma-points-predict-update: почему KF/EKF ломаются на нелинейности, как sigma points заменяют якобиан, шаги predict и update с χ²-гейтом в трекинге."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/kalman-filter
  - concept/tracking
aliases:
  - UKF storyboard
  - ukf-sigma-points-predict-update
related:
  - unscented-kalman-filter-and-tracking
  - action-recognition-and-object-tracking-metrics
status: notes
lang: ru
type: note
slug: unscented-kalman-filter-and-tracking/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# UKF: sigma points, predict и update — сториборд

Клип `assets/visualizations/ukf-sigma-points-predict-update.{mp4,gif}`, 44 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Классический Kalman Filter (KF)», «Extended Kalman Filter (EKF)», «Unscented Kalman Filter (UKF)», «Статистика Хи-квадрат для Обнаружения Выбросов», «Современные Методы Отслеживания»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Почему Kalman Filter ломается на нелинейности? | Гауссиана на входе (эллипс в осях r, θ), стрелка f, «банан» на выходе, красный эллипс EKF от линеаризации и точка f(x̄); карточки «KF: линейный случай» и «EKF: F = ∂f/∂x, H = ∂h/∂x» | KF точен, пока f и h линейны · нелинейная f гнёт облако в «банан», EKF заменяет его эллипсом по якобиану · Тейлор 1-го порядка даёт ошибку линеаризации, нужны якобианы; UKF обходится без производных |
| 2 | 14–30 с | Как sigma points заменяют якобиан? | Тот же эллипс с 5 sigma points (χ₀ = x̄, χᵢ = x̄ ± √((n+λ)P)ᵢ) и весами 1/3, 1/6; точки летят через f на панель выхода (Y₀…Y₄); зелёный крест ȳ и эллипс P_y, красный EKF для сравнения; карточки «Sigma points», «Веса», «Unscented transform» | 2n+1 sigma points вокруг x̄ · веса W₀ = λ/(n+λ), Wᵢ = 1/(2(n+λ)) · каждая точка проходит через f, без производных · ȳ и P_y по образам, точность до 2-го порядка против 1-го у EKF |
| 3 | 30–44 с | Как UKF ведёт трек: predict → update? | Плоскость кадра: оценка в t−1 (синий эллипс), 5 sigma points летят через f в предсказание (фиолетовый эллипс, +Q), детекция z и инновация ν, зелёный эллипс после update (сдвиг и сжатие), пунктирный χ²-гейт и красный выброс; карточки «Predict», «Update», «χ²-гейт»; заметка про SORT / DeepSORT | Predict: sigma points → f → x̂ и P, +Q · Update: sigma points → h, gain K тянет оценку к z, P сжимается · χ²-гейт: νᵀS⁻¹ν > 5.99 (m = 2, α = 0.05) — выброс, update пропускаем · **ключевая идея**: 2n+1 sigma points вместо якобианов; predict через f, update через h и K; χ²-гейт против выбросов |

Числа и формулы (все из README): UKF предложен Julier & Uhlmann в 1997 (раздел «Введение»); 2n+1 sigma points, λ = α²(n+κ) − n, χ₀ = x̄, χᵢ = x̄ ± √((n+λ)P)ᵢ; веса W₀⁽ᵐ⁾ = λ/(n+λ), Wᵢ = 1/(2(n+λ)), W₀⁽ᶜ⁾ = W₀⁽ᵐ⁾ + (1 − α² + β), β = 2 для гауссиан; α = 1, κ = 0 либо 3 − n; predict: x̂ₜ|ₜ₋₁ = Σ Wᵢ⁽ᵐ⁾χ*ᵢ, Pₜ|ₜ₋₁ = Σ Wᵢ⁽ᶜ⁾(…)(…)ᵀ + Q; update: K = P_xz·P_zz⁻¹, x̂ₜ|ₜ = x̂ₜ|ₜ₋₁ + K(z − ẑ), Pₜ|ₜ = Pₜ|ₜ₋₁ − K·P_zz·Kᵀ; χ² = νᵀS⁻¹ν, порог χ²₀.₀₅(2) = 5.99; EKF: 1-й порядок, якобианы F = ∂f/∂x, H = ∂h/∂x; UKF: точность до 2-го порядка; SORT: Kalman Filter для предсказания + венгерский алгоритм для ассоциации.

Иллюстративно (помечено на экране): нелинейная f — переход из полярных координат (r, θ) в декартовы (x, y); среднее r = 5, θ = 55°, σ_r = 1, σ_θ = 26°; для n = 2 взяты α = 1, κ = 3 − n = 1, откуда λ = 1, W₀ = 1/3, Wᵢ = 1/6, W₀⁽ᶜ⁾ = 7/3; размеры эллипсов и положения точек в сцене 3 нарисованы схематично, не из расчёта.

## Сборка

```bash
cd topics/unscented-kalman-filter-and-tracking/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,9.5,12,18,21,29,34,36.5,41,43
npx -y hyperframes@0.8.81 render --quality looks --output renders/ukf-sigma-points-predict-update.mp4
ffmpeg -i renders/ukf-sigma-points-predict-update.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/ukf-sigma-points-predict-update.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/ukf-sigma-points-predict-update.mp4 -o ../../assets/visualizations/ukf-sigma-points-predict-update.gif
```
