---
title: "Сториборд клипа: Grid / Random → Bayesian → Hyperband"
description: "Сториборд и команды сборки HyperFrames-клипа grid-random-bayesian-hyperband: grid vs random search на 2D-пространстве, суррогат + acquisition в Bayesian Optimization и раннее отсечение Successive Halving / Hyperband."
tags:
  - kb/note
  - kb/visualization
  - domain/mlops
  - concept/hyperparameter-tuning
  - concept/bayesian-optimization
aliases:
  - hyperparameter tuning storyboard
  - grid-random-bayesian-hyperband
related:
  - hyperparameter-tuning
  - bayes-theorem-and-probability-foundations
status: notes
lang: ru
type: note
slug: hyperparameter-tuning/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Grid / Random → Bayesian → Hyperband — сториборд

Клип `assets/visualizations/grid-random-bayesian-hyperband.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Grid Search», «Random Search», «Bayesian Optimization» (Surrogate Model, Acquisition Function, Optuna), «Bandit-based методы: Hyperband и BOHB», «Практические рекомендации», «Сравнение методов»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Почему Random Search бьёт Grid Search? | Слева две квадратные панели 2D-пространства (ось x — важный гиперпараметр, ось y — неважный): сетка 3 × 3 = 9 синих точек и 9 оранжевых случайных точек; под каждой — полоса «качество (иллюстративно)» с зелёной кривой, проекции точек на важную ось: у сетки 3 тика (пик «пропущен»), у random 9 тиков («рядом с пиком»). Справа карточки: «Grid Search: nᵈ экспериментов» (таблица 25 / 125 / 3 125 / 390 625 при n = 5, пример lr × max_depth = 9) и «Random Search · Bergstra & Bengio, 2012» (nᵈ точек → nᵈ уникальных значений каждого гиперпараметра; d_eff ≪ d) | Grid Search перебирает все комбинации: 3 × 3 = 9, при d гиперпараметрах — nᵈ · проекция на важную ось: у сетки лишь 3 уникальных значения, пик между узлами пропущен · Random Search: те же 9 точек дают 9 уникальных значений важного параметра |
| 2 | 14–29 с | Как Bayesian Optimization выбирает λ? | Верхний график loss f(λ) (пунктиром — «настоящая f, неизвестна»): 3 оранжевые warm-up точки → синяя кривая μ(λ) с полосой ± σ (GP с RBF-ядром, считается в сцене детерминированно) и линия y*; нижний график acquisition EI(λ) с оранжевой вертикалью λ_next = argmax α(λ); падает новая точка, суррогат и EI перестраиваются (два шага: λ ≈ 0.64, затем ≈ 0.71 у минимума). Справа карточки: «Цикл Bayesian Optimization» (4 шага из README) и «Acquisition: Expected Improvement» EI(λ) = 𝔼[max(0, y* − f(λ))], TPE: EI ∝ ℓ(λ)/g(λ), бюджет 10–200 экспериментов | один запуск f(λ) — полное обучение, начинаем с нескольких случайных точек (warm-up) · суррогат (GP) даёт μ(λ) и σ(λ) — дёшево · acquisition (EI) балансирует exploitation и exploration: следующая точка — argmax α(λ) · новое наблюдение уточняет суррогат; в Optuna по умолчанию TPE |
| 3 | 29–43 с | Зачем Hyperband обрывает слабые конфигурации? | График val accuracy vs бюджет (старт, 1, 2, 4, 8 эпох — удваивается): 8 иллюстративных кривых обучения; на чекпоинтах красные пунктиры «оставить 4 / 2 / 1», проигравшие тускнеют с ✕, счётчик «живых конфигураций» 8 → 4 → 2 → 1, победитель зелёный. Справа карточки: «Successive Halving · Jamieson & Talwalkar, 2016» (64 × 1 → 32, 32 × 2 → 16, …, 2 × 32 → 1; строки подсвечиваются по раундам) и «Hyperband · Li et al., 2018» (brackets от 81 × 1 эпоха до 5 × 81; BOHB = TPE + Hyperband; Optuna pruning) | BO тратит полный бюджет на каждую λ; Successive Halving стартует 64 конфигурации × 1 эпоха · каждый раунд оставляем лучшую половину и удваиваем бюджет: 64 → 32 → … → 1 · **ключевая идея**: Hyperband гоняет несколько раундов с разным стартовым бюджетом; BOHB = TPE (умный выбор λ) + Hyperband (раннее отсечение) |

Числа из README: Grid Search nᵈ — 25 / 125 / 3 125 / 390 625 при n = 5 и d = 2 / 3 / 5 / 8; пример learning_rate ∈ {0.001, 0.01, 0.1} × max_depth ∈ {3, 5, 7} → 9 комбинаций; Random Search: nᵈ уникальных значений каждого гиперпараметра вместо n (Bergstra & Bengio, 2012), в схеме README — 9 значений вместо 3; EI(λ) = 𝔼[max(0, y* − f(λ))], TPE: EI ∝ ℓ(λ)/g(λ), бюджет BO 10–200 экспериментов; Successive Halving 64 × 1 → 32 × 2 → 16 × 4 → 8 × 8 → 4 × 16 → 2 × 32 → 1; Hyperband brackets 81 × 1 эпоха … 5 × 81 эпоха; BOHB = TPE + Hyperband.
Иллюстративные величины (так и подписаны на экране): форма кривой «качество» в сцене 1 и координаты случайных точек; целевая функция f(λ), warm-up точки λ = 0.10 / 0.40 / 0.95 и GP-суррогат в сцене 2; 8 кривых обучения и чекпоинты 1 / 2 / 4 / 8 эпох в сцене 3.

## Сборка

```bash
cd topics/hyperparameter-tuning/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7,12,17,21,27,32,36,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/grid-random-bayesian-hyperband.mp4
ffmpeg -i renders/grid-random-bayesian-hyperband.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/grid-random-bayesian-hyperband.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/grid-random-bayesian-hyperband.mp4 -o ../../assets/visualizations/grid-random-bayesian-hyperband.gif
```
