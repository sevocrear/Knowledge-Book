---
title: "Сториборд клипа: Bagging, boosting, stacking"
description: "Сториборд и команды сборки HyperFrames-клипа bagging-boosting-stacking: бутстрэп-выборки и усреднение независимых моделей (variance), последовательный gradient boosting по псевдо-остаткам (bias) и двухуровневый stacking с мета-моделью на OOF-предсказаниях."
tags:
  - kb/note
  - kb/visualization
  - domain/classical-ml
  - concept/ensemble
  - concept/boosting
aliases:
  - bagging-boosting-stacking
related:
  - ensemble-methods-model-combination
  - decision-trees
status: notes
lang: ru
type: note
slug: ensemble-methods-model-combination/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Bagging, boosting, stacking — сториборд

Клип `assets/visualizations/bagging-boosting-stacking.{mp4,gif}`, 44 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Зачем комбинировать модели», «Bagging (Bootstrap Aggregating)» / «Random Forest»,
«Boosting» / «AdaBoost» / «Gradient Boosting» / «XGBoost, LightGBM, CatBoost», «Stacking (Stacked Generalization)», «Сравнение методов»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Bagging: зачем усреднять много моделей? | Набор D (20 объектов, иллюстративно) → три бутстрэп-выборки D₁…D₃, объекты «летят» в слоты, повторы — оранжевые → три дерева f₁…f₃ → агрегация ŷ (среднее / голосование); карточки «Bootstrap» (≈ 63.2% уникальных, ≈ 36.8% OOB), «Агрегация» (ŷ = (1/M)·Σ fᵢ(x) или mode{fᵢ(x)}), «Почему падает variance» (Var(ŷ) = ρσ² + (1 − ρ)σ²/M); заметка про Random Forest (m = √p признаков на сплите) | M бутстрэп-выборок с возвращением, ≈ 63.2% уникальных, остальные OOB · на каждой Dᵢ — независимая модель fᵢ, ответы усредняем или голосуем · усреднение снижает variance (ρ → 0 ⇒ σ²/M), Random Forest декоррелирует деревья |
| 2 | 14–30 с | Boosting: как модели чинят друг друга? | График y(x) с 12 точками (иллюстративно): F₀ = среднее, оранжевые остатки rᵢ, затем три «пенька» h₁…h₃ с η = 0.5 — ступеньки F₁, F₂, F₃ приближаются к точкам, остатки укорачиваются; столбики Σrᵢ² 100% → 42% → 15% → 6% (иллюстративно); карточки «Псевдо-остатки» (rᵢ = −∂L/∂F_{m−1}; MSE: yᵢ − F_{m−1}(xᵢ); log-loss: yᵢ − pᵢ) и «Шаг бустинга» (F_m = F_{m−1} + η·γ_m·h_m, η обычно 0.01–0.3); в финале объекты с наибольшим остатком «тяжелеют» (AdaBoost: wᵢ ↑) | F_M(x) = Σ α_m·h_m(x), каждая новая h_m ловит ошибки предыдущих → снижает bias · Gradient Boosting: h_m учится на псевдо-остатках · шаг с learning rate η, остатки и ошибка падают с каждым m · AdaBoost растит веса ошибочных объектов; XGBoost / LightGBM / CatBoost — промышленный gradient boosting |
| 3 | 30–44 с | Stacking: кто решает, какой модели верить? | Объект x → уровень 0: RF, XGBoost, SVM, kNN → pred₁…pred₄ «едут» в мета-модель (LogReg) уровня 1 → ŷ (final_pred); полоса K = 5 фолдов, оранжевая рамка «фолд k: предсказать → OOF» пробегает по фолдам; карточки «Два уровня», «Мета-признаки без утечки», таблица «Что снижает каждый метод» (Bagging — variance, Boosting — bias, Stacking — bias и variance, инференс M + 1) | базовые модели уровня 0 дают pred₁…pred₄, мета-модель уровня 1 учится их комбинировать · K фолдов: fⱼ учат без фолда k и предсказывают на нём — OOF-предсказания = мета-признаки · **ключевая идея**: bagging усредняет независимые модели (↓ variance), boosting чинит ошибки по очереди (↓ bias), stacking учит мета-модель, кому верить |

Числа и формулы из README: ≈ 63.2% уникальных объектов в бутстрэп-выборке и ≈ 36.8% OOB; ŷ = (1/M)·Σ fᵢ(x) (регрессия), ŷ = mode{f₁(x), …, f_M(x)} (классификация);
Var((1/M)·Σ fᵢ) = ρ·σ² + (1 − ρ)/M · σ² (ρ = 0 → σ²/M, ρ = 1 → σ²); Random Forest — m = √p признаков на сплите (классификация);
F_M(x) = Σ α_m·h_m(x); rᵢ = −∂L(yᵢ, F_{m−1}(xᵢ))/∂F_{m−1}(xᵢ), для MSE rᵢ = yᵢ − F_{m−1}(xᵢ), для log-loss rᵢ = yᵢ − pᵢ; F_m = F_{m−1} + η·γ_m·h_m, η обычно 0.01–0.3;
AdaBoost — перевзвешивание объектов; stacking — уровень 0 (RF, XGBoost, SVM, kNN) → мета-модель (LogReg), K фолдов, OOF-предсказания; таблица «Сравнение методов» (bias / variance, инференс M + 1).

Иллюстративные величины (помечены на экране): N = 20 объектов и состав бутстрэп-выборок в сцене 1; 12 точек y(x), η = 0.5 и проценты Σrᵢ² (100 → 42 → 15 → 6) в сцене 2 —
посчитаны детерминированным градиентным бустингом на «пеньках» (MSE, γ = 1) прямо в сцене; K = 5 фолдов в сцене 3 (как `cv=5` в примере кода README).

## Сборка

```bash
cd topics/ensemble-methods-model-combination/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 4.5,7.5,12,17,21.5,28.5,33,36.5,42
npx -y hyperframes@0.8.81 render --quality looks --output renders/bagging-boosting-stacking.mp4
ffmpeg -i renders/bagging-boosting-stacking.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/bagging-boosting-stacking.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/bagging-boosting-stacking.mp4 -o ../../assets/visualizations/bagging-boosting-stacking.gif
```
