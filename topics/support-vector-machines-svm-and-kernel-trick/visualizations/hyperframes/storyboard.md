---
title: "Сториборд клипа: SVM — максимальный запас и kernel trick"
description: "Сториборд и команды сборки HyperFrames-клипа svm-margin-and-kernel-trick: разделяющая гиперплоскость с максимальным запасом 2/‖w‖ и опорные векторы, soft margin со slack ξ и компромиссом C, kernel trick — отображение φ и ядро k(x, x′) без явного φ."
tags:
  - kb/note
  - kb/visualization
  - domain/classical-ml
  - concept/svm
  - concept/kernel-methods
aliases:
  - SVM storyboard
  - svm-margin-and-kernel-trick
related:
  - support-vector-machines-svm-and-kernel-trick
  - decision-trees
status: notes
lang: ru
type: note
slug: support-vector-machines-svm-and-kernel-trick/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# SVM: максимальный запас и kernel trick — сториборд

Клип `assets/visualizations/svm-margin-and-kernel-trick.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Линейно разделимый случай и запас (margin)», «Оптимизационная задача
и двойственная форма», «Support Vectors», «Мягкий запас (soft margin) и параметр C», «Kernel Trick», «Типичные ядра»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Какую разделяющую прямую выбрать? | Два класса точек (синие y = −1, оранжевые y = +1); четыре серые прямые-кандидаты; затем белая гиперплоскость wᵀx + b = 0, полоса запаса с границами wᵀx + b = ±1, стрелка «2/‖w‖»; бирюзовые кольца на 4 опорных векторах; карточки «Гиперплоскость» и «Максимальный запас» | классы разделимы, подходят многие прямые · SVM берёт гиперплоскость с максимальным запасом до ближайших точек · ширина полосы 2/‖w‖ → min ½‖w‖²; точки на границе — опорные векторы |
| 2 | 13–27 с | Что делать, если данные не разделимы? | Та же полоса; три выброса с красными кольцами (два внутри полосы, один на чужой стороне); красные отрезки slack ξ₁, ξ₂, ξ₃ до «своей» границы; карточка «Soft margin» (ограничение и целевая функция); карточка «Компромисс C»; полоса сужается (большой C) и расширяется (маленький C, slack появляется и у бывших опорных точек) | условие yᵢ(wᵀxᵢ + b) ≥ 1 выполнить нельзя · slack ξᵢ ≥ 0 ослабляет ограничение, сумма нарушений штрафуется с весом C · большой C — узкий запас, мало нарушений, риск переобучения; маленький C — шире запас, больше ошибок на train, часто лучше обобщение |
| 3 | 27–42 с | Как разделить то, что не делится прямой? | Два концентрических класса (как `make_circles` из README); попытка провести прямую → красный ✗; точки «поднимаются» в координаты (x₁, x₁² + x₂²), ось x₂ меняется на x₁² + x₂²; горизонтальная гиперплоскость wᵀφ(x) + b = 0; карточки «Kernel trick» и «Типичные ядра» (Linear, Polynomial, RBF) | никакая прямая не разделяет · отображение φ: добавим признак x₁² + x₂² (иллюстративно) — классы делит обычная гиперплоскость · данные входят только через xᵢᵀxⱼ → заменяем на ядро k(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩ · **ключевая идея**: максимальный запас на опорных векторах; ядро даёт нелинейную границу, а задача остаётся линейной в H |

Формулы (из README): wᵀx + b = 0; ŷ = sign(wᵀx + b); min ½‖w‖² при yᵢ(wᵀxᵢ + b) ≥ 1; геометрический запас yᵢ(wᵀxᵢ + b)/‖w‖
(при функциональном запасе 1 расстояние до границы полосы 1/‖w‖, ширина полосы 2/‖w‖); soft margin yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0,
min ½‖w‖² + C Σ ξᵢ, 0 ≤ αᵢ ≤ C; k(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩; f(x) = Σ αᵢ yᵢ k(xᵢ, x) + b; ядра xᵀz, (γ xᵀz + r)ᵖ, exp(−γ‖x − z‖²).
Координаты точек, ширина полосы при разных C и признак φ(x) = (x₁, x₂, x₁² + x₂²) — иллюстративные, помечены на экране.

## Сборка

```bash
cd topics/support-vector-machines-svm-and-kernel-trick/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.5,10.5,15,20,23.5,26.5,29.5,33,37,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/svm-margin-and-kernel-trick.mp4
ffmpeg -i renders/svm-margin-and-kernel-trick.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/svm-margin-and-kernel-trick.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/svm-margin-and-kernel-trick.mp4 -o ../../assets/visualizations/svm-margin-and-kernel-trick.gif
```
