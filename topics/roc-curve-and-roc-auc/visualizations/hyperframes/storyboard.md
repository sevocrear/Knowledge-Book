---
title: "Сториборд клипа: ROC-кривая и ROC AUC — порог, кривая, площадь"
description: "Сториборд и команды сборки HyperFrames-клипа roc-curve-threshold-sweep-auc: два холма скоров и движущийся порог t (TPR/FPR), прогон порогов рисует ROC-кривую с точкой Youden’s J, AUC как площадь и вероятность P(s(x⁺) > s(x⁻)), контраст с PR-кривой."
tags:
  - kb/note
  - kb/visualization
  - domain/ml-foundations
  - concept/metrics
  - concept/roc
aliases:
  - roc-curve-threshold-sweep-auc
  - ROC AUC storyboard
related:
  - roc-curve-and-roc-auc
  - how-models-predict-confidence-and-calibration
status: notes
lang: ru
type: note
slug: roc-curve-and-roc-auc/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# ROC-кривая и ROC AUC: порог → кривая → площадь — сториборд

Клип `assets/visualizations/roc-curve-threshold-sweep-auc.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Что Такое TPR и FPR», «Определение ROC-кривой», «Определение ROC AUC и Интуиция»,
«Как Строить ROC и ROC AUC на Практике», «Как Выбрать Порог Классификации», «ROC vs PR-кривые»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Что меняется, когда двигаем порог t? | Два холма скоров s(x) ∈ [0, 1]: негативы (класс 0, синий) и позитивы (класс 1, оранжевый); пунктирный порог t; справа от t закрашены TP (оранжевый) и FP (красный); карточка правила s(x) ≥ t → класс 1; карточка с живыми числами t / TPR / FPR; порог едет 0.80 → 0.50 → 0.30, TPR 0.10 → 0.80 → 0.99, FPR 0.00 → 0.20 → 0.72 (иллюстративно) | модель даёт скор s(x): два холма, перекрываются · справа от t — класс 1; TPR = TP/(TP+FN), FPR = FP/(FP+TN) · опускаем порог: TPR ↑, но и FPR ↑; каждый t — своя пара (FPR, TPR) |
| 2 | 13–29 с | Как из порогов получается ROC-кривая? | Оси FPR (X) и TPR (Y), диагональ «монетка: AUC = 0.5»; справа мини-холмы с порогом t; порог проходит 0.95 → 0.05, оранжевая точка едет из (0, 0) в (1, 1) и дорисовывает синюю ROC; маркеры t = 0.7 / 0.5 / 0.3; зелёная точка «идеал (0, 1)»; фиолетовый отрезок Youden’s J = TPR − FPR = 0.61 при t* = 0.5 | перебираем пороги от +∞ к −∞ (все уникальные скоры), для каждого t — пара FPR(t), TPR(t) · каждый порог — точка; точка едет из (0, 0) в (1, 1) и прочерчивает ROC · идеал — угол (0, 1), монетка — диагональ, ниже диагонали — классы перепутаны · Youden’s J = TPR − FPR, t* = argmax J |
| 3 | 29–42 с | Что измеряет ROC AUC? | Площадь под ROC закрашивается, «AUC = 0.89 (иллюстративно)», AUC = ∫₀¹ TPR d(FPR); карточка шкалы 1.0 / 0.5 / < 0.5; панель «случайная пара x⁺, x⁻»: (0.72, 0.31) ✓, (0.55, 0.61) ✗, (0.81, 0.44) ✓; карточка AUC = P(s(x⁺) > s(x⁻)); карточка «vs PR-кривая» | AUC — площадь под ROC: 1.0 идеал, 0.5 монетка, < 0.5 классы перепутаны · AUC = P(s(x⁺) > s(x⁻)) — метрика ранжирования · **ключевая идея**: ROC AUC — про ранжирование при всех порогах; при сильном дисбалансе (1 % позитивов) ROC AUC обманчив — смотри и PR AUC |

Числа из README: TPR = TP/(TP+FN), FPR = FP/(FP+TN), правило s(x) ≥ t → класс 1; ROC — точки (FPR(t), TPR(t)), идеал в (0, 1),
случайная модель — диагональ, ниже диагонали — инвертировать; AUC ∈ [0, 1], 1.0 / 0.5 / < 0.5, AUC = ∫₀¹ TPR d(FPR) = P(s(x⁺) > s(x⁻));
Youden’s J(t) = TPR(t) − FPR(t), t* = argmax J; PR-кривая: X = Recall (= TPR), Y = Precision, при 1 % позитивов ROC AUC обманчиво высок.

Иллюстративные величины (не из README, помечены на экране): скоры негативов ~ N(0.38, 0.14²), позитивов ~ N(0.62, 0.14²);
TPR(t) = 1 − Φ((t − 0.62)/0.14), FPR(t) = 1 − Φ((t − 0.38)/0.14); AUC = Φ(0.24 / (0.14·√2)) ≈ 0.89; J(0.5) = 0.804 − 0.196 ≈ 0.61;
пары скоров (0.72, 0.31), (0.55, 0.61), (0.81, 0.44).

## Сборка

```bash
cd topics/roc-curve-and-roc-auc/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 4,8,11.5,16,21,24.5,27.5,31,35.5,37.5,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/roc-curve-threshold-sweep-auc.mp4
ffmpeg -i renders/roc-curve-threshold-sweep-auc.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/roc-curve-threshold-sweep-auc.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/roc-curve-threshold-sweep-auc.mp4 -o ../../assets/visualizations/roc-curve-threshold-sweep-auc.gif
```
