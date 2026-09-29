---
title: "Тег: domain/ml-foundations"
description: Заметки с тегом domain/ml-foundations в книге знаний.
tags:
  - domain/ml-foundations
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `domain/ml-foundations`

## Заметки

- [Теорема Байеса и основы теории вероятностей](../../topics/bayes-theorem-and-probability-foundations/README.md) — Аксиомы Колмогорова, условная вероятность, формула полной вероятности, теорема Байеса, MAP/MLE и наивный Байес в ML.
- [Cross Entropy и Focal Loss](../../topics/classification-losses-cross-entropy-focal-loss/README.md) — Бинарная и многоклассовая кросс-энтропия, Focal Loss (α, γ) для дисбаланса и детекции (RetinaNet), когда выбирать CE vs Focal.
- [Гауссово распределение (Normal Distribution)](../../topics/gaussian-distribution/README.md) — Одномерное и многомерное нормальное распределение, PDF/CDF и роль гауссианы в VAE, diffusion и Kalman filtering.
- [Сториборд клипа: Гауссиана — PDF, правило 68-95-99.7 и многомерный случай](../../topics/gaussian-distribution/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа gaussian-pdf-and-multivariate: одномерная плотность и роль μ и σ, правило 68-95-99.7 со стандартизацией и CDF Φ(z), эллипсы равной плотности многомерной гауссианы и её роль в ML.
- [Уверенность, калибровка и неопределённость](../../topics/how-models-predict-confidence-and-calibration/README.md) — Logits→softmax/sigmoid, reliability diagrams, ECE/Brier, temperature scaling и aleatoric/epistemic uncertainty (ensembles, MC Dropout).
- [Сториборд клипа: калибровка — softmax, reliability diagram, temperature scaling](../../topics/how-models-predict-confidence-and-calibration/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа softmax-calibration-temperature-scaling: overconfident softmax из логитов, reliability diagram с ECE и temperature scaling softmax(z/T), которое меняет уверенность, но не argmax.
- [Настройка гиперпараметров (Hyperparameter Tuning)](../../topics/hyperparameter-tuning/README.md) — Grid/Random search, Bayesian Optimization (Optuna/TPE), Hyperband/BOHB, PBT, CMA-ES, NAS и LR schedules.
- [ROC-кривые и ROC AUC](../../topics/roc-curve-and-roc-auc/README.md) — TPR/FPR, построение ROC, AUC как метрика ранжирования, выбор порога (Youden’s J) и связь с PR-кривыми.
- [Сториборд клипа: ROC-кривая и ROC AUC — порог, кривая, площадь](../../topics/roc-curve-and-roc-auc/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа roc-curve-threshold-sweep-auc: два холма скоров и движущийся порог t (TPR/FPR), прогон порогов рисует ROC-кривую с точкой Youden’s J, AUC как площадь и вероятность P(s(x⁺) > s(x⁻)), контраст с PR-кривой.

