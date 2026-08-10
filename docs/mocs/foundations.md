---
title: "MOC: Mathematical & ML Foundations"
description: "Вероятность, метрики, классический ML и базовые строительные блоки."
tags:
  - kb/moc
  - domain/ml-foundations
type: moc
status: canonical
updated: 2026-08-10
---

# MOC: Mathematical & ML Foundations

Вероятность, метрики, классический ML и базовые строительные блоки.

## Topics

- [Теорема Байеса и основы теории вероятностей](../../topics/bayes-theorem-and-probability-foundations/README.md) — Аксиомы Колмогорова, условная вероятность, формула полной вероятности, теорема Байеса, MAP/MLE и наивный Байес в ML.
- [Гауссово распределение (Normal Distribution)](../../topics/gaussian-distribution/README.md) — Одномерное и многомерное нормальное распределение, PDF/CDF и роль гауссианы в VAE, diffusion и Kalman filtering.
- [ROC-кривые и ROC AUC](../../topics/roc-curve-and-roc-auc/README.md) — TPR/FPR, построение ROC, AUC как метрика ранжирования, выбор порога (Youden’s J) и связь с PR-кривыми.
- [Confidence, Calibration and Uncertainty](../../topics/how-models-predict-confidence-and-calibration/README.md) — Logits→softmax/sigmoid, reliability diagrams, ECE/Brier, temperature scaling и aleatoric/epistemic uncertainty (ensembles, MC Dropout).
- [Деревья решений (Decision Trees)](../../topics/decision-trees/README.md) — Структура дерева, Gini/энтропия/Information Gain, ID3/C4.5/CART, переобучение и связь с ансамблями Random Forest/XGBoost.
- [Support Vector Machines (SVM) и Kernel Trick](../../topics/support-vector-machines-svm-and-kernel-trick/README.md) — Max-margin классификация, soft-margin C, dual formulation и kernel trick (linear/poly/RBF) без явного φ(x).
- [Настройка гиперпараметров (Hyperparameter Tuning)](../../topics/hyperparameter-tuning/README.md) — Grid/Random search, Bayesian Optimization (Optuna/TPE), Hyperband/BOHB, PBT, CMA-ES, NAS и LR schedules.
- [Методы комбинирования моделей (Ensemble Methods)](../../topics/ensemble-methods-model-combination/README.md) — Bagging/boosting/stacking, XGBoost/LightGBM/CatBoost, MoE, distillation и model merging (TIES/DARE/SLERP) для LLM.
- [Batch Normalization и Layer Normalization](../../topics/normalization-layers-batchnorm-layernorm/README.md) — Нормализация активаций: формулы BatchNorm vs LayerNorm, влияние на обучение, выбор для CNN и Transformer.

## See also

- [All topics](../index.md)
- [Tags](../tags/README.md)
