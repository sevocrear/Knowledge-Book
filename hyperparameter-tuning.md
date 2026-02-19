# Настройка гиперпараметров (Hyperparameter Tuning)

## Как объяснить 5-летнему ребёнку

Представь, что ты печёшь торт. Рецепт говорит: «добавь сахар» — но не говорит, сколько. Слишком мало — невкусно, слишком много — приторно. Ты пробуешь: одну ложку, две, три... и находишь самое вкусное количество. Настройка гиперпараметров — это то же самое: мы пробуем разные «рецепты» для нашей модели и выбираем тот, при котором она работает лучше всего.

---

## Table of Contents

1. [Параметры vs Гиперпараметры](#параметры-vs-гиперпараметры)
2. [Пространство поиска и типы гиперпараметров](#пространство-поиска-и-типы-гиперпараметров)
3. [Оценка качества: кросс-валидация](#оценка-качества-кросс-валидация)
4. [Grid Search (Полный перебор)](#grid-search-полный-перебор)
5. [Random Search (Случайный поиск)](#random-search-случайный-поиск)
6. [Bayesian Optimization (Байесовская оптимизация)](#bayesian-optimization-байесовская-оптимизация)
   - [Surrogate Model (Суррогатная модель)](#surrogate-model-суррогатная-модель)
   - [Acquisition Function](#acquisition-function)
   - [Optuna](#optuna)
   - [Hyperopt](#hyperopt)
7. [Bandit-based методы: Hyperband и BOHB](#bandit-based-методы-hyperband-и-bohb)
   - [Successive Halving](#successive-halving)
   - [Hyperband](#hyperband)
   - [BOHB](#bohb)
8. [Population-Based Training (PBT)](#population-based-training-pbt)
9. [Эволюционные алгоритмы (Evolutionary Strategies)](#эволюционные-алгоритмы-evolutionary-strategies)
10. [Neural Architecture Search (NAS)](#neural-architecture-search-nas)
11. [Автоматический подбор гиперпараметров learning rate](#автоматический-подбор-learning-rate)
    - [LR Range Test / LR Finder](#lr-range-test--lr-finder)
    - [Cosine Annealing, OneCycleLR](#cosine-annealing-onecyclelr)
12. [Практические рекомендации](#практические-рекомендации)
13. [Сравнение методов](#сравнение-методов)
14. [Что используется больше всего (2024-2026)](#что-используется-больше-всего-2024-2026)
15. [Примеры кода](#примеры-кода)
16. [References](#references)

---

## Параметры vs Гиперпараметры

| | Параметры модели | Гиперпараметры |
|---|---|---|
| **Что это** | Обучаемые веса ($W$, $b$) | «Настройки рецепта» |
| **Кто задаёт** | Алгоритм обучения (gradient descent) | Человек / алгоритм тюнинга |
| **Когда определяются** | Во время обучения | **До** обучения |
| **Примеры** | Веса нейросети, коэффициенты регрессии | Learning rate, число слоёв, batch size |

**Формально**: модель $f(x; \theta, \lambda)$, где $\theta$ — параметры (обучаются), $\lambda$ — гиперпараметры (задаются извне).

Задача тюнинга:

$$\lambda^* = \arg\min_{\lambda \in \Lambda} \mathcal{L}_{\text{val}}\big(f(\cdot\,; \hat{\theta}(\lambda), \lambda)\big)$$

где $\hat{\theta}(\lambda) = \arg\min_\theta \mathcal{L}_{\text{train}}(f(\cdot\,; \theta, \lambda))$ — параметры, обученные при фиксированных $\lambda$.

---

## Пространство поиска и типы гиперпараметров

### Типы по шкале

| Тип | Примеры | Шкала поиска |
|---|---|---|
| **Вещественные** | Learning rate, weight decay, dropout | Часто **логарифмическая**: $10^{-5} \ldots 10^{-1}$ |
| **Целые** | Число слоёв, число нейронов, `n_estimators` | Линейная или логарифмическая |
| **Категориальные** | Тип оптимизатора (Adam/SGD), ядро SVM (rbf/poly) | Перебор вариантов |
| **Булевы** | Использовать ли BatchNorm, augmentation | True / False |
| **Условные** | `num_layers` зависит от `architecture_type` | Условное пространство |

### Типичные гиперпараметры по моделям

**Gradient Boosting (XGBoost/LightGBM/CatBoost):**

| Гиперпараметр | Диапазон | Шкала |
|---|---|---|
| `learning_rate` | $[0.001, 0.3]$ | log |
| `n_estimators` | $[50, 3000]$ | log |
| `max_depth` | $[3, 12]$ | int |
| `num_leaves` (LightGBM) | $[15, 255]$ | int |
| `subsample` | $[0.5, 1.0]$ | uniform |
| `colsample_bytree` | $[0.5, 1.0]$ | uniform |
| `reg_alpha` (L1) | $[10^{-8}, 10]$ | log |
| `reg_lambda` (L2) | $[10^{-8}, 10]$ | log |
| `min_child_weight` | $[1, 100]$ | log |

**Нейронные сети:**

| Гиперпараметр | Диапазон | Шкала |
|---|---|---|
| `learning_rate` | $[10^{-5}, 10^{-2}]$ | log |
| `batch_size` | $\{16, 32, 64, 128, 256\}$ | categorical |
| `weight_decay` | $[10^{-6}, 10^{-2}]$ | log |
| `dropout` | $[0.0, 0.5]$ | uniform |
| `num_layers` | $[2, 8]$ | int |
| `hidden_dim` | $\{64, 128, 256, 512, 1024\}$ | categorical |
| `optimizer` | $\{\text{Adam, AdamW, SGD}\}$ | categorical |

---

## Оценка качества: кросс-валидация

Любой метод тюнинга требует **честной оценки** каждой конфигурации гиперпараметров.

### K-Fold Cross-Validation

```
Данные: [████████████████████████████████████████]

Fold 1: [VAL ][  TRAIN  ][  TRAIN  ][  TRAIN  ][  TRAIN  ]  → score₁
Fold 2: [TRAIN][  VAL   ][  TRAIN  ][  TRAIN  ][  TRAIN  ]  → score₂
Fold 3: [TRAIN][  TRAIN ][  VAL    ][  TRAIN  ][  TRAIN  ]  → score₃
Fold 4: [TRAIN][  TRAIN ][  TRAIN  ][  VAL    ][  TRAIN  ]  → score₄
Fold 5: [TRAIN][  TRAIN ][  TRAIN  ][  TRAIN  ][  VAL    ]  → score₅

Итоговый score = mean(score₁ … score₅)  ±  std(score₁ … score₅)
```

$$\text{CV}(\lambda) = \frac{1}{K}\sum_{k=1}^{K} \mathcal{L}\big(f_{-k}(\cdot\,; \hat{\theta}_k, \lambda), D_k\big)$$

где $f_{-k}$ — модель обученная на всех фолдах, кроме $k$-го, $D_k$ — валидационный фолд.

**Типичные значения $K$**: 5 или 10. Для маленьких датасетов — Leave-One-Out (LOO), для больших — достаточно одного hold-out split.

### Stratified K-Fold

Для задач классификации — сохраняет пропорции классов в каждом фолде. **Обязателен** при несбалансированных классах.

### Nested Cross-Validation

Для корректной оценки всей процедуры (тюнинг + обучение) — вложенная CV:

```
Внешний цикл (оценка): K₁ фолдов
  └── Внутренний цикл (тюнинг): K₂ фолдов
       └── Для каждой конфигурации λ:
            обучить на K₂-1 фолдах, оценить на 1 фолде
       → выбрать лучший λ*
  → обучить на всех K₁-1 фолдах с λ*, оценить на оставленном
```

---

## Grid Search (Полный перебор)

### Принцип работы

Задаём **сетку** значений для каждого гиперпараметра и перебираем **все возможные комбинации**.

$$\Lambda_{\text{grid}} = \lambda_1^{(1)} \times \lambda_2^{(1)} \times \ldots = \{(\lambda_1, \lambda_2, \ldots) : \lambda_i \in S_i\}$$

**Пример**: `learning_rate` ∈ {0.001, 0.01, 0.1}, `max_depth` ∈ {3, 5, 7} → 9 комбинаций.

### Сложность

Если $d$ гиперпараметров, каждый с $n$ значениями → $n^d$ экспериментов. **Проклятие размерности** — экспоненциальный рост.

| Число гиперпараметров | Значений на каждый | Всего экспериментов |
|---|---|---|
| 2 | 5 | 25 |
| 3 | 5 | 125 |
| 5 | 5 | 3 125 |
| 8 | 5 | 390 625 |

### Плюсы и минусы

| + | − |
|---|---|
| Гарантированно находит лучшее из сетки | Экспоненциальный рост |
| Простой и воспроизводимый | Может пропустить «между узлами» |
| Параллелизуется | Неэффективен при > 3-4 гиперпараметрах |

---

## Random Search (Случайный поиск)

### Принцип работы

Вместо перебора сетки — **случайная** выборка $N$ конфигураций из пространства $\Lambda$.

$$\lambda_i \sim p(\lambda), \quad i = 1, \ldots, N$$

где $p(\lambda)$ — распределение на пространстве гиперпараметров (uniform, log-uniform и т.д.).

### Почему Random Search лучше Grid Search

**Ключевой результат** (Bergstra & Bengio, 2012): если только $d_{\text{eff}} \ll d$ гиперпараметров реально важны (что типично!), то Random Search **экспоненциально эффективнее** Grid Search.

**Интуиция**: при Grid Search с $n$ точками по каждой оси мы получаем только $n$ уникальных значений каждого гиперпараметра. При Random Search с $n^d$ точками мы получаем $n^d$ уникальных значений каждого — гораздо лучше покрываем важные оси.

![Random vs Grid Search](https://miro.medium.com/v2/resize:fit:1400/1*ZTlQm_WRcrNqL-nLnx6GJA.png)

### Плюсы и минусы

| + | − |
|---|---|
| Лучше покрывает важные измерения | Нет гарантии оптимума |
| Легко масштабируется (добавить ещё N точек) | Не использует результаты предыдущих проб |
| Параллелизуется тривиально | При малом бюджете может «промахнуться» |
| Работает при любом $d$ | |

---

## Bayesian Optimization (Байесовская оптимизация)

### Принцип работы

**Bayesian Optimization** — итеративный метод, который **использует результаты предыдущих экспериментов** для выбора следующей точки. Строит **суррогатную модель** функции качества и на её основе решает, где пробовать дальше.

**Алгоритм:**
1. Провести несколько случайных экспериментов (warm-up)
2. Повторять:
   a. Построить/обновить суррогатную модель $\hat{f}(\lambda)$ по наблюдениям $\{(\lambda_i, y_i)\}$
   b. Используя acquisition function, выбрать $\lambda_{\text{next}} = \arg\max_\lambda \alpha(\lambda)$
   c. Провести эксперимент: $y_{\text{next}} = f(\lambda_{\text{next}})$
   d. Добавить $(\lambda_{\text{next}}, y_{\text{next}})$ в историю

```
Итерация 1:   ●        ●    ●         (случайные точки)
               ↓
Surrogate:  ~~●~~~~~~~~●~~~~●~~~~~~~~~  (модель + неопределённость)
               ↓
Acquisition:      ↑ (следующая точка — максимум acquisition)
               ↓
Итерация 2:   ●    ●   ●    ●         (добавили новую точку)
               ↓
Surrogate:  ~~●~~~~●~~~●~~~~●~~~~~~~~~  (модель уточнилась)
              ...
```

### Surrogate Model (Суррогатная модель)

Суррогат — дешёвая аппроксимация дорогой целевой функции $f(\lambda)$.

#### Gaussian Process (GP)

Классический суррогат для Bayesian Optimization. Даёт **предсказание + неопределённость** в каждой точке:

$$f(\lambda) \sim \mathcal{GP}\big(m(\lambda), k(\lambda, \lambda')\big)$$

где $m(\lambda)$ — среднее (обычно 0), $k(\lambda, \lambda')$ — ковариационная функция (ядро, например, Matérn 5/2).

**Posterior** после наблюдений $(X, \mathbf{y})$:

$$\mu(\lambda) = k(\lambda, X)\,[k(X, X) + \sigma_n^2 I]^{-1}\,\mathbf{y}$$

$$\sigma^2(\lambda) = k(\lambda, \lambda) - k(\lambda, X)\,[k(X, X) + \sigma_n^2 I]^{-1}\,k(X, \lambda)$$

**Плюс**: полная вероятностная модель с доверительными интервалами.
**Минус**: масштабируется как $O(n^3)$ по числу наблюдений, плохо работает с $d > 20$.

#### Tree-structured Parzen Estimator (TPE)

Используется в **Optuna** и **Hyperopt**. Вместо моделирования $p(y|\lambda)$ моделирует $p(\lambda|y)$:

$$p(\lambda | y) = \begin{cases} \ell(\lambda) & \text{если } y < y^* \quad (\text{«хорошие» конфигурации}) \\ g(\lambda) & \text{если } y \geq y^* \quad (\text{«плохие» конфигурации}) \end{cases}$$

где $y^*$ — порог (обычно квантиль наблюдений, напр. 25-й перцентиль).

Acquisition function для TPE:

$$\text{EI}(\lambda) \propto \frac{\ell(\lambda)}{g(\lambda)}$$

**Плюс**: хорошо работает с условными и категориальными гиперпараметрами, масштабируется лучше GP.

#### Random Forest / Gradient Boosted Trees

Используются в **SMAC** (Sequential Model-based Algorithm Configuration). Суррогат — ансамбль деревьев, неопределённость оценивается через дисперсию предсказаний деревьев.

### Acquisition Function

Acquisition function $\alpha(\lambda)$ балансирует **exploitation** (исследование уже хороших областей) и **exploration** (исследование неизвестных областей).

#### Expected Improvement (EI)

Самая популярная. «Насколько в среднем мы ожидаем улучшение по сравнению с текущим лучшим $y^*$?»

$$\text{EI}(\lambda) = \mathbb{E}\big[\max(0, y^* - f(\lambda))\big]$$

Для GP имеет аналитическую формулу:

$$\text{EI}(\lambda) = (y^* - \mu(\lambda))\,\Phi(Z) + \sigma(\lambda)\,\phi(Z), \quad Z = \frac{y^* - \mu(\lambda)}{\sigma(\lambda)}$$

где $\Phi$ — CDF, $\phi$ — PDF стандартного нормального.

#### Другие acquisition functions

| Функция | Формула (для GP) | Свойство |
|---|---|---|
| **Probability of Improvement (PI)** | $\Phi\big(\frac{y^* - \mu}{\sigma}\big)$ | Exploitation-oriented |
| **Upper Confidence Bound (UCB)** | $\mu(\lambda) - \kappa \cdot \sigma(\lambda)$ | Настраиваемый баланс ($\kappa$) |
| **Expected Improvement (EI)** | см. выше | Баланс explore/exploit |
| **Knowledge Gradient** | — | Теоретически оптимальный, дорогой |

### Optuna

**Optuna** (Akiba et al., 2019) — самый популярный фреймворк для Bayesian optimization в 2024-2026.

**Ключевые особенности:**
- **Define-by-Run** API — пространство поиска определяется в коде (а не в конфиге)
- **TPE** по умолчанию, также доступны GP, CMA-ES, Random Search
- **Pruning** — ранняя остановка неперспективных экспериментов (через callbacks)
- **Multi-objective** — оптимизация нескольких метрик одновременно (Pareto-front)
- **Distributed** — параллельный запуск на нескольких воркерах через RDB-хранилище
- **Dashboard** — визуализация через optuna-dashboard

### Hyperopt

**Hyperopt** (Bergstra et al., 2013) — более старый фреймворк, тоже использует TPE.

**Плюсы**: простой API, хорошая интеграция со Spark (`hyperopt.SparkTrials`).
**Минусы**: менее активно развивается, нет pruning из коробки. Сейчас **Optuna более популярен**.

---

## Bandit-based методы: Hyperband и BOHB

### Проблема

Bayesian Optimization тратит **полный бюджет** на каждую конфигурацию (обучает до конца). Но часто уже после нескольких эпох видно, что конфигурация плохая. Можно ли прекращать плохие раньше?

### Successive Halving

**Successive Halving** (Jamieson & Talwalkar, 2016) — последовательное отсечение половины конфигураций:

1. Стартовать $n$ случайных конфигураций, каждой дать бюджет $B/n$
2. Оценить качество, оставить лучшую половину
3. Удвоить бюджет оставшимся → повторить
4. Пока не останется одна конфигурация

```
Раунд 0:  64 конфигурации ×  1 эпоха   →  оценить, оставить 32
Раунд 1:  32 конфигурации ×  2 эпохи   →  оценить, оставить 16
Раунд 2:  16 конфигураций ×  4 эпохи   →  оценить, оставить 8
Раунд 3:   8 конфигураций ×  8 эпох    →  оценить, оставить 4
Раунд 4:   4 конфигурации × 16 эпох    →  оценить, оставить 2
Раунд 5:   2 конфигурации × 32 эпохи   →  оценить, оставить 1 ← победитель
```

### Hyperband

**Hyperband** (Li et al., 2017) — решает дилемму Successive Halving: «много конфигураций с малым бюджетом» vs «мало конфигураций с большим бюджетом».

**Решение**: запустить **несколько раундов** Successive Halving с разными начальными бюджетами:

| Bracket | Начальных конфигураций | Начальный бюджет | Стратегия |
|---|---|---|---|
| $s = 4$ | 81 | 1 эпоха | Агрессивное отсечение |
| $s = 3$ | 27 | 3 эпохи | |
| $s = 2$ | 9 | 9 эпох | |
| $s = 1$ | 6 | 27 эпох | |
| $s = 0$ | 5 | 81 эпоха | Полное обучение |

Каждый bracket — один запуск Successive Halving. Общий бюджет примерно одинаков для всех brackets.

### BOHB (Bayesian Optimization + HyperBand)

**BOHB** (Falkner et al., 2018) — комбинация: вместо случайного выбора конфигураций (как в Hyperband) используется **TPE** для умного выбора + **Hyperband** для ранней остановки.

$$\text{BOHB} = \underbrace{\text{Bayesian Opt (TPE)}}_{\text{умный выбор}\,\lambda} + \underbrace{\text{Hyperband}}_{\text{раннее отсечение}}$$

**Преимущество**: получает лучшее от обоих миров — эффективный выбор конфигураций и раннюю остановку.

> В Optuna есть встроенная поддержка pruning, которая по сути реализует аналог Successive Halving / Hyperband.

---

## Population-Based Training (PBT)

### Принцип работы

**PBT** (Jaderberg et al., 2017, DeepMind) — **эволюционный** подход к тюнингу, где популяция моделей обучается **параллельно**, периодически обмениваясь гиперпараметрами.

**Алгоритм:**
1. Инициализировать популяцию из $P$ моделей с разными $\lambda$
2. Обучать все модели параллельно
3. Периодически (каждые $T$ шагов):
   - **Exploit**: если модель в нижних 20% по качеству — заменить её веса на веса случайной модели из верхних 20%
   - **Explore**: мутировать гиперпараметры (случайное возмущение)
4. Продолжать обучение

```
Модель 1: ●───●───●───●═══●═══●═══●══→  (заменена на копию модели 3)
Модель 2: ●───●───●───●───●───●───●──→
Модель 3: ●───●───●───●───●───●───●──→  (лидер, копируется в модель 1)
Модель 4: ●───●───●───●───●═══●═══●══→  (заменена на копию модели 2)
          0   T   2T  3T  4T  5T  6T
```

**Ключевое отличие от других методов**: гиперпараметры меняются **во время обучения** (learning rate schedule обучается автоматически!).

**Где используется:**
- DeepMind для AlphaStar, обучения RL-агентов
- Подбор LR schedule для больших нейросетей
- Fine-tuning LLM

---

## Эволюционные алгоритмы (Evolutionary Strategies)

### Принцип работы

Наследуют идеи из биологической эволюции: **популяция → мутация → отбор → скрещивание**.

#### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**CMA-ES** (Hansen & Ostermeier, 2001) — один из лучших derivative-free оптимизаторов для непрерывных пространств.

**Алгоритм:**
1. Сэмплировать популяцию из многомерного нормального распределения:
   $$\lambda_i \sim \mathcal{N}\big(m, \sigma^2 C\big), \quad i = 1, \ldots, P$$
2. Оценить $f(\lambda_i)$ для каждого
3. Обновить среднее $m$ в сторону лучших:
   $$m \leftarrow \sum_{i=1}^{\mu} w_i \lambda_{i:\text{sorted}}$$
4. Адаптировать ковариационную матрицу $C$ и шаг $\sigma$

**Плюс**: не требует градиентов, хорошо работает в пространствах до ~100 измерений.
**Доступен в Optuna** как `CmaEsSampler`.

---

## Neural Architecture Search (NAS)

### Принцип работы

**NAS** — автоматический поиск **архитектуры** нейросети (не только гиперпараметров, но и самой структуры).

### Основные подходы

#### 1. NAS с Reinforcement Learning (Zoph & Le, 2017)

Контроллер (RNN) генерирует описание архитектуры → обучается через RL (reward = accuracy на валидации).

**Минус**: астрономические вычислительные затраты (800 GPU-дней в оригинальной работе).

#### 2. DARTS — Differentiable Architecture Search (Liu et al., 2019)

**Ключевая идея**: сделать выбор архитектуры **дифференцируемым**. Вместо дискретного выбора операций — взвешенная сумма всех кандидатов:

$$\bar{o}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o)}{\sum_{o'}\exp(\alpha_{o'})} \cdot o(x)$$

Оптимизация — **bilevel**:
- Внутренний уровень: обновляем веса $\theta$ (SGD)
- Внешний уровень: обновляем архитектурные параметры $\alpha$ (SGD на валидации)

**Плюс**: поиск за ~1 GPU-день (vs ~800 для RL-NAS).

#### 3. Efficient NAS (2023-2026)

- **Once-for-All (OFA)**: обучить одну суперсеть, из которой извлекаются подсети
- **Zero-Cost NAS**: оценка архитектуры **без обучения** (по градиентам, спектру якобиана)
- **Hardware-Aware NAS**: оптимизация не только accuracy, но и latency/memory

### Где используется NAS

| Модель | Метод NAS | Результат |
|---|---|---|
| NASNet | RL-NAS | Архитектура для ImageNet |
| EfficientNet | Compound scaling + NAS | Семейство эффективных CNN |
| MobileNetV3 | Platform-Aware NAS | Мобильные архитектуры |
| Once-for-All | OFA | Адаптация под hardware |

---

## Автоматический подбор Learning Rate

Learning rate — самый важный гиперпараметр нейросети. Существуют методы его полуавтоматического подбора.

### LR Range Test / LR Finder

**LR Finder** (Smith, 2017) — метод визуального определения оптимального LR:

1. Начать с очень маленького LR ($10^{-7}$)
2. Постепенно увеличивать LR за одну эпоху (линейно или экспоненциально)
3. Записывать loss на каждом шаге
4. Оптимальный LR — в точке **наибольшего убывания** loss (не минимума!)

```
Loss
 ↑
 │\
 │ \        (хороший LR — здесь, наибольший наклон)
 │  \_____
 │        \
 │         \___/  ← loss начинает расти (LR слишком большой)
 └──────────────→ LR
  10⁻⁷          10⁻¹
```

### Learning Rate Schedules

| Schedule | Формула | Когда |
|---|---|---|
| **Step Decay** | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$ | Классические CNN |
| **Cosine Annealing** | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos\frac{\pi t}{T})$ | Современные модели |
| **OneCycleLR** | Warm-up → max → cosine decay | Быстрая конвергенция |
| **Warmup + Linear Decay** | $\eta_t = \eta_{\max} \cdot \min\big(\frac{t}{T_w}, 1 - \frac{t - T_w}{T - T_w}\big)$ | LLM pre-training |
| **Warmup + Cosine** | Warm-up, затем cosine decay | Transformers, ViT |

**OneCycleLR** (Smith & Topin, 2019) — особенно популярен:
- Фаза 1: LR растёт от $\eta_0/\text{div}$ до $\eta_0$
- Фаза 2: LR уменьшается от $\eta_0$ до $\eta_0/\text{div}$
- Фаза 3: LR уменьшается до $\eta_0 / (10 \cdot \text{div})$

---

## Практические рекомендации

### Порядок тюнинга (по важности)

**Для нейросетей:**
1. **Learning rate** — самый важный, начните с него
2. **Batch size** — влияет на обобщение и скорость
3. **Число слоёв / hidden dim** — ёмкость модели
4. **Weight decay** — регуляризация
5. **Dropout** — если есть переобучение
6. **Остальные** — scheduler, warm-up steps и т.д.

**Для GBDT (XGBoost/LightGBM):**
1. **n_estimators + learning_rate** — вместе (больше деревьев + меньший LR)
2. **max_depth / num_leaves** — сложность деревьев
3. **subsample + colsample_bytree** — регуляризация
4. **min_child_weight** — предотвращение переобучения
5. **reg_alpha, reg_lambda** — L1/L2 регуляризация

### Общие советы

1. **Начинайте с Random Search** для разведки пространства, затем Bayesian Optimization для уточнения
2. **Логарифмическая шкала** для LR, weight decay, регуляризации
3. **Не тюньте всё сразу** — фиксируйте одни, тюньте другие
4. **Early stopping** — по валидационной метрике, экономит ресурсы
5. **Воспроизводимость** — фиксируйте seed, логируйте всё
6. **Бюджет**: обычно 50-200 экспериментов с Bayesian Opt дают хороший результат

---

## Сравнение методов

| Метод | Число экспериментов | Использует историю | Раннее отсечение | Параллелизм | Сложность реализации |
|---|---|---|---|---|---|
| **Grid Search** | $n^d$ (экспоненциально) | Нет | Нет | Полный | Минимальная |
| **Random Search** | Любое | Нет | Нет | Полный | Минимальная |
| **Bayesian Opt (GP)** | 10-200 | **Да** | Нет | Ограничен | Средняя |
| **Bayesian Opt (TPE/Optuna)** | 10-200 | **Да** | **Да** (pruning) | **Да** | Низкая (Optuna) |
| **Hyperband** | Зависит от brackets | Нет | **Да** | В рамках bracket | Средняя |
| **BOHB** | Зависит от brackets | **Да** | **Да** | Да | Средняя |
| **PBT** | Популяция × эпохи | Да (неявно) | Да (exploit) | Полный | Высокая |
| **CMA-ES** | 50-500 | **Да** | Нет | Ограничен | Низкая (Optuna) |
| **NAS (DARTS)** | 1 (дифференцируемый) | — | — | — | Высокая |

---

## Что используется больше всего (2024-2026)

### Табличные данные / ML

| Место | Метод | Инструмент | Комментарий |
|---|---|---|---|
| 1 | **Optuna (TPE)** | `optuna` | Стандарт индустрии, define-by-run |
| 2 | **Random Search + Early Stopping** | sklearn, manual | Простой baseline |
| 3 | **Hyperband / ASHA** | Ray Tune, Optuna | Для нейросетей с ранней остановкой |

### Deep Learning

| Место | Метод | Инструмент | Комментарий |
|---|---|---|---|
| 1 | **Optuna + Pruning** | `optuna` | TPE + MedianPruner/SuccessiveHalving |
| 2 | **LR Finder + OneCycleLR** | `torch-lr-finder`, PyTorch | Полуавтоматический для LR |
| 3 | **Ray Tune + ASHA** | `ray[tune]` | Масштабируемый, distributed |
| 4 | **W&B Sweeps** | `wandb` | Bayesian + визуализация |
| 5 | **PBT** | Ray Tune | Для RL и больших моделей |

### LLM Fine-Tuning

| Место | Метод | Комментарий |
|---|---|---|
| 1 | **Manual + LR Finder** | Мало гиперпараметров (LR, LoRA rank, epochs) |
| 2 | **Optuna** | Если бюджет позволяет |
| 3 | **Grid Search** | Мало параметров → допустимо |

### Рекомендации по выбору

```
Сколько у вас бюджета (GPU-часов)?
├── Мало (< 20 экспериментов)
│   ├── Мало гиперпараметров (< 4) → Grid Search или Random Search
│   └── Много гиперпараметров → Random Search
├── Средне (20-200 экспериментов)
│   └── Optuna (TPE) + Pruning
├── Много (> 200 экспериментов)
│   ├── Нейросети → Hyperband / BOHB / ASHA
│   ├── RL → PBT
│   └── Ищете архитектуру → NAS (DARTS)
└── Гиперпараметров > 20, непрерывные → CMA-ES
```

---

## Примеры кода

### Grid Search (sklearn)

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

print(f"Best ROC AUC: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")
```

### Random Search (sklearn)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform

param_distributions = {
    'n_estimators': randint(50, 1000),
    'max_depth': randint(3, 12),
    'learning_rate': loguniform(1e-3, 3e-1),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_samples_leaf': randint(1, 20),
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X_train, y_train)

print(f"Best ROC AUC: {random_search.best_score_:.4f}")
print(f"Best params: {random_search.best_params_}")
```

### Optuna (Bayesian Optimization)

```python
import optuna
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 100, log=True),
        'verbose': -1,
    }

    model = lgb.LGBMClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    return scores.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best ROC AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Optuna с Pruning для нейросетей (PyTorch)

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import torch.nn as nn

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    n_layers = trial.suggest_int('n_layers', 2, 6)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    layers = []
    in_dim = input_dim
    for i in range(n_layers):
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(50):
        train_loss = train_one_epoch(model, optimizer, train_loader)
        val_acc = evaluate(model, val_loader)

        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc


study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
study.optimize(objective, n_trials=50)
```

### Optuna — визуализация результатов

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_slice,
)

plot_optimization_history(study).show()

plot_param_importances(study).show()

plot_contour(study, params=['learning_rate', 'max_depth']).show()

plot_slice(study).show()
```

### LR Finder (PyTorch)

```python
from torch_lr_finder import LRFinder

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
lr_finder.range_test(train_loader, end_lr=1, num_iter=200, step_mode='exp')
lr_finder.plot()
lr_finder.reset()

# Типично: выбрать LR ≈ на порядок меньше минимума loss
# Если минимум при 0.01 → используйте 0.001
```

### Ray Tune (Distributed HPO)

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

search_space = {
    'lr': tune.loguniform(1e-5, 1e-2),
    'batch_size': tune.choice([32, 64, 128, 256]),
    'hidden_dim': tune.choice([128, 256, 512]),
    'n_layers': tune.randint(2, 7),
}

scheduler = ASHAScheduler(
    max_t=100,            # макс. эпох
    grace_period=10,      # мин. эпох до отсечения
    reduction_factor=2,   # отсекать половину
)

analysis = tune.run(
    train_fn,
    config=search_space,
    num_samples=50,
    scheduler=scheduler,
    metric='val_acc',
    mode='max',
    resources_per_trial={'cpu': 2, 'gpu': 0.5},
)

print(f"Best config: {analysis.best_config}")
```

---

## References

### Внутренние ссылки (knowledge-book)
- [Decision Trees](./decision-trees.md) — гиперпараметры деревьев (max_depth, min_samples_leaf)
- [Ensemble Methods & Model Combination](./ensemble-methods-model-combination.md) — тюнинг ансамблей (n_estimators, learning_rate для GBDT)
- [ROC Curves and ROC AUC](./roc-curve-and-roc-auc.md) — ROC AUC как метрика для тюнинга
- [Cross Entropy and Focal Loss](./classification-losses-cross-entropy-focal-loss.md) — loss-функции, через которые оценивается качество
- [Low-Rank Adaptation (LoRA)](./low-rank-adaptation-lora.md) — гиперпараметры LoRA (rank, alpha)
- [Normalization Layers](./normalization-layers-batchnorm-layernorm.md) — BatchNorm/LayerNorm как часть архитектуры
- [Bayes' Theorem](./bayes-theorem-and-probability-foundations.md) — математическая основа Bayesian Optimization

### Внешние ссылки
- Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR
- Bergstra, J. et al. (2013). *Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions*. ICML (Hyperopt)
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD
- Snoek, J. et al. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS
- Li, L. et al. (2017). *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*. JMLR
- Falkner, S. et al. (2018). *BOHB: Robust and Efficient Hyperparameter Optimization at Scale*. ICML
- Jaderberg, M. et al. (2017). *Population Based Training of Neural Networks*. arXiv
- Hansen, N. & Ostermeier, A. (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evolutionary Computation
- Liu, H. et al. (2019). *DARTS: Differentiable Architecture Search*. ICLR
- Smith, L. (2017). *Cyclical Learning Rates for Training Neural Networks*. WACV
- Smith, L. & Topin, N. (2019). *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*. SPIE
- Zoph, B. & Le, Q. (2017). *Neural Architecture Search with Reinforcement Learning*. ICLR
- Optuna documentation: https://optuna.readthedocs.io/
- Ray Tune documentation: https://docs.ray.io/en/latest/tune/
