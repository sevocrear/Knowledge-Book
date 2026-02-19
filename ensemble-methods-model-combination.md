# Методы комбинирования моделей (Ensemble Methods)

## Как объяснить 5-летнему ребёнку

Представь, ты хочешь угадать, сколько конфет в банке. Если спросишь одного друга — он может ошибиться. Но если спросишь 10 друзей и возьмёшь среднее число — ответ будет гораздо точнее! Вот это и есть ансамбль: вместо одного «угадывателя» мы берём много и объединяем их ответы. Одни друзья лучше считают маленькие банки, другие — большие, а вместе они почти не ошибаются.

---

## Table of Contents

1. [Зачем комбинировать модели](#зачем-комбинировать-модели)
2. [Таксономия методов](#таксономия-методов)
3. [Bagging (Bootstrap Aggregating)](#bagging-bootstrap-aggregating)
   - [Random Forest](#random-forest)
4. [Boosting](#boosting)
   - [AdaBoost](#adaboost)
   - [Gradient Boosting](#gradient-boosting)
   - [XGBoost, LightGBM, CatBoost](#xgboost-lightgbm-catboost)
5. [Stacking (Stacked Generalization)](#stacking-stacked-generalization)
6. [Voting и Averaging](#voting-и-averaging)
7. [Mixture of Experts (MoE)](#mixture-of-experts-moe)
8. [Knowledge Distillation](#knowledge-distillation)
9. [Model Merging для LLM](#model-merging-для-llm)
10. [Ансамбли в Deep Learning](#ансамбли-в-deep-learning)
    - [Snapshot Ensembles](#snapshot-ensembles)
    - [Test-Time Augmentation (TTA)](#test-time-augmentation-tta)
    - [Stochastic Weight Averaging (SWA)](#stochastic-weight-averaging-swa)
    - [Multi-head / Multi-task подходы](#multi-head--multi-task-подходы)
11. [Сравнение методов](#сравнение-методов)
12. [Примеры кода на Python](#примеры-кода-на-python)
13. [Что используется больше всего сейчас (2024-2026)](#что-используется-больше-всего-сейчас-2024-2026)
14. [References](#references)

---

## Зачем комбинировать модели

Каждая отдельная модель имеет три источника ошибки:

| Источник ошибки | Описание | Что помогает |
|---|---|---|
| **Bias** (смещение) | Модель слишком проста, не может уловить закономерности | Boosting, более сложные базовые модели |
| **Variance** (дисперсия) | Модель слишком чувствительна к конкретным данным | Bagging, усреднение |
| **Noise** (шум) | Неустранимая ошибка в данных | Ничего (ни один метод не уберёт) |

**Bias-Variance Decomposition** — для MSE ожидаемая ошибка модели $f$ на точке $x$:

$$\text{Error}(x) = \text{Bias}^2(x) + \text{Variance}(x) + \sigma^2_{\text{noise}}$$

Основная идея ансамблей: **объединение нескольких моделей снижает variance и/или bias**, при этом noise остаётся неизменным.

### Почему это работает (математически)

Пусть у нас $M$ моделей $f_1, \ldots, f_M$ с одинаковой дисперсией $\sigma^2$ и попарной корреляцией $\rho$. Дисперсия усреднённого предсказания:

$$\text{Var}\left(\frac{1}{M}\sum_{i=1}^{M} f_i\right) = \rho \cdot \sigma^2 + \frac{1-\rho}{M}\cdot\sigma^2$$

- Если модели **независимы** ($\rho = 0$), дисперсия уменьшается в $M$ раз: $\frac{\sigma^2}{M}$
- Если модели **полностью коррелированы** ($\rho = 1$), дисперсия не уменьшается: $\sigma^2$

**Вывод**: чем **разнообразнее** модели в ансамбле, тем сильнее эффект.

---

## Таксономия методов

```
Методы комбинирования моделей
├── Параллельные (независимые модели)
│   ├── Bagging (Random Forest)
│   ├── Voting / Averaging
│   └── Stacking / Blending
├── Последовательные (каждая следующая исправляет предыдущую)
│   ├── AdaBoost
│   ├── Gradient Boosting (XGBoost, LightGBM, CatBoost)
│   └── Residual connections (в DL — остаточные связи)
├── Learned комбинирование
│   ├── Mixture of Experts (MoE)
│   ├── Stacking (мета-обучение)
│   └── Multi-head / Multi-task
├── Компрессия ансамблей
│   ├── Knowledge Distillation
│   └── Model Merging (TIES, DARE, SLERP)
└── Техники для DL
    ├── Snapshot Ensembles
    ├── Test-Time Augmentation (TTA)
    ├── Stochastic Weight Averaging (SWA)
    └── MC Dropout
```

---

## Bagging (Bootstrap Aggregating)

### Принцип работы

**Bagging** (Breiman, 1996) — параллельное обучение $M$ моделей на **бутстрэп-выборках** (случайные выборки с возвращением из обучающего набора), затем усреднение предсказаний.

**Алгоритм:**
1. Из обучающего набора $D$ размера $N$ создать $M$ бутстрэп-выборок $D_1, \ldots, D_M$ (каждая размера $N$, сэмплирование с возвращением)
2. Обучить на каждой выборке $D_i$ независимую модель $f_i$
3. Объединить предсказания:
   - **Регрессия**: $\hat{y} = \frac{1}{M}\sum_{i=1}^{M} f_i(x)$
   - **Классификация**: $\hat{y} = \text{mode}\{f_1(x), \ldots, f_M(x)\}$ (голосование большинства)

> **Факт**: в каждой бутстрэп-выборке ~63.2% уникальных объектов (остальные ~36.8% — Out-of-Bag, OOB — можно использовать для оценки качества без отдельной валидации).

### Random Forest

**Random Forest** (Breiman, 2001) — Bagging + **рандомизация признаков** при каждом разбиении в дереве решений.

На каждом сплите узла:
1. Случайно выбрать $m$ признаков из $p$ доступных (обычно $m = \sqrt{p}$ для классификации, $m = p/3$ для регрессии)
2. Найти лучший сплит только среди этих $m$ признаков
3. Повторять до остановки

**Почему это работает:** рандомизация признаков **декоррелирует** деревья (снижает $\rho$ в формуле выше), что ещё сильнее снижает дисперсию.

**Гиперпараметры:**

| Параметр | Описание | Типичное значение |
|---|---|---|
| `n_estimators` | Число деревьев | 100–1000 |
| `max_features` | Число признаков на сплите | $\sqrt{p}$ или $p/3$ |
| `max_depth` | Максимальная глубина дерева | None (без ограничений) |
| `min_samples_leaf` | Минимум объектов в листе | 1–5 |

**Feature Importance** в Random Forest:

$$\text{Importance}(j) = \frac{1}{M} \sum_{m=1}^{M} \sum_{t \in T_m} \Delta I(t) \cdot \mathbb{1}[\text{feature}(t) = j]$$

где $\Delta I(t)$ — уменьшение impurity при сплите $t$, а $T_m$ — множество всех узлов дерева $m$.

---

## Boosting

### Общая идея

**Boosting** — семейство **последовательных** методов, где каждая новая модель фокусируется на ошибках предыдущих. В отличие от Bagging, Boosting снижает **bias** (смещение).

$$F_M(x) = \sum_{m=1}^{M} \alpha_m \cdot h_m(x)$$

где $h_m$ — «слабый» базовый классификатор, $\alpha_m$ — его вес.

### AdaBoost

**AdaBoost** (Freund & Schapire, 1997) — адаптивный бустинг с перевзвешиванием объектов.

**Алгоритм:**
1. Инициализировать веса объектов: $w_i^{(1)} = \frac{1}{N}$
2. Для $m = 1, \ldots, M$:
   - Обучить слабый классификатор $h_m$ на взвешенной выборке
   - Вычислить взвешенную ошибку:
     $$\epsilon_m = \frac{\sum_{i=1}^{N} w_i^{(m)} \cdot \mathbb{1}[y_i \ne h_m(x_i)]}{\sum_{i=1}^{N} w_i^{(m)}}$$
   - Вычислить вес классификатора:
     $$\alpha_m = \frac{1}{2}\ln\frac{1 - \epsilon_m}{\epsilon_m}$$
   - Обновить веса объектов:
     $$w_i^{(m+1)} = w_i^{(m)} \cdot \exp(-\alpha_m \cdot y_i \cdot h_m(x_i))$$
   - Нормировать веса
3. Финальное предсказание: $F(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m \cdot h_m(x)\right)$

**Интуиция**: объекты, на которых ошиблась предыдущая модель, получают больший вес → следующая модель «концентрируется» на сложных случаях.

### Gradient Boosting

**Gradient Boosting** (Friedman, 2001) — обобщение бустинга через **градиентный спуск в пространстве функций**.

**Ключевая идея**: вместо перевзвешивания объектов, каждое новое дерево обучается предсказывать **отрицательный градиент функции потерь** (псевдо-остатки):

$$r_i^{(m)} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$

**Алгоритм:**
1. Инициализация: $F_0(x) = \arg\min_c \sum_{i=1}^{N} L(y_i, c)$
2. Для $m = 1, \ldots, M$:
   - Вычислить псевдо-остатки $r_i^{(m)}$
   - Обучить дерево $h_m$ на $(x_i, r_i^{(m)})$
   - Найти оптимальный шаг: $\gamma_m = \arg\min_\gamma \sum_{i=1}^{N} L(y_i, F_{m-1}(x_i) + \gamma \cdot h_m(x_i))$
   - Обновить модель: $F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m \cdot h_m(x)$

где $\eta$ — learning rate (обычно 0.01–0.3).

> Для **MSE** псевдо-остатки — это буквально остатки $r_i = y_i - F_{m-1}(x_i)$.
> Для **log-loss** псевдо-остатки — это $r_i = y_i - p_i$, где $p_i = \sigma(F_{m-1}(x_i))$.

### XGBoost, LightGBM, CatBoost

Это три самых популярных промышленных реализации градиентного бустинга (и одни из самых используемых ML-моделей в мире).

#### XGBoost (Chen & Guestrin, 2016)

**Ключевые улучшения:**
- **Регуляризация** — L1 и L2 штрафы на веса листьев
- **Оптимизация второго порядка** — использует не только градиент, но и гессиан:
  $$\mathcal{L}^{(m)} \approx \sum_{i=1}^{N} \left[ g_i h_m(x_i) + \frac{1}{2} h_i h_m^2(x_i) \right] + \Omega(h_m)$$
  где $g_i = \partial L / \partial F_{m-1}$, $h_i = \partial^2 L / \partial F_{m-1}^2$
- **Column subsampling** — как в Random Forest, случайный выбор признаков
- **Histogram-based splits** — быстрый поиск сплитов
- **Параллелизм** на уровне построения отдельных деревьев

#### LightGBM (Ke et al., 2017)

**Ключевые улучшения:**
- **Leaf-wise** рост дерева (вместо level-wise у XGBoost) — быстрее и точнее, но может переобучаться
- **GOSS** (Gradient-based One-Side Sampling) — сохраняет объекты с большим градиентом, сэмплирует из остальных
- **EFB** (Exclusive Feature Bundling) — объединяет разреженные взаимоисключающие признаки
- **Значительно быстрее** XGBoost на больших датасетах

#### CatBoost (Prokhorenkova et al., 2018, Яндекс)

**Ключевые улучшения:**
- **Ordered Target Encoding** — умная обработка категориальных признаков без target leakage
- **Ordered Boosting** — борьба с target leakage при вычислении остатков
- **Symmetric trees** — все сплиты на одном уровне одинаковые → быстрый инференс
- **Лучшее качество из коробки** на данных с категориальными признаками

#### Сравнение

| Критерий | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Скорость обучения | Средняя | **Высокая** | Средняя |
| Скорость инференса | Средняя | Высокая | **Очень высокая** |
| Категориальные признаки | Требует кодирование | Поддерживает | **Нативная поддержка** |
| Качество из коробки | Хорошее | Хорошее | **Отличное** |
| GPU-ускорение | Да | Да | Да |
| Использование в Kaggle | Очень популярен | **Самый популярный** | Популярен |

---

## Stacking (Stacked Generalization)

### Принцип работы

**Stacking** (Wolpert, 1992) — обучение **мета-модели**, которая учится комбинировать предсказания базовых моделей.

**Двухуровневая архитектура:**

```
Уровень 0 (базовые модели):
  Модель 1 (RF)       → pred_1
  Модель 2 (XGBoost)  → pred_2
  Модель 3 (SVM)      → pred_3
  Модель 4 (kNN)      → pred_4

Уровень 1 (мета-модель):
  Мета-модель (LogReg) ← [pred_1, pred_2, pred_3, pred_4] → final_pred
```

**Алгоритм с кросс-валидацией (для предотвращения переобучения):**

1. Разделить обучающий набор на $K$ фолдов
2. Для каждого базового алгоритма $f_j$:
   - Для каждого фолда $k$:
     - Обучить $f_j$ на всех фолдах, кроме $k$
     - Предсказать $f_j$ на фолде $k$
   - Собрать OOF-предсказания (out-of-fold) — это мета-признаки
3. Обучить мета-модель на мета-признаках
4. Для тестовых данных: получить предсказания всех базовых моделей (обученных на всём train), подать в мета-модель

**Почему работает**: мета-модель учится, **когда и какой** базовой модели доверять. Например, если Random Forest хорошо работает для одного типа данных, а SVM — для другого, мета-модель это обнаружит.

---

## Voting и Averaging

Самые простые способы комбинирования:

### Hard Voting (Жёсткое голосование)

$$\hat{y} = \text{mode}\{f_1(x), f_2(x), \ldots, f_M(x)\}$$

Каждая модель «голосует» за класс, побеждает класс с большинством голосов.

### Soft Voting (Мягкое голосование)

$$\hat{y} = \arg\max_c \sum_{i=1}^{M} w_i \cdot P_i(y = c \mid x)$$

Усредняются **вероятности** (возможно, с весами $w_i$). Обычно лучше Hard Voting, т.к. учитывает «уверенность» моделей.

### Weighted Averaging (для регрессии)

$$\hat{y} = \sum_{i=1}^{M} w_i \cdot f_i(x), \quad \sum w_i = 1$$

Веса можно подобрать через оптимизацию на валидации.

---

## Mixture of Experts (MoE)

### Принцип работы

**Mixture of Experts** — архитектура, в которой **маршрутизатор** (gating network) направляет каждый вход к подмножеству «экспертов». В отличие от обычного ансамбля, **не все эксперты активируются для каждого входа** (sparse activation).

### Классический MoE

$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

где:
- $E_i(x)$ — выход $i$-го эксперта
- $g_i(x)$ — вес маршрутизатора для $i$-го эксперта, $\sum_i g_i(x) = 1$

Маршрутизатор:

$$g(x) = \text{Softmax}(W_g \cdot x)$$

### Sparse MoE (Top-k routing)

В современных LLM используется **sparse** вариант — активируются только $k$ из $N$ экспертов:

$$y = \sum_{i \in \text{TopK}(g(x), k)} g_i(x) \cdot E_i(x)$$

**Пример**: Mixtral 8×7B — 8 экспертов, для каждого токена активируются 2 ($k=2$). Общее число параметров ~47B, но для каждого токена используется только ~13B.

### Архитектура MoE в Transformer

```
Input Token
    ↓
[Self-Attention]
    ↓
[Router / Gating Network]  ← Решает, какие эксперты активировать
    ↓
┌──────┬──────┬──────┬──────┐
│ FFN₁ │ FFN₂ │ FFN₃ │ FFN₄ │  ← Эксперты (каждый — FFN блок)
└──────┴──────┴──────┴──────┘
    ↓ (weighted sum top-k)
Output Token
```

**Ключевые проблемы и решения:**

| Проблема | Решение |
|---|---|
| Load balancing (все токены идут к одному эксперту) | Auxiliary loss для балансировки нагрузки |
| Нестабильность обучения | Router z-loss, Expert Capacity Factor |
| Коммуникация между GPU | Expert Parallelism |

### Современные MoE модели (2024-2026)

| Модель | Эксперты | Top-k | Параметры (total / active) |
|---|---|---|---|
| Mixtral 8×7B | 8 | 2 | 47B / 13B |
| Mixtral 8×22B | 8 | 2 | 141B / 39B |
| DeepSeek-V2 | 160 | 6 | 236B / 21B |
| DeepSeek-V3 | 256 | 8 | 671B / 37B |
| Grok-1 | 8 | 2 | 314B / ~86B |
| DBRX | 16 | 4 | 132B / 36B |

---

## Knowledge Distillation

### Принцип работы

**Knowledge Distillation** (Hinton et al., 2015) — перенос «знаний» из большой модели-учителя (teacher) в маленькую модель-ученика (student).

**Ключевая идея**: soft-labels (вероятности) учителя содержат больше информации, чем hard-labels (0/1). Например, если учитель предсказывает "кот: 0.7, тигр: 0.2, собака: 0.1" — это говорит ученику, что кот похож на тигра.

### Функция потерь

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\big(p_T(x; T) \| p_S(x; T)\big) + (1-\alpha) \cdot \text{CE}\big(y, p_S(x; 1)\big)$$

где:
- $p_T(x; T)$ — soft-labels учителя при температуре $T$
- $p_S(x; T)$ — soft-labels ученика при температуре $T$
- $T$ — температура (обычно 2–20), «размягчает» распределение вероятностей:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $\alpha$ — баланс между imitation loss и task loss (обычно 0.5–0.9)

### Типы дистилляции

```
1. Response-based: ученик повторяет выходы учителя
   Teacher output → Student output (soft labels)

2. Feature-based: ученик повторяет промежуточные представления
   Teacher features → Student features (intermediate layers)

3. Relation-based: ученик повторяет отношения между примерами
   Teacher relations → Student relations (similarity matrices)
```

### Дистилляция в LLM (2024-2026)

В мире LLM дистилляция приобрела новые формы:

- **On-policy distillation** — ученик генерирует текст, учитель оценивает (DeepSeek-R1)
- **Chain-of-Thought distillation** — передача цепочек рассуждений от большой модели
- **Self-distillation** — модель дистиллирует саму себя (Born-Again Networks)

---

## Model Merging для LLM

### Принцип работы

**Model Merging** — объединение весов нескольких fine-tuned моделей **без дополнительного обучения**. Очень популярная техника в 2024-2026 для LLM.

### Методы

#### 1. Linear Interpolation (LERP)

$$\theta_{\text{merged}} = (1 - \lambda) \cdot \theta_A + \lambda \cdot \theta_B$$

Простое линейное смешивание весов двух моделей. Работает, если модели fine-tuned из одной базовой.

#### 2. SLERP (Spherical Linear Interpolation)

Интерполяция по поверхности гиперсферы — сохраняет нормы и более гладко смешивает:

$$\theta_{\text{merged}} = \frac{\sin((1-t)\Omega)}{\sin\Omega} \theta_A + \frac{\sin(t\Omega)}{\sin\Omega} \theta_B$$

где $\Omega = \arccos\left(\frac{\theta_A \cdot \theta_B}{\|\theta_A\| \|\theta_B\|}\right)$.

#### 3. TIES-Merging (Yadav et al., 2023)

**T**rim, **I**ncrease, **E**lect **S**ign:
1. **Trim**: обнулить мелкие дельты (параметры, близкие к base model)
2. **Elect Sign**: для каждого параметра выбрать знак, за который «голосует» большинство моделей
3. **Merge**: усреднить, оставив только параметры с выбранным знаком

#### 4. DARE (Yu et al., 2024)

**D**rop **A**nd **RE**scale:
1. Случайно обнулить часть дельт (параметров, отличающихся от base model) с вероятностью $p$
2. Пере-масштабировать оставшиеся: $\delta_{\text{new}} = \frac{\delta}{1-p}$
3. Объединить с TIES или линейно

#### 5. Model Soups (Wortsman et al., 2022)

Усреднение весов нескольких моделей, обученных с разными гиперпараметрами:
- **Uniform soup**: простое усреднение всех
- **Greedy soup**: добавлять модели по одной, только если качество растёт

### Почему Model Merging работает

- Fine-tuned модели из одной базовой лежат в **одном бассейне loss landscape** (линейное соединение не проходит через барьеры)
- Разные fine-tuned модели выучивают **разные знания** (код, математика, диалог) — слияние объединяет их
- Это **бесплатный ансамбль** — не требует дополнительной памяти при инференсе

---

## Ансамбли в Deep Learning

### Snapshot Ensembles

**Snapshot Ensembles** (Huang et al., 2017) — ансамбль из чекпоинтов **одного обучения** с циклическим learning rate.

**Идея**: при обучении с cosine annealing LR модель проходит через несколько «бассейнов» в loss landscape. Сохраняем snapshot в каждом минимуме → получаем ансамбль бесплатно.

```
Learning Rate
  ↑
  │  /\   /\   /\
  │ /  \ /  \ /  \
  │/    ↓    ↓    ↓
  └─────────────────→ Epochs
        s₁   s₂   s₃  ← snapshots
```

### Test-Time Augmentation (TTA)

Применение аугментаций к тестовому изображению и усреднение предсказаний:

1. Взять входное изображение $x$
2. Создать $K$ аугментаций: $\{x, \text{flip}(x), \text{rotate}(x, 90°), \ldots\}$
3. Получить предсказания для каждой
4. Усреднить: $\hat{y} = \frac{1}{K} \sum_{k=1}^{K} f(x_k)$

**Типичные аугментации для TTA**: горизонтальный flip, multi-scale resize, crop, rotation.

### Stochastic Weight Averaging (SWA)

**SWA** (Izmailov et al., 2018) — усреднение весов модели с нескольких эпох обучения:

$$\theta_{\text{SWA}} = \frac{1}{n} \sum_{i=1}^{n} \theta_i$$

где $\theta_i$ — веса после $i$-й эпохи с циклическим или постоянным LR.

Даёт **более плоские минимумы** → лучшая обобщающая способность. В PyTorch есть встроенная поддержка `torch.optim.swa_utils`.

### MC Dropout

**Monte Carlo Dropout** (Gal & Ghahramani, 2016) — dropout **остаётся включённым** при инференсе. Делаем $T$ прогонов → получаем ансамбль + оценку неопределённости:

$$\hat{y} = \frac{1}{T}\sum_{t=1}^{T} f(x; \theta, z_t), \quad z_t \sim \text{Bernoulli}(p)$$

$$\text{Uncertainty}(x) = \text{Var}\left[\{f(x; \theta, z_t)\}_{t=1}^{T}\right]$$

### Multi-head / Multi-task подходы

Одна shared backbone, несколько голов для разных задач:

```
          Input
            ↓
    [Shared Backbone]
     ↓      ↓      ↓
  Head₁  Head₂  Head₃
  (cls)  (det)  (seg)
```

Общие признаки действуют как **регуляризация**, а специализированные головы решают свои задачи. Это не совсем ансамбль, но комбинирование моделей в одной архитектуре.

---

## Сравнение методов

| Метод | Снижает Bias | Снижает Variance | Вычисления (train) | Вычисления (infer) | Когда использовать |
|---|---|---|---|---|---|
| **Bagging / RF** | Нет | **Да** | $M \times$ (параллельно) | $M \times$ | Высокая дисперсия, много данных |
| **Boosting** | **Да** | Немного | $M \times$ (последовательно) | $M \times$ | Недообучение, структурные данные |
| **Stacking** | **Да** | **Да** | $K \times M$ | $M + 1$ | Kaggle, максимум качества |
| **Voting** | Немного | **Да** | $M \times$ | $M \times$ | Простая комбинация разных моделей |
| **MoE** | **Да** | **Да** | $N \times$ (но sparse) | $k \times$ (только top-k) | LLM, масштабирование |
| **Distillation** | — | — | $2 \times$ | $1 \times$ (student) | Сжатие модели |
| **Model Merging** | — | — | $0 \times$ (нет доп. обучения) | $1 \times$ | LLM, бесплатный ансамбль |
| **TTA** | Немного | **Да** | $0 \times$ | $K \times$ | CV, максимум качества |
| **SWA** | Немного | **Да** | $\sim 1 \times$ | $1 \times$ | DL, бесплатное улучшение |

---

## Примеры кода на Python

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)

scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"RF ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

### XGBoost vs LightGBM vs CatBoost

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.1,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss'
)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.1,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    verbose=-1
)

# CatBoost
cb_model = CatBoostClassifier(
    iterations=300, learning_rate=0.1,
    depth=6, verbose=0
)

for name, model in [("XGBoost", xgb_model), ("LightGBM", lgb_model), ("CatBoost", cb_model)]:
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"{name} ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(),
    cv=5,  # кросс-валидация для мета-признаков
    n_jobs=-1
)

scores = cross_val_score(stacking, X, y, cv=5, scoring='roc_auc')
print(f"Stacking ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Soft Voting

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
    ],
    voting='soft',  # 'hard' для жёсткого голосования
    weights=[2, 3, 1]  # веса моделей (опционально)
)

scores = cross_val_score(voting, X, y, cv=5, scoring='roc_auc')
print(f"Soft Voting ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Knowledge Distillation (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    T — температура, alpha — вес мягких лейблов.
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# Пример использования в цикле обучения
# teacher_model.eval()
# for x, y in dataloader:
#     with torch.no_grad():
#         teacher_out = teacher_model(x)
#     student_out = student_model(x)
#     loss = distillation_loss(student_out, teacher_out, y, T=4.0, alpha=0.7)
#     loss.backward()
#     optimizer.step()
```

### Stochastic Weight Averaging (PyTorch)

```python
import torch
from torch.optim.swa_utils import AveragedModel, SWALR

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.01)
swa_start = 75

for epoch in range(100):
    train_one_epoch(model, optimizer, dataloader)

    if epoch >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()

# Обновить BatchNorm-статистики для SWA-модели
torch.optim.swa_utils.update_bn(dataloader, swa_model)
```

### Model Merging (mergekit)

```yaml
# config.yaml для mergekit
models:
  - model: base_model/math-expert
    parameters:
      weight: 0.4
  - model: base_model/code-expert
    parameters:
      weight: 0.3
  - model: base_model/chat-expert
    parameters:
      weight: 0.3
merge_method: ties  # или slerp, dare_ties, linear
base_model: base_model/llama-3-8b
parameters:
  density: 0.5
  normalize: true
dtype: float16
```

```bash
# Запуск mergekit
mergekit-yaml config.yaml ./merged_model --cuda
```

---

## Что используется больше всего сейчас (2024-2026)

### Табличные данные (Kaggle, продакшн)

| Место | Метод | Почему |
|---|---|---|
| 1 | **LightGBM / XGBoost / CatBoost** | Быстро, качественно, interpretable |
| 2 | **Stacking из GBDT моделей** | Максимум качества на соревнованиях |
| 3 | **Blending с нейросетями** | TabNet, FT-Transformer + GBDT |

> **Факт**: На Kaggle в 2024-2025 ~80% winning solutions на табличных данных используют ансамбли из LightGBM/XGBoost/CatBoost.

### Computer Vision

| Место | Метод | Почему |
|---|---|---|
| 1 | **Test-Time Augmentation (TTA)** | Бесплатное улучшение на инференсе |
| 2 | **SWA / EMA** | Exponential Moving Average весов — стандарт в CV |
| 3 | **Model Soups** | Усреднение нескольких обученных моделей |
| 4 | **Multi-scale inference** | Предсказание на разных масштабах |

### LLM и NLP

| Место | Метод | Почему |
|---|---|---|
| 1 | **Mixture of Experts (MoE)** | Масштабирование без линейного роста compute |
| 2 | **Model Merging (TIES, DARE, SLERP)** | Бесплатное объединение навыков |
| 3 | **Knowledge Distillation** | GPT-4 → маленькие модели, DeepSeek-R1 → открытые модели |
| 4 | **Speculative Decoding** | Маленькая модель-draft + большая модель-верификатор |

### Рекомендации по выбору

```
Есть ли у вас табличные данные?
  ├── Да → Начните с LightGBM/CatBoost
  │        └── Нужно максимальное качество? → Stacking
  └── Нет (изображения, текст, etc.)
       ├── Computer Vision → TTA + SWA/EMA
       ├── NLP/LLM задача
       │   ├── Обучаете с нуля → MoE
       │   ├── Fine-tuning → Model Merging
       │   └── Деплой → Knowledge Distillation
       └── Production с ограничениями
            └── Distillation → одна маленькая модель
```

---

## References

### Внутренние ссылки (knowledge-book)
- [Decision Trees](./decision-trees.md) — базовые модели для Bagging и Boosting
- [ROC Curves and ROC AUC](./roc-curve-and-roc-auc.md) — метрика оценки моделей в ансамблях
- [Cross Entropy and Focal Loss](./classification-losses-cross-entropy-focal-loss.md) — loss-функции, используемые в Boosting
- [Transformers, Attention and ViT](./transformers-attention-and-vision-transformers-vit.md) — архитектура для MoE
- [Low-Rank Adaptation (LoRA)](./low-rank-adaptation-lora.md) — fine-tuning перед Model Merging
- [Normalization Layers](./normalization-layers-batchnorm-layernorm.md) — BatchNorm и SWA
- [Hyperparameter Tuning](./hyperparameter-tuning.md) — настройка гиперпараметров ансамблей (n_estimators, learning_rate и т.д.)

### Внешние ссылки
- Breiman, L. (1996). *Bagging Predictors*. Machine Learning, 24(2)
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1)
- Freund, Y., & Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*
- Friedman, J. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS
- Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS
- Wolpert, D. (1992). *Stacked Generalization*. Neural Networks
- Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network*
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*
- Yadav, P. et al. (2023). *TIES-Merging: Resolving Interference When Merging Models*. NeurIPS
- Yu, L. et al. (2024). *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (DARE)
- Wortsman, M. et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time*. ICML
- Huang, G. et al. (2017). *Snapshot Ensembles: Train 1, Get M for Free*. ICLR
- Izmailov, P. et al. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization*. UAI
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML
