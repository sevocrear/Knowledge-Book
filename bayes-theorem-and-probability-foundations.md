# Теорема Байеса и основы теории вероятностей

## Как объяснить 5-летнему ребёнку

Представь, что у тебя есть коробка с конфетами двух цветов: красные и синие. Ты знаешь, что красных конфет больше. Теперь кто-то достал конфету, но ты не видел какую — тебе сказали только, что конфета сладкая. Теорема Байеса — это как волшебная формула, которая помогает угадать, какого цвета конфету достали, используя то, что ты уже знаешь (красных больше) и новую подсказку (она сладкая).

---

## Table of Contents

1. [Введение](#введение)
2. [Теоретико-множественные основы](#теоретико-множественные-основы)
3. [Аксиомы вероятности Колмогорова](#аксиомы-вероятности-колмогорова)
4. [Условная вероятность](#условная-вероятность)
5. [Независимость событий](#независимость-событий)
6. [Формула полной вероятности](#формула-полной-вероятности)
7. [Теорема Байеса](#теорема-байеса)
8. [Интуиция: почему теорема Байеса работает](#интуиция-почему-теорема-байеса-работает)
9. [Примеры и расчёты](#примеры-и-расчёты)
10. [Байесовский вывод в машинном обучении](#байесовский-вывод-в-машинном-обучении)
11. [Наивный Байесовский классификатор](#наивный-байесовский-классификатор)
12. [Связь с другими концепциями](#связь-с-другими-концепциями)
13. [Реализация на Python](#реализация-на-python)
14. [References](#references)

---

## Введение

**Теорема Байеса** — один из фундаментальных результатов теории вероятностей, названный в честь английского математика и пресвитерианского священника **Томаса Байеса** (1701–1761). Теорема описывает, как **обновлять вероятность гипотезы** при получении новых данных (evidence).

В машинном обучении и AI теорема Байеса лежит в основе:
- Наивного Байесовского классификатора
- Байесовских нейронных сетей
- Вариационного вывода (VAE)
- Гауссовских процессов
- Байесовской оптимизации гиперпараметров
- Фильтров Калмана и частичных фильтров

---

## Теоретико-множественные основы

Прежде чем перейти к вероятностям, нужно понять язык, на котором формулируется теория — **теорию множеств**.

### Пространство элементарных исходов

**Пространство элементарных исходов** $\Omega$ (omega) — это множество всех возможных результатов эксперимента.

**Примеры:**
- Бросок монеты: $\Omega = \{H, T\}$ (орёл, решка)
- Бросок кубика: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Измерение температуры: $\Omega = \mathbb{R}$ (все действительные числа)

### Событие

**Событие** $A$ — это подмножество $\Omega$, т.е. $A \subseteq \Omega$.

**Примеры:**
- Событие «выпало чётное число»: $A = \{2, 4, 6\}$
- Событие «температура выше 20°C»: $A = (20, +\infty)$

### Операции над событиями

| Операция | Обозначение | Смысл |
|----------|-------------|-------|
| Объединение | $A \cup B$ | Произошло $A$ **или** $B$ (или оба) |
| Пересечение | $A \cap B$ | Произошли **и** $A$, **и** $B$ |
| Дополнение | $\bar{A}$ или $A^c$ | $A$ **не** произошло |
| Разность | $A \setminus B$ | $A$ произошло, но $B$ — нет |

### Несовместные события

События $A$ и $B$ **несовместны** (mutually exclusive), если $A \cap B = \emptyset$.

### Разбиение пространства

Множество событий $\{H_1, H_2, \ldots, H_n\}$ образует **полную группу событий** (разбиение), если:
1. $H_i \cap H_j = \emptyset$ для $i \neq j$ (попарно несовместны)
2. $H_1 \cup H_2 \cup \ldots \cup H_n = \Omega$ (покрывают всё пространство)

![Диаграмма Венна](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Venn_diagram_ABC_BW_Explanation.png/440px-Venn_diagram_ABC_BW_Explanation.png)

---

## Аксиомы вероятности Колмогорова

В 1933 году **Андрей Николаевич Колмогоров** предложил аксиоматический подход к теории вероятностей, который лёг в основу современной математической статистики.

### Определение вероятностной меры

**Вероятность** — это функция $P: \mathcal{F} \to [0, 1]$, где $\mathcal{F}$ — σ-алгебра событий на $\Omega$, удовлетворяющая трём аксиомам:

### Аксиома 1: Неотрицательность

$$P(A) \geq 0 \quad \text{для любого события } A$$

Вероятность не может быть отрицательной.

### Аксиома 2: Нормировка

$$P(\Omega) = 1$$

Вероятность того, что произойдёт хоть что-то из возможного, равна 1.

### Аксиома 3: Счётная аддитивность

Для любой последовательности **попарно несовместных** событий $A_1, A_2, \ldots$:

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

Если события не пересекаются, вероятность их объединения равна сумме вероятностей.

### Следствия из аксиом

Из аксиом следуют важные свойства:

1. **Вероятность невозможного события:**
   $$P(\emptyset) = 0$$

2. **Вероятность дополнения:**
   $$P(\bar{A}) = 1 - P(A)$$

3. **Монотонность:**
   $$\text{Если } A \subseteq B, \text{ то } P(A) \leq P(B)$$

4. **Формула включений-исключений:**
   $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

5. **Ограниченность:**
   $$0 \leq P(A) \leq 1$$

---

## Условная вероятность

**Условная вероятность** — ключевое понятие, на котором строится теорема Байеса.

### Определение

**Условная вероятность** события $A$ при условии, что произошло событие $B$ (где $P(B) > 0$):

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### Интуиция

Условная вероятность отвечает на вопрос: **«Какова вероятность $A$, если мы уже знаем, что $B$ произошло?»**

Мы как бы «сужаем» пространство исходов с $\Omega$ до $B$ и смотрим, какую часть от $B$ составляет пересечение $A \cap B$.

![Условная вероятность](https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Venn_diagram_for_conditional_probability.svg/440px-Venn_diagram_for_conditional_probability.svg.png)

### Пример

Бросаем кубик. Событие $A$ = «выпало 6», событие $B$ = «выпало чётное число».

$$P(A) = \frac{1}{6}, \quad P(B) = \frac{3}{6} = \frac{1}{2}, \quad P(A \cap B) = P(A) = \frac{1}{6}$$

$$P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{1/6}{1/2} = \frac{1}{3}$$

Интуиция: если мы знаем, что выпало чётное число (2, 4 или 6), то из трёх равновероятных исходов только один — шестёрка.

### Правило умножения

Из определения условной вероятности следует **правило умножения**:

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

Это равенство — ключ к выводу теоремы Байеса.

### Цепное правило (Chain Rule)

Для нескольких событий:

$$P(A_1 \cap A_2 \cap \ldots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdot \ldots \cdot P(A_n|A_1 \cap \ldots \cap A_{n-1})$$

---

## Независимость событий

### Определение

События $A$ и $B$ **независимы**, если выполнение одного из них не влияет на вероятность другого:

$$P(A|B) = P(A) \quad \text{или эквивалентно} \quad P(A \cap B) = P(A) \cdot P(B)$$

### Условная независимость

События $A$ и $B$ **условно независимы при условии $C$**, если:

$$P(A \cap B | C) = P(A|C) \cdot P(B|C)$$

Это ключевое понятие для **Наивного Байесовского классификатора**.

### Важно!

Независимость — это **не** то же самое, что несовместность:
- **Несовместные события** ($A \cap B = \emptyset$): если произошло $A$, то $B$ точно не произошло → они **зависимы**.
- **Независимые события**: знание о $A$ не меняет вероятность $B$.

---

## Формула полной вероятности

Формула полной вероятности позволяет вычислить вероятность события через разбиение пространства на части.

### Формулировка

Пусть $\{H_1, H_2, \ldots, H_n\}$ — полная группа гипотез (разбиение $\Omega$), и $P(H_i) > 0$ для всех $i$. Тогда для любого события $A$:

$$P(A) = \sum_{i=1}^{n} P(A|H_i) \cdot P(H_i)$$

### Интуиция

Чтобы найти вероятность $A$, мы:
1. Разбиваем все способы, которыми может произойти $A$, на непересекающиеся случаи (гипотезы $H_i$).
2. Для каждой гипотезы вычисляем вероятность $A$ при условии этой гипотезы.
3. Взвешиваем эти условные вероятности на вероятности самих гипотез.

### Пример

Фабрика имеет 3 станка: станок 1 производит 50% продукции (брак 1%), станок 2 — 30% (брак 2%), станок 3 — 20% (брак 3%).

Какова общая доля брака?

Гипотезы: $H_1$ — изделие со станка 1, $H_2$ — со станка 2, $H_3$ — со станка 3.

$$P(\text{брак}) = P(\text{брак}|H_1) \cdot P(H_1) + P(\text{брак}|H_2) \cdot P(H_2) + P(\text{брак}|H_3) \cdot P(H_3)$$

$$P(\text{брак}) = 0.01 \cdot 0.5 + 0.02 \cdot 0.3 + 0.03 \cdot 0.2 = 0.005 + 0.006 + 0.006 = 0.017 = 1.7\%$$

---

## Теорема Байеса

### Вывод теоремы

Начнём с правила умножения:

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

Приравняем правые части:

$$P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

Разделим на $P(B)$ (при $P(B) > 0$):

$$\boxed{P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}}$$

Это и есть **теорема Байеса**.

### Терминология

В контексте байесовского вывода компоненты имеют специальные названия:

| Компонент | Обозначение | Название | Интерпретация |
|-----------|-------------|----------|---------------|
| $P(H)$ | Prior | **Априорная вероятность** | Наше убеждение о гипотезе $H$ **до** наблюдения данных |
| $P(D|H)$ | Likelihood | **Правдоподобие** | Вероятность наблюдать данные $D$, если гипотеза $H$ верна |
| $P(H|D)$ | Posterior | **Апостериорная вероятность** | Наше убеждение о гипотезе $H$ **после** наблюдения данных |
| $P(D)$ | Evidence / Marginal Likelihood | **Маргинальное правдоподобие** | Нормировочная константа |

### Формула Байеса с явным знаменателем

Используя формулу полной вероятности для $P(D)$:

$$P(H|D) = \frac{P(D|H) \cdot P(H)}{\sum_{i} P(D|H_i) \cdot P(H_i)}$$

### Запись через пропорциональность

Часто удобно записывать теорему Байеса через пропорциональность, опуская нормировочную константу:

$$P(H|D) \propto P(D|H) \cdot P(H)$$

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

---

## Интуиция: почему теорема Байеса работает

### Обновление убеждений

Теорема Байеса — это формальный способ **обновлять убеждения** при получении новой информации.

**Prior** → **Наблюдение данных** → **Posterior**

Prior (априорная вероятность) отражает то, что мы знаем **до** эксперимента. Likelihood говорит, насколько вероятно наблюдать полученные данные при разных гипотезах. Posterior (апостериорная вероятность) — это наше обновлённое убеждение.

### Геометрическая интерпретация

Рассмотрим диаграмму Венна:

```
┌─────────────────────────────────┐
│           Ω (всё пространство)  │
│    ┌───────────────┐            │
│    │      B        │            │
│    │   ┌──────┐    │            │
│    │   │ A∩B  │    │            │
│    │   └──────┘    │            │
│    └───────────────┘            │
│         ┌──────┐                │
│         │ A\B  │                │
│         └──────┘                │
└─────────────────────────────────┘
```

$P(A|B)$ — это отношение площади $A \cap B$ к площади $B$. Мы «сжимаем» вселенную до $B$ и смотрим, какую долю занимает $A$.

### Пример: медицинский тест

Допустим:
- Болезнь встречается у 1% населения: $P(\text{болен}) = 0.01$
- Тест имеет чувствительность 99%: $P(\text{положительный}|\text{болен}) = 0.99$
- Тест имеет специфичность 95%: $P(\text{отрицательный}|\text{здоров}) = 0.95$

Вопрос: если тест положительный, какова вероятность, что человек болен?

**Решение:**

$$P(\text{болен}|\text{+}) = \frac{P(\text{+}|\text{болен}) \cdot P(\text{болен})}{P(\text{+})}$$

Вычислим $P(\text{+})$ по формуле полной вероятности:

$$P(\text{+}) = P(\text{+}|\text{болен}) \cdot P(\text{болен}) + P(\text{+}|\text{здоров}) \cdot P(\text{здоров})$$

$$P(\text{+}) = 0.99 \cdot 0.01 + 0.05 \cdot 0.99 = 0.0099 + 0.0495 = 0.0594$$

$$P(\text{болен}|\text{+}) = \frac{0.99 \cdot 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.167 = 16.7\%$$

**Удивительный результат!** Даже при положительном тесте вероятность болезни всего ~17%. Это потому, что априорная вероятность болезни очень мала (1%), и большинство положительных тестов — ложноположительные.

---

## Примеры и расчёты

### Пример 1: Две урны

В урне 1 — 3 белых и 2 чёрных шара. В урне 2 — 1 белый и 4 чёрных. Случайно выбрали урну и достали белый шар. Какова вероятность, что это урна 1?

**Дано:**
- $P(U_1) = P(U_2) = 0.5$ (урны равновероятны)
- $P(\text{белый}|U_1) = 3/5 = 0.6$
- $P(\text{белый}|U_2) = 1/5 = 0.2$

**Решение:**

$$P(U_1|\text{белый}) = \frac{P(\text{белый}|U_1) \cdot P(U_1)}{P(\text{белый}|U_1) \cdot P(U_1) + P(\text{белый}|U_2) \cdot P(U_2)}$$

$$P(U_1|\text{белый}) = \frac{0.6 \cdot 0.5}{0.6 \cdot 0.5 + 0.2 \cdot 0.5} = \frac{0.3}{0.3 + 0.1} = \frac{0.3}{0.4} = 0.75$$

### Пример 2: Спам-фильтр

Пусть 20% писем — спам. Слово «бесплатно» встречается в 80% спам-писем и в 5% обычных писем.

Если письмо содержит «бесплатно», какова вероятность, что это спам?

**Дано:**
- $P(\text{spam}) = 0.2$, $P(\text{ham}) = 0.8$
- $P(\text{«бесплатно»}|\text{spam}) = 0.8$
- $P(\text{«бесплатно»}|\text{ham}) = 0.05$

**Решение:**

$$P(\text{spam}|\text{«бесплатно»}) = \frac{0.8 \cdot 0.2}{0.8 \cdot 0.2 + 0.05 \cdot 0.8} = \frac{0.16}{0.16 + 0.04} = \frac{0.16}{0.20} = 0.8$$

---

## Байесовский вывод в машинном обучении

### Оценка параметров

В ML мы часто оцениваем параметры модели $\theta$ по данным $D$:

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

- $P(\theta)$ — **prior** (наше априорное знание о параметрах)
- $P(D|\theta)$ — **likelihood** (как хорошо модель с параметрами $\theta$ объясняет данные)
- $P(\theta|D)$ — **posterior** (апостериорное распределение параметров)

### Maximum A Posteriori (MAP)

**MAP оценка** — это значение $\theta$, максимизирующее апостериорную вероятность:

$$\theta_{\text{MAP}} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} P(D|\theta) \cdot P(\theta)$$

Логарифмируя:

$$\theta_{\text{MAP}} = \arg\max_{\theta} \left[\log P(D|\theta) + \log P(\theta)\right]$$

**Связь с регуляризацией:**
- Если $P(\theta) \sim \mathcal{N}(0, \sigma^2)$ (гауссовский prior), то $\log P(\theta) \propto -\|\theta\|_2^2$ — это **L2 регуляризация** (Ridge)
- Если $P(\theta) \sim \text{Laplace}(0, b)$, то $\log P(\theta) \propto -\|\theta\|_1$ — это **L1 регуляризация** (Lasso)

### Maximum Likelihood Estimation (MLE)

**MLE** — это частный случай MAP с равномерным (неинформативным) prior:

$$\theta_{\text{MLE}} = \arg\max_{\theta} P(D|\theta)$$

### Полный байесовский вывод

Вместо точечной оценки можно использовать всё апостериорное распределение для предсказаний:

$$P(y^*|x^*, D) = \int P(y^*|x^*, \theta) \cdot P(\theta|D) \, d\theta$$

Это даёт **uncertainty quantification** — оценку неопределённости предсказаний.

---

## Наивный Байесовский классификатор

### Постановка задачи

Задача классификации: дан объект с признаками $\mathbf{x} = (x_1, x_2, \ldots, x_d)$, нужно определить класс $C$.

По теореме Байеса:

$$P(C|\mathbf{x}) = \frac{P(\mathbf{x}|C) \cdot P(C)}{P(\mathbf{x})}$$

### Проблема: проклятие размерности

Для оценки $P(\mathbf{x}|C)$ нужно экспоненциально много данных при высокой размерности.

### Наивное предположение

**Наивный Байес** предполагает **условную независимость** признаков при заданном классе:

$$P(x_1, x_2, \ldots, x_d | C) = \prod_{i=1}^{d} P(x_i | C)$$

Тогда:

$$P(C|\mathbf{x}) \propto P(C) \cdot \prod_{i=1}^{d} P(x_i | C)$$

### Классификация

$$\hat{C} = \arg\max_C P(C) \cdot \prod_{i=1}^{d} P(x_i | C)$$

Или в логарифмической форме (для численной стабильности):

$$\hat{C} = \arg\max_C \left[\log P(C) + \sum_{i=1}^{d} \log P(x_i | C)\right]$$

### Варианты Naive Bayes

| Вариант | Распределение признаков | Применение |
|---------|------------------------|------------|
| Gaussian NB | $P(x_i|C) \sim \mathcal{N}(\mu_{iC}, \sigma_{iC}^2)$ | Непрерывные признаки |
| Multinomial NB | Мультиномиальное | Текст (подсчёт слов) |
| Bernoulli NB | $P(x_i|C) \sim \text{Bernoulli}(p_{iC})$ | Бинарные признаки |

---

## Связь с другими концепциями

### Теорема Байеса и VAE

В **Вариационных автоэнкодерах** (VAE) используется байесовский вывод:

$$P(z|x) = \frac{P(x|z) \cdot P(z)}{P(x)}$$

- $P(z)$ — prior на латентное пространство (обычно $\mathcal{N}(0, I)$)
- $P(x|z)$ — decoder (likelihood)
- $P(z|x)$ — истинный posterior (недоступен напрямую)
- $q_\phi(z|x)$ — приближение posterior через encoder

ELBO (Evidence Lower Bound) получается из байесовского вывода.

### Теорема Байеса и фильтр Калмана

**Фильтр Калмана** — это рекурсивное применение теоремы Байеса для линейных гауссовских систем:

1. **Prediction step** (prior): $P(x_t | y_{1:t-1})$
2. **Update step** (posterior): $P(x_t | y_{1:t}) \propto P(y_t | x_t) \cdot P(x_t | y_{1:t-1})$

### Теорема Байеса и RAG

В **Retrieval-Augmented Generation** байесовская интерпретация:

$$P(\text{ответ}|\text{вопрос}) = \sum_{\text{док}} P(\text{ответ}|\text{вопрос}, \text{док}) \cdot P(\text{док}|\text{вопрос})$$

- $P(\text{док}|\text{вопрос})$ — retrieval (какие документы релевантны)
- $P(\text{ответ}|\text{вопрос}, \text{док})$ — generation (генерация на основе документа)

---

## Реализация на Python

### Базовый пример: теорема Байеса

```python
def bayes_theorem(prior: float, likelihood: float, marginal: float) -> float:
    """
    Вычисление апостериорной вероятности по теореме Байеса.
    
    Args:
        prior: P(H) - априорная вероятность гипотезы
        likelihood: P(D|H) - правдоподобие данных при гипотезе
        marginal: P(D) - маргинальная вероятность данных
    
    Returns:
        P(H|D) - апостериорная вероятность
    """
    return (likelihood * prior) / marginal


def total_probability(priors: list, likelihoods: list) -> float:
    """
    Формула полной вероятности.
    
    Args:
        priors: список P(H_i) - априорные вероятности гипотез
        likelihoods: список P(D|H_i) - правдоподобия при каждой гипотезе
    
    Returns:
        P(D) - маргинальная вероятность
    """
    return sum(p * l for p, l in zip(priors, likelihoods))


# Пример: медицинский тест
prior_sick = 0.01  # P(болен)
prior_healthy = 0.99  # P(здоров)

# Правдоподобия
likelihood_positive_sick = 0.99  # P(+|болен) - чувствительность
likelihood_positive_healthy = 0.05  # P(+|здоров) - 1 - специфичность

# Маргинальная вероятность положительного теста
marginal_positive = total_probability(
    [prior_sick, prior_healthy],
    [likelihood_positive_sick, likelihood_positive_healthy]
)

# Апостериорная вероятность болезни при положительном тесте
posterior_sick = bayes_theorem(
    prior=prior_sick,
    likelihood=likelihood_positive_sick,
    marginal=marginal_positive
)

print(f"P(болен|+) = {posterior_sick:.4f} = {posterior_sick*100:.2f}%")
# Output: P(болен|+) = 0.1667 = 16.67%
```

### Naive Bayes с нуля

```python
import numpy as np
from collections import defaultdict
from typing import Dict, List

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes классификатор с нуля.
    """
    
    def __init__(self):
        self.classes = None
        self.priors: Dict[int, float] = {}  # P(C)
        self.means: Dict[int, np.ndarray] = {}  # μ_iC
        self.vars: Dict[int, np.ndarray] = {}  # σ²_iC
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение: оценка параметров распределений.
        
        Args:
            X: признаки, shape (n_samples, n_features)
            y: метки классов, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / n_samples  # P(C)
            self.means[c] = X_c.mean(axis=0)  # μ для каждого признака
            self.vars[c] = X_c.var(axis=0) + 1e-9  # σ² + сглаживание
        
        return self
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """
        Плотность гауссовского распределения.
        
        P(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        """
        coeff = 1.0 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent
    
    def _log_likelihood(self, x: np.ndarray, c: int) -> float:
        """
        Лог-правдоподобие: log P(x|C) = Σ log P(x_i|C)
        (используем log для численной стабильности)
        """
        pdf_values = self._gaussian_pdf(x, self.means[c], self.vars[c])
        return np.sum(np.log(pdf_values + 1e-300))
    
    def _log_posterior(self, x: np.ndarray, c: int) -> float:
        """
        Лог-апостериорная вероятность (без нормировки):
        log P(C|x) ∝ log P(C) + log P(x|C)
        """
        return np.log(self.priors[c]) + self._log_likelihood(x, c)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов для новых объектов.
        
        Args:
            X: признаки, shape (n_samples, n_features)
        
        Returns:
            Предсказанные метки классов
        """
        predictions = []
        for x in X:
            # Выбираем класс с максимальной апостериорной вероятностью
            posteriors = {c: self._log_posterior(x, c) for c in self.classes}
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Вероятности классов (softmax от log-posteriors).
        """
        probas = []
        for x in X:
            log_posts = np.array([self._log_posterior(x, c) for c in self.classes])
            # Softmax для нормировки
            log_posts -= log_posts.max()  # для численной стабильности
            posts = np.exp(log_posts)
            posts /= posts.sum()
            probas.append(posts)
        return np.array(probas)


# Пример использования
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение и предсказание
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Вероятности первых 3 примеров:\n{gnb.predict_proba(X_test[:3])}")
```

### Визуализация байесовского обновления

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def visualize_bayesian_update():
    """
    Визуализация последовательного байесовского обновления
    при оценке параметра θ (вероятность выпадения орла).
    """
    # Истинное значение параметра
    true_theta = 0.7
    
    # Prior: Beta(1, 1) = равномерное распределение
    alpha_prior, beta_prior = 1, 1
    
    theta = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    # Симулируем подбрасывания монеты
    np.random.seed(42)
    n_flips_sequence = [0, 1, 5, 10, 50, 200]
    
    alpha, beta = alpha_prior, beta_prior
    heads_total, tails_total = 0, 0
    
    for idx, n_flips in enumerate(n_flips_sequence):
        # Генерируем новые данные
        if n_flips > 0:
            new_flips = n_flips - (heads_total + tails_total)
            results = np.random.binomial(1, true_theta, new_flips)
            heads_total += results.sum()
            tails_total += len(results) - results.sum()
        
        # Posterior: Beta(alpha + heads, beta + tails)
        alpha_post = alpha_prior + heads_total
        beta_post = beta_prior + tails_total
        
        # Построение
        ax = axes[idx]
        
        # Prior (если первый график)
        if idx == 0:
            prior = stats.beta(alpha_prior, beta_prior)
            ax.fill_between(theta, prior.pdf(theta), alpha=0.3, label='Prior')
            ax.plot(theta, prior.pdf(theta), 'b-', lw=2)
        else:
            # Posterior
            posterior = stats.beta(alpha_post, beta_post)
            ax.fill_between(theta, posterior.pdf(theta), alpha=0.3, 
                           color='green', label='Posterior')
            ax.plot(theta, posterior.pdf(theta), 'g-', lw=2)
        
        # Истинное значение
        ax.axvline(true_theta, color='red', linestyle='--', lw=2, 
                   label=f'True θ = {true_theta}')
        
        # MAP оценка
        if heads_total + tails_total > 0:
            map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
            ax.axvline(map_estimate, color='purple', linestyle=':', lw=2,
                      label=f'MAP = {map_estimate:.3f}')
        
        ax.set_xlabel('θ')
        ax.set_ylabel('Плотность')
        ax.set_title(f'n = {n_flips}, Орлов: {heads_total}, Решек: {tails_total}')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)
    
    plt.suptitle('Байесовское обновление: оценка вероятности орла\n'
                 'Prior: Beta(1,1), Posterior: Beta(1+heads, 1+tails)', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('bayesian_update.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_bayesian_update()
```

### Сравнение со sklearn

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Генерация данных
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Наша реализация
gnb_custom = GaussianNaiveBayes()
gnb_custom.fit(X[:800], y[:800])
custom_acc = accuracy_score(y[800:], gnb_custom.predict(X[800:]))

# sklearn
gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X[:800], y[:800])
sklearn_acc = accuracy_score(y[800:], gnb_sklearn.predict(X[800:]))

print(f"Custom GNB accuracy: {custom_acc:.4f}")
print(f"Sklearn GNB accuracy: {sklearn_acc:.4f}")

# Cross-validation
scores = cross_val_score(GaussianNB(), X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## References

### Внутренние ссылки (Knowledge Book)

- [Gaussian Distribution (Normal Distribution)](./gaussian-distribution.md) — фундаментальное распределение, используемое в байесовском выводе
- [Variational Autoencoders (VAEs)](./variational-autoencoders-vaes.md) — применение теоремы Байеса для генеративных моделей
- [Unscented Kalman Filter](./unscented-kalman-filter-and-tracking.md) — рекурсивное байесовское обновление для фильтрации
- [Retrieval-Augmented Generation (RAG)](./retrieval-augmented-generation-rag.md) — байесовская интерпретация retrieval
- [ROC Curves and ROC AUC](./roc-curve-and-roc-auc.md) — метрики классификации, связанные с оценкой вероятностей

### Внешние ресурсы

- [Bayes' theorem - Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Kolmogorov axioms - Wikipedia](https://en.wikipedia.org/wiki/Probability_axioms)
- [Naive Bayes classifier - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Bishop C.M. "Pattern Recognition and Machine Learning", Chapter 2](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Murphy K.P. "Machine Learning: A Probabilistic Perspective", Chapters 2-3](https://probml.github.io/pml-book/)
- [3Blue1Brown: Bayes theorem (YouTube)](https://www.youtube.com/watch?v=HZGCoVF3YvM)
