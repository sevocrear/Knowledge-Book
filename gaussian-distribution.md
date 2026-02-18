# Гауссово Распределение (Normal Distribution): Основы

## Table of Contents

1. [Введение](#введение)
2. [Определение и Формула](#определение-и-формула)
3. [Свойства Гауссова Распределения](#свойства-гауссова-распределения)
4. [Стандартное Нормальное Распределение](#стандартное-нормальное-распределение)
5. [Многомерное Гауссово Распределение](#многомерное-гауссово-распределение)
6. [Применение в Машинном Обучении](#применение-в-машинном-обучении)
7. [Связь с Diffusion Models](#связь-с-diffusion-models)
8. [Визуализация и Примеры](#визуализация-и-примеры)
9. [References](#references)

---

![image](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/500px-Normal_Distribution_PDF.svg.png)

## Введение

**Гауссово распределение** (также называемое **нормальным распределением**) - это одно из самых важных и широко используемых распределений в статистике, теории вероятностей и машинном обучении. Оно названо в честь Карла Фридриха Гаусса, хотя было открыто независимо несколькими математиками.

### Почему Гауссово Распределение Важно?

1. **Центральная Предельная Теорема**: Сумма большого числа независимых случайных величин стремится к нормальному распределению
2. **Естественное явление**: Многие природные процессы следуют нормальному распределению (рост, измерения, ошибки)
3. **Математическая удобность**: Имеет множество полезных свойств, которые упрощают вычисления
4. **Фундамент ML**: Основа для многих алгоритмов машинного обучения, включая diffusion models, VAEs, и др.

---

## Определение и Формула

### Одномерное Гауссово Распределение

**Плотность вероятности (Probability Density Function, PDF)** для одномерного нормального распределения:

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

где:
- $\mu$ (mu) - **среднее значение** (mean) - центр распределения
- $\sigma$ (sigma) - **стандартное отклонение** (standard deviation) - мера разброса
- $\sigma^2$ - **дисперсия** (variance)

**Обозначение**: $X \sim \mathcal{N}(\mu, \sigma^2)$ означает, что случайная величина $X$ следует нормальному распределению со средним $\mu$ и дисперсией $\sigma^2$.

### Параметры Распределения

#### Среднее ($\mu$)

Среднее значение определяет **центр** распределения - где находится "пик" кривой:

$$\mu = \mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) dx$$

#### Дисперсия ($\sigma^2$) и Стандартное Отклонение ($\sigma$)

Дисперсия определяет **ширину** распределения - насколько данные разбросаны вокруг среднего:

$$\sigma^2 = \text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot p(x) dx$$

$$\sigma = \sqrt{\sigma^2}$$

**Интуиция**:
- Маленькое $\sigma$ → узкое распределение (данные сконцентрированы)
- Большое $\sigma$ → широкое распределение (данные разбросаны)

---

## Свойства Гауссова Распределения

### 1. Симметричность

Гауссово распределение **симметрично** относительно среднего $\mu$:

$$p(\mu + a) = p(\mu - a)$$

### 2. Правило 68-95-99.7 (Empirical Rule)

Для нормального распределения:
- **68%** данных находится в пределах $[\mu - \sigma, \mu + \sigma]$
- **95%** данных находится в пределах $[\mu - 2\sigma, \mu + 2\sigma]$
- **99.7%** данных находится в пределах $[\mu - 3\sigma, \mu + 3\sigma]$

### 3. Линейные Преобразования

Если $X \sim \mathcal{N}(\mu, \sigma^2)$, то:

$$Y = aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$$

**Пример**: Если $X \sim \mathcal{N}(0, 1)$, то $2X + 3 \sim \mathcal{N}(3, 4)$

### 4. Сумма Независимых Гауссовых Величин

Если $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ и $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$ независимы, то:

$$X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

### 5. Максимум Плотности

Максимум плотности достигается при $x = \mu$:

$$p(\mu) = \frac{1}{\sigma\sqrt{2\pi}}$$

### 6. Моменты

- **Первый момент (среднее)**: $\mathbb{E}[X] = \mu$
- **Второй момент (дисперсия)**: $\text{Var}(X) = \sigma^2$
- **Третий момент (асимметрия)**: 0 (симметричное распределение)
- **Четвертый момент (эксцесс)**: 3$\sigma^4$

---

## Стандартное Нормальное Распределение

**Стандартное нормальное распределение** - это частный случай с $\mu = 0$ и $\sigma = 1$:

$$Z \sim \mathcal{N}(0, 1)$$

**PDF**:

$$p(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)$$

### Стандартизация

Любое нормальное распределение можно преобразовать в стандартное:

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

Это называется **z-score** или **стандартизация**.

### Кумулятивная Функция Распределения (CDF)

**CDF** для стандартного нормального распределения:

$$\Phi(z) = P(Z \leq z) = \int_{-\infty}^{z} \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{t^2}{2}\right) dt$$

Значения $\Phi(z)$ табулированы или вычисляются численно.

---

## Многомерное Гауссово Распределение

### Определение

Для **вектора** $\mathbf{x} \in \mathbb{R}^d$ многомерное гауссово распределение:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

где:
- $\boldsymbol{\mu} \in \mathbb{R}^d$ - **вектор средних**
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ - **ковариационная матрица** (симметричная, положительно определенная)
- $|\boldsymbol{\Sigma}|$ - определитель матрицы $\boldsymbol{\Sigma}$

**Обозначение**: $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

### Ковариационная Матрица

Ковариационная матрица $\boldsymbol{\Sigma}$ содержит:
- **Диагональные элементы**: дисперсии каждой переменной
- **Внедиагональные элементы**: ковариации между переменными

$$\boldsymbol{\Sigma}_{ij} = \text{Cov}(X_i, X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$$

### Изотропное Гауссово Распределение

Если $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$ (диагональная матрица с одинаковыми значениями), то:

$$p(\mathbf{x}) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{||\mathbf{x} - \boldsymbol{\mu}||^2}{2\sigma^2}\right)$$

Это называется **изотропным** (isotropic) распределением - одинаковый разброс во всех направлениях.

### Стандартное Многомерное Распределение

Если $\boldsymbol{\mu} = \mathbf{0}$ и $\boldsymbol{\Sigma} = \mathbf{I}$ (единичная матрица):

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{||\mathbf{x}||^2}{2}\right)$$

**Обозначение**: $\mathbf{X} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

---

## Применение в Машинном Обучении

### 1. Инициализация Весов

Веса нейронных сетей часто инициализируются из нормального распределения:

```python
# PyTorch пример
import torch.nn as nn

# Инициализация весов из N(0, 0.01)
nn.init.normal_(layer.weight, mean=0.0, std=0.1)
```

### 2. Регуляризация

**L2 регуляризация** эквивалентна априорному распределению весов как $\mathcal{N}(0, \lambda^{-1})$:

$$p(\mathbf{w}) \propto \exp\left(-\frac{\lambda}{2}||\mathbf{w}||^2\right)$$

### 3. Шум в Обучении

Добавление гауссова шума используется для:
- **Data augmentation**: Добавление шума к входным данным
- **Dropout альтернатива**: Гауссов шум вместо dropout
- **Exploration в RL**: Гауссов шум для исследования пространства действий

### 4. Вероятностные Модели

Многие вероятностные модели используют гауссово распределение:
- **VAEs**: Latent space моделируется как $\mathcal{N}(0, \mathbf{I})$
- **Diffusion Models**: Шум добавляется из $\mathcal{N}(0, \mathbf{I})$
- **Gaussian Processes**: Байесовская непараметрическая модель
- **Bayesian Neural Networks**: Апостериорное распределение весов

---

## Связь с Diffusion Models

### Почему Гауссово Распределение в Diffusion Models?

1. **Математическая Удобность**: Гауссовы распределения имеют замкнутую форму для многих операций
2. **Центральная Предельная Теорема**: Сумма многих маленьких изменений → нормальное распределение
3. **Свойства Сложения**: Сумма независимых гауссовых величин снова гауссова

### Forward Process

В diffusion models на каждом шаге добавляется **гауссов шум**:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

Это означает, что мы добавляем шум из $\mathcal{N}(\mathbf{0}, \beta_t \mathbf{I})$.

### Closed-Form Solution

Благодаря свойствам гауссовых распределений, можно напрямую вычислить:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

### Reverse Process

Модель учится предсказывать параметры гауссова распределения:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

### Sampling

Генерация начинается с **стандартного гауссова шума**:

$$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Затем итеративно удаляется шум, пока не получим $\mathbf{x}_0$.

---

## Визуализация и Примеры

### Python Пример: Визуализация Гауссова Распределения

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Параметры
mu = 0      # среднее
sigma = 1   # стандартное отклонение

# Генерация данных
x = np.linspace(-5, 5, 1000)
y = norm.pdf(x, mu, sigma)  # PDF

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label=f'$\mathcal{{N}}({mu}, {sigma}^2)$')
plt.axvline(mu, color='r', linestyle='--', label=f'$\mu = {mu}$')
plt.fill_between(x, 0, y, where=(x >= mu-sigma) & (x <= mu+sigma), 
                 alpha=0.3, label='68% данных')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.title('Гауссово (Нормальное) Распределение')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Пример: Разные Параметры

```python
# Разные средние и стандартные отклонения
params = [
    (0, 0.5, '$\mu=0, \sigma=0.5$'),
    (0, 1, '$\mu=0, \sigma=1$'),
    (0, 2, '$\mu=0, \sigma=2$'),
    (-2, 1, '$\mu=-2, \sigma=1$'),
    (2, 1, '$\mu=2, \sigma=1$')
]

x = np.linspace(-6, 6, 1000)
plt.figure(figsize=(12, 6))

for mu, sigma, label in params:
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, label=label, linewidth=2)

plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.title('Гауссовы Распределения с Разными Параметрами')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Пример: Сэмплирование

```python
# Генерация случайных сэмплов
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 10000)

# Гистограмма
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Гистограмма сэмплов')

# Теоретическая кривая
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'r-', linewidth=2, label='Теоретическая PDF')

plt.xlabel('x')
plt.ylabel('Плотность')
plt.title('Сэмплирование из $\mathcal{N}(0, 1)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Многомерное Распределение

```python
from scipy.stats import multivariate_normal

# Параметры
mu = np.array([0, 0])
sigma = np.array([[1, 0.5], [0.5, 1]])  # Ковариационная матрица

# Создание распределения
rv = multivariate_normal(mu, sigma)

# Сэмплирование
samples = rv.rvs(1000)

# Визуализация
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Многомерное Гауссово Распределение')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

---

## References

### Related Documents

- **[Diffusion Models](./diffusion-models.md)**: Использование гауссова распределения для добавления и удаления шума
- **[Variational Autoencoders (VAEs)](./variational-autoencoders-vaes.md)**: Гауссово распределение в latent space

### Key Concepts

- **Probability Theory**: Основы теории вероятностей
- **Central Limit Theorem**: Центральная предельная теорема
- **Maximum Likelihood Estimation**: Оценка параметров через максимизацию правдоподобия
- **Bayesian Inference**: Байесовский вывод с гауссовыми априорами

### Further Reading

1. **Bishop (2006)**: "Pattern Recognition and Machine Learning" - Глава 2
2. **Murphy (2022)**: "Probabilistic Machine Learning" - Глава 2
3. **Wikipedia**: "Normal Distribution" - Подробная математическая информация

---

*Документ создан: 2025*
*Последнее обновление: 2025*
