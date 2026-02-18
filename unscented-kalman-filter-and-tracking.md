# Unscented Kalman Filter и Современные Методы Отслеживания Объектов

## Table of Contents

1. [Введение](#введение)
2. [Основы Фильтрации и Отслеживания](#основы-фильтрации-и-отслеживания)
3. [Классический Kalman Filter (KF)](#классический-kalman-filter-kf)
4. [Extended Kalman Filter (EKF)](#extended-kalman-filter-ekf)
   - [Настройка Параметров EKF](#настройка-параметров-ekf)
5. [Unscented Kalman Filter (UKF)](#unscented-kalman-filter-ukf)
   - [Статистика Хи-квадрат для Обнаружения Выбросов](#статистика-хи-квадрат-для-обнаружения-выбросов-outlier-detection)
   - [Настройка Параметров UKF](#настройка-параметров-ukf)
6. [Particle Filter (PF)](#particle-filter-pf)
7. [Сравнение Методов](#сравнение-методов)
8. [Современные Методы Отслеживания](#современные-методы-отслеживания)
9. [Реализация UKF](#реализация-ukf)
10. [Применения и Примеры](#применения-и-примеры)
11. [Текущее Состояние (2023-2026)](#текущее-состояние-2023-2026)
12. [References](#references)

---

## Введение

**Unscented Kalman Filter (UKF)** — это мощный алгоритм фильтрации для нелинейных систем, предложенный Julier и Uhlmann в 1997 году. UKF решает проблему нелинейной фильтрации, используя технику **Unscented Transform (UT)**, которая более точно аппроксимирует распределения вероятностей при нелинейных преобразованиях по сравнению с Extended Kalman Filter (EKF).

### Ключевые Характеристики

- **Точность**: Более точная аппроксимация нелинейных преобразований по сравнению с EKF
- **Вычислительная Эффективность**: Обычно быстрее Particle Filter при сопоставимой точности
- **Устойчивость**: Более устойчив к начальным условиям, чем EKF
- **Применимость**: Эффективен для умеренно нелинейных систем

### Исторический Контекст

- **1960**: Kalman Filter для линейных систем
- **1960-1970**: Extended Kalman Filter для нелинейных систем (линеаризация)
- **1997**: Unscented Kalman Filter (Julier & Uhlmann)
- **2000-е**: Particle Filters для сильно нелинейных систем
- **2010-2020**: Deep Learning подходы к отслеживанию
- **2020-2026**: Transformer-based tracking, end-to-end learning

---

## Основы Фильтрации и Отслеживания

### Проблема Фильтрации

Задача фильтрации состоит в оценке скрытого состояния системы $\mathbf{x}_t$ на основе наблюдений $\mathbf{z}_t$:

**State Space Model:**
- **State Equation**: $\mathbf{x}_t = f(\mathbf{x}_{t-1}, \mathbf{u}_t, \mathbf{w}_t)$
- **Observation Equation**: $\mathbf{z}_t = h(\mathbf{x}_t, \mathbf{v}_t)$

где:
- $\mathbf{x}_t$ — состояние системы в момент времени $t$
- $\mathbf{z}_t$ — наблюдения в момент времени $t$
- $\mathbf{u}_t$ — управляющий вход
- $\mathbf{w}_t$ — шум процесса (process noise)
- $\mathbf{v}_t$ — шум наблюдений (observation noise)
- $f(\cdot)$ — функция перехода состояния
- $h(\cdot)$ — функция наблюдения

### Типы Систем

1. **Линейные системы**: $f$ и $h$ — линейные функции
2. **Нелинейные системы**: $f$ и/или $h$ — нелинейные функции
3. **Мультимодальные системы**: распределение вероятностей имеет несколько мод

### Основные Задачи

- **Фильтрация**: $p(\mathbf{x}_t | \mathbf{z}_{1:t})$ — оценка текущего состояния
- **Предсказание**: $p(\mathbf{x}_{t+k} | \mathbf{z}_{1:t})$ — прогноз будущего состояния
- **Сглаживание**: $p(\mathbf{x}_t | \mathbf{z}_{1:T})$ — ретроспективная оценка

---

## Классический Kalman Filter (KF)

### Предположения

Kalman Filter работает для **линейных систем** с **гауссовскими шумами**:

$$\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_{t-1} + \mathbf{B}_t \mathbf{u}_t + \mathbf{w}_t$$
$$\mathbf{z}_t = \mathbf{H}_t \mathbf{x}_t + \mathbf{v}_t$$

где:
- $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)$ — шум процесса
- $\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)$ — шум наблюдений
- $\mathbf{F}_t$ — матрица перехода состояния
- $\mathbf{H}_t$ — матрица наблюдений

### Алгоритм Kalman Filter

**Predict Step (Предсказание):**

$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}_t \hat{\mathbf{x}}_{t-1|t-1} + \mathbf{B}_t \mathbf{u}_t$$
$$\mathbf{P}_{t|t-1} = \mathbf{F}_t \mathbf{P}_{t-1|t-1} \mathbf{F}_t^T + \mathbf{Q}_t$$

**Update Step (Обновление):**

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^T (\mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^T + \mathbf{R}_t)^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H}_t \hat{\mathbf{x}}_{t|t-1})$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1}$$

где:
- $\hat{\mathbf{x}}_{t|t-1}$ — предсказанное состояние
- $\hat{\mathbf{x}}_{t|t}$ — отфильтрованное состояние
- $\mathbf{P}_{t|t-1}$ — ковариация предсказания
- $\mathbf{P}_{t|t}$ — ковариация фильтрации
- $\mathbf{K}_t$ — матрица усиления Kalman (Kalman gain)

### Ограничения

- Работает только для **линейных систем**
- Требует **гауссовских распределений**
- Не может обрабатывать **мультимодальные распределения**

---

## Extended Kalman Filter (EKF)

### Основная Идея

EKF расширяет Kalman Filter на нелинейные системы через **линеаризацию** с использованием **разложения Тейлора первого порядка**.

### Линеаризация

Для нелинейных функций $f$ и $h$:

$$\mathbf{x}_t = f(\mathbf{x}_{t-1}, \mathbf{u}_t, \mathbf{w}_t)$$
$$\mathbf{z}_t = h(\mathbf{x}_t, \mathbf{v}_t)$$

Линеаризуем вокруг текущей оценки:

$$\mathbf{F}_t = \frac{\partial f}{\partial \mathbf{x}} \Big|_{\hat{\mathbf{x}}_{t-1|t-1}}$$
$$\mathbf{H}_t = \frac{\partial h}{\partial \mathbf{x}} \Big|_{\hat{\mathbf{x}}_{t|t-1}}$$

### Алгоритм EKF

**Predict Step:**

$$\hat{\mathbf{x}}_{t|t-1} = f(\hat{\mathbf{x}}_{t-1|t-1}, \mathbf{u}_t, \mathbf{0})$$
$$\mathbf{P}_{t|t-1} = \mathbf{F}_t \mathbf{P}_{t-1|t-1} \mathbf{F}_t^T + \mathbf{Q}_t$$

**Update Step:**

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^T (\mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^T + \mathbf{R}_t)^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - h(\hat{\mathbf{x}}_{t|t-1}, \mathbf{0}))$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1}$$

### Проблемы EKF

1. **Ошибки линеаризации**: Разложение Тейлора первого порядка может быть неточным
2. **Вычисление якобианов**: Требует аналитического или численного дифференцирования
3. **Нестабильность**: Может расходиться при сильной нелинейности
4. **Асимметрия**: Линеаризация может привести к асимметричным распределениям

### Настройка Параметров EKF

Настройка параметров EKF — критически важный этап для достижения хорошей производительности. Основные параметры, которые требуют настройки:

#### Основные Параметры EKF

1. **$\mathbf{Q}_t$ — Ковариация шума процесса (Process Noise Covariance)**
   - Описывает неопределенность модели движения
   - Влияет на то, насколько быстро фильтр адаптируется к изменениям
   - **Большие значения**: Фильтр быстрее реагирует на изменения, но более шумный
   - **Малые значения**: Фильтр более сглаженный, но медленнее реагирует

2. **$\mathbf{R}_t$ — Ковариация шума наблюдений (Measurement Noise Covariance)**
   - Описывает точность сенсоров
   - Обычно можно оценить из характеристик сенсоров или данных
   - **Большие значения**: Меньше доверия к наблюдениям
   - **Малые значения**: Больше доверия к наблюдениям

3. **$\mathbf{P}_{0|0}$ — Начальная ковариация состояния (Initial State Covariance)**
   - Описывает неопределенность начального состояния
   - Влияет на скорость сходимости в начале
   - Обычно устанавливается как диагональная матрица с большими значениями

4. **$\hat{\mathbf{x}}_{0|0}$ — Начальное состояние (Initial State)**
   - Начальная оценка состояния системы
   - Может быть получена из первого наблюдения или априорной информации

#### Процесс Настройки Параметров

**Шаг 1: Анализ Системы**

Определите:
- Какие компоненты состояния (позиция, скорость, ориентация и т.д.)
- Какие наблюдения доступны
- Физические характеристики системы

**Шаг 2: Начальная Оценка $\mathbf{R}$**

Оцените $\mathbf{R}$ из:
- **Характеристик сенсоров**: Точность GPS, камеры, IMU и т.д.
- **Статистики наблюдений**: Вычислите ковариацию ошибок наблюдений на валидационных данных
- **Эмпирических данных**: Измерьте стандартное отклонение ошибок сенсоров

Пример для 2D позиции:
```python
# Если GPS имеет точность ±2 метра
R_gps = np.diag([2.0**2, 2.0**2])  # Дисперсия = стандартное отклонение^2

# Если камера имеет точность ±5 пикселей при разрешении 1920x1080
# и объект находится на расстоянии, где 1 пиксель = 0.1 метра
R_camera = np.diag([(5 * 0.1)**2, (5 * 0.1)**2])
```

**Шаг 3: Начальная Оценка $\mathbf{Q}$**

Оцените $\mathbf{Q}$ из:
- **Физической модели**: Как быстро может изменяться состояние?
- **Эмпирических данных**: Измерьте вариативность изменений состояния
- **Эвристики**: Начните с малых значений и увеличивайте при необходимости

Общий подход:
```python
# Для модели постоянной скорости
# Q описывает неопределенность в ускорении
dt = 0.1  # Интервал времени
process_noise_vel = 0.5  # Стандартное отклонение ускорения (м/с²)

# Q для состояния [x, y, vx, vy]
Q = np.array([
    [dt**4/4, 0, dt**3/2, 0],           # Ковариация позиции x
    [0, dt**4/4, 0, dt**3/2],           # Ковариация позиции y
    [dt**3/2, 0, dt**2, 0],              # Ковариация скорости vx
    [0, dt**3/2, 0, dt**2]               # Ковариация скорости vy
]) * process_noise_vel**2
```

**Шаг 4: Настройка через Валидацию**

Используйте валидационные данные для тонкой настройки:

1. **Разделите данные**: Train/validation/test sets
2. **Определите метрику**: RMSE, MAE, или специфичную для задачи
3. **Градиентный поиск или grid search**:

```python
def tune_ekf_params(train_data, val_data, param_ranges):
    """
    Поиск оптимальных параметров EKF
    """
    best_params = None
    best_score = float('inf')
    
    for Q_scale in param_ranges['Q']:
        for R_scale in param_ranges['R']:
            Q = Q_scale * base_Q
            R = R_scale * base_R
            
            # Оценка на валидационных данных
            score = evaluate_ekf(val_data, Q, R)
            
            if score < best_score:
                best_score = score
                best_params = {'Q': Q, 'R': R}
    
    return best_params
```

#### Практические Рекомендации

**1. Начните с Консервативных Значений**

```python
# Начальные значения (консервативные)
P0 = np.eye(n) * 100  # Большая начальная неопределенность
Q = np.eye(n) * 0.1   # Малый шум процесса
R = np.eye(m) * 1.0   # Средний шум наблюдений
```

**2. Используйте Диагональные Матрицы**

Если нет информации о корреляциях, используйте диагональные матрицы:

```python
# Для состояния [px, py, vx, vy]
Q = np.diag([
    process_noise_pos**2,   # Шум позиции
    process_noise_pos**2,
    process_noise_vel**2,    # Шум скорости
    process_noise_vel**2
])

R = np.diag([
    measurement_noise_pos**2,  # Шум наблюдения позиции
    measurement_noise_pos**2
])
```

**3. Масштабирование по Компонентам**

Разные компоненты состояния могут иметь разную неопределенность:

```python
# Позиция более стабильна, скорость более изменчива
Q = np.diag([
    0.01,  # process_noise_pos_x
    0.01,  # process_noise_pos_y
    0.5,   # process_noise_vel_x (больше, т.к. скорость изменяется быстрее)
    0.5    # process_noise_vel_y
])
```

**4. Адаптация к Частоте Обновлений**

Если частота наблюдений изменяется, масштабируйте $\mathbf{Q}$:

```python
# Q пропорционален dt
Q = Q_base * dt
# или для модели постоянного ускорения
Q = Q_base * dt**2
```

**5. Проверка Устойчивости**

Убедитесь, что:
- $\mathbf{P}_{t|t}$ остается положительно определенной
- $\mathbf{K}_t$ не становится слишком большим или слишком малым
- Оценки не расходятся

#### Диагностика Проблем

**Проблема: Фильтр слишком медленно реагирует**

**Симптомы:**
- Большая задержка в отслеживании изменений
- Остаточные ошибки не уменьшаются

**Решение:**
- Увеличьте $\mathbf{Q}$ (больше доверия к модели)
- Уменьшите $\mathbf{R}$ (больше доверия к наблюдениям)

**Проблема: Фильтр слишком шумный**

**Симптомы:**
- Оценки сильно колеблются
- Высокая дисперсия в результатах

**Решение:**
- Уменьшите $\mathbf{Q}$ (меньше доверия к модели)
- Увеличьте $\mathbf{R}$ (меньше доверия к наблюдениям)

**Проблема: Фильтр расходится**

**Симптомы:**
- $\mathbf{P}_{t|t}$ растет без ограничений
- Ошибки увеличиваются со временем

**Решение:**
- Проверьте правильность якобианов $\mathbf{F}_t$ и $\mathbf{H}_t$
- Увеличьте $\mathbf{R}$ (больше сглаживания)
- Проверьте, что система наблюдаема

**Проблема: Фильтр не сходится**

**Симптомы:**
- Ошибки не уменьшаются со временем
- Фильтр не использует наблюдения эффективно

**Решение:**
- Уменьшите начальную $\mathbf{P}_{0|0}$
- Проверьте правильность модели наблюдений $h(\cdot)$
- Убедитесь, что система наблюдаема

#### Примеры Настройки для Разных Задач

**Пример 1: Отслеживание 2D Позиции с Постоянной Скоростью**

```python
# Состояние: [x, y, vx, vy]
dt = 0.1  # 10 Hz обновления

# Q: неопределенность в ускорении
accel_std = 0.5  # м/с²
Q = np.array([
    [dt**4/4, 0, dt**3/2, 0],
    [0, dt**4/4, 0, dt**3/2],
    [dt**3/2, 0, dt**2, 0],
    [0, dt**3/2, 0, dt**2]
]) * accel_std**2

# R: точность GPS ±2 метра
R = np.diag([2.0**2, 2.0**2])

# Начальная неопределенность
P0 = np.diag([10.0**2, 10.0**2, 1.0**2, 1.0**2])
```

**Пример 2: Отслеживание с Ориентацией (6 DOF)**

```python
# Состояние: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
# где q - кватернион ориентации

dt = 0.05  # 20 Hz

# Q для позиции/скорости
Q_pos_vel = np.diag([
    0.01, 0.01, 0.01,  # process_noise_pos
    0.1, 0.1, 0.1      # process_noise_vel
])

# Q для ориентации (в радианах)
Q_orient = np.diag([
    0.001, 0.001, 0.001  # process_noise_orient (мало, т.к. ориентация стабильна)
])

Q = np.block([
    [Q_pos_vel, np.zeros((6, 4))],
    [np.zeros((4, 6)), Q_orient]
])

# R: GPS позиция ±2м, IMU ориентация ±0.01 рад
R = np.diag([
    2.0**2, 2.0**2, 2.0**2,  # measurement_noise_pos
    0.01**2, 0.01**2, 0.01**2  # measurement_noise_orient
])
```

**Пример 3: Отслеживание Размеров Объекта**

```python
# Состояние: [x, y, width, height, vx, vy]
# Размеры объекта могут изменяться медленнее, чем позиция

Q = np.diag([
    0.1,    # process_noise_pos_x
    0.1,    # process_noise_pos_y
    0.01,   # process_noise_dim (размеры изменяются медленнее)
    0.01,
    0.5,    # process_noise_vel_x
    0.5     # process_noise_vel_y
])

# R: детектор имеет точность ±5 пикселей для позиции, ±2 пикселя для размеров
R = np.diag([
    5.0**2,   # measurement_noise_pos_x
    5.0**2,   # measurement_noise_pos_y
    2.0**2,   # measurement_noise_dim (размеры измеряются точнее)
    2.0**2
])
```

#### Автоматическая Настройка (Advanced)

**EM Algorithm для EKF:**

Expectation-Maximization алгоритм может автоматически настраивать $\mathbf{Q}$ и $\mathbf{R}$:

```python
def em_ekf_tuning(observations, initial_Q, initial_R, max_iter=10):
    """
    EM алгоритм для настройки Q и R
    """
    Q = initial_Q
    R = initial_R
    
    for iteration in range(max_iter):
        # E-step: запустить EKF с текущими Q и R
        states, covariances = run_ekf(observations, Q, R)
        
        # M-step: обновить Q и R на основе остатков
        innovations = compute_innovations(observations, states)
        Q, R = update_noise_covariances(innovations, states, covariances)
    
    return Q, R
```

**Адаптивные Методы:**

Для систем с изменяющимися характеристиками можно использовать адаптивные методы:

```python
def adaptive_ekf_update(ekf, z_t, window_size=10):
    """
    Адаптивная настройка R на основе недавних инноваций
    """
    innovation = z_t - ekf.h(ekf.x)
    
    # Обновить оценку R на основе скользящего окна
    ekf.innovation_history.append(innovation)
    if len(ekf.innovation_history) > window_size:
        ekf.innovation_history.pop(0)
    
    # Вычислить эмпирическую ковариацию инноваций
    innovations_array = np.array(ekf.innovation_history)
    R_adaptive = np.cov(innovations_array.T)
    
    # Сглаживание: смесь старого и нового R
    ekf.R = 0.9 * ekf.R + 0.1 * R_adaptive
```

#### Чеклист Настройки EKF

- [ ] Определить структуру состояния и наблюдений
- [ ] Оценить $\mathbf{R}$ из характеристик сенсоров или данных
- [ ] Начать с консервативных значений $\mathbf{Q}$
- [ ] Использовать диагональные матрицы, если нет информации о корреляциях
- [ ] Масштабировать параметры по компонентам состояния
- [ ] Валидировать на отдельном наборе данных
- [ ] Проверить устойчивость (положительная определенность $\mathbf{P}$)
- [ ] Диагностировать проблемы (медленная реакция, шум, расходимость)
- [ ] Итеративно улучшать параметры
- [ ] Документировать выбранные значения и обоснование

---

## Unscented Kalman Filter (UKF)

### Основная Идея

UKF использует **Unscented Transform (UT)** вместо линеаризации. UT выбирает минимальный набор **sigma points**, которые точно захватывают среднее и ковариацию распределения, и затем пропускает их через нелинейную функцию.

### Unscented Transform

**Шаг 1: Выбор Sigma Points**

Для $n$-мерного состояния выбираем $2n+1$ sigma points:

$$\boldsymbol{\chi}_0 = \bar{\mathbf{x}}$$
$$\boldsymbol{\chi}_i = \bar{\mathbf{x}} + \sqrt{(n+\lambda)\mathbf{P}}_i, \quad i = 1, ..., n$$
$$\boldsymbol{\chi}_i = \bar{\mathbf{x}} - \sqrt{(n+\lambda)\mathbf{P}}_{i-n}, \quad i = n+1, ..., 2n$$

где:
- $\bar{\mathbf{x}}$ — среднее распределения
- $\mathbf{P}$ — ковариационная матрица
- $\sqrt{(n+\lambda)\mathbf{P}}_i$ — $i$-я строка/столбец матричного квадратного корня
- $\lambda = \alpha^2(n+\kappa) - n$ — параметр масштабирования
- $\alpha$ — параметр распространения (обычно $10^{-3} \leq \alpha \leq 1$)
- $\kappa$ — вторичный параметр масштабирования (обычно $\kappa = 0$ или $3-n$)

**Шаг 2: Веса**

$$W_0^{(m)} = \frac{\lambda}{n+\lambda}$$
$$W_0^{(c)} = \frac{\lambda}{n+\lambda} + (1-\alpha^2+\beta)$$
$$W_i^{(m)} = W_i^{(c)} = \frac{1}{2(n+\lambda)}, \quad i = 1, ..., 2n$$

где:
- $W_i^{(m)}$ — веса для среднего
- $W_i^{(c)}$ — веса для ковариации
- $\beta$ — параметр для учета априорной информации (для гауссовских распределений $\beta=2$)

**Шаг 3: Трансформация через Нелинейную Функцию**

$$\boldsymbol{\mathcal{Y}}_i = f(\boldsymbol{\chi}_i), \quad i = 0, ..., 2n$$

**Шаг 4: Вычисление Трансформированного Среднего и Ковариации**

$$\bar{\mathbf{y}} = \sum_{i=0}^{2n} W_i^{(m)} \boldsymbol{\mathcal{Y}}_i$$
$$\mathbf{P}_y = \sum_{i=0}^{2n} W_i^{(c)} (\boldsymbol{\mathcal{Y}}_i - \bar{\mathbf{y}})(\boldsymbol{\mathcal{Y}}_i - \bar{\mathbf{y}})^T$$

### Алгоритм UKF

**Инициализация:**

$$\hat{\mathbf{x}}_{0|0} = \mathbb{E}[\mathbf{x}_0]$$
$$\mathbf{P}_{0|0} = \mathbb{E}[(\mathbf{x}_0 - \hat{\mathbf{x}}_{0|0})(\mathbf{x}_0 - \hat{\mathbf{x}}_{0|0})^T]$$

**Predict Step (Time Update):**

1. **Генерация Sigma Points:**
   $$\boldsymbol{\chi}_{t-1|t-1} = [\hat{\mathbf{x}}_{t-1|t-1}, \hat{\mathbf{x}}_{t-1|t-1} \pm \sqrt{(n+\lambda)\mathbf{P}_{t-1|t-1}}]$$

2. **Трансформация через Функцию Перехода:**
   $$\boldsymbol{\chi}_{t|t-1}^* = f(\boldsymbol{\chi}_{t-1|t-1}, \mathbf{u}_t)$$

3. **Предсказание Среднего:**
   $$\hat{\mathbf{x}}_{t|t-1} = \sum_{i=0}^{2n} W_i^{(m)} \boldsymbol{\chi}_{i,t|t-1}^*$$

4. **Предсказание Ковариации:**
   $$\mathbf{P}_{t|t-1} = \sum_{i=0}^{2n} W_i^{(c)} (\boldsymbol{\chi}_{i,t|t-1}^* - \hat{\mathbf{x}}_{t|t-1})(\boldsymbol{\chi}_{i,t|t-1}^* - \hat{\mathbf{x}}_{t|t-1})^T + \mathbf{Q}_t$$

**Update Step (Measurement Update):**

1. **Трансформация через Функцию Наблюдения:**
   $$\boldsymbol{\mathcal{Z}}_{t|t-1} = h(\boldsymbol{\chi}_{t|t-1}^*)$$

2. **Предсказание Наблюдения:**
   $$\hat{\mathbf{z}}_{t|t-1} = \sum_{i=0}^{2n} W_i^{(m)} \boldsymbol{\mathcal{Z}}_{i,t|t-1}$$

3. **Ковариация Наблюдения:**
   $$\mathbf{P}_{zz} = \sum_{i=0}^{2n} W_i^{(c)} (\boldsymbol{\mathcal{Z}}_{i,t|t-1} - \hat{\mathbf{z}}_{t|t-1})(\boldsymbol{\mathcal{Z}}_{i,t|t-1} - \hat{\mathbf{z}}_{t|t-1})^T + \mathbf{R}_t$$

4. **Кросс-ковариация:**
   $$\mathbf{P}_{xz} = \sum_{i=0}^{2n} W_i^{(c)} (\boldsymbol{\chi}_{i,t|t-1}^* - \hat{\mathbf{x}}_{t|t-1})(\boldsymbol{\mathcal{Z}}_{i,t|t-1} - \hat{\mathbf{z}}_{t|t-1})^T$$

5. **Kalman Gain:**
   $$\mathbf{K}_t = \mathbf{P}_{xz} \mathbf{P}_{zz}^{-1}$$

6. **Обновление Состояния:**
   $$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \hat{\mathbf{z}}_{t|t-1})$$

7. **Обновление Ковариации:**
   $$\mathbf{P}_{t|t} = \mathbf{P}_{t|t-1} - \mathbf{K}_t \mathbf{P}_{zz} \mathbf{K}_t^T$$

### Статистика Хи-квадрат для Обнаружения Выбросов (Outlier Detection)

**Статистика хи-квадрат** (также называемая **Normalized Innovation Squared, NIS** или **Normalized Residual Squared**) — это мощный инструмент для обнаружения выбросов (outliers) в измерениях при использовании UKF. Она позволяет определить, является ли текущее измерение статистически согласованным с предсказанием фильтра или это аномальное наблюдение.

#### Математическая Формулировка

**Инновация (Innovation)** — это разница между фактическим измерением и предсказанным измерением:

$$\boldsymbol{\nu}_t = \mathbf{z}_t - \hat{\mathbf{z}}_{t|t-1}$$

где:
- $\mathbf{z}_t$ — фактическое измерение в момент времени $t$
- $\hat{\mathbf{z}}_{t|t-1}$ — предсказанное измерение (из шага 2 Update Step)
- $\boldsymbol{\nu}_t$ — инновация (innovation)

**Ковариация инноваций (Innovation Covariance)** уже вычисляется в алгоритме UKF:

$$\mathbf{S}_t = \mathbf{P}_{zz} = \sum_{i=0}^{2n} W_i^{(c)} (\boldsymbol{\mathcal{Z}}_{i,t|t-1} - \hat{\mathbf{z}}_{t|t-1})(\boldsymbol{\mathcal{Z}}_{i,t|t-1} - \hat{\mathbf{z}}_{t|t-1})^T + \mathbf{R}_t$$

**Статистика хи-квадрат** определяется как:

$$\chi^2_t = \boldsymbol{\nu}_t^T \mathbf{S}_t^{-1} \boldsymbol{\nu}_t$$

#### Теоретическое Обоснование

Если фильтр работает корректно и измерения соответствуют модели, то:

1. **Инновация** $\boldsymbol{\nu}_t$ имеет гауссовское распределение:
   $$\boldsymbol{\nu}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{S}_t)$$

2. **Нормализованная инновация** имеет стандартное гауссовское распределение:
   $$\mathbf{S}_t^{-1/2} \boldsymbol{\nu}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

3. **Статистика хи-квадрат** следует распределению хи-квадрат с $m$ степенями свободы:
   $$\chi^2_t \sim \chi^2(m)$$

где $m = \dim(\mathbf{z}_t)$ — размерность измерения.

#### Использование для Обнаружения Выбросов

**Правило принятия решения:**

Для заданного уровня значимости $\alpha$ (обычно $\alpha = 0.05$ или $0.01$), измерение считается выбросом, если:

$$\chi^2_t > \chi^2_{\alpha}(m)$$

где $\chi^2_{\alpha}(m)$ — критическое значение распределения хи-квадрат с $m$ степенями свободы и уровнем значимости $\alpha$.

**Интерпретация:**

- **$\chi^2_t \leq \chi^2_{\alpha}(m)$**: Измерение согласовано с моделью, используем его для обновления
- **$\chi^2_t > \chi^2_{\alpha}(m)$**: Измерение является выбросом, возможно:
  - Ошибочное измерение (шум сенсора, ошибка детектора)
  - Изменение модели (объект изменил поведение)
  - Неправильная модель фильтра

#### Критические Значения Хи-квадрат

Для типичных уровней значимости и размерностей измерений:

| Размерность $m$ | $\alpha = 0.05$ | $\alpha = 0.01$ | $\alpha = 0.001$ |
|-----------------|----------------|-----------------|------------------|
| 1 | 3.84 | 6.63 | 10.83 |
| 2 | 5.99 | 9.21 | 13.82 |
| 3 | 7.81 | 11.34 | 16.27 |
| 4 | 9.49 | 13.28 | 18.47 |
| 5 | 11.07 | 15.09 | 20.52 |

**Практическое правило:** Для 2D позиции ($m=2$) обычно используется порог $\chi^2_{0.05}(2) = 5.99$.

#### Реализация в UKF

```python
from scipy.stats import chi2

class UnscentedKalmanFilter:
    # ... (предыдущий код) ...
    
    def update(self, z, gate_threshold=None):
        """
        Update step с обнаружением выбросов
        
        Parameters:
        -----------
        z : np.array
            Измерение
        gate_threshold : float, optional
            Порог хи-квадрат для обнаружения выбросов
            Если None, используется chi2.ppf(0.95, dim_z)
        """
        # Генерация sigma points для предсказания
        sigma_points = self._compute_sigma_points(self.x, self.P)
        
        # Трансформация через функцию наблюдения
        sigma_points_z = np.zeros((self.dim_z, 2*self.dim_x + 1))
        for i in range(sigma_points.shape[1]):
            sigma_points_z[:, i] = self.hx(sigma_points[:, i])
        
        # Предсказание наблюдения
        z_pred = np.dot(sigma_points_z, self.Wm)
        
        # Ковариация наблюдения (Innovation Covariance)
        Pzz = np.zeros((self.dim_z, self.dim_z))
        for i in range(sigma_points_z.shape[1]):
            diff = sigma_points_z[:, i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R
        
        # Вычисление инновации
        innovation = z - z_pred
        
        # Вычисление статистики хи-квадрат
        try:
            Pzz_inv = np.linalg.inv(Pzz)
            chi_squared = np.dot(innovation.T, np.dot(Pzz_inv, innovation))
        except np.linalg.LinAlgError:
            # Если матрица вырождена, используем псевдообратную
            Pzz_inv = np.linalg.pinv(Pzz)
            chi_squared = np.dot(innovation.T, np.dot(Pzz_inv, innovation))
        
        # Обнаружение выбросов
        if gate_threshold is None:
            # Использовать 95% доверительный интервал
            gate_threshold = chi2.ppf(0.95, self.dim_z)
        
        is_outlier = chi_squared > gate_threshold
        
        if is_outlier:
            # Выброс обнаружен - можно:
            # 1. Пропустить обновление
            # 2. Использовать только предсказание
            # 3. Увеличить неопределенность
            # 4. Записать в лог для анализа
            
            # В этом примере пропускаем обновление
            print(f"Outlier detected! chi^2 = {chi_squared:.2f} > {gate_threshold:.2f}")
            return {
                'updated': False,
                'chi_squared': chi_squared,
                'is_outlier': True,
                'innovation': innovation
            }
        
        # Нормальное обновление
        # Кросс-ковариация
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(sigma_points.shape[1]):
            diff_x = sigma_points[:, i] - self.x
            diff_z = sigma_points_z[:, i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # Kalman gain
        K = np.dot(Pxz, np.linalg.inv(Pzz))
        
        # Обновление состояния
        self.x += np.dot(K, innovation)
        
        # Обновление ковариации
        self.P -= np.dot(K, np.dot(Pzz, K.T))
        
        return {
            'updated': True,
            'chi_squared': chi_squared,
            'is_outlier': False,
            'innovation': innovation
        }
```

#### Стратегии Обработки Выбросов

**1. Пропуск Обновления (Gating)**

Если обнаружен выброс, просто пропустить обновление и использовать только предсказание:

```python
if chi_squared > threshold:
    # Не обновлять состояние, использовать только предсказание
    pass
else:
    # Нормальное обновление
    ukf.update(z)
```

**2. Адаптивное Увеличение Неопределенности**

Увеличить ковариацию наблюдений для выбросов:

```python
if chi_squared > threshold:
    # Временно увеличить R для учета неопределенности
    R_original = ukf.R.copy()
    ukf.R = ukf.R * 10  # Увеличить в 10 раз
    ukf.update(z)
    ukf.R = R_original  # Восстановить
```

**3. Использование Множественных Гипотез**

Для множественного отслеживания объектов можно использовать несколько гипотез:

```python
# Для каждого трека проверяем все детекции
for track in tracks:
    for detection in detections:
        chi_squared = compute_chi_squared(track, detection)
        if chi_squared < threshold:
            # Возможная ассоциация
            associations.append((track, detection, chi_squared))
```

**4. Адаптивный Порог**

Динамически изменять порог на основе истории:

```python
# Вычислить среднее значение хи-квадрат за последние N шагов
chi_squared_history = []
chi_squared_history.append(chi_squared)

if len(chi_squared_history) > window_size:
    chi_squared_history.pop(0)

mean_chi_squared = np.mean(chi_squared_history)
adaptive_threshold = mean_chi_squared * 2  # Порог = 2 * среднее
```

#### Пример: Отслеживание с Обнаружением Выбросов

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Создание UKF с обнаружением выбросов
ukf = UnscentedKalmanFilter(
    dim_x=4, dim_z=2, dt=0.1,
    fx=motion_model, hx=observation_model,
    Q=Q, R=R
)

# Порог для обнаружения выбросов (95% доверительный интервал для 2D)
chi2_threshold = chi2.ppf(0.95, 2)  # ≈ 5.99

# Симуляция с выбросами
true_states = []
estimated_states = []
observations = []
chi_squared_values = []
outlier_flags = []

x_true = np.array([0, 0, 1, 0.5])
ukf.x = np.array([0, 0, 1, 0.5])
ukf.P = np.eye(4) * 10

for t in np.arange(0, 10, 0.1):
    # Истинное движение
    x_true = motion_model(x_true, 0.1) + np.random.multivariate_normal(
        np.zeros(4), Q
    )
    
    # Наблюдение (иногда добавляем выброс)
    z = observation_model(x_true) + np.random.multivariate_normal(
        np.zeros(2), R
    )
    
    # Добавляем выброс в 10% случаев
    if np.random.rand() < 0.1:
        z += np.random.multivariate_normal(
            np.zeros(2), np.eye(2) * 20  # Большой выброс
        )
    
    # UKF с обнаружением выбросов
    ukf.predict()
    result = ukf.update(z, gate_threshold=chi2_threshold)
    
    true_states.append(x_true.copy())
    estimated_states.append(ukf.x.copy())
    observations.append(z.copy())
    chi_squared_values.append(result['chi_squared'])
    outlier_flags.append(result['is_outlier'])

# Визуализация
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Траектории
true_states = np.array(true_states)
estimated_states = np.array(estimated_states)
observations = np.array(observations)
outlier_flags = np.array(outlier_flags)

axes[0].plot(true_states[:, 0], true_states[:, 1], 'g-', 
             label='Истинная траектория', linewidth=2)
axes[0].plot(estimated_states[:, 0], estimated_states[:, 1], 'b--', 
             label='UKF оценка', linewidth=2)
axes[0].plot(observations[~outlier_flags, 0], 
             observations[~outlier_flags, 1], 'b.', 
             label='Нормальные наблюдения', alpha=0.5)
axes[0].plot(observations[outlier_flags, 0], 
             observations[outlier_flags, 1], 'rx', 
             label='Выбросы', markersize=10)
axes[0].set_xlabel('X позиция')
axes[0].set_ylabel('Y позиция')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title('Отслеживание с Обнаружением Выбросов')

# Статистика хи-квадрат
axes[1].plot(chi_squared_values, 'b-', label='χ² статистика', linewidth=1)
axes[1].axhline(y=chi2_threshold, color='r', linestyle='--', 
                label=f'Порог (χ²₀.₀₅(2) = {chi2_threshold:.2f})')
axes[1].fill_between(range(len(chi_squared_values)), 
                      chi2_threshold, max(chi_squared_values), 
                      where=np.array(chi_squared_values) > chi2_threshold,
                      alpha=0.3, color='red', label='Область выбросов')
axes[1].set_xlabel('Время (шаги)')
axes[1].set_ylabel('χ² значение')
axes[1].legend()
axes[1].grid(True)
axes[1].set_title('Статистика Хи-квадрат для Обнаружения Выбросов')

plt.tight_layout()
plt.show()
```

#### Анализ Производительности

**Метрики для оценки качества обнаружения выбросов:**

1. **True Positive Rate (TPR)**: Доля правильно обнаруженных выбросов
2. **False Positive Rate (FPR)**: Доля нормальных измерений, ошибочно помеченных как выбросы
3. **Precision**: Точность обнаружения выбросов
4. **Recall**: Полнота обнаружения выбросов

```python
def evaluate_outlier_detection(true_outliers, detected_outliers):
    """
    Оценка качества обнаружения выбросов
    """
    tp = np.sum(true_outliers & detected_outliers)  # True Positives
    fp = np.sum(~true_outliers & detected_outliers)  # False Positives
    fn = np.sum(true_outliers & ~detected_outliers)  # False Negatives
    tn = np.sum(~true_outliers & ~detected_outliers)  # True Negatives
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        'TPR': tpr,
        'FPR': fpr,
        'Precision': precision,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
    }
```

#### Практические Рекомендации

**1. Выбор Уровня Значимости**

- **$\alpha = 0.05$ (95% доверительный интервал)**: Стандартный выбор для большинства задач
- **$\alpha = 0.01$ (99% доверительный интервал)**: Более строгий, меньше ложных срабатываний
- **$\alpha = 0.001$ (99.9% доверительный интервал)**: Очень строгий, только явные выбросы

**2. Адаптация к Размерности**

Для высоких размерностей ($m > 5$) статистика хи-квадрат может быть менее чувствительной. Рассмотрите:
- Использование компонентного анализа (PCA) для снижения размерности
- Разделение измерения на подкомпоненты и проверка каждой отдельно

**3. Временная Фильтрация**

Используйте скользящее окно для анализа трендов:

```python
# Вычислить среднее значение хи-квадрат за окно
window_size = 10
if len(chi_squared_history) >= window_size:
    recent_mean = np.mean(chi_squared_history[-window_size:])
    if recent_mean > threshold:
        # Систематическая проблема, не просто выброс
        print("Warning: Systematic measurement issues detected")
```

**4. Интеграция с Multiple Object Tracking**

В системах множественного отслеживания (MOT) статистика хи-квадрат используется для:
- **Gating**: Исключение невозможных ассоциаций
- **Data Association**: Выбор лучшей ассоциации на основе минимального $\chi^2$
- **Track Management**: Решение о создании/удалении треков

```python
# Пример для MOT
def associate_detections_to_tracks(tracks, detections, chi2_threshold):
    """
    Ассоциация детекций с треками на основе хи-квадрат
    """
    cost_matrix = np.full((len(tracks), len(detections)), np.inf)
    
    for i, track in enumerate(tracks):
        track.predict()
        for j, detection in enumerate(detections):
            # Вычислить хи-квадрат для этой пары
            innovation = detection - track.z_pred
            chi_squared = innovation.T @ track.S_inv @ innovation
            
            if chi_squared < chi2_threshold:
                cost_matrix[i, j] = chi_squared
    
    # Использовать венгерский алгоритм для оптимальной ассоциации
    # (реализация опущена для краткости)
    return cost_matrix
```

#### Ограничения и Предостережения

1. **Предположение о Гауссовости**: Статистика хи-квадрат предполагает гауссовские распределения. Для сильно нелинейных систем это может быть неточно.

2. **Коррелированные Выбросы**: Если выбросы происходят последовательно, стандартный порог может быть недостаточным.

3. **Размерность**: Для очень высоких размерностей ($m > 10$) распределение хи-квадрат становится менее информативным.

4. **Вычислительная Сложность**: Инверсия матрицы $\mathbf{S}_t$ может быть дорогой для больших размерностей.

#### Связь с Другими Методами

**Validation Gate (Валидационные Ворота):**
Статистика хи-квадрат реализует концепцию "validation gate" — области в пространстве измерений, внутри которой измерения считаются валидными. Эта область представляет собой эллипсоид:

$$\{\mathbf{z} : (\mathbf{z} - \hat{\mathbf{z}}_{t|t-1})^T \mathbf{S}_t^{-1} (\mathbf{z} - \hat{\mathbf{z}}_{t|t-1}) \leq \chi^2_{\alpha}(m)\}$$

**Mahalanobis Distance:**
Статистика хи-квадрат — это квадрат расстояния Махаланобиса между измерением и предсказанием:

$$\chi^2_t = D_M^2(\mathbf{z}_t, \hat{\mathbf{z}}_{t|t-1}) = (\mathbf{z}_t - \hat{\mathbf{z}}_{t|t-1})^T \mathbf{S}_t^{-1} (\mathbf{z}_t - \hat{\mathbf{z}}_{t|t-1})$$

**Как бы я объяснил это пятилетнему ребенку:**

Представь, что ты играешь в игру, где нужно угадать, где находится мяч. Ты делаешь предсказание, а потом смотришь, где мяч на самом деле. Если мяч находится очень далеко от твоего предсказания (дальше, чем обычно), то, возможно, что-то пошло не так — может быть, кто-то сдвинул мяч, или ты ошибся. Статистика хи-квадрат — это способ измерить, насколько далеко находится мяч от твоего предсказания, и решить, стоит ли доверять этому измерению или это ошибка.

### Преимущества UKF

1. **Точность**: Точнее EKF для нелинейных систем (точность до второго порядка)
2. **Без производных**: Не требует вычисления якобианов
3. **Устойчивость**: Более устойчив к начальным условиям
4. **Вычислительная эффективность**: $O(n^3)$ как у EKF, но без вычисления производных

### Ограничения UKF

1. **Гауссовские распределения**: Предполагает гауссовские распределения
2. **Мультимодальность**: Не может обрабатывать мультимодальные распределения
3. **Высокая размерность**: Может быть неэффективен для очень высоких размерностей

### Настройка Параметров UKF

UKF имеет те же основные параметры, что и EKF ($\mathbf{Q}$, $\mathbf{R}$, $\mathbf{P}_0$, $\hat{\mathbf{x}}_0$), плюс три специфичных параметра Unscented Transform: **$\alpha$**, **$\beta$**, и **$\kappa$**.

#### Основные Параметры UKF

**1. Параметры Шумов (как в EKF):**

- **$\mathbf{Q}_t$ — Process Noise Covariance**: Настраивается так же, как в EKF
- **$\mathbf{R}_t$ — Measurement Noise Covariance**: Настраивается так же, как в EKF
- **$\mathbf{P}_{0|0}$ — Initial State Covariance**: Настраивается так же, как в EKF
- **$\hat{\mathbf{x}}_{0|0}$ — Initial State**: Настраивается так же, как в EKF

**2. Параметры Unscented Transform:**

- **$\alpha$ (ukf_alpha)** — Параметр распространения sigma points
  - Контролирует, насколько далеко от среднего размещаются sigma points
  - Диапазон: $10^{-3} \leq \alpha \leq 1$
  - **Малые значения** ($\alpha \approx 0.001$): Sigma points близко к среднему, более консервативно
  - **Большие значения** ($\alpha \approx 1$): Sigma points дальше от среднего, лучше для сильной нелинейности
  - **Рекомендуемое значение**: $\alpha = 1$ для большинства задач

- **$\beta$ (ukf_beta)** — Параметр для учета априорной информации
  - Используется для учета информации о распределении (например, моменты высших порядков)
  - Для гауссовских распределений: $\beta = 2$ (оптимально)
  - Для других распределений: может быть настроен эмпирически
  - **Рекомендуемое значение**: $\beta = 2$ для гауссовских распределений

- **$\kappa$ (ukf_kappa)** — Вторичный параметр масштабирования
  - Обычно устанавливается как $\kappa = 0$ или $\kappa = 3 - n$ (где $n$ — размерность состояния)
  - Влияет на расположение sigma points через $\lambda = \alpha^2(n+\kappa) - n$
  - **Рекомендуемое значение**: $\kappa = 0$ или $\kappa = 3 - n$

#### Настройка Параметров Unscented Transform

**Шаг 1: Начните со Стандартных Значений**

Для большинства задач используйте стандартные значения:

```python
# Стандартные параметры UKF
ukf_alpha = 1.0      # Полное распространение sigma points
ukf_beta = 2.0       # Оптимально для гауссовских распределений
ukf_kappa = 0.0      # Или 3 - dim_x для обеспечения положительной определенности
```

**Шаг 2: Настройка $\alpha$ (ukf_alpha)**

$\alpha$ — самый важный параметр для настройки:

```python
# Консервативный подход (для стабильных систем)
ukf_alpha = 0.001  # Sigma points очень близко к среднему

# Стандартный подход (для большинства задач)
ukf_alpha = 1.0    # Полное использование ковариации

# Агрессивный подход (для сильно нелинейных систем)
ukf_alpha = 1.0    # Максимальное значение (обычно достаточно)
```

**Правило выбора $\alpha$:**
- Если фильтр слишком консервативен (медленно реагирует): увеличьте $\alpha$ до 1.0
- Если фильтр нестабилен: уменьшите $\alpha$ до 0.1-0.5
- Для большинства задач: $\alpha = 1.0$ работает хорошо

**Шаг 3: Настройка $\beta$ (ukf_beta)**

$\beta$ обычно не требует настройки:

```python
# Для гауссовских распределений (стандартный случай)
ukf_beta = 2.0

# Если известно, что распределение имеет тяжелые хвосты
# можно попробовать другие значения, но обычно не требуется
```

**Шаг 4: Настройка $\kappa$ (ukf_kappa)**

$\kappa$ обычно устанавливается автоматически:

```python
dim_x = 4  # Размерность состояния

# Вариант 1: kappa = 0 (стандартный)
ukf_kappa = 0.0

# Вариант 2: kappa = 3 - dim_x (обеспечивает положительную определенность)
ukf_kappa = 3 - dim_x  # Для dim_x=4: kappa = -1

# Обычно оба варианта работают одинаково хорошо
```

#### Практические Рекомендации для UKF

**1. Стандартная Конфигурация**

Для большинства задач используйте:

```python
# Стандартные параметры UKF
ukf_params = {
    'alpha': 1.0,
    'beta': 2.0,
    'kappa': 0.0
}

# Параметры шумов (настраиваются как в EKF)
Q = np.diag([process_noise_pos**2, process_noise_pos**2, 
             process_noise_vel**2, process_noise_vel**2])
R = np.diag([measurement_noise_pos**2, measurement_noise_pos**2])
```

**2. Настройка для Сильно Нелинейных Систем**

Если система сильно нелинейна:

```python
# Увеличьте alpha для лучшего покрытия нелинейности
ukf_alpha = 1.0  # Уже максимальное значение

# Увеличьте Q для учета неопределенности модели
Q = Q * 1.5  # Увеличить на 50%
```

**3. Настройка для Высокой Размерности**

Для систем с высокой размерностью ($n > 10$):

```python
# Уменьшите alpha для предотвращения численной нестабильности
ukf_alpha = 0.5  # Вместо 1.0

# Или используйте масштабированный UKF
# (более сложный вариант, требует модификации алгоритма)
```

**4. Компонентная Настройка Параметров**

Как и в EKF, можно настроить параметры по компонентам:

```python
# Для состояния [px, py, vx, vy, orient]
Q = np.diag([
    process_noise_pos**2,      # process_noise_pos
    process_noise_pos**2,
    process_noise_vel**2,      # process_noise_vel
    process_noise_vel**2,
    process_noise_orient**2    # process_noise_orient
])

R = np.diag([
    measurement_noise_pos**2,      # measurement_noise_pos
    measurement_noise_pos**2,
    measurement_noise_orient**2,   # measurement_noise_orient
    measurement_noise_dim**2       # measurement_noise_dim (если есть)
])
```

#### Пример Полной Конфигурации UKF

```python
class UKFConfig:
    """Конфигурация UKF для отслеживания объекта с ориентацией"""
    
    # Параметры Unscented Transform
    ukf_alpha = 1.0      # Полное распространение
    ukf_beta = 2.0       # Для гауссовских распределений
    ukf_kappa = 0.0      # Стандартное значение
    
    # Process Noise (шум процесса)
    process_noise_pos = 0.1      # Неопределенность позиции (м)
    process_noise_vel = 0.5       # Неопределенность скорости (м/с)
    process_noise_orient = 0.01  # Неопределенность ориентации (рад)
    process_noise_dim = 0.05     # Неопределенность размеров (м)
    
    # Measurement Noise (шум наблюдений)
    measurement_noise_pos = 2.0      # Точность GPS (м)
    measurement_noise_orient = 0.01 # Точность IMU (рад)
    measurement_noise_dim = 0.1      # Точность детектора размеров (м)
    
    @classmethod
    def get_Q(cls, dt):
        """Вычислить Q матрицу для модели постоянной скорости"""
        n = 7  # [x, y, vx, vy, orient, width, height]
        Q = np.zeros((n, n))
        
        # Позиция и скорость (модель постоянной скорости)
        Q[0:2, 0:2] = np.eye(2) * (dt**4/4) * cls.process_noise_pos**2
        Q[2:4, 2:4] = np.eye(2) * (dt**2) * cls.process_noise_vel**2
        Q[0:2, 2:4] = np.eye(2) * (dt**3/2) * cls.process_noise_pos * cls.process_noise_vel
        Q[2:4, 0:2] = Q[0:2, 2:4].T
        
        # Ориентация (изменяется медленно)
        Q[4, 4] = cls.process_noise_orient**2
        
        # Размеры (изменяются очень медленно)
        Q[5:7, 5:7] = np.eye(2) * cls.process_noise_dim**2
        
        return Q
    
    @classmethod
    def get_R(cls):
        """Вычислить R матрицу"""
        R = np.diag([
            cls.measurement_noise_pos**2,
            cls.measurement_noise_pos**2,
            cls.measurement_noise_orient**2,
            cls.measurement_noise_dim**2,
            cls.measurement_noise_dim**2
        ])
        return R
```

#### Диагностика Проблем UKF

**Проблема: Численная Нестабильность**

**Симптомы:**
- Ошибки при вычислении матричного квадратного корня
- Отрицательные значения в $\mathbf{P}$

**Решение:**
- Уменьшите $\alpha$ до 0.5 или меньше
- Увеличьте $\kappa$ до $3-n$ для обеспечения положительной определенности
- Добавьте небольшой шум к диагонали $\mathbf{P}$: $\mathbf{P} = \mathbf{P} + \epsilon \mathbf{I}$

**Проблема: Недостаточное Покрытие Нелинейности**

**Симптомы:**
- Фильтр работает хуже, чем ожидалось для нелинейных систем
- Большие ошибки в областях высокой нелинейности

**Решение:**
- Увеличьте $\alpha$ до 1.0 (максимальное значение)
- Проверьте, что $\mathbf{Q}$ достаточно большой для учета неопределенности модели

**Проблема: Слишком Консервативное Поведение**

**Симптомы:**
- Фильтр медленно реагирует на изменения
- Sigma points слишком близко к среднему

**Решение:**
- Увеличьте $\alpha$ до 1.0
- Уменьшите $\mathbf{R}$ (больше доверия к наблюдениям)

#### Сравнение Настройки EKF vs UKF

| Параметр | EKF | UKF | Комментарий |
|----------|-----|-----|-------------|
| **Q, R, P₀** | Требуют настройки | Требуют настройки | Одинаково для обоих |
| **Якобианы** | Требуют вычисления | Не требуются | UKF проще |
| **$\alpha$** | Не применимо | Требует настройки | Обычно $\alpha = 1.0$ |
| **$\beta$** | Не применимо | Обычно $\beta = 2.0$ | Редко требует настройки |
| **$\kappa$** | Не применимо | Обычно $\kappa = 0$ | Редко требует настройки |

**Вывод:** UKF требует настройки дополнительных параметров ($\alpha$, $\beta$, $\kappa$), но они обычно устанавливаются на стандартные значения и не требуют частой настройки. Основная работа по настройке — это настройка $\mathbf{Q}$ и $\mathbf{R}$, что одинаково для EKF и UKF.

#### Чеклист Настройки UKF

- [ ] Установить стандартные значения UKF параметров ($\alpha=1.0$, $\beta=2.0$, $\kappa=0.0$)
- [ ] Настроить $\mathbf{Q}$ и $\mathbf{R}$ (как в EKF)
- [ ] Настроить $\mathbf{P}_0$ и $\hat{\mathbf{x}}_0$
- [ ] Если система сильно нелинейна: проверить, что $\alpha = 1.0$
- [ ] Если возникают проблемы с численной стабильностью: уменьшить $\alpha$ или увеличить $\kappa$
- [ ] Валидировать на отдельном наборе данных
- [ ] Сравнить с EKF для проверки улучшения
- [ ] Документировать выбранные значения

---

## Particle Filter (PF)

### Основная Идея

Particle Filter (также известный как **Sequential Monte Carlo, SMC**) использует **набор частиц** (samples) для представления распределения вероятностей. Это позволяет обрабатывать **нелинейные** и **негауссовские** системы, включая **мультимодальные распределения**.

### Алгоритм Particle Filter

**Инициализация:**

Генерируем $N$ частиц из начального распределения:
$$\mathbf{x}_0^{(i)} \sim p(\mathbf{x}_0), \quad i = 1, ..., N$$

**Predict Step:**

Для каждой частицы:
$$\mathbf{x}_t^{(i)} \sim p(\mathbf{x}_t | \mathbf{x}_{t-1}^{(i)}) = f(\mathbf{x}_{t-1}^{(i)}, \mathbf{w}_t^{(i)})$$

**Update Step:**

1. **Вычисление весов:**
   $$w_t^{(i)} = p(\mathbf{z}_t | \mathbf{x}_t^{(i)}) = h(\mathbf{x}_t^{(i)}, \mathbf{v}_t)$$
   $$\tilde{w}_t^{(i)} = \frac{w_t^{(i)}}{\sum_{j=1}^N w_t^{(j)}}$$

2. **Resampling (перевыборка):**
   Перевыбираем частицы согласно их весам для предотвращения вырождения (degeneracy).

### Преимущества Particle Filter

1. **Универсальность**: Работает с любыми нелинейностями и распределениями
2. **Мультимодальность**: Может обрабатывать мультимодальные распределения
3. **Точность**: При достаточном количестве частиц может быть очень точным

### Ограничения Particle Filter

1. **Вычислительная сложность**: $O(N)$ где $N$ — количество частиц (обычно $N = 100-10000$)
2. **Вырождение**: Требует resampling, который может привести к потере разнообразия
3. **Высокая размерность**: Требует экспоненциально больше частиц для высоких размерностей

---

## Сравнение Методов

### Таблица Сравнения

| Характеристика | KF | EKF | UKF | PF |
|----------------|----|-----|-----|----|
| **Тип систем** | Линейные | Нелинейные | Нелинейные | Нелинейные |
| **Распределения** | Гауссовские | Гауссовские | Гауссовские | Любые |
| **Мультимодальность** | Нет | Нет | Нет | Да |
| **Точность** | Точный | 1-й порядок | 2-й порядок | Точный (при $N \to \infty$) |
| **Вычислительная сложность** | $O(n^3)$ | $O(n^3)$ | $O(n^3)$ | $O(N)$ |
| **Требует производные** | Нет | Да | Нет | Нет |
| **Устойчивость** | Высокая | Средняя | Высокая | Средняя |

### Когда Использовать Каждый Метод

**Kalman Filter (KF):**
- Линейные системы с гауссовскими шумами
- Низкая вычислительная сложность
- Примеры: отслеживание с постоянной скоростью, навигация

**Extended Kalman Filter (EKF):**
- Слабо нелинейные системы
- Когда производные легко вычисляются
- Примеры: отслеживание с постоянным ускорением, GPS навигация

**Unscented Kalman Filter (UKF):**
- Умеренно нелинейные системы
- Когда производные сложны или неточны
- **Рекомендуется вместо EKF** для большинства нелинейных задач
- Примеры: отслеживание объектов в компьютерном зрении, робототехника

**Particle Filter (PF):**
- Сильно нелинейные системы
- Негауссовские распределения
- Мультимодальные распределения
- Примеры: отслеживание множественных объектов, SLAM

### UKF vs EKF: Детальное Сравнение

**Точность:**

UKF точнее EKF, потому что:
- EKF использует линеаризацию первого порядка (ошибка $O(\Delta x^2)$)
- UKF использует Unscented Transform (точность до второго порядка)
- UKF лучше сохраняет свойства распределения (симметрия, моменты)

**Вычислительная Сложность:**

- EKF: $O(n^3)$ + вычисление якобианов
- UKF: $O(n^3)$ + $2n+1$ вычислений функции
- Обычно UKF быстрее, так как вычисление функции проще, чем вычисление производных

**Устойчивость:**

- EKF может расходиться при сильной нелинейности
- UKF более устойчив благодаря лучшей аппроксимации

**Практические Рекомендации:**

**Используйте UKF вместо EKF**, если:
- Система нелинейна
- Производные сложны для вычисления
- Нужна более высокая точность
- Система умеренно нелинейна (не требует Particle Filter)

---

## Современные Методы Отслеживания

### Deep Learning Подходы

#### 1. DeepSORT и SORT

**SORT (Simple Online and Realtime Tracking)** — классический метод:
- Использует Kalman Filter для предсказания
- Венгерский алгоритм для ассоциации
- Простой и эффективный

**DeepSORT** — улучшенная версия:
- Добавляет deep learning модель для re-identification
- Более точная ассоциация объектов
- Использует CNN для извлечения признаков

#### 2. Transformer-based Tracking

**TransTrack** (2020):
- Использует Transformer для отслеживания
- End-to-end обучение
- Одновременное обнаружение и отслеживание

**TrackFormer** (2021):
- Transformer-based архитектура
- Использует attention механизмы
- Высокая точность на стандартных бенчмарках

**MOTR** (2021):
- End-to-end multiple object tracking
- Использует DETR (Detection Transformer)
- Не требует отдельного модуля ассоциации

#### 3. Joint Detection and Tracking

**FairMOT** (2020):
- Единая сеть для обнаружения и отслеживания
- Использует anchor-free detection
- Эффективен для множественного отслеживания

**CenterTrack** (2020):
- Отслеживание через центры объектов
- Простая и эффективная архитектура
- Хорошая производительность

#### 4. Graph Neural Networks для Tracking

**GNN-based Tracking**:
- Использует графы для представления связей между объектами
- GNN для ассоциации треков
- Эффективен для сложных сцен

### Современные Гибридные Подходы (2023-2026)

#### 1. ByteTrack (2021)

- Использует все детекции (включая низкоуверенные)
- Простая ассоциация на основе IoU
- Очень эффективен и точный
- Стандарт для множественного отслеживания

#### 2. StrongSORT (2022)

- Улучшенная версия DeepSORT
- Более точная ассоциация
- Использует appearance features
- Kalman Filter с улучшенной моделью движения

#### 3. OC-SORT (2022)

- Observation-Centric SORT
- Фокусируется на наблюдениях, а не на предсказаниях
- Более устойчив к occlusions
- Использует Kalman Filter с модификациями

#### 4. Bot-SORT (2022)

- ByteTrack + StrongSORT гибрид
- Использует camera motion compensation
- Высокая точность на стандартных бенчмарках

#### 5. Transformer-based Multi-Object Tracking

**TransMOT** (2023):
- Transformer для множественного отслеживания
- Использует temporal attention
- End-to-end обучение

**MOTRv2** (2023):
- Улучшенная версия MOTR
- Более эффективная архитектура
- Лучшая точность

### Современные Методы для Специфических Задач

#### Visual Object Tracking (Single Object)

**Siamese Networks:**
- SiamRPN, SiamMask, SiamBAN
- Используют Siamese архитектуру для отслеживания
- Очень эффективны для single object tracking

**Transformer-based Trackers:**
- **STARK** (2021): Spatio-Temporal Transformer
- **OSTrack** (2022): One-Stream Transformer
- **MixFormer** (2022): Mixed attention mechanism
- **SimTrack** (2022): Simple and effective

**Latest (2024-2026):**
- **SeqTrack** (2023): Sequence-to-sequence tracking
- **GRM** (2023): Global Response Mechanism
- **AiATrack** (2023): Attention in Attention

#### 3D Object Tracking

**3D Kalman Filter:**
- Расширение KF для 3D пространства
- Используется в автономных транспортных средствах

**LiDAR-based Tracking:**
- AB3DMOT (2020)
- CenterPoint (2021)
- 3D SORT

#### Multi-Modal Tracking

**RGB-D Tracking:**
- Использует RGB и depth информацию
- Более точное отслеживание

**RGB-Thermal Tracking:**
- Использует видимый и тепловой спектры
- Эффективен в сложных условиях

### Сравнение Современных Методов

| Метод | Тип | Точность | Скорость | Применимость |
|-------|-----|----------|----------|--------------|
| **SORT** | Traditional | Средняя | Очень высокая | Простые сцены |
| **DeepSORT** | Hybrid | Высокая | Высокая | Общие задачи |
| **ByteTrack** | Hybrid | Очень высокая | Высокая | Стандарт 2022-2024 |
| **StrongSORT** | Hybrid | Очень высокая | Средняя | Высокая точность |
| **OC-SORT** | Hybrid | Высокая | Высокая | Occlusions |
| **MOTR** | Deep Learning | Высокая | Средняя | End-to-end |
| **TransTrack** | Deep Learning | Высокая | Средняя | Transformer-based |

### Рекомендации по Выбору Метода (2024-2026)

**Для Production:**
- **ByteTrack** или **OC-SORT** — лучший баланс точности и скорости
- **StrongSORT** — если нужна максимальная точность

**Для Research:**
- **Transformer-based методы** (MOTR, TransMOT) — для изучения end-to-end подходов
- **Graph-based методы** — для сложных сцен с множественными взаимодействиями

**Для Single Object Tracking:**
- **OSTrack** или **MixFormer** — state-of-the-art точность
- **SeqTrack** — для sequence-to-sequence подходов

**Для 3D Tracking:**
- **3D Kalman Filter** или **CenterPoint** — для LiDAR данных
- **AB3DMOT** — для множественного 3D отслеживания

---

## Реализация UKF

### Python Реализация

```python
import numpy as np
from scipy.linalg import cholesky

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter для нелинейных систем
    """
    
    def __init__(self, dim_x, dim_z, dt, fx, hx, Q, R, alpha=1e-3, beta=2, kappa=0):
        """
        Parameters:
        -----------
        dim_x : int
            Размерность состояния
        dim_z : int
            Размерность наблюдений
        dt : float
            Интервал времени
        fx : callable
            Функция перехода состояния: x = fx(x, dt)
        hx : callable
            Функция наблюдения: z = hx(x)
        Q : np.array
            Ковариация шума процесса
        R : np.array
            Ковариация шума наблюдений
        alpha : float
            Параметр распространения (обычно 1e-3)
        beta : float
            Параметр для учета априорной информации (2 для гауссовских)
        kappa : float
            Вторичный параметр масштабирования (0 или 3-dim_x)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R
        
        # Параметры Unscented Transform
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa != 0 else 3 - dim_x
        self.lambda_ = alpha**2 * (dim_x + kappa) - dim_x
        
        # Веса
        self.Wm = np.zeros(2*dim_x + 1)
        self.Wc = np.zeros(2*dim_x + 1)
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (dim_x + self.lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2*dim_x + 1):
            self.Wm[i] = 1 / (2 * (dim_x + self.lambda_))
            self.Wc[i] = 1 / (2 * (dim_x + self.lambda_))
        
        # Инициализация состояния
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
    
    def predict(self, u=None):
        """
        Predict step (time update)
        """
        # Генерация sigma points
        sigma_points = self._compute_sigma_points(self.x, self.P)
        
        # Трансформация через функцию перехода
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[1]):
            sigma_points_pred[:, i] = self.fx(sigma_points[:, i], self.dt, u)
        
        # Предсказание среднего
        self.x = np.dot(sigma_points_pred, self.Wm)
        
        # Предсказание ковариации
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(sigma_points_pred.shape[1]):
            diff = sigma_points_pred[:, i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q
    
    def update(self, z):
        """
        Update step (measurement update)
        """
        # Генерация sigma points для предсказания
        sigma_points = self._compute_sigma_points(self.x, self.P)
        
        # Трансформация через функцию наблюдения
        sigma_points_z = np.zeros((self.dim_z, 2*self.dim_x + 1))
        for i in range(sigma_points.shape[1]):
            sigma_points_z[:, i] = self.hx(sigma_points[:, i])
        
        # Предсказание наблюдения
        z_pred = np.dot(sigma_points_z, self.Wm)
        
        # Ковариация наблюдения
        Pzz = np.zeros((self.dim_z, self.dim_z))
        for i in range(sigma_points_z.shape[1]):
            diff = sigma_points_z[:, i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R
        
        # Кросс-ковариация
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(sigma_points.shape[1]):
            diff_x = sigma_points[:, i] - self.x
            diff_z = sigma_points_z[:, i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # Kalman gain
        K = np.dot(Pxz, np.linalg.inv(Pzz))
        
        # Обновление состояния
        self.x += np.dot(K, z - z_pred)
        
        # Обновление ковариации
        self.P -= np.dot(K, np.dot(Pzz, K.T))
    
    def _compute_sigma_points(self, x, P):
        """
        Вычисление sigma points
        """
        n = self.dim_x
        sigma_points = np.zeros((n, 2*n + 1))
        sigma_points[:, 0] = x
        
        # Вычисление матричного квадратного корня
        try:
            L = cholesky((n + self.lambda_) * P)
        except:
            # Если Cholesky не работает, используем SVD
            U, s, V = np.linalg.svd((n + self.lambda_) * P)
            L = U @ np.diag(np.sqrt(s))
        
        for i in range(n):
            sigma_points[:, i+1] = x + L[:, i]
            sigma_points[:, i+n+1] = x - L[:, i]
        
        return sigma_points
```

### Пример Использования: Отслеживание Движущегося Объекта

```python
import matplotlib.pyplot as plt

# Модель движения: постоянная скорость с поворотами
def motion_model(x, dt, u=None):
    """
    x = [px, py, vx, vy] - позиция и скорость
    """
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return F @ x

# Модель наблюдения: измеряем только позицию
def observation_model(x):
    """
    z = [px, py] - наблюдаем только позицию
    """
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return H @ x

# Параметры
dim_x = 4  # [px, py, vx, vy]
dim_z = 2  # [px, py]
dt = 0.1

# Шумы
Q = np.eye(dim_x) * 0.1  # Шум процесса
R = np.eye(dim_z) * 1.0  # Шум наблюдений

# Создание фильтра
ukf = UnscentedKalmanFilter(
    dim_x=dim_x,
    dim_z=dim_z,
    dt=dt,
    fx=motion_model,
    hx=observation_model,
    Q=Q,
    R=R
)

# Инициализация
ukf.x = np.array([0, 0, 1, 0.5])  # Начальное состояние
ukf.P = np.eye(dim_x) * 10

# Симуляция
true_states = []
estimated_states = []
observations = []

x_true = np.array([0, 0, 1, 0.5])
for t in np.arange(0, 10, dt):
    # Истинное движение (с небольшим поворотом)
    if t > 5:
        # Поворот после 5 секунд
        angle = 0.1
        vx_new = x_true[2] * np.cos(angle) - x_true[3] * np.sin(angle)
        vy_new = x_true[2] * np.sin(angle) + x_true[3] * np.cos(angle)
        x_true[2] = vx_new
        x_true[3] = vy_new
    
    x_true = motion_model(x_true, dt) + np.random.multivariate_normal(
        np.zeros(dim_x), Q
    )
    
    # Наблюдение
    z = observation_model(x_true) + np.random.multivariate_normal(
        np.zeros(dim_z), R
    )
    
    # UKF
    ukf.predict()
    ukf.update(z)
    
    true_states.append(x_true.copy())
    estimated_states.append(ukf.x.copy())
    observations.append(z.copy())

# Визуализация
true_states = np.array(true_states)
estimated_states = np.array(estimated_states)
observations = np.array(observations)

plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0], true_states[:, 1], 'g-', label='Истинная траектория', linewidth=2)
plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'b--', label='UKF оценка', linewidth=2)
plt.plot(observations[:, 0], observations[:, 1], 'r.', label='Наблюдения', alpha=0.5)
plt.xlabel('X позиция')
plt.ylabel('Y позиция')
plt.legend()
plt.grid(True)
plt.title('Unscented Kalman Filter: Отслеживание Объекта')
plt.show()
```

---

## Применения и Примеры

### 1. Отслеживание Объектов в Компьютерном Зрении

**Задача:** Отслеживание людей, автомобилей, животных в видео

**Применение UKF:**
- Модель движения: постоянная скорость или постоянное ускорение
- Наблюдения: детекции из object detection модели (YOLO, Faster R-CNN)
- UKF используется для сглаживания траекторий и предсказания позиций

**Пример с YOLO + UKF:**

```python
# Псевдокод для отслеживания с YOLO
def track_objects_with_ukf(video, yolo_model):
    ukf_trackers = {}
    
    for frame in video:
        # Детекция объектов
        detections = yolo_model.detect(frame)
        
        # Обновление существующих трекеров
        for track_id, ukf in ukf_trackers.items():
            ukf.predict()
            # Ассоциация с детекциями
            best_detection = associate(ukf, detections)
            if best_detection:
                ukf.update(best_detection)
        
        # Создание новых трекеров для неассоциированных детекций
        for detection in unassociated_detections:
            new_tracker = create_ukf_tracker(detection)
            ukf_trackers[new_id] = new_tracker
```

### 2. Робототехника и Навигация

**SLAM (Simultaneous Localization and Mapping):**
- UKF для оценки позиции робота
- Объединение данных с IMU, одометрии, LiDAR

**Автономные Транспортные Средства:**
- Отслеживание других транспортных средств
- Предсказание траекторий
- Планирование пути

### 3. GPS и Навигационные Системы

**Интеграция GPS + IMU:**
- UKF для объединения данных GPS и инерциальных датчиков
- Более точная оценка позиции и ориентации

### 4. Финансовое Моделирование

**Отслеживание Финансовых Инструментов:**
- Моделирование цен акций
- Фильтрация шума в финансовых данных

### 5. Биомедицинские Применения

**Отслеживание Движения:**
- Анализ походки
- Отслеживание маркеров в motion capture

---

## Текущее Состояние (2023-2026)

### Тренды в Отслеживании Объектов

#### 1. End-to-End Learning

**Тенденция:** Переход от модульных систем к end-to-end обучению
- **MOTR, TransTrack**: Полностью обучаемые системы
- **Преимущества**: Оптимизация всей системы совместно
- **Недостатки**: Требуют больших датасетов и вычислительных ресурсов

#### 2. Transformer Dominance

**Тенденция:** Transformer архитектуры доминируют в tracking
- **Attention механизмы**: Эффективны для временных зависимостей
- **Self-attention**: Моделирование взаимодействий между объектами
- **Cross-attention**: Связь между детекциями и треками

#### 3. Мультимодальное Отслеживание

**Тенденция:** Использование множественных модальностей
- **RGB + Depth**: Более точное 3D отслеживание
- **RGB + Thermal**: Отслеживание в сложных условиях
- **Camera + LiDAR**: Для автономных транспортных средств

#### 4. Эффективность и Реальное Время

**Тенденция:** Оптимизация для реального времени
- **Lightweight модели**: MobileNet, EfficientNet backbones
- **Квантизация и дистилляция**: Сжатие моделей
- **Edge computing**: Развертывание на мобильных устройствах

### State-of-the-Art Методы (2024-2026)

#### Multiple Object Tracking (MOT)

**Лидеры по точности:**
1. **ByteTrack** (2021) — все еще актуален
2. **OC-SORT** (2022) — лучший для occlusions
3. **StrongSORT** (2022) — максимальная точность
4. **Bot-SORT** (2022) — гибридный подход

**Новые методы:**
- **MOTRv2** (2023) — улучшенный Transformer-based
- **GTR** (2023) — Graph Transformer для tracking
- **QDTrack** (2023) — Query-based detection и tracking

#### Single Object Tracking (SOT)

**Лидеры:**
1. **OSTrack** (2022) — One-Stream Transformer
2. **MixFormer** (2022) — Mixed attention
3. **SeqTrack** (2023) — Sequence-to-sequence
4. **AiATrack** (2023) — Attention in Attention

### Роль UKF в Современных Системах

**UKF все еще актуален для:**
1. **Гибридные системы**: Комбинация с deep learning детекторами
2. **Реальное время**: Когда нужна низкая задержка
3. **Ресурсоограниченные системы**: Мобильные устройства, embedded системы
4. **Робототехника**: SLAM, навигация, контроль

**Примеры современных систем, использующих UKF:**
- **DeepSORT**: Использует Kalman Filter (можно заменить на UKF)
- **SORT**: Основан на Kalman Filter
- **ByteTrack**: Использует Kalman Filter для предсказания

### Будущие Направления

1. **Нейро-символические системы**: Комбинация символических методов (UKF) с нейросетями
2. **Uncertainty quantification**: Более точная оценка неопределенности
3. **Continual learning**: Адаптация к новым типам объектов
4. **Multi-agent tracking**: Отслеживание в мультиагентных системах

---

## References

### Основные Работы по UKF

1. **Julier, S. J., & Uhlmann, J. K.** (1997). "New extension of the Kalman filter to nonlinear systems." *Signal processing, sensor fusion, and target recognition VI*, 3068, 182-193.

2. **Julier, S. J., & Uhlmann, J. K.** (2004). "Unscented filtering and nonlinear estimation." *Proceedings of the IEEE*, 92(3), 401-422.

### Сравнительные Исследования

3. **Wan, E. A., & Van Der Merwe, R.** (2000). "The unscented Kalman filter for nonlinear estimation." *Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium*.

4. **Gustafsson, F., et al.** (2002). "Particle filters for positioning, navigation, and tracking." *IEEE Transactions on signal processing*, 50(2), 425-437.

### Современные Методы Отслеживания

5. **Zhang, Y., et al.** (2021). "ByteTrack: Multi-object tracking by associating every detection box." *arXiv preprint arXiv:2110.06864*.

6. **Cao, J., et al.** (2022). "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking." *arXiv preprint arXiv:2203.14360*.

7. **Du, Y., et al.** (2023). "StrongSORT: Make DeepSORT Great Again." *IEEE Transactions on Multimedia*.

8. **Zeng, F., et al.** (2022). "MOTR: End-to-End Multiple-Object Tracking with Transformer." *ECCV*.

9. **Ye, B., et al.** (2022). "OSTrack: One-Stream Transformer for Visual Object Tracking." *CVPR*.

10. **Cui, Y., et al.** (2022). "MixFormer: End-to-End Tracking with Iterative Mixed Attention." *CVPR*.

### Deep Learning и Tracking

11. **Wojke, N., Bewley, A., & Paulus, D.** (2017). "Simple online and realtime tracking with a deep association metric." *ICIP*.

12. **Bewley, A., et al.** (2016). "Simple online and realtime tracking." *ICIP*.

### Робототехника и SLAM

13. **Thrun, S., et al.** (2005). "Probabilistic robotics." *MIT press*.

14. **Durrant-Whyte, H., & Bailey, T.** (2006). "Simultaneous localization and mapping: part I." *IEEE robotics & automation magazine*, 13(2), 99-110.

### Связанные Темы

- **[Gaussian Distribution](./gaussian-distribution.md)**: Математические основы для Kalman фильтров
- **[Variational Autoencoders](./variational-autoencoders-vaes.md)**: Вероятностные модели для генерации
- **[Diffusion Models](./diffusion-models.md)**: Стохастические процессы в генеративных моделях

---

## Заключение

**Unscented Kalman Filter (UKF)** представляет собой мощный и точный метод для нелинейной фильтрации, который превосходит Extended Kalman Filter по точности и устойчивости, оставаясь при этом вычислительно эффективным. UKF особенно полезен для задач отслеживания объектов, где требуется баланс между точностью и скоростью.

**Современные методы отслеживания** все чаще используют deep learning подходы, но классические методы фильтрации, такие как UKF, остаются важными компонентами гибридных систем, особенно для приложений реального времени и ресурсоограниченных систем.

**Рекомендации:**
- Используйте **UKF** вместо EKF для нелинейных систем
- Для production систем рассмотрите **ByteTrack** или **OC-SORT**
- Для research изучите **Transformer-based методы** (MOTR, TransTrack)
- Для single object tracking используйте **OSTrack** или **MixFormer**
