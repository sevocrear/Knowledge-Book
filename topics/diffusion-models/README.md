---
title: Diffusion Models (диффузионные модели)
description: "Прямой и обратный процесс диффузии, DDPM/DDIM, latent diffusion (Stable Diffusion), Consistency Models, Flow Matching и DiT."
tags:
  - kb/topic
  - domain/generative
  - concept/diffusion
  - concept/score-matching
  - concept/latent-diffusion
aliases:
  - diffusion models
  - DDPM
  - DDIM
  - Stable Diffusion
  - Flow Matching
  - DiT
related:
  - variational-autoencoders-vaes
  - generative-adversarial-networks-gans
  - gaussian-distribution
status: canonical
lang: ru
type: topic
slug: diffusion-models
updated: 2026-09-18
---
# Diffusion Models (диффузионные модели)

## Оглавление

1. [Как объяснить 5-летнему ребёнку](#как-объяснить-5-летнему-ребёнку)
2. [Введение в диффузионные модели](#введение-в-диффузионные-модели)
3. [Основная идея и интуиция](#основная-идея-и-интуиция)
4. [Математические основы](#математические-основы)
5. [Прямой и обратный процессы диффузии](#прямой-и-обратный-процессы-диффузии)
6. [Процесс обучения](#процесс-обучения)
7. [Сэмплирование и генерация](#сэмплирование-и-генерация)
8. [Пример реализации](#пример-реализации)
9. [Ключевые варианты и расширения](#ключевые-варианты-и-расширения)
10. [Применения](#применения)
11. [Текущее состояние (2023-2026)](#текущее-состояние-2023-2026)
12. [Сравнение с другими генеративными моделями](#сравнение-с-другими-генеративными-моделями)
13. [Источники](#источники)

---

## Как объяснить 5-летнему ребёнку

Возьми красивую картинку и по чуть-чуть засыпай её песком, пока не останется только шум. Потом учи робота убирать песок шаг за шагом. Когда он научится — можно начать с кучи песка и медленно «вычищать» её, пока не проявится новая картинка. Так работают diffusion models (как Stable Diffusion).

---

## Введение в диффузионные модели

**Diffusion Models** (также **Denoising Diffusion Probabilistic Models, DDPM**) — класс генеративных моделей, которые дали выдающиеся результаты в генерации изображений, текста, аудио и других типов данных. Впервые их представили Sohl-Dickstein et al. (2015), а популяризировали Ho et al. (2020). С тех пор diffusion models стали основой многих современных систем генерации, включая DALL-E 2, Stable Diffusion, Midjourney и Imagen.

### Ключевые свойства

- **Вероятностный каркас (probabilistic framework)**: опираются на теорию стохастических процессов
- **Высокое качество генерации**: дают детализированные изображения высокого качества
- **Стабильное обучение**: обучение обычно стабильнее, чем у GAN
- **Гибкое conditioning**: легко адаптируются к условной генерации (текст, классы, изображения)
- **Теоретическая база**: имеют прочную основу в теории вероятностей

### Исторический контекст

Diffusion models берут начало в физике (процессы диффузии) и были адаптированы для машинного обучения. Ключевые вехи:

- **2015**: Sohl-Dickstein et al. вводят концепцию diffusion models
- **2020**: Ho et al. представляют DDPM с упрощённой формулировкой
- **2021**: Nichol & Dhariwal улучшают DDPM (DDIM, classifier guidance)
- **2022**: Rombach et al. представляют Latent Diffusion Models (Stable Diffusion)
- **2023-2024**: быстрый прогресс в text-to-image, генерации видео и 3D

---

## Основная идея и интуиция

### Базовая идея

Diffusion models работают по принципу **постепенного добавления и удаления шума**:

1. **Прямой процесс (forward diffusion)**: постепенно добавляем шум к данным, пока они не превратятся в чистый шум
2. **Обратный процесс (reverse diffusion)**: обучаем нейросеть предсказывать, как удалить шум, чтобы восстановить исходные данные

### Наглядная аналогия

Представьте процесс создания картины в обратном порядке:

- **Прямой процесс**: начинаем с чёткой картины и постепенно размазываем краски, добавляя случайные мазки, пока не получим полностью случайный набор цветов
- **Обратный процесс**: обучаем художника (нейросеть) восстанавливать картину, глядя на размазанные краски и предсказывая, какие мазки нужно убрать, чтобы вернуться к исходному изображению

### Почему это работает

Ключевая интуиция: **удаление шума проще, чем прямое генерирование**. Вместо того чтобы учиться генерировать сложное изображение с нуля, модель учится выполнять последовательность простых операций удаления шума.

### Визуализация процесса диффузии

```
Исходное изображение → [Добавить шум] → [Добавить шум] → ... → [Добавить шум] → Чистый шум
     x₀                     x₁               x₂                       xₜ

Чистый шум → [Убрать шум] → [Убрать шум] → ... → [Убрать шум] → Сгенерированное изображение
    xₜ          xₜ₋₁             xₜ₋₂                    x₀
```

---

## Математические основы

### Прямой процесс диффузии

Прямой процесс (forward process) постепенно добавляет гауссовский шум к данным по заранее заданному расписанию (noise schedule).

#### Один шаг

На каждом шаге $t$ мы добавляем шум:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

где:
- $\beta_t$ — расписание шума (noise schedule), обычно $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$
- $\mathbf{x}_0$ — исходные данные
- $\mathbf{x}_t$ — данные на шаге $t$

#### Замкнутая форма

Благодаря свойствам гауссовских распределений мы можем напрямую получить $\mathbf{x}_t$ из $\mathbf{x}_0$:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

где:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

Это означает:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$

где $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

#### Расписание шума (noise schedule)

Типичные расписания:
- **Linear**: $\beta_t = \text{linear}(0.0001, 0.02, T)$
- **Cosine**: $\bar{\alpha}_t = \frac{\cos(\pi t / 2T + s)}{1+s}$, где $s$ — небольшой offset

### Обратный процесс диффузии

Обратный процесс (reverse process) пытается инвертировать прямой процесс, удаляя шум:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

где $\boldsymbol{\mu}_\theta$ и $\boldsymbol{\Sigma}_\theta$ — параметры, предсказанные нейросетью.

### Целевая функция обучения

#### Упрощённый loss (DDPM)

Ho et al. показали, что можно использовать упрощённую функцию потерь:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)||^2 \right]$$

где:
- $t \sim \text{Uniform}(1, T)$ — случайный временной шаг
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ — случайный шум
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ — зашумлённые данные
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ — предсказание шума нейросетью

#### Интуиция за функцией потерь

Модель учится предсказывать шум $\boldsymbol{\epsilon}$, который был добавлен к $\mathbf{x}_0$ для получения $\mathbf{x}_t$. Зная предсказанный шум, мы можем восстановить $\mathbf{x}_0$:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

---

## Прямой и обратный процессы диффузии

### Прямой процесс: добавление шума

Прямой процесс — это марковская цепь, которая постепенно разрушает структуру данных:

```python
def forward_diffusion(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Прямая диффузия: добавляет шум к данным
    
    Args:
        x0: исходные данные [B, C, H, W]
        t: временной шаг [B]
        sqrt_alphas_cumprod: sqrt(alpha_bar_t) [T]
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t) [T]
    
    Returns:
        xt: зашумлённые данные
        noise: добавленный шум
    """
    # Извлекаем коэффициенты для батча
    sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    
    # Генерируем случайный шум
    noise = torch.randn_like(x0)
    
    # Добавляем шум
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    return xt, noise
```

### Обратный процесс: удаление шума

Обратный процесс использует обученную модель, чтобы постепенно убирать шум:

```python
def reverse_diffusion_step(xt, t, model, sqrt_alphas_cumprod, 
                          sqrt_one_minus_alphas_cumprod, 
                          posterior_variance, posterior_mean_coef1, 
                          posterior_mean_coef2):
    """
    Один шаг обратной диффузии (reverse diffusion)
    
    Args:
        xt: данные на шаге t
        t: текущий временной шаг
        model: обученная модель для предсказания шума
        ...: параметры для вычисления mean и variance
    
    Returns:
        x_prev: данные на шаге t-1
    """
    # Предсказываем шум
    predicted_noise = model(xt, t)
    
    # Вычисляем предсказание x0
    sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    
    pred_x0 = (xt - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
    
    # Вычисляем mean для p(x_{t-1} | x_t, x_0)
    posterior_mean = (
        posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * pred_x0 +
        posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * xt
    )
    
    # Вычисляем variance
    posterior_var = posterior_variance[t].reshape(-1, 1, 1, 1)
    
    # Сэмплируем x_{t-1}
    if t[0] == 0:
        return posterior_mean
    else:
        noise = torch.randn_like(xt)
        return posterior_mean + torch.sqrt(posterior_var) * noise
```

---

## Процесс обучения

### Алгоритм обучения

Алгоритм обучения diffusion model:

1. **Взять данные (sample data)**: выбираем случайный батч $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
2. **Выбрать timestep**: выбираем случайный временной шаг $t \sim \text{Uniform}(1, T)$
3. **Добавить шум**: генерируем зашумлённые данные $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$
4. **Предсказать шум**: модель предсказывает шум $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
5. **Посчитать loss**: вычисляем MSE между истинным и предсказанным шумом
6. **Обратное распространение**: обновляем параметры модели

### На чём учится diffusion model?

#### Типы данных для обучения

Diffusion models могут обучаться на разных типах данных:

**1. Изображения (image diffusion)**
- **Датасеты**: 
  - ImageNet (1.2M изображений, 1000 классов)
  - LAION-5B (5.85 миллиардов изображений с текстовыми описаниями)
  - COCO (330K изображений с аннотациями)
  - CelebA (200K лиц)
  - FFHQ (70K высококачественных лиц)
- **Формат**: обычно RGB-изображения, нормализованные в диапазон $[-1, 1]$ или $[0, 1]$
- **Разрешение**: от 64x64 до 1024x1024 и выше

**2. Видео (video diffusion)**
- **Датасеты**:
  - WebVid (10M видео с текстовыми описаниями)
  - Kinetics (400K видео, 400 классов действий)
  - UCF-101, HMDB-51 (видео с действиями)
  - InternVid (236M пар видео–текст)
- **Формат**: последовательность кадров (frames), обычно 16–128 кадров
- **Разрешение**: от 128x128 до 1024x1024 на кадр

**3. Текст (text diffusion)**
- **Датасеты**: 
  - Common Crawl
  - Wikipedia
  - книги, статьи
- **Формат**: токенизированный текст

**4. Аудио (audio diffusion)**
- **Датасеты**:
  - AudioSet (2M аудиоклипов)
  - LibriSpeech (1000 часов речи)
- **Формат**: спектрограммы или raw-аудио

#### Процесс обучения: детальный разбор

**Шаг 1: подготовка данных**

```python
# Пример для изображений
def prepare_image_data(image_path):
    """
    Подготовка изображения для обучения
    """
    # Загрузка изображения
    image = Image.open(image_path)
    
    # Ресайз до нужного разрешения (например, 256x256)
    image = image.resize((256, 256))
    
    # Преобразование в тензор
    image_tensor = transforms.ToTensor()(image)  # [0, 1]
    
    # Нормализация в [-1, 1]
    image_tensor = image_tensor * 2.0 - 1.0
    
    return image_tensor  # Shape: [3, 256, 256]
```

**Шаг 2: выбор случайного временного шага**

Модель учится на **всех временных шагах одновременно**:

```python
# Для каждого батча выбираем случайные временные шаги
t = torch.randint(0, timesteps, (batch_size,))  # [0, T-1]
```

Это позволяет модели:
- быстро обучаться (не нужно проходить все шаги последовательно)
- изучать разные уровни шума одновременно
- эффективно использовать данные

**Шаг 3: добавление шума**

Для каждого изображения в батче:
- выбираем случайный временной шаг $t$
- генерируем случайный шум $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- вычисляем зашумлённое изображение: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$

```python
def add_noise(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Добавляет шум к изображению согласно прямому процессу (forward process)
    """
    # Извлекаем коэффициенты для каждого элемента батча
    sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    
    # Генерируем случайный гауссов шум
    noise = torch.randn_like(x0)
    
    # Добавляем шум
    x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    return x_t, noise
```

**Шаг 4: предсказание шума**

Модель (обычно U-Net) получает:
- **Вход**: зашумлённое изображение $\mathbf{x}_t$ (shape: [B, C, H, W])
- **Условие**: временной шаг $t$ (shape: [B])
- **Выход**: предсказанный шум $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ (shape: [B, C, H, W])

```python
# Forward pass через модель
predicted_noise = model(x_t, t)  # [B, C, H, W]
```

**Шаг 5: вычисление потерь**

Функция потерь — это **MSE между истинным и предсказанным шумом**:

$$\mathcal{L} = ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)||^2$$

```python
# Вычисление потерь
loss = F.mse_loss(noise, predicted_noise)
```

**Почему именно предсказание шума?**

1. **Проще для модели**: предсказать шум проще, чем предсказать исходное изображение напрямую
2. **Стабильность**: это приводит к более стабильному обучению
3. **Математическая обоснованность**: связано со score matching и оптимальным транспортом

#### Что изучает модель?

Модель учится **обратному процессу диффузии**:

- **На ранних шагах** (большой $t$, много шума): модель учится распознавать общую структуру и композицию
- **На средних шагах**: модель учится восстанавливать детали и формы
- **На поздних шагах** (малый $t$, мало шума): модель учится финальным деталям и текстурам

**Аналогия**: как художник, который:
- сначала намечает общую композицию (ранние шаги)
- затем добавляет основные формы (средние шаги)
- в конце прорабатывает детали (поздние шаги)

#### Условное обучение

Модель может обучаться с **условиями** (conditioning):

**1. Class-conditional**: генерация определённого класса
```python
predicted_noise = model(x_t, t, class_label)
```

**2. Text-conditional**: генерация по текстовому описанию
```python
# Текст кодируется через CLIP или T5
text_embedding = text_encoder(prompt)
predicted_noise = model(x_t, t, text_embedding)
```

**3. Image-conditional**: генерация на основе другого изображения
```python
predicted_noise = model(x_t, t, condition_image)
```

#### Объём данных

Типичные объёмы данных для обучения:
- **Базовые модели**: 1–10 миллионов изображений
- **Крупные модели** (Stable Diffusion): 100+ миллионов изображений
- **Очень крупные** (DALL-E 2, Imagen): 1+ миллиард изображений

**Время обучения**:
- небольшие модели (64x64): несколько дней на 1–4 GPU
- средние модели (256x256): недели на 8–16 GPU
- крупные модели (1024x1024): месяцы на десятках/сотнях GPU

### Ключевые решения в дизайне

#### Архитектура сети

Типичная архитектура — **U-Net** с временными embeddings:

- **Структура encoder–decoder**: для обработки изображений
- **Time embeddings**: sinusoidal или learned embeddings для временного шага $t$
- **Слои attention**: self-attention для глобального контекста
- **Residual connections**: для стабильного обучения

#### Time embedding

Временной шаг $t$ кодируется с помощью sinusoidal embeddings:

```python
def get_timestep_embedding(timesteps, dim):
    """
    Sinusoidal positional embeddings для временных шагов
    """
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

---

## Сэмплирование и генерация

### Алгоритм сэмплирования

Процесс генерации (sampling) — это обратный diffusion-процесс:

1. **Начать с шума**: начинаем с чистого шума $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. **Итеративный denoising**: для $t = T, T-1, ..., 1$:
   - предсказываем шум: $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
   - вычисляем $\mathbf{x}_{t-1}$ по предсказанному шуму
3. **Итоговый сэмпл**: $\mathbf{x}_0$ — сгенерированное изображение

### Сэмплирование DDPM

```python
def sample_ddpm(model, shape, device, timesteps=1000, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                posterior_variance, posterior_mean_coef1, 
                posterior_mean_coef2):
    """
    Генерация сэмплов с помощью DDPM
    """
    # Начинаем с чистого шума
    x = torch.randn(shape, device=device)
    
    # Итеративно удаляем шум
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Предсказываем шум
        predicted_noise = model(x, t_tensor)
        
        # Вычисляем x_{t-1}
        x = reverse_diffusion_step(
            x, t_tensor, model, sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod, posterior_variance,
            posterior_mean_coef1, posterior_mean_coef2
        )
    
    return x
```

### Сэмплирование DDIM (детерминированное)

DDIM (Denoising Diffusion Implicit Models) даёт детерминированную генерацию и более быстрый sampling:

```python
def sample_ddim(model, shape, device, timesteps=50, eta=0.0):
    """
    DDIM sampling — быстрее и детерминированно (если eta=0)
    
    Args:
        eta: параметр стохастичности (0 = детерминированный, 1 = стохастический)
    """
    x = torch.randn(shape, device=device)
    
    # Используем подпоследовательность временных шагов
    step_size = timesteps // 50  # 50 шагов вместо 1000
    
    for i in reversed(range(0, timesteps, step_size)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        predicted_noise = model(x, t)
        
        # Правило обновления DDIM
        alpha_bar_t = alphas_cumprod[i]
        alpha_bar_t_prev = alphas_cumprod[max(0, i - step_size)]
        
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod[i] * predicted_noise) / sqrt_alphas_cumprod[i]
        
        direction_point = sqrt_one_minus_alphas_cumprod[max(0, i - step_size)] * predicted_noise
        
        if eta > 0:
            noise = eta * torch.randn_like(x) * sqrt_one_minus_alphas_cumprod[max(0, i - step_size)]
        else:
            noise = 0
        
        x = sqrt_alphas_cumprod[max(0, i - step_size)] * pred_x0 + direction_point + noise
    
    return x
```

---

## Пример реализации

### Полная реализация DDPM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings для временных шагов"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class Block(nn.Module):
    """Базовый блок для U-Net"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Первая свёртка
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Расширяем последние 2 размерности
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Добавляем канал времени
        h = h + time_emb
        # Вторая свёртка
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Downsample или upsample
        return self.transform(h)


class SimpleUNet(nn.Module):
    """Упрощённая архитектура U-Net для diffusion model"""
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Начальная проекция
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Эмбеддинг времени
        t = self.time_mlp(timestep)
        # Начальная свёртка
        x = self.conv0(x)
        # U-Net
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Добавляем residual x как дополнительные каналы
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


class DiffusionModel:
    """Класс для обучения и генерации с помощью DDPM"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Вычисляем расписание шума
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """Добавляет шум к изображениям"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """Выбирает случайные временные шаги"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """Генерирует новые изображения"""
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((beta) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(model, dataloader, optimizer, device, epochs=100):
    """Функция обучения"""
    mse = nn.MSELoss()
    diffusion = DiffusionModel(device=device)
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}:")
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")


# Пример использования
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Загрузка данных
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # нормализация в [-1, 1]
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Инициализация модели
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Обучение
    train(model, dataloader, optimizer, device, epochs=100)
    
    # Генерация
    diffusion = DiffusionModel(device=device)
    generated_images = diffusion.sample(model, n=8)
```

---

## Ключевые варианты и расширения

### 1. DDIM (Denoising Diffusion Implicit Models)

**Ключевые особенности:**
- детерминированная генерация (при $\eta = 0$)
- более быстрый sampling (меньше шагов)
- обратимость процесса

**Формула обновления:**

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

### 2. Latent Diffusion Models (Stable Diffusion)

**Идея:** работают в латентном пространстве VAE вместо пиксельного пространства.

**Преимущества:**
- быстрее (меньше размерность)
- меньше памяти
- высокое качество

**Архитектура:**
1. VAE encoder: изображение → латентное представление
2. Diffusion в латентном пространстве
3. VAE decoder: латентное представление → изображение

### 3. Classifier Guidance

**Идея:** использовать предобученный классификатор, чтобы улучшить генерацию.

**Score function:**

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + s \cdot \nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t)$$

где $s$ — guidance scale.

### 4. Classifier-Free Guidance

**Идея:** обучать условную и безусловную модели одновременно, без классификатора.

**Предсказание:**

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, y) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset) + s \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset))$$

где $s$ — guidance scale, $\emptyset$ — пустое условие.

### 5. Score-Based Generative Models (SGM)

**Альтернативная формулировка** через score matching:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_t} \left[ \lambda(t) ||\mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)||^2 \right]$$

где $\mathbf{s}_\theta$ — score network.

### 6. Progressive Distillation

**Идея:** обучать быстрые модели через дистилляцию медленных.

**Результат:** генерация за 4–8 шагов вместо 1000.

### 7. Rectified Flow / Flow Matching

**Новый подход (2023-2024):** прямой путь от шума к данным.

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t)$$

где $\mathbf{v}_\theta$ — velocity field.

---

## Применения

### 1. Генерация изображений

**Text-to-image:**
- DALL-E 2 (OpenAI)
- Stable Diffusion (Stability AI)
- Midjourney
- Imagen (Google)

**Image-to-image:**
- inpainting (заполнение пропусков)
- super-resolution
- style transfer
- colorization (раскрашивание)

### 2. Генерация видео

**Text-to-video:**
- Runway Gen-2
- Pika Labs
- Stable Video Diffusion
- Sora (OpenAI, 2024)

**Редактирование видео:**
- inpainting в видео
- video-to-video translation

#### Как diffusion models генерируют видео?

Генерация видео с помощью diffusion models — это расширение image generation на временную размерность. Основная идея: **модель учится генерировать последовательность кадров, сохраняя временную согласованность**.

##### Архитектурные подходы

**1. Генерация кадр за кадром (frame-by-frame, базовый вариант)**

Самый простой подход — генерировать каждый кадр независимо:

```python
def generate_video_frames(model, text_prompt, num_frames=16):
    """
    Генерация видео кадр за кадром
    """
    frames = []
    for i in range(num_frames):
        # Генерируем каждый кадр независимо
        frame = sample_ddpm(model, text_prompt)
        frames.append(frame)
    
    return frames  # Проблема: нет временной согласованности
```

**Проблема**: кадры не связаны между собой, получается «дрожащее» видео.

**2. Temporal conditioning (временное условие)**

Добавляем информацию о временной позиции кадра:

```python
def generate_video_with_temporal(model, text_prompt, num_frames=16):
    """
    Генерация с учётом временной позиции
    """
    frames = []
    for frame_idx in range(num_frames):
        # Добавляем временное условие
        temporal_embedding = get_temporal_embedding(frame_idx, num_frames)
        
        # Генерируем кадр с учётом времени
        frame = sample_ddpm(model, text_prompt, temporal_embedding)
        frames.append(frame)
    
    return frames
```

**3. 3D-свёртки / spatio-temporal attention**

Используем 3D-свёртки или spatio-temporal attention, чтобы обрабатывать видео как единый объём:

```python
class VideoDiffusionModel(nn.Module):
    """
    Модель для генерации видео с 3D-свёртками
    """
    def __init__(self):
        super().__init__()
        # 3D-свёртки для обработки пространства-времени
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        # ... остальные слои
        
    def forward(self, video_noise, t, text_embedding):
        """
        video_noise: [B, C, T, H, W] — зашумлённое видео
        t: временной шаг диффузии
        text_embedding: текстовое условие
        """
        # Обработка видео как 3D-объёма
        x = self.conv3d_1(video_noise)
        x = self.conv3d_2(x)
        # ...
        return predicted_noise
```

**4. Latent video diffusion (Stable Video Diffusion)**

Работа в латентном пространстве VAE:

```python
# 1. Кодируем видео в латентное пространство
video_latents = vae_encoder(video_frames)  # [B, C, T, H', W']

# 2. Diffusion в латентном пространстве
denoised_latents = diffusion_model(video_latents, text_prompt)

# 3. Декодируем обратно в пиксели
generated_frames = vae_decoder(denoised_latents)  # [B, C, T, H, W]
```

##### Процесс обучения для видео

**Прямой процесс для видео:**

Аналогично изображениям, но применяем к каждому кадру:

$$q(\mathbf{v}_t | \mathbf{v}_{t-1}) = \prod_{i=1}^{F} \mathcal{N}(\mathbf{v}_{t,i}; \sqrt{1-\beta_t}\mathbf{v}_{t-1,i}, \beta_t \mathbf{I})$$

где $\mathbf{v}_t = [\mathbf{x}_{t,1}, \mathbf{x}_{t,2}, ..., \mathbf{x}_{t,F}]$ — видео с $F$ кадрами.

**Ключевое отличие**: нужно сохранять **временную согласованность** между кадрами.

##### Техники обеспечения временной согласованности

**1. Temporal attention**

Механизм внимания между кадрами:

```python
class TemporalAttention(nn.Module):
    """
    Attention между кадрами для сохранения согласованности
    """
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        
    def forward(self, frames):
        """
        frames: [B, T, C, H, W]
        """
        B, T, C, H, W = frames.shape
        
        # Reshape для attention: [B*H*W, T, C]
        frames_flat = frames.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, C)
        
        # Self-attention между кадрами
        attended, _ = self.attention(frames_flat, frames_flat, frames_flat)
        
        # Reshape обратно
        attended = attended.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        
        return attended
```

**2. Conditioning по optical flow**

Использование optical flow для обеспечения плавности:

```python
def compute_optical_flow(frame1, frame2):
    """
    Вычисляет optical flow между кадрами
    """
    # Используем метод типа Lucas-Kanade или deep learning
    flow = optical_flow_model(frame1, frame2)
    return flow

# При генерации используем flow для предсказания следующего кадра
```

**3. Интерполяция кадров (frame interpolation)**

Генерация промежуточных кадров для плавности:

```python
def interpolate_frames(frame1, frame2, num_intermediate=2):
    """
    Генерирует промежуточные кадры между двумя кадрами
    """
    # Используем diffusion model для генерации промежуточных кадров
    intermediate_frames = []
    for alpha in np.linspace(0, 1, num_intermediate + 2)[1:-1]:
        # Условная генерация с интерполяцией
        frame = conditional_sample(model, frame1, frame2, alpha)
        intermediate_frames.append(frame)
    
    return [frame1] + intermediate_frames + [frame2]
```

##### Современные модели (2024)

**1. Sora (OpenAI, 2024)**

Ключевые особенности:
- **Diffusion Transformer (DiT)**: использует Transformer вместо U-Net
- **Spacetime patches**: разбивает видео на пространственно-временные патчи
- **Scaling**: масштабируется до очень больших моделей
- **Long videos**: может генерировать видео до 60 секунд
- **Физика**: понимает физические законы (гравитация, отражения)

**Архитектура Sora:**

```python
class SoraModel(nn.Module):
    """
    Упрощённая версия архитектуры Sora
    """
    def __init__(self):
        super().__init__()
        # VAE для работы в латентном пространстве
        self.vae_encoder = VideoVAEEncoder()
        self.vae_decoder = VideoVAEDecoder()
        
        # Diffusion Transformer
        self.dit = DiffusionTransformer(
            input_size=(16, 256, 256),  # T, H, W в латентном пространстве
            patch_size=(1, 2, 2),  # Spacetime patches
            in_channels=4,
            hidden_size=1152,
            depth=24,
            num_heads=16
        )
        
    def forward(self, video_latents, t, text_embedding):
        """
        video_latents: [B, C, T, H, W] в латентном пространстве
        """
        # Разбиваем на патчи
        patches = self.patchify(video_latents)
        
        # Добавляем позиционные embeddings (пространственные + временные)
        patches = patches + self.spatial_pos_emb + self.temporal_pos_emb
        
        # Diffusion Transformer
        denoised_patches = self.dit(patches, t, text_embedding)
        
        # Собираем обратно
        video_latents = self.unpatchify(denoised_patches)
        
        return video_latents
```

**2. Stable Video Diffusion (Stability AI, 2024)**

- основан на Stable Diffusion
- генерирует короткие видео (обычно 4–25 кадров)
- открытая модель
- хорошее качество для коротких клипов

**3. Runway Gen-2**

- коммерческая модель
- хорошее качество генерации
- поддержка разных условий (текст, изображение)

##### Процесс генерации видео

**Полный pipeline:**

```python
def generate_video_from_text(model, text_prompt, num_frames=16, 
                            resolution=(256, 256)):
    """
    Генерация видео из текстового промпта
    """
    # 1. Кодируем текст
    text_embedding = text_encoder(text_prompt)  # [B, text_dim]
    
    # 2. Начинаем с шума в латентном пространстве
    # Shape: [B, C, T, H', W'] где T=num_frames
    video_latents = torch.randn(
        (1, 4, num_frames, resolution[0]//8, resolution[1]//8)
    )
    
    # 3. Diffusion-процесс (обратный)
    for t in reversed(range(timesteps)):
        # Предсказываем шум
        predicted_noise = model(
            video_latents, 
            torch.tensor([t]), 
            text_embedding
        )
        
        # Обновляем латентное представление
        video_latents = denoise_step(
            video_latents, 
            predicted_noise, 
            t
        )
    
    # 4. Декодируем в пиксели
    video_frames = vae_decoder(video_latents)  # [B, 3, T, H, W]
    
    # 5. Постобработка (нормализация, интерполяция)
    video_frames = postprocess_video(video_frames)
    
    return video_frames
```

##### Вызовы генерации видео

1. **Временная согласованность**: кадры должны плавно переходить друг в друга
2. **Длинные видео**: сложно генерировать длинные последовательности
3. **Вычислительная сложность**: видео требует намного больше памяти и вычислений
4. **Физическая реалистичность**: движения должны подчиняться физическим законам
5. **Текстура и детали**: сохранение деталей во времени

##### Будущие направления

- **Более длинные видео**: генерация минутных и часовых роликов
- **Лучшая физика**: более реалистичное моделирование физики
- **Контроль движения**: точный контроль над движениями объектов
- **Мультимодальность**: генерация видео с синхронизацией аудио

### 3. Генерация 3D

**Text-to-3D:**
- DreamFusion
- Magic3D
- Point-E

**Image-to-3D:**
- Zero-1-to-3

### 4. Генерация аудио

**Text-to-speech:**
- AudioLM (Google)
- MusicLM

**Редактирование аудио:**
- audio inpainting
- style transfer для аудио

### 5. Медицинская визуализация

- генерация медицинских изображений
- data augmentation
- anomaly detection

### 6. Научные применения

- генерация молекулярных структур
- protein folding
- material design

---

## Текущее состояние (2023-2026)

### Модели state-of-the-art (2024-2025)

#### Генерация изображений

1. **Stable Diffusion 3 (2024)**
   - улучшенная архитектура (MMDiT)
   - лучшее понимание текста
   - более детализированная генерация

2. **DALL-E 3 (2023)**
   - интеграция с GPT-4
   - улучшенное следование промптам
   - более безопасная генерация

3. **Midjourney v6 (2024)**
   - фотореалистичная генерация
   - улучшенная композиция

#### Генерация видео

1. **Sora (OpenAI, 2024)**
   - генерация видео до 60 секунд
   - понимание физики и пространства
   - мультимодальные условия

2. **Stable Video Diffusion (2024)**
   - открытая модель для video generation
   - хорошее качество и контроль

#### Генерация 3D

1. **3D Gaussian Splatting + Diffusion**
   - быстрая генерация 3D-сцен
   - высокое качество рендеринга

2. **Triplane Diffusion**
   - эффективное представление 3D

### Недавние достижения

#### 1. Consistency Models (2023)

**Идея:** прямое отображение шума в данные за один шаг.

$$\mathbf{x}_0 = f_\theta(\mathbf{x}_t, t)$$

**Преимущества:**
- очень быстрая генерация
- детерминированная
- можно использовать как few-step diffusion

#### 2. Latent Consistency Models (LCM, 2024)

- работают в латентном пространстве
- генерация за 4 шага
- используются в Stable Diffusion

#### 3. Flow Matching (2023-2024)

**Rectified Flow / Flow Matching:**
- прямой путь от шума к данным
- более эффективное обучение
- быстрая генерация

#### 4. Diffusion Transformers (DiT, 2023)

**Идея:** заменить U-Net на архитектуру Transformer.

**Преимущества:**
- масштабируемость
- лучшее качество при больших моделях
- используется в Sora

#### 5. Мультимодальная диффузия

- **Text + Image**: text-to-image, image-to-text
- **Audio + Text**: генерация аудио
- **Video + Text**: генерация видео
- **3D + Text**: генерация 3D

### Улучшения качества и скорости

**Скорость:**
- 2020: 1000 шагов (медленно)
- 2022: 50 шагов (DDIM)
- 2023: 4–8 шагов (LCM, Progressive Distillation)
- 2024: 1 шаг (Consistency Models)

**Качество:**
- постоянное улучшение FID, IS scores
- лучшее понимание текста
- более детализированная генерация

### Открытые проблемы

1. **Компромисс скорость vs качество**: быстрая генерация часто жертвует качеством
2. **Контроль**: точный контроль над генерацией всё ещё сложен
3. **Согласованность (consistency)**: трудно удерживать согласованность в длинных последовательностях
4. **Память**: большие модели требуют много памяти
5. **Bias и безопасность**: проблемы со смещениями и безопасной генерацией

---

## Сравнение с другими генеративными моделями

### Diffusion Models vs GAN

| Аспект | Diffusion Models | GAN |
|--------|------------------|-----|
| **Стабильность обучения** | стабильное обучение | может быть нестабильным |
| **Mode collapse** | нет этой проблемы | может страдать от mode collapse |
| **Качество сэмплов** | очень высокое | высокое (но возможны артефакты) |
| **Разнообразие** | высокое | зависит от архитектуры |
| **Скорость sampling** | медленное (но улучшается) | быстрое |
| **Likelihood** | можно оценить (через ELBO) | нет явного likelihood |
| **Conditioning** | легко добавляется | нужны специальные техники |

### Diffusion Models vs VAE

| Аспект | Diffusion Models | VAE |
|--------|------------------|-----|
| **Качество сэмплов** | очень высокое | часто размытое |
| **Латентное пространство** | нет явного latent space | структурированный latent space |
| **Интерполяция** | сложнее | легко в latent space |
| **Обучение** | стабильное | может быть нестабильным |
| **Likelihood** | можно оценить | явный ELBO |
| **Скорость** | медленное | быстрое |

### Diffusion Models vs авторегрессионные модели

| Аспект | Diffusion Models | Авторегрессионные (PixelCNN и др.) |
|--------|------------------|-----------------------------------|
| **Параллельная генерация** | можно генерировать параллельно | последовательная |
| **Длинные зависимости** | хорошо | ограничено |
| **Качество сэмплов** | очень высокое | хорошее |
| **Скорость** | медленное | медленное (последовательное) |

### Когда использовать diffusion models

**Используйте diffusion models, когда:**
- нужно очень высокое качество генерации
- важна стабильность обучения
- нужна условная генерация (текст, классы)
- можно позволить медленную генерацию (или использовать быстрые варианты)

**Рассмотрите альтернативы, когда:**
- нужна очень быстрая генерация (GAN, VAE)
- нужен структурированный latent space (VAE)
- ограничены ресурсы (VAE, небольшие GAN)

---

## Источники

### Связанные документы

- **[Gaussian Distribution (Normal Distribution)](../gaussian-distribution/README.md)**: фундаментальное распределение, используемое для добавления и удаления шума в diffusion models
- **[Variational Autoencoders (VAEs)](../variational-autoencoders-vaes/README.md)**: альтернативный подход к генеративному моделированию с явным latent space
- **[Generative Adversarial Networks (GANs)](../generative-adversarial-networks-gans/README.md)**: adversarial-подход к генерации, сравнение с diffusion models

### Ключевые статьи

1. **Sohl-Dickstein et al. (2015)**: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" — первая работа по diffusion models

2. **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" — популяризация и упрощение формулировки

3. **Song et al. (2021)**: "Denoising Diffusion Implicit Models" — DDIM, детерминированная генерация

4. **Rombach et al. (2022)**: "High-Resolution Image Synthesis with Latent Diffusion Models" — Stable Diffusion

5. **Ho & Salimans (2022)**: "Classifier-Free Diffusion Guidance" — classifier-free guidance

6. **Song et al. (2023)**: "Consistency Models" — одношаговая генерация

7. **Song et al. (2023)**: "Consistency Trajectory Models" — улучшенные consistency models

8. **Lipman et al. (2023)**: "Flow Matching for Generative Modeling" — подход flow matching

9. **Peebles & Xie (2023)**: "Scalable Diffusion Models with Transformers" — архитектура DiT

10. **Luo et al. (2023)**: "Latent Consistency Models" — LCM для быстрой генерации

### Недавние статьи (2024-2025)

1. **OpenAI (2024)**: "Sora: Creating Video from Text" — модель генерации видео

2. **Stability AI (2024)**: "Stable Diffusion 3" — улучшенная версия Stable Diffusion

3. **Google (2024)**: "Imagen 3" — улучшенная text-to-image модель

### Ресурсы

- **Hugging Face Diffusers**: библиотека для работы с diffusion models
- **Stable Diffusion WebUI**: пользовательский интерфейс для Stable Diffusion
- **Papers with Code**: актуальные результаты и реализации

### Математический фон

- **Stochastic Processes**: теория марковских процессов
- **Variational Inference**: ELBO и вариационные методы
- **Score Matching**: альтернативная формулировка через score functions
- **Optimal Transport**: связь с теорией оптимального транспорта

---

*Документ создан: 2025*
*Последнее обновление: 2025*
