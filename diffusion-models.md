# Diffusion Models: A Comprehensive Guide

## Table of Contents

1. [Introduction to Diffusion Models](#introduction-to-diffusion-models)
2. [Core Idea and Intuition](#core-idea-and-intuition)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Forward and Reverse Diffusion Processes](#forward-and-reverse-diffusion-processes)
5. [Training Process](#training-process)
6. [Sampling and Generation](#sampling-and-generation)
7. [Implementation Example](#implementation-example)
8. [Key Variants and Extensions](#key-variants-and-extensions)
9. [Applications](#applications)
10. [Current Status (2023-2026)](#current-status-2023-2026)
11. [Comparison with Other Generative Models](#comparison-with-other-generative-models)
12. [References](#references)

---

## Introduction to Diffusion Models

**Diffusion Models** (также известные как **Denoising Diffusion Probabilistic Models, DDPM**) представляют собой класс генеративных моделей, которые достигли выдающихся результатов в генерации изображений, текста, аудио и других типов данных. Впервые представленные в работе Sohl-Dickstein et al. (2015) и популяризированные Ho et al. (2020), diffusion models стали основой для многих современных систем генерации, включая DALL-E 2, Stable Diffusion, Midjourney и Imagen.

### Key Characteristics

- **Probabilistic Framework**: Основаны на теории стохастических процессов
- **High-Quality Generation**: Производят высококачественные, детализированные изображения
- **Stable Training**: Более стабильное обучение по сравнению с GANs
- **Flexible Conditioning**: Легко адаптируются для условной генерации (текст, классы, изображения)
- **Theoretical Foundation**: Имеют прочную теоретическую основу в теории вероятностей

### Historical Context

Diffusion models берут начало в физике (процессы диффузии) и были адаптированы для машинного обучения. Ключевые вехи:

- **2015**: Sohl-Dickstein et al. вводят концепцию diffusion models
- **2020**: Ho et al. представляют DDPM с упрощенной формулировкой
- **2021**: Nichol & Dhariwal улучшают DDPM (DDIM, classifier guidance)
- **2022**: Rombach et al. представляют Latent Diffusion Models (Stable Diffusion)
- **2023-2024**: Rapid progress в text-to-image, video generation, 3D generation

---

## Core Idea and Intuition

### The Fundamental Concept

Diffusion models работают по принципу **постепенного добавления и удаления шума**:

1. **Forward Process (Forward Diffusion)**: Постепенно добавляем шум к данным, пока они не превратятся в чистый шум
2. **Reverse Process (Reverse Diffusion)**: Обучаем нейросеть предсказывать, как удалить шум, чтобы восстановить исходные данные

### Intuitive Analogy

Представьте процесс создания картины в обратном порядке:

- **Forward Process**: Начинаем с четкой картины и постепенно размазываем краски, добавляя случайные мазки, пока не получим полностью случайный набор цветов
- **Reverse Process**: Обучаем художника (нейросеть) восстанавливать картину, глядя на размазанные краски и предсказывая, какие мазки нужно убрать, чтобы вернуться к исходному изображению

### Why This Works

Ключевая интуиция: **удаление шума проще, чем прямое генерирование**. Вместо того чтобы учиться генерировать сложное изображение с нуля, модель учится выполнять последовательность простых операций удаления шума.

### The Diffusion Process Visualization

```
Original Image → [Add Noise] → [Add Noise] → ... → [Add Noise] → Pure Noise
     x₀              x₁              x₂                    xₜ

Pure Noise → [Remove Noise] → [Remove Noise] → ... → [Remove Noise] → Generated Image
    xₜ            xₜ₋₁              xₜ₋₂                    x₀
```

---

## Mathematical Foundations

### Forward Diffusion Process

Forward process постепенно добавляет гауссовский шум к данным согласно предопределенному расписанию (noise schedule).

#### Single Step

На каждом шаге $t$ мы добавляем шум:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

где:
- $\beta_t$ - расписание шума (noise schedule), обычно $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$
- $\mathbf{x}_0$ - исходные данные
- $\mathbf{x}_t$ - данные на шаге $t$

#### Closed-Form Solution

Благодаря свойствам гауссовских распределений, мы можем напрямую получить $\mathbf{x}_t$ из $\mathbf{x}_0$:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

где:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

Это означает:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$

где $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

#### Noise Schedule

Типичные расписания:
- **Linear**: $\beta_t = \text{linear}(0.0001, 0.02, T)$
- **Cosine**: $\bar{\alpha}_t = \frac{\cos(\pi t / 2T + s)}{1+s}$, где $s$ - небольшой offset

### Reverse Diffusion Process

Reverse process пытается инвертировать forward process, удаляя шум:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

где $\boldsymbol{\mu}_\theta$ и $\boldsymbol{\Sigma}_\theta$ - параметры, предсказанные нейросетью.

### Training Objective

#### Simplified Loss (DDPM)

Ho et al. показали, что можно использовать упрощенную функцию потерь:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)||^2 \right]$$

где:
- $t \sim \text{Uniform}(1, T)$ - случайный временной шаг
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ - случайный шум
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ - зашумленные данные
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ - предсказание шума нейросетью

#### Intuition Behind the Loss

Модель учится предсказывать шум $\boldsymbol{\epsilon}$, который был добавлен к $\mathbf{x}_0$ для получения $\mathbf{x}_t$. Зная предсказанный шум, мы можем восстановить $\mathbf{x}_0$:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

---

## Forward and Reverse Diffusion Processes

### Forward Process: Adding Noise

Forward process - это марковская цепь, которая постепенно разрушает структуру данных:

```python
def forward_diffusion(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Forward diffusion: добавляет шум к данным
    
    Args:
        x0: исходные данные [B, C, H, W]
        t: временной шаг [B]
        sqrt_alphas_cumprod: sqrt(alpha_bar_t) [T]
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t) [T]
    
    Returns:
        xt: зашумленные данные
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

### Reverse Process: Removing Noise

Reverse process использует обученную модель для постепенного удаления шума:

```python
def reverse_diffusion_step(xt, t, model, sqrt_alphas_cumprod, 
                          sqrt_one_minus_alphas_cumprod, 
                          posterior_variance, posterior_mean_coef1, 
                          posterior_mean_coef2):
    """
    Один шаг reverse diffusion
    
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

## Training Process

### Training Algorithm

Алгоритм обучения diffusion model:

1. **Sample Data**: Выбираем случайный батч данных $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
2. **Sample Timestep**: Выбираем случайный временной шаг $t \sim \text{Uniform}(1, T)$
3. **Add Noise**: Генерируем зашумленные данные $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$
4. **Predict Noise**: Модель предсказывает шум $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
5. **Compute Loss**: Вычисляем MSE между истинным и предсказанным шумом
6. **Backpropagate**: Обновляем параметры модели

### На Чем Учится Diffusion Model?

#### Типы Данных для Обучения

Diffusion models могут обучаться на различных типах данных:

**1. Изображения (Image Diffusion)**
- **Датасеты**: 
  - ImageNet (1.2M изображений, 1000 классов)
  - LAION-5B (5.85 миллиардов изображений с текстовыми описаниями)
  - COCO (330K изображений с аннотациями)
  - CelebA (200K лиц)
  - FFHQ (70K высококачественных лиц)
- **Формат**: Обычно RGB изображения, нормализованные в диапазон $[-1, 1]$ или $[0, 1]$
- **Разрешение**: От 64x64 до 1024x1024 и выше

**2. Видео (Video Diffusion)**
- **Датасеты**:
  - WebVid (10M видео с текстовыми описаниями)
  - Kinetics (400K видео, 400 классов действий)
  - UCF-101, HMDB-51 (видео с действиями)
  - InternVid (236M видео-текст пар)
- **Формат**: Последовательность кадров (frames), обычно 16-128 кадров
- **Разрешение**: От 128x128 до 1024x1024 на кадр

**3. Текст (Text Diffusion)**
- **Датасеты**: 
  - Common Crawl
  - Wikipedia
  - Книги, статьи
- **Формат**: Токенизированный текст

**4. Аудио (Audio Diffusion)**
- **Датасеты**:
  - AudioSet (2M аудио клипов)
  - LibriSpeech (1000 часов речи)
- **Формат**: Спектрограммы или raw аудио

#### Процесс Обучения: Детальный Разбор

**Шаг 1: Подготовка Данных**

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

**Шаг 2: Выбор Случайного Временного Шага**

Модель учится на **всех временных шагах одновременно**:

```python
# Для каждого батча выбираем случайные временные шаги
t = torch.randint(0, timesteps, (batch_size,))  # [0, T-1]
```

Это позволяет модели:
- Быстро обучаться (не нужно проходить все шаги последовательно)
- Изучать разные уровни шума одновременно
- Эффективно использовать данные

**Шаг 3: Добавление Шума**

Для каждого изображения в батче:
- Выбираем случайный временной шаг $t$
- Генерируем случайный шум $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Вычисляем зашумленное изображение: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$

```python
def add_noise(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Добавляет шум к изображению согласно forward process
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

**Шаг 4: Предсказание Шума**

Модель (обычно U-Net) получает:
- **Вход**: Зашумленное изображение $\mathbf{x}_t$ (shape: [B, C, H, W])
- **Условие**: Временной шаг $t$ (shape: [B])
- **Выход**: Предсказанный шум $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ (shape: [B, C, H, W])

```python
# Forward pass через модель
predicted_noise = model(x_t, t)  # [B, C, H, W]
```

**Шаг 5: Вычисление Потерь**

Функция потерь - это **MSE между истинным и предсказанным шумом**:

$$\mathcal{L} = ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)||^2$$

```python
# Вычисление потерь
loss = F.mse_loss(noise, predicted_noise)
```

**Почему именно предсказание шума?**

1. **Проще для модели**: Предсказать шум проще, чем предсказать исходное изображение напрямую
2. **Стабильность**: Это приводит к более стабильному обучению
3. **Математическая обоснованность**: Связано с score matching и оптимальным транспортом

#### Что Изучает Модель?

Модель учится **обратному процессу диффузии**:

- **На ранних шагах** (большой $t$, много шума): Модель учится распознавать общую структуру и композицию
- **На средних шагах**: Модель учится восстанавливать детали и формы
- **На поздних шагах** (малый $t$, мало шума): Модель учится финальным деталям и текстурам

**Аналогия**: Как художник, который:
- Сначала намечает общую композицию (ранние шаги)
- Затем добавляет основные формы (средние шаги)
- В конце прорабатывает детали (поздние шаги)

#### Условное Обучение

Модель может обучаться с **условиями** (conditioning):

**1. Class-Conditional**: Генерация определенного класса
```python
predicted_noise = model(x_t, t, class_label)
```

**2. Text-Conditional**: Генерация по текстовому описанию
```python
# Текст кодируется через CLIP или T5
text_embedding = text_encoder(prompt)
predicted_noise = model(x_t, t, text_embedding)
```

**3. Image-Conditional**: Генерация на основе другого изображения
```python
predicted_noise = model(x_t, t, condition_image)
```

#### Объем Данных

Типичные объемы данных для обучения:
- **Базовые модели**: 1-10 миллионов изображений
- **Крупные модели** (Stable Diffusion): 100+ миллионов изображений
- **Очень крупные** (DALL-E 2, Imagen): 1+ миллиард изображений

**Время обучения**:
- Небольшие модели (64x64): Несколько дней на 1-4 GPU
- Средние модели (256x256): Недели на 8-16 GPU
- Крупные модели (1024x1024): Месяцы на десятках/сотнях GPU

### Key Design Choices

#### Network Architecture

Типичная архитектура - **U-Net** с временными embeddings:

- **Encoder-Decoder Structure**: Для обработки изображений
- **Time Embeddings**: Sinusoidal или learned embeddings для временного шага $t$
- **Attention Layers**: Self-attention для глобального контекста
- **Residual Connections**: Для стабильного обучения

#### Time Embedding

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

## Sampling and Generation

### Sampling Algorithm

Процесс генерации (sampling) - это обратный diffusion процесс:

1. **Start from Noise**: Начинаем с чистого шума $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. **Iterative Denoising**: Для $t = T, T-1, ..., 1$:
   - Предсказываем шум: $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
   - Вычисляем $\mathbf{x}_{t-1}$ используя предсказанный шум
3. **Final Sample**: $\mathbf{x}_0$ - сгенерированное изображение

### DDPM Sampling

```python
def sample_ddpm(model, shape, device, timesteps=1000, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                posterior_variance, posterior_mean_coef1, 
                posterior_mean_coef2):
    """
    Генерация сэмплов используя DDPM
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

### DDIM Sampling (Deterministic)

DDIM (Denoising Diffusion Implicit Models) позволяет детерминированную генерацию и более быстрый sampling:

```python
def sample_ddim(model, shape, device, timesteps=50, eta=0.0):
    """
    DDIM sampling - быстрее и детерминированно (если eta=0)
    
    Args:
        eta: параметр стохастичности (0 = детерминированный, 1 = стохастический)
    """
    x = torch.randn(shape, device=device)
    
    # Используем подпоследовательность временных шагов
    step_size = timesteps // 50  # 50 шагов вместо 1000
    
    for i in reversed(range(0, timesteps, step_size)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        predicted_noise = model(x, t)
        
        # DDIM update rule
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

## Implementation Example

### Complete DDPM Implementation

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
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SimpleUNet(nn.Module):
    """Упрощенная U-Net архитектура для diffusion model"""
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

        # Initial projection
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
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
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
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
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

## Key Variants and Extensions

### 1. DDIM (Denoising Diffusion Implicit Models)

**Ключевые особенности:**
- Детерминированная генерация (при $\eta = 0$)
- Более быстрый sampling (меньше шагов)
- Обратимость процесса

**Формула обновления:**

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

### 2. Latent Diffusion Models (Stable Diffusion)

**Идея:** Работают в латентном пространстве VAE вместо пиксельного пространства.

**Преимущества:**
- Быстрее (меньше размерность)
- Меньше памяти
- Высокое качество

**Архитектура:**
1. VAE encoder: изображение → латентное представление
2. Diffusion в латентном пространстве
3. VAE decoder: латентное представление → изображение

### 3. Classifier Guidance

**Идея:** Использование предобученного классификатора для улучшения генерации.

**Score function:**

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + s \cdot \nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t)$$

где $s$ - guidance scale.

### 4. Classifier-Free Guidance

**Идея:** Обучение условной и безусловной моделей одновременно, без классификатора.

**Предсказание:**

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, y) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset) + s \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset))$$

где $s$ - guidance scale, $\emptyset$ - пустое условие.

### 5. Score-Based Generative Models (SGM)

**Альтернативная формулировка** через score matching:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_t} \left[ \lambda(t) ||\mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)||^2 \right]$$

где $\mathbf{s}_\theta$ - score network.

### 6. Progressive Distillation

**Идея:** Обучение быстрых моделей через дистилляцию медленных.

**Результат:** Генерация за 4-8 шагов вместо 1000.

### 7. Rectified Flow / Flow Matching

**Новый подход (2023-2024):** Прямой путь от шума к данным.

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t)$$

где $\mathbf{v}_\theta$ - velocity field.

---

## Applications

### 1. Image Generation

**Text-to-Image:**
- DALL-E 2 (OpenAI)
- Stable Diffusion (Stability AI)
- Midjourney
- Imagen (Google)

**Image-to-Image:**
- Inpainting (заполнение пропусков)
- Super-resolution
- Style transfer
- Colorization

### 2. Video Generation

**Text-to-Video:**
- Runway Gen-2
- Pika Labs
- Stable Video Diffusion
- Sora (OpenAI, 2024)

**Video Editing:**
- Inpainting в видео
- Video-to-video translation

#### Как Diffusion Models Генерируют Видео?

Генерация видео с помощью diffusion models - это расширение image generation на временную размерность. Основная идея: **модель учится генерировать последовательность кадров, сохраняя временную согласованность**.

##### Архитектурные Подходы

**1. Frame-by-Frame Generation (Базовый)**

Самый простой подход - генерировать каждый кадр независимо:

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

**Проблема**: Кадры не связаны между собой, получается "дрожащее" видео.

**2. Temporal Conditioning (Временное Условие)**

Добавляем информацию о временной позиции кадра:

```python
def generate_video_with_temporal(model, text_prompt, num_frames=16):
    """
    Генерация с учетом временной позиции
    """
    frames = []
    for frame_idx in range(num_frames):
        # Добавляем временное условие
        temporal_embedding = get_temporal_embedding(frame_idx, num_frames)
        
        # Генерируем кадр с учетом времени
        frame = sample_ddpm(model, text_prompt, temporal_embedding)
        frames.append(frame)
    
    return frames
```

**3. 3D Convolutions / Spatio-Temporal Attention**

Используем 3D свертки или spatio-temporal attention для обработки видео как единого объема:

```python
class VideoDiffusionModel(nn.Module):
    """
    Модель для генерации видео с 3D свертками
    """
    def __init__(self):
        super().__init__()
        # 3D свертки для обработки пространства-времени
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        # ... остальные слои
        
    def forward(self, video_noise, t, text_embedding):
        """
        video_noise: [B, C, T, H, W] - зашумленное видео
        t: временной шаг диффузии
        text_embedding: текстовое условие
        """
        # Обработка видео как 3D объема
        x = self.conv3d_1(video_noise)
        x = self.conv3d_2(x)
        # ...
        return predicted_noise
```

**4. Latent Video Diffusion (Stable Video Diffusion)**

Работа в латентном пространстве VAE:

```python
# 1. Кодируем видео в латентное пространство
video_latents = vae_encoder(video_frames)  # [B, C, T, H', W']

# 2. Diffusion в латентном пространстве
denoised_latents = diffusion_model(video_latents, text_prompt)

# 3. Декодируем обратно в пиксели
generated_frames = vae_decoder(denoised_latents)  # [B, C, T, H, W]
```

##### Процесс Обучения для Видео

**Forward Process для Видео:**

Аналогично изображениям, но применяем к каждому кадру:

$$q(\mathbf{v}_t | \mathbf{v}_{t-1}) = \prod_{i=1}^{F} \mathcal{N}(\mathbf{v}_{t,i}; \sqrt{1-\beta_t}\mathbf{v}_{t-1,i}, \beta_t \mathbf{I})$$

где $\mathbf{v}_t = [\mathbf{x}_{t,1}, \mathbf{x}_{t,2}, ..., \mathbf{x}_{t,F}]$ - видео с $F$ кадрами.

**Ключевое отличие**: Нужно сохранять **временную согласованность** между кадрами.

##### Техники Обеспечения Временной Согласованности

**1. Temporal Attention**

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

**2. Optical Flow Conditioning**

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

**3. Frame Interpolation**

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

##### Современные Модели (2024)

**1. Sora (OpenAI, 2024)**

Ключевые особенности:
- **Diffusion Transformer (DiT)**: Использует Transformer вместо U-Net
- **Spacetime Patches**: Разбивает видео на пространственно-временные патчи
- **Scaling**: Масштабируется до очень больших моделей
- **Long Videos**: Может генерировать видео до 60 секунд
- **Физика**: Понимает физические законы (гравитация, отражения)

**Архитектура Sora:**

```python
class SoraModel(nn.Module):
    """
    Упрощенная версия архитектуры Sora
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

- Основан на Stable Diffusion
- Генерирует короткие видео (обычно 4-25 кадров)
- Открытая модель
- Хорошее качество для коротких клипов

**3. Runway Gen-2**

- Коммерческая модель
- Хорошее качество генерации
- Поддержка различных условий (текст, изображение)

##### Процесс Генерации Видео

**Полный Pipeline:**

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
    
    # 3. Diffusion процесс (обратный)
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

##### Вызовы Генерации Видео

1. **Временная Согласованность**: Кадры должны плавно переходить друг в друга
2. **Долгие Видео**: Сложно генерировать длинные последовательности
3. **Вычислительная Сложность**: Видео требует намного больше памяти и вычислений
4. **Физическая Реалистичность**: Движения должны подчиняться физическим законам
5. **Текстура и Детали**: Сохранение деталей во времени

##### Будущие Направления

- **Более Длинные Видео**: Генерация минутных и часовых видео
- **Лучшая Физика**: Более реалистичное моделирование физики
- **Контроль Движения**: Точный контроль над движениями объектов
- **Мультимодальность**: Генерация видео с аудио синхронизацией

### 3. 3D Generation

**Text-to-3D:**
- DreamFusion
- Magic3D
- Point-E

**Image-to-3D:**
- Zero-1-to-3

### 4. Audio Generation

**Text-to-Speech:**
- AudioLM (Google)
- MusicLM

**Audio Editing:**
- Audio inpainting
- Style transfer для аудио

### 5. Medical Imaging

- Генерация медицинских изображений
- Data augmentation
- Anomaly detection

### 6. Scientific Applications

- Генерация молекулярных структур
- Protein folding
- Material design

---

## Current Status (2023-2026)

### State-of-the-Art Models (2024-2025)

#### Image Generation

1. **Stable Diffusion 3 (2024)**
   - Улучшенная архитектура (MMDiT)
   - Лучшее понимание текста
   - Более детализированная генерация

2. **DALL-E 3 (2023)**
   - Интеграция с GPT-4
   - Улучшенное следование промптам
   - Более безопасная генерация

3. **Midjourney v6 (2024)**
   - Фотореалистичная генерация
   - Улучшенная композиция

#### Video Generation

1. **Sora (OpenAI, 2024)**
   - Генерация видео до 60 секунд
   - Понимание физики и пространства
   - Мультимодальные условия

2. **Stable Video Diffusion (2024)**
   - Открытая модель для video generation
   - Хорошее качество и контроль

#### 3D Generation

1. **3D Gaussian Splatting + Diffusion**
   - Быстрая генерация 3D сцен
   - Высокое качество рендеринга

2. **Triplane Diffusion**
   - Эффективное представление 3D

### Recent Advances

#### 1. Consistency Models (2023)

**Идея:** Прямое отображение шума в данные за один шаг.

$$\mathbf{x}_0 = f_\theta(\mathbf{x}_t, t)$$

**Преимущества:**
- Очень быстрая генерация
- Детерминированная
- Можно использовать как few-step diffusion

#### 2. Latent Consistency Models (LCM, 2024)

- Работают в латентном пространстве
- Генерация за 4 шага
- Используется в Stable Diffusion

#### 3. Flow Matching (2023-2024)

**Rectified Flow / Flow Matching:**
- Прямой путь от шума к данным
- Более эффективное обучение
- Быстрая генерация

#### 4. Diffusion Transformers (DiT, 2023)

**Идея:** Замена U-Net на Transformer архитектуру.

**Преимущества:**
- Масштабируемость
- Лучшее качество при больших моделях
- Используется в Sora

#### 5. Multimodal Diffusion

- **Text + Image**: Text-to-image, image-to-text
- **Audio + Text**: Audio generation
- **Video + Text**: Video generation
- **3D + Text**: 3D generation

### Performance Improvements

**Speed:**
- 2020: 1000 шагов (медленно)
- 2022: 50 шагов (DDIM)
- 2023: 4-8 шагов (LCM, Progressive Distillation)
- 2024: 1 шаг (Consistency Models)

**Quality:**
- Постоянное улучшение FID, IS scores
- Лучшее понимание текста
- Более детализированная генерация

### Open Challenges

1. **Speed vs Quality Trade-off**: Быстрая генерация часто жертвует качеством
2. **Control**: Точный контроль над генерацией все еще сложен
3. **Consistency**: Поддержание консистентности в длинных последовательностях
4. **Memory**: Большие модели требуют много памяти
5. **Bias and Safety**: Проблемы с bias и безопасной генерацией

---

## Comparison with Other Generative Models

### Diffusion Models vs GANs

| Аспект | Diffusion Models | GANs |
|--------|------------------|------|
| **Training Stability** | Стабильное обучение | Может быть нестабильным |
| **Mode Collapse** | Нет проблемы | Может страдать от mode collapse |
| **Sample Quality** | Очень высокое | Высокое (но может быть артефакты) |
| **Diversity** | Высокая | Зависит от архитектуры |
| **Sampling Speed** | Медленное (но улучшается) | Быстрое |
| **Likelihood** | Можно оценить (через ELBO) | Нет явного likelihood |
| **Conditioning** | Легко добавляется | Требует специальных техник |

### Diffusion Models vs VAEs

| Аспект | Diffusion Models | VAEs |
|--------|------------------|------|
| **Sample Quality** | Очень высокое | Часто размытое |
| **Latent Space** | Нет явного latent space | Структурированный latent space |
| **Interpolation** | Сложнее | Легко в latent space |
| **Training** | Стабильное | Может быть нестабильным |
| **Likelihood** | Можно оценить | Явный ELBO |
| **Speed** | Медленное | Быстрое |

### Diffusion Models vs Autoregressive Models

| Аспект | Diffusion Models | Autoregressive (PixelCNN, etc.) |
|--------|------------------|--------------------------------|
| **Parallel Generation** | Можно генерировать параллельно | Последовательное |
| **Long-range Dependencies** | Хорошо | Ограничено |
| **Sample Quality** | Очень высокое | Хорошее |
| **Speed** | Медленное | Медленное (последовательное) |

### When to Use Diffusion Models

**Используйте Diffusion Models когда:**
- Нужно очень высокое качество генерации
- Важна стабильность обучения
- Нужна условная генерация (текст, классы)
- Можно позволить медленную генерацию (или использовать быстрые варианты)

**Рассмотрите альтернативы когда:**
- Нужна очень быстрая генерация (GANs, VAEs)
- Нужен структурированный latent space (VAEs)
- Ограничены ресурсы (VAEs, малые GANs)

---

## References

### Related Documents

- **[Gaussian Distribution (Normal Distribution)](./gaussian-distribution.md)**: Фундаментальное распределение, используемое для добавления и удаления шума в diffusion models
- **[Variational Autoencoders (VAEs)](./variational-autoencoders-vaes.md)**: Альтернативный подход к генеративному моделированию с явным latent space
- **[Generative Adversarial Networks (GANs)](./generative-adversarial-networks-gans.md)**: Adversarial подход к генерации, сравнение с diffusion models

### Key Papers

1. **Sohl-Dickstein et al. (2015)**: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" - Первая работа по diffusion models

2. **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" - Популяризация и упрощение формулировки

3. **Song et al. (2021)**: "Denoising Diffusion Implicit Models" - DDIM, детерминированная генерация

4. **Rombach et al. (2022)**: "High-Resolution Image Synthesis with Latent Diffusion Models" - Stable Diffusion

5. **Ho & Salimans (2022)**: "Classifier-Free Diffusion Guidance" - Classifier-free guidance

6. **Song et al. (2023)**: "Consistency Models" - Одношаговая генерация

7. **Song et al. (2023)**: "Consistency Trajectory Models" - Улучшенные consistency models

8. **Lipman et al. (2023)**: "Flow Matching for Generative Modeling" - Flow matching подход

9. **Peebles & Xie (2023)**: "Scalable Diffusion Models with Transformers" - DiT архитектура

10. **Luo et al. (2023)**: "Latent Consistency Models" - LCM для быстрой генерации

### Recent Papers (2024-2025)

1. **OpenAI (2024)**: "Sora: Creating Video from Text" - Video generation model

2. **Stability AI (2024)**: "Stable Diffusion 3" - Улучшенная версия Stable Diffusion

3. **Google (2024)**: "Imagen 3" - Улучшенная text-to-image модель

### Resources

- **Hugging Face Diffusers**: Библиотека для работы с diffusion models
- **Stable Diffusion WebUI**: Пользовательский интерфейс для Stable Diffusion
- **Papers with Code**: Актуальные результаты и реализации

### Mathematical Background

- **Stochastic Processes**: Теория марковских процессов
- **Variational Inference**: ELBO и вариационные методы
- **Score Matching**: Альтернативная формулировка через score functions
- **Optimal Transport**: Связь с теорией оптимального транспорта

---

*Документ создан: 2025*
*Последнее обновление: 2025*
