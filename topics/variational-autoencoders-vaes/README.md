---
title: Variational Autoencoders (VAE)
description: "ELBO, encoder/decoder, reparameterization trick, латентное пространство и роль VAE в современных generative pipelines."
tags:
  - kb/topic
  - domain/generative
  - concept/vae
  - concept/latent-variable
  - concept/elbo
aliases:
  - VAE
  - Variational Autoencoder
  - ELBO
  - reparameterization trick
related:
  - gaussian-distribution
  - generative-adversarial-networks-gans
  - diffusion-models
  - bayes-theorem-and-probability-foundations
status: canonical
lang: ru
type: topic
slug: variational-autoencoders-vaes
updated: 2026-09-18
---
# Variational Autoencoders (VAE)

## Оглавление

1. [Как объяснить 5-летнему ребёнку](#как-объяснить-5-летнему-ребёнку)
2. [Введение в VAE](#введение-в-vae)
3. [Основная идея и интуиция](#основная-идея-и-интуиция)
4. [Математические основы](#математические-основы)
5. [Архитектура и компоненты](#архитектура-и-компоненты)
6. [Процесс обучения](#процесс-обучения)
7. [Пример реализации](#пример-реализации)
8. [Варианты и расширения](#варианты-и-расширения)
9. [Применения](#применения)
10. [Текущий статус (2025-2026)](#текущий-статус-2025-2026)
11. [Источники](#источники)
12. [Ключевые выводы](#ключевые-выводы)

---

## Как объяснить 5-летнему ребёнку

Представь машину, которая учится рисовать похожие картинки. Сначала она смотрит на настоящую картинку и записывает не «каждую точку», а короткое описание «о чём она» — как будто шёпотом. Потом по этому шёпоту рисует картинку заново. Если шёпот чуть изменить, получится *новая*, но всё ещё понятная картинка. Так VAE учится придумывать похожие вещи, а не только копировать.

---

## Введение в VAE

**Variational Autoencoders (VAE)** — класс генеративных моделей (generative models), предложенный Kingma & Welling (2013). Они соединяют идеи вариационного вывода (variational inference) и автокодировщиков (autoencoders). В отличие от обычных autoencoders, которые учат детерминированные отображения, VAE учит вероятностное представление латентного пространства (latent space) и поэтому может порождать новые примеры данных.

### Ключевые особенности

- **Вероятностное латентное пространство (probabilistic latent space)**: вход кодируется в распределение вероятностей, а не в фиксированный вектор
- **Генеративная способность**: можно сэмплировать из выученного распределения и получать новые данные
- **Регуляризованное латентное пространство**: пространство структурировано и непрерывно, поэтому возможна плавная интерполяция
- **Variational inference**: латентное представление учат через приближённый байесовский вывод

---

## Основная идея и интуиция

### Фундаментальная проблема

Обычный autoencoder сжимает данные в фиксированное латентное представление и восстанавливает их. Новые примеры из него получить нельзя, потому что:
1. В латентном пространстве могут быть «дыры» (области, куда ничего не закодировано)
2. Нет вероятностной модели распределения данных
3. Сэмплирование из латентного пространства не гарантирует осмысленный выход

### Решение VAE

VAE решает это так:
1. **Кодирование в распределения**: вместо одной точки кодируем в распределение вероятностей (обычно Gaussian)
2. **Регуляризация**: заставляем латентные распределения быть близкими к стандартному нормальному
3. **Сэмплирование**: новые примеры получают, сэмплируя из prior и прогоняя через decoder

### Интуитивная аналогия

VAE можно представить как переводчика, который учит язык:
- **Encoder**: учится переводить предложения (данные) в структурированное «пространство мыслей» (latent space)
- **Latent space**: непрерывное упорядоченное пространство, где похожие мысли лежат рядом
- **Decoder**: учится переводить мысли обратно в предложения
- **Регуляризация**: следит, чтобы «пространство мыслей» следовало стандартной структуре (как правилам грамматики)
- **Генерация**: новые предложения можно получать, сэмплируя мысли из этого структурированного пространства

---

## Математические основы

### Вероятностная модель

VAE моделирует процесс порождения данных так:

$$
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}
$$

Где:
- $p(\mathbf{z})$ — prior по латентным переменным (обычно $\mathcal{N}(0, I)$)
- $p_\theta(\mathbf{x}|\mathbf{z})$ — decoder (generative model)
- $\mathbf{z}$ — латентная переменная
- $\mathbf{x}$ — наблюдаемые данные

### Вариационная нижняя граница (ELBO)

Истинный posterior $p(\mathbf{z}|\mathbf{x})$ вычислить нельзя, поэтому VAE приближает его encoder-сетью $q_\phi(\mathbf{z}|\mathbf{x})$. Цель обучения — максимизировать Evidence Lower BOund (ELBO):

$$
\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))
$$

Это можно переписать так:

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))
$$

**Составляющие:**
1. **Член реконструкции (reconstruction term)**: $\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]$ — поощряет точное восстановление
2. **Член регуляризации (regularization term)**: $-D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$ — прижимает латентное распределение к prior

### Трюк репараметризации (reparameterization trick)

Чтобы провести backpropagation через операцию сэмплирования, VAE использует reparameterization trick:

$$
\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Так градиенты проходят через детерминированные функции $\mu_\phi$ и $\sigma_\phi$, а $\mathbf{z}$ остаётся случайным.

---

## Архитектура и компоненты

### Сеть encoder

Encoder $q_\phi(\mathbf{z}|\mathbf{x})$ отображает вход $\mathbf{x}$ в параметры латентного распределения:

$$
\mu_\phi(\mathbf{x}),\ \log \sigma^2_\phi(\mathbf{x}) = \text{Encoder}_\phi(\mathbf{x})
$$

- На выходе — среднее $\mu$ и лог-дисперсия $\log \sigma^2$ (для численной стабильности)
- Обычно это нейронная сеть (CNN для изображений, MLP для других данных)

### Латентное пространство (latent space)

- **Размерность**: обычно намного меньше размерности входа
- **Распределение**: предполагается Gaussian: $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mu_\phi(\mathbf{x}), \sigma_\phi^2(\mathbf{x})I)$
- **Prior**: $p(\mathbf{z}) = \mathcal{N}(0, I)$

### Сеть decoder

Decoder $p_\theta(\mathbf{x}|\mathbf{z})$ отображает латентный код $\mathbf{z}$ в распределение данных:

$$
\hat{\mathbf{x}} = \text{Decoder}_\theta(\mathbf{z})
$$

- Для изображений: выдаёт значения пикселей (часто с активацией sigmoid/tanh)
- Может моделировать разные распределения (Gaussian для непрерывных данных, Bernoulli для бинарных)

### Функция потерь

Для изображений со значениями пикселей в [0, 1] reconstruction loss обычно такой:

$$
\mathcal{L}_{\text{recon}} = -\log p_\theta(\mathbf{x}|\mathbf{z}) = \text{BCE}(\mathbf{x}, \hat{\mathbf{x}}) \text{ или } \text{MSE}(\mathbf{x}, \hat{\mathbf{x}})
$$

Член KL-дивергенции:

$$
\mathcal{L}_{\text{KL}} = D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) = \frac{1}{2}\sum_{i=1}^{d} [\sigma_i^2 + \mu_i^2 - 1 - \log(\sigma_i^2)]
$$

---

## Процесс обучения

### Прямой проход (forward pass)

1. Вход $\mathbf{x}$ проходит через encoder
2. Encoder выдаёт $\mu_\phi(\mathbf{x})$ и $\log \sigma^2_\phi(\mathbf{x})$
3. Сэмплируем $\epsilon \sim \mathcal{N}(0, I)$
4. Считаем $\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon$, где $\sigma_\phi = \exp(\tfrac{1}{2}\log\sigma^2_\phi)$
5. Декодируем: $\hat{\mathbf{x}} = \text{Decoder}_\theta(\mathbf{z})$

### Обратный проход (backward pass)

1. Считаем reconstruction loss: $\mathcal{L}_{\text{recon}}$
2. Считаем KL-дивергенцию: $\mathcal{L}_{\text{KL}}$
3. Полный loss: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$
4. Backpropagation через encoder и decoder

### Beta-VAE

Вариант, который добавляет вес $\beta$ к KL-члену:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}
$$

- $\beta = 1$: обычный VAE
- $\beta > 1$: сильнее регуляризация, лучше disentanglement
- $\beta < 1$: лучше реконструкция, менее структурированное латентное пространство

---

## Пример реализации

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=400):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For images in [0, 1]
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent code to data"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """Compute VAE loss"""
        # Reconstruction loss (BCE for binary data)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def sample(self, num_samples=64, device='cuda'):
        """Generate samples from prior"""
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)

# Training loop example
def train_vae(model, train_loader, optimizer, device, beta=1.0, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = model.loss_function(
                recon_batch, data, mu, logvar, beta
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, '
                      f'KL: {kl_loss.item():.4f}')
```

---

## Варианты и расширения

### 1. Beta-VAE
- Добавляет вес $\beta$ к KL-члену для лучшего disentanglement
- Полезен, когда нужны интерпретируемые латентные факторы

### 2. VQ-VAE (Vector Quantized VAE)
- Использует дискретные латентные коды вместо непрерывных
- Лучше ловит дискретные структуры в данных

### 3. VAE-GAN
- Сочетает VAE с discriminator из GAN
- Adversarial loss даёт более качественные изображения

### 4. Conditional VAE (CVAE)
- Обусловливает генерацию дополнительной информацией (метки, атрибуты)
- Позволяет управляемую генерацию (controlled generation)

### 5. Hierarchical VAE
- Несколько уровней латентных переменных
- Лучше подходит для сложных иерархических данных

### 6. NVAE (Nouveau VAE)
- Глубокая иерархическая архитектура с residual-ячейками (Vahdat & Kautz, 2020)
- На момент выхода — одна из сильнейших «чистых» VAE для изображений высокого разрешения (256×256 CelebA-HQ, FFHQ)

---

## Применения

### Исторические применения (2013-2020)

1. **Генерация изображений**: MNIST, CelebA, CIFAR-10
2. **Сжатие данных**: learned compression
3. **Детекция аномалий (anomaly detection)**: выбросы в латентном пространстве
4. **Обучение представлений (representation learning)**: unsupervised feature learning
5. **Аугментация данных**: синтетические обучающие примеры

### Текущие применения (2021–2026)

1. **Молекулярный дизайн**: drug discovery, material science
2. **Генерация 3D-форм**: point clouds, meshes
3. **Генерация аудио**: музыка, speech synthesis
4. **Генерация текста**: variational text models
5. **Рекомендательные системы**: моделирование предпочтений пользователя

---

## Текущий статус (2025-2026)

### Используются ли VAE до сих пор?

**Да, но в конкретных нишах:**

1. **Исследования**: область всё ещё живая, особенно:
   - Disentangled representation learning
   - Иерархическое generative modeling
   - Задачи, где нужно структурированное латентное пространство

2. **Индустриальные применения**:
   - **Молекулярный / белковый дизайн**: VAE-модели для drug discovery
   - **Детекция аномалий**: промышленные задачи, где важна интерпретируемость
   - **Сжатие данных**: системы learned compression
   - **Управляемая генерация**: когда нужно структурированное, интерпретируемое латентное пространство

3. **Гибридные модели**: часто сочетают с:
   - Diffusion models (как encoder/decoder)
   - Transformers (для последовательных данных)
   - GAN (архитектуры VAE-GAN)

### Сравнение с современными альтернативами

| Тип модели | Сильные стороны | Слабые стороны | Лучше всего для |
|------------|-----------------|----------------|-----------------|
| **VAE** | Структурированное латентное пространство, интерпретируемость, стабильное обучение | Размытые реконструкции, posterior collapse (KL-член «выключает» латенты) | Disentangled representations, детекция аномалий |
| **GAN** | Высококачественные сэмплы, резкие изображения | Нестабильное обучение, mode collapse | Генерация изображений высокой точности |
| **Diffusion Models** | Качество state-of-the-art, стабильное обучение | Медленная генерация, большие вычислительные затраты | Текущий SOTA для изображений и аудио |
| **Flow Models** | Точный likelihood, обратимость | Ограниченная выразительность | Оценка плотности, задачи на правдоподобие |

### Почему VAE важны в 2025-2026

1. **Интерпретируемость**: структурированное латентное пространство даёт понимание и контроль
2. **Стабильность**: обучение стабильнее, чем у GAN
3. **Теоретический фундамент**: сильная вероятностная основа
4. **Гибридные архитектуры**: компоненты входят в современные системы (например, VAE-encoder в diffusion models)
5. **Отдельные домены**: всё ещё лучший выбор для ряда задач (молекулярный дизайн, детекция аномалий)

---

## Источники

### Основополагающие статьи

1. **Kingma & Welling (2013)**: "Auto-Encoding Variational Bayes" — оригинальная статья про VAE
2. **Rezende et al. (2014)**: "Stochastic Backpropagation and Approximate Inference in Deep Generative Models"
3. **Higgins et al. (2017)**: "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
4. **van den Oord et al. (2017)**: "Neural Discrete Representation Learning" (VQ-VAE)

### Современные расширения

5. **Vahdat & Kautz (2020)**: "NVAE: A Deep Hierarchical Variational Autoencoder"
6. **Rombach et al. (2022)**: "High-Resolution Image Synthesis with Latent Diffusion Models" (использует VAE encoder)

### Связанные темы

- См.: [Generative Adversarial Networks (GANs)](../generative-adversarial-networks-gans/README.md)
- См.: [Diffusion Models](../diffusion-models/README.md)
- См.: disentangled representation learning (например, $\beta$-VAE, ICLR 2017: https://openreview.net/forum?id=Sy2fzU9gl)

---

## Ключевые выводы

1. **VAE учит вероятностные латентные представления**, которые позволяют и генерировать, и интерполировать
2. **Целевая функция ELBO** балансирует качество реконструкции и структуру латентного пространства
3. **Reparameterization trick** делает возможным градиентную оптимизацию
4. **Структурированное латентное пространство** делает VAE ценным для интерпретируемой генерации
5. **В 2025–2026 всё ещё актуальны** для задач, где нужно структурированное, интерпретируемое латентное пространство
6. **Часто входят в гибридные архитектуры** вместе с современными generative models
