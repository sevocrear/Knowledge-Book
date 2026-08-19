---
title: Generative Adversarial Networks (GAN)
description: "Состязательное обучение generator/discriminator, mode collapse, современные варианты GAN и сравнение с VAE и diffusion."
tags:
  - kb/topic
  - domain/generative
  - concept/gan
  - concept/adversarial-training
aliases:
  - GAN
  - Generative Adversarial Networks
  - StyleGAN
  - mode collapse
related:
  - variational-autoencoders-vaes
  - diffusion-models
  - gaussian-distribution
status: canonical
lang: ru
type: topic
slug: generative-adversarial-networks-gans
updated: 2026-08-19
---
# Generative Adversarial Networks (GAN)

## Оглавление

1. [Как объяснить 5-летнему ребёнку](#как-объяснить-5-летнему-ребёнку)
2. [Введение в GAN](#введение-в-gan)
3. [Основная идея и интуиция](#основная-идея-и-интуиция)
4. [Математические основы](#математические-основы)
5. [Архитектура и обучение](#архитектура-и-обучение)
6. [Пример реализации](#пример-реализации)
7. [Проблемы и решения](#проблемы-и-решения)
8. [Современные варианты GAN](#современные-варианты-gan)
9. [Применения](#применения)
10. [Текущий статус (2025-2026)](#текущий-статус-2025-2026)
11. [Сравнение VAE и GAN](#сравнение-vae-и-gan)
12. [Источники](#источники)

---

## Как объяснить 5-летнему ребёнку

Два робота играют в игру. Один рисует поддельные картинки, другой угадывает: настоящая это или подделка. Чем лучше угадывает «детектив», тем лучше учится рисовать «художник». Когда детектив уже почти не отличает рисунок от настоящей фотографии — художник научился придумывать очень правдоподобные картинки. Это и есть GAN.

---

## Введение в GAN

**Generative Adversarial Networks (GAN)** предложили Goodfellow et al. в 2014 году как новый способ обучать генеративные модели. GAN используют состязательное обучение (adversarial training): две нейросети соревнуются. Генератор (generator) создаёт поддельные данные, а дискриминатор (discriminator) пытается отличить настоящие данные от поддельных.

### Ключевые свойства

- **Состязательное обучение (adversarial training)**: две сети обучаются друг против друга
- **Качественные сэмплы**: могут давать очень реалистичные, резкие изображения
- **Неявное распределение (implicit distribution)**: моделирует распределение данных неявно (без явного правдоподобия, likelihood)
- **Теоретико-игровая постановка**: в основе — минимаксная оптимизация (minimax)

---

## Основная идея и интуиция

### Состязательная игра

GAN формулируют генерацию как минимаксную игру двух игроков:

1. **Generator (G)**: старается создать реалистичные подделки, чтобы обмануть дискриминатор
2. **Discriminator (D)**: старается правильно отличить настоящие данные от поддельных

### Интуитивная аналогия

GAN похожи на фальшивомонетчика и детектива:
- **Generator (фальшивомонетчик)**: печатает фальшивые деньги и старается сделать их неотличимыми от настоящих
- **Discriminator (детектив)**: рассматривает купюры и ищет подделки
- **Процесс обучения**: чем лучше становится фальшивомонетчик, тем сильнее должен стать детектив. Получается гонка, в итоге подделки выглядят очень правдоподобно

### Равновесие

Обучение сходится, когда:
- генератор выдаёт данные, неотличимые от настоящих
- дискриминатор уже не может их различить (даёт вероятность 0.5 и для настоящих, и для поддельных)
- это равновесие Нэша (Nash equilibrium) этой игры

---

## Математические основы

### Минимаксный критерий

Целевая функция GAN:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$

Где:
- $D(\mathbf{x})$: вероятность, которую дискриминатор приписывает тому, что $\mathbf{x}$ настоящее
- $G(\mathbf{z})$: выход генератора по шуму $\mathbf{z}$
- $p_{\text{data}}(\mathbf{x})$: распределение настоящих данных
- $p_{\mathbf{z}}(\mathbf{z})$: априорное распределение шума (обычно $\mathcal{N}(0, I)$)

### Цель дискриминатора

Дискриминатор хочет максимизировать:

$$
\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}[\log(1 - D(G(\mathbf{z})))]
$$

- Максимизировать $\log D(\mathbf{x})$ для настоящих данных (значение должно быть близко к 1)
- Максимизировать $\log(1 - D(G(\mathbf{z})))$ для поддельных данных (значение $D$ должно быть близко к 0)

### Цель генератора

Генератор хочет минимизировать:

$$
\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}[\log(1 - D(G(\mathbf{z})))]
$$

Или, эквивалентно, максимизировать (ненасыщающаяся потеря, non-saturating loss):

$$
\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}[\log D(G(\mathbf{z}))]
$$

### Оптимальный дискриминатор

В оптимуме дискриминатор имеет вид:

$$
D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}
$$

Когда $p_g = p_{\text{data}}$, $D^*(\mathbf{x}) = \frac{1}{2}$ всюду.

### Глобальный оптимум

Глобальный минимум для генератора достигается при $p_g = p_{\text{data}}$, то есть когда генератор точно воспроизводит распределение данных.

---

## Архитектура и обучение

### Архитектура генератора

- **Вход**: случайный вектор шума $\mathbf{z} \sim \mathcal{N}(0, I)$ (обычно 100–512 измерений)
- **Архитектура**:
  - Для изображений: транспонированные свёртки (transposed convolutions / deconvolutions) или upsampling + convolutions
  - Пространственные размеры постепенно растут
  - Используются batch normalization, активации ReLU/LeakyReLU
- **Выход**: сгенерированные данные (например, изображения)

### Архитектура дискриминатора

- **Вход**: настоящие или сгенерированные данные
- **Архитектура**:
  - Для изображений: обычная CNN с downsampling
  - Пространственные размеры постепенно уменьшаются
  - Используются batch normalization, активации LeakyReLU
- **Выход**: одна вероятность (настоящее vs поддельное)

### Алгоритм обучения

```
1. Sample minibatch of noise: {z₁, z₂, ..., zₘ} ~ p_z(z)
2. Sample minibatch of real data: {x₁, x₂, ..., xₘ} ~ p_data(x)
3. Update Discriminator (maximize):
   - Forward pass: D(x) and D(G(z))
   - Compute loss: -[log D(x) + log(1 - D(G(z)))]
   - Backpropagate and update D
4. Update Generator (minimize):
   - Forward pass: D(G(z))
   - Compute loss: -log D(G(z))  (non-saturating)
   - Backpropagate and update G
5. Repeat until convergence
```

### Практические советы по обучению

1. **Чередующиеся обновления**: обычно обновляют D чаще, чем G (например, в отношении 5:1)
2. **Learning rates**: разные learning rate для G и D
3. **Batch Normalization**: помогает стабилизировать обучение
4. **Label smoothing**: для настоящих меток использовать 0.9 вместо 1.0
5. **Шум**: добавлять шум ко входам дискриминатора

---

## Пример реализации

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Generator Network
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (nc) x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# Training Function
def train_gan(generator, discriminator, dataloader, device, epochs=50, lr=0.0002, beta1=0.5):
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Labels
    real_label = 0.9  # Label smoothing
    fake_label = 0.0
    
    for epoch in range(epochs):
        for i, (data, _) in enumerate(dataloader):
            batch_size = data.size(0)
            data = data.to(device)
            
            # ========== Train Discriminator ==========
            # Train on real data
            discriminator.zero_grad()
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Train on fake data
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # ========== Train Generator ==========
            generator.zero_grad()
            label.fill_(real_label)  # Generator wants to fool discriminator
            output = discriminator(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Print statistics
            if i % 50 == 0:
                print(f'[{epoch}/{epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
```

---

## Проблемы и решения

### 1. Нестабильность обучения

**Проблема**: GAN печально известны тем, что их трудно обучать. Типичные симптомы:
- генератор или дискриминатор становится слишком сильным
- значение loss плохо коррелирует с качеством сэмплов
- обучение схлопывается (training collapse)

**Решения**:
- **Progressive GAN**: постепенно повышать разрешение
- **Wasserstein GAN (WGAN)**: вместо дивергенции Йенсена–Шеннона (JS divergence) использовать расстояние Васерштейна (Wasserstein distance)
- **Gradient Penalty**: WGAN-GP добавляет штраф по градиенту для стабильности
- **Spectral Normalization**: ограничивает константу Липшица дискриминатора

### 2. Схлопывание мод (mode collapse)

**Проблема**: генератор выдаёт ограниченное разнообразие сэмплов

**Решения**:
- **Unrolled GANs**: разворачивать несколько шагов обновления дискриминатора
- **Mini-batch Discrimination**: поощрять разнообразие внутри батча
- **Feature Matching**: согласовывать промежуточные признаки, а не только финальный выход

### 3. Оценка качества

**Проблема**: нет явного правдоподобия (likelihood), оценивать сложно

**Решения**:
- **Inception Score (IS)**: измеряет качество и разнообразие
- **Fréchet Inception Distance (FID)**: сравнивает распределения в пространстве признаков
- **Оценка людьми**: субъективная оценка качества

---

## Современные варианты GAN

### 1. DCGAN (2015)
- Deep Convolutional GAN
- Зафиксировал архитектурные рекомендации
- Использует strided convolutions, batch norm

### 2. WGAN / WGAN-GP (2017)
- Расстояние Васерштейна для стабильности
- Gradient penalty как способ обеспечить ограничение Липшица
- Более стабильное обучение

### 3. Progressive GAN (2017)
- Постепенно повышает разрешение
- Начинает с 4×4 и удваивает разрешение шаг за шагом
- Позволяет генерировать изображения высокого разрешения

### 4. StyleGAN (2019) / StyleGAN2 (2020) / StyleGAN3 (2021)
- Архитектура генератора на основе стиля (style-based)
- Отделяет латентный код (latent code) от шума
- Качество изображений на уровне state-of-the-art
- StyleGAN3 лучше справляется с алиасингом (aliasing)

### 5. BigGAN (2018)
- Обучение GAN в большом масштабе
- Условная генерация по классу (class-conditional)
- Truncation trick для компромисса качество/разнообразие

### 6. Self-Attention GAN (SAGAN) (2018)
- Добавляет слои self-attention
- Лучше моделирует дальнодействующие зависимости

### 7. Projected GANs (2021)
- Использует предобученные сети признаков
- Более быстрое обучение и лучше качество

---

## Применения

### Исторические применения (2014–2020)

1. **Генерация изображений**: CelebA, LSUN, ImageNet
2. **Image-to-image translation**: Pix2Pix, CycleGAN
3. **Super-resolution**: SRGAN
4. **Перенос стиля (style transfer)**: разные методы на основе GAN
5. **Аугментация данных**: генерация обучающих примеров

### Текущие применения (2021–2025)

1. **Синтез изображений высокого качества**: StyleGAN3 для лиц и объектов
2. **Редактирование изображений**: GAN inversion для манипуляций
3. **3D-генерация**: 3D-GAN, GRAF
4. **Генерация видео**: Video GANs
5. **Адаптация домена (domain adaptation)**: unsupervised domain transfer

---

## Текущий статус (2025-2026)

### Используют ли GAN до сих пор?

**Да, но они уже не так доминируют, как раньше:**

1. **Конкретные применения**:
   - **StyleGAN3**: всё ещё state-of-the-art для качественной генерации лиц
   - **Редактирование изображений**: GAN inversion для семантического редактирования
   - **Перенос домена**: unsupervised domain adaptation
   - **Аугментация данных**: генерация синтетических обучающих примеров

2. **Исследования**:
   - Тема живая, но уже не главная
   - Фокус на точечных улучшениях (например, эффективность, управляемость)
   - Гибридные архитектуры, где GAN сочетают с другими методами

3. **Индустрия**:
   - **Развлечения**: генерация лиц, создание персонажей
   - **Мода**: виртуальная примерка, генерация дизайна
   - **Игры**: генерация ассетов, процедурный контент

### Почему GAN уже не доминируют

1. **Diffusion models**: лучше качество, стабильнее обучение
2. **Сложность обучения**: GAN по-прежнему труднее обучать
3. **Оценка**: отсутствие явного правдоподобия усложняет оценку
4. **Mode collapse**: всё ещё проблема во многих задачах

### Когда использовать GAN в 2025–2026

- **Качественная генерация лиц**: StyleGAN3 всё ещё конкурентоспособен
- **Быстрая генерация**: GAN быстрее, чем diffusion models
- **Редактирование изображений**: GAN inversion даёт семантическое редактирование
- **Узкие домены**: там, где GAN уже хорошо себя показали

---

## Сравнение VAE и GAN

### Принципиальные различия

| Аспект | VAE | GAN |
|--------|-----|-----|
| **Цель** | Максимизация ELBO (вариационная нижняя оценка) | Минимаксная игра (состязательная) |
| **Обучение** | Стабильное, совместная оптимизация | Нестабильное, чередующаяся оптимизация |
| **Латентное пространство** | Явное, структурированное, непрерывное | Неявное, менее структурированное |
| **Правдоподобие (likelihood)** | Явное (нижняя оценка) | Нет явного правдоподобия |
| **Качество сэмплов** | Часто размытые | Резкие, высокого качества |
| **Mode collapse** | Редко | Частая проблема |
| **Интерпретируемость** | Высокая (структурированное латентное пространство) | Ниже |
| **Интерполяция** | Плавная (непрерывный латентный код) | Менее плавная |
| **Скорость обучения** | Умеренная | Может быть медленной (из-за чередования) |
| **Теоретическая база** | Сильная (вариационный вывод) | Теоретико-игровая |

### Математическое сравнение

**Целевая функция VAE**:
$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))
$$

**Целевая функция GAN**:
$$
\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}[\log(1 - D(G(\mathbf{z})))]
$$

### Когда что выбирать

**VAE уместен, когда**:
- нужно интерпретируемое, структурированное латентное пространство
- важна плавная интерполяция
- нужна явная оценка правдоподобия
- важна стабильность
- вы занимаетесь поиском аномалий (anomaly detection)
- нужны распутанные представления (disentangled representations)

**GAN уместен, когда**:
- нужны качественные, резкие изображения
- качество сэмплов — главный приоритет
- явное правдоподобие не требуется
- вы готовы мириться с нестабильным обучением
- задача — image-to-image translation
- нужна быстрая генерация

### Гибридные подходы

1. **VAE-GAN**: сочетает encoder/decoder из VAE с дискриминатором GAN
2. **Adversarial Autoencoders**: состязательное обучение в латентном пространстве
3. **BEGAN**: использует autoencoder как дискриминатор

---

## Источники

### Основополагающие статьи

1. **Goodfellow et al. (2014)**: "Generative Adversarial Nets" — оригинальная статья про GAN
2. **Radford et al. (2015)**: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (DCGAN)
3. **Arjovsky et al. (2017)**: "Wasserstein GAN"
4. **Gulrajani et al. (2017)**: "Improved Training of Wasserstein GANs" (WGAN-GP)

### Современные варианты

5. **Karras et al. (2019)**: "A Style-Based Generator Architecture for Generative Adversarial Networks" (StyleGAN)
6. **Karras et al. (2020)**: "Analyzing and Improving the Image Quality of StyleGAN" (StyleGAN2)
7. **Karras et al. (2021)**: "Alias-Free Generative Adversarial Networks" (StyleGAN3)
8. **Sauer et al. (2021)**: "Projected GANs Converge Faster"

### Связанные темы

- См.: [Variational Autoencoders (VAEs)](../variational-autoencoders-vaes/README.md)
- См.: [Diffusion Models](../diffusion-models/README.md)
- См.: [Knowledge-book Generative Models index](../../README.md#generative-models)

---

## Ключевые выводы

1. **GAN используют состязательное обучение** между generator и discriminator
2. **Минимаксный критерий** задаёт теоретико-игровую постановку
3. **Сэмплы высокого качества**, но нестабильность обучения — главная трудность
4. **В 2025–2026 всё ещё используются** в узких задачах (StyleGAN3, редактирование изображений)
5. **Уже не доминируют** относительно diffusion models в общей генерации изображений
6. **Дополняют VAE**: у каждого свои сильные стороны под разные сценарии
