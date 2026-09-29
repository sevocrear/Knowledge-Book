---
title: Low-Rank Adaptation (LoRA)
description: "PEFT через низкоранговые адаптеры ΔW≈BA: математика, QLoRA/AdaLoRA/DoRA, эффективность памяти и практика в Hugging Face PEFT."
tags:
  - kb/topic
  - domain/llm
  - concept/peft
  - concept/lora
  - concept/fine-tuning
aliases:
  - LoRA
  - QLoRA
  - AdaLoRA
  - DoRA
  - PEFT
related:
  - transformers-attention-and-vision-transformers-vit
  - retrieval-augmented-generation-rag
  - vision-language-action-models-vla
  - ensemble-methods-model-combination
status: canonical
lang: ru
type: topic
slug: low-rank-adaptation-lora
updated: 2026-09-18
---
# Low-Rank Adaptation (LoRA)

## Оглавление

1. Введение: проблема тонкой настройки больших моделей
2. Математическая основа LoRA: разложение матриц низкого ранга
3. Архитектура LoRA: как это работает
4. Параметры и эффективность: сравнение с полной тонкой настройкой
5. Варианты LoRA: QLoRA, AdaLoRA, DoRA
6. Практическое применение: когда использовать LoRA
7. Реализация в PyTorch
8. Связанные темы и источники
9. Как объяснить это 5‑летнему ребёнку

---

## 1. Введение: проблема тонкой настройки больших моделей

**Проблема:**
- Современные LLM (Llama 3.x/4, Qwen2.5/3, Mistral/Mixtral, Gemma, GPT-class API bases) содержат миллиарды–триллионы параметров
- Полная тонкая настройка (full fine-tuning) требует обновления всех весов модели
- Это требует огромных вычислительных ресурсов и памяти (часто десятки–сотни GB GPU-памяти)
- Для каждой новой задачи нужно хранить полную копию модели

**Решение LoRA:**
- Вместо обновления всех весов, LoRA добавляет **небольшие адаптивные матрицы** к существующим слоям
- Обучаются только эти новые матрицы (обычно ≪1% от исходных параметров)
- Можно переиспользовать базовую модель для множества задач
- Экономия памяти и времени обучения на порядки относительно full FT

**История и статус (2021–2026):**
- LoRA предложена в работе "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Де-факто стандарт PEFT для LLM/VLM вместе с QLoRA; распространённые варианты — AdaLoRA, DoRA, LoRA+
- В экосистеме Hugging Face обычно используют через `peft` + (опционально) `bitsandbytes` для QLoRA

---

## 2. Математическая основа LoRA: разложение матриц низкого ранга

### 2.1. Интуиция: почему низкий ранг?

**Ключевая гипотеза LoRA:**
> При адаптации модели к новой задаче, изменения весов происходят в **низкоранговом подпространстве**. То есть, реальные изменения можно представить как произведение двух маленьких матриц.

**Математически:**
Пусть $W \in \mathbb{R}^{d \times k}$ — матрица весов слоя (например, линейный слой в Transformer).

При полной тонкой настройке:
$$
W_{\text{new}} = W_{\text{old}} + \Delta W
$$

**Гипотеза LoRA:** матрица изменений $\Delta W$ имеет **низкий ранг** $r \ll \min(d, k)$.

По теореме о разложении матриц, любую матрицу ранга $r$ можно представить как:
$$
\Delta W = BA
$$

где:
- $A \in \mathbb{R}^{r \times k}$ — матрица «вниз» (down-projection: $k \to r$)
- $B \in \mathbb{R}^{d \times r}$ — матрица «вверх» (up-projection: $r \to d$)
- $r$ — **rank** (ранг адаптации), обычно $r \in \{1, 2, 4, 8, 16, 32, 64\}$

**Количество параметров:**
- Полная настройка: $d \times k$ параметров
- LoRA: $d \times r + r \times k = r(d + k)$ параметров

**Экономия:**
Если $r \ll \min(d, k)$, то $r(d + k) \ll d \times k$.

**Пример:**
- Слой: $d = 4096$, $k = 4096$ → $16,777,216$ параметров
- LoRA с $r = 8$: $8 \times (4096 + 4096) = 65,536$ параметров
- Экономия: **256 раз меньше параметров!**

---

## 3. Архитектура LoRA: как это работает

### 3.1. Применение LoRA к линейным слоям

В Transformer модели LoRA обычно применяется к:
- **Query, Key, Value проекциям** в attention слоях
- **Feed-forward сетям** (MLP слоям)
- **Output проекциям** в attention

**Формула для одного слоя:**

Пусть вход: $\mathbf{x} \in \mathbb{R}^k$

**Обычный слой:**
$$
\mathbf{h} = W \mathbf{x} + \mathbf{b}
$$

**Слой с LoRA:**
$$
\mathbf{h} = W \mathbf{x} + \mathbf{b} + \underbrace{B A \mathbf{x}}_{\text{LoRA адаптация}}
$$

где:
- $W$ — **замороженные** (frozen) веса базовой модели
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ — **обучаемые** LoRA матрицы
- $\mathbf{b}$ — bias (если есть)

**Важно:**
- $W$ **не обновляется** во время обучения (gradient = 0)
- Обучаются только $A$ и $B$
- При инференсе: $\mathbf{h} = (W + BA) \mathbf{x} + \mathbf{b}$ можно вычислить заранее

### 3.2. Инициализация LoRA матриц

**Стандартная инициализация:**
- $A$ инициализируется **случайными значениями** (обычно из нормального распределения)
- $B$ инициализируется **нулями**: $B = \mathbf{0}$

**Почему $B = 0$?**
- При старте обучения: $\Delta W = BA = 0 \times A = 0$
- Модель начинает с исходных весов $W$ (без изменений)
- Градиенты постепенно "включают" адаптацию

**Альтернативная инициализация (в некоторых реализациях):**
- $A$: случайная инициализация с малым масштабом
- $B$: нули (как и раньше)

### 3.3. Масштабирование (scaling)

Часто используется **коэффициент масштабирования** $\alpha$:

$$
\mathbf{h} = W \mathbf{x} + \frac{\alpha}{r} \cdot BA \mathbf{x}
$$

где:
- $\alpha$ — гиперпараметр масштабирования (обычно $\alpha = r$ или $\alpha = 2r$)
- $r$ — rank адаптации

**Смысл:**
- Позволяет контролировать "силу" адаптации
- При $\alpha = r$: масштабирование компенсирует уменьшение ранга
- Упрощает подбор гиперпараметров при изменении $r$

---

## 4. Параметры и эффективность: сравнение с полной тонкой настройкой

### 4.1. Количество параметров

**Пример: LLaMA-7B**

Базовая модель:
- Attention слои: ~$4 \times 4096 \times 4096$ параметров на слой (Q, K, V, O)
- MLP слои (SwiGLU: gate, up, down): ~$3 \times 4096 \times 11008$ параметров на слой
- Всего: ~7 миллиардов параметров

**LoRA настройка:**
- Применяется к Q, K, V, O (attention-only вариант)
- Rank $r = 8$, $\alpha = 16$
- Количество LoRA параметров: $32 \times 4 \times 8 \times (4096 + 4096) \approx 8.4$ миллиона (только Q, V — ~4.2M; вместе с MLP-слоями gate/up/down — ~20M)

**Сравнение:**
- Полная настройка: 7B параметров
- LoRA: ~8M параметров
- **Экономия: ~800 раз меньше параметров!**

### 4.2. Память во время обучения

**Полная тонкая настройка:**
- Хранение весов: 7B параметров × 4 bytes (FP32) = 28 GB
- Градиенты: +28 GB
- Оптимизатор (Adam): +56 GB (моменты)
- **Итого: ~112 GB GPU памяти**

**LoRA:**
- Базовые веса (frozen): 28 GB
- LoRA веса: 8M × 4 bytes = 32 MB
- Градиенты LoRA: +32 MB
- Оптимизатор LoRA: +64 MB
- **Итого: ~28.1 GB GPU памяти**

**Экономия памяти: ~4 раза** (можно использовать меньшие GPU)

### 4.3. Скорость обучения

- **Forward pass:** почти такая же (добавляется только $BA\mathbf{x}$)
- **Backward pass:** вычисляются градиенты только для $A$ и $B$
- **Обновление весов:** обновляются только LoRA параметры
- **Ускорение:** умеренное — backward всё равно проходит через замороженные слои (нужны градиенты по активациям); в статье LoRA — ~25% на GPT-3 175B, основная выгода — память и объём чекпойнтов

### 4.4. Качество модели

**Результаты исследований:**
- LoRA часто достигает **95-99% качества** полной тонкой настройки
- При правильном выборе $r$ и $\alpha$ разница минимальна
- Для некоторых задач LoRA даже лучше (регуляризация через низкий ранг)

---

## 5. Варианты LoRA: QLoRA, AdaLoRA, DoRA

### 5.1. QLoRA (Quantized LoRA)

**Проблема:** даже с LoRA базовые веса занимают много памяти (28 GB для LLaMA-7B)

**Решение QLoRA:**
1. **Квантизация базовой модели** в 4-bit (NF4)
2. Применение LoRA к квантизированным весам
3. Использование **Paged Optimizers** для экономии памяти

**Результат:**
- Базовые веса: 28 GB → ~4 GB (4-bit квантизация)
- LoRA веса: ~32 MB
- **Итого: ~4 GB GPU памяти** под веса для обучения LLaMA-7B (плюс активации; на практике 7B в QLoRA укладывается в ~6–8 GB)

**Формула:**
$$
\mathbf{h} = Q(W) \mathbf{x} + BA \mathbf{x}
$$

где $Q(W)$ — квантизированные (NF4) веса базовой модели, деквантизируемые «на лету» в bf16 при прямом проходе; ключевые компоненты QLoRA — NF4, double quantization и paged optimizers.

### 5.2. AdaLoRA (Adaptive LoRA)

**Проблема:** фиксированный rank $r$ для всех слоёв может быть неоптимальным

**Решение AdaLoRA:**
- **Адаптивный rank** для разных слоёв
- Важные слои получают больший rank, менее важные — меньший
- Использует **SVD разложение** для динамической настройки ранга

**Алгоритм:**
1. Начинает с одинакового ранга для всех слоёв
2. Периодически переоценивает важность параметров
3. Перераспределяет параметры между слоями

### 5.3. DoRA (Weight-Decomposed Low-Rank Adaptation)

**Идея:** разделить адаптацию на **масштаб** (magnitude) и **направление** (direction)

**Формула:**
$$
W' = m \cdot \frac{W + BA}{\|W + BA\|_c}
$$

где:
- $m$ — обучаемый масштабный параметр
- $\| \cdot \|_c$ — норма по столбцам
- $BA$ — LoRA адаптация (направление)

**Преимущества:**
- Более гибкая адаптация
- Лучше работает на сложных задачах
- Сохраняет эффективность LoRA

---

## 6. Практическое применение: когда использовать LoRA

### 6.1. Когда LoRA подходит

✅ **Идеальные случаи:**
- Ограниченные вычислительные ресурсы (мало GPU памяти)
- Нужно быстро адаптировать модель под несколько задач
- Базовая модель уже хорошо обучена (pre-trained)
- Задача специфична, но не требует кардинальных изменений архитектуры

✅ **Типичные задачи:**
- Fine-tuning LLM под конкретный домен (медицина, право, код)
- Адаптация под новый язык или стиль текста
- Создание специализированных чат-ботов
- Мульти-таск обучение (несколько задач с одной базовой моделью)

### 6.2. Когда LoRA может быть недостаточно

❌ **Ограничения:**
- Задачи, требующие кардинальных архитектурных изменений
- Когда нужна максимальная производительность (полная настройка может быть лучше)
- Очень специфичные задачи, где низкоранговое предположение не выполняется

### 6.3. Выбор гиперпараметров

**Rank $r$:**
- **Маленький** ($r = 1, 2, 4$): меньше параметров, быстрее, но может быть недостаточно
- **Средний** ($r = 8, 16$): хороший баланс (рекомендуется начать с $r = 8$)
- **Большой** ($r = 32, 64$): больше параметров, ближе к полной настройке

**Alpha $\alpha$:**
- Обычно: $\alpha = r$ или $\alpha = 2r$
- Больше $\alpha$ → сильнее адаптация
- Можно подбирать экспериментально

**Target modules (куда применять LoRA):**
- `q_proj, k_proj, v_proj, o_proj` — attention слои
- `gate_proj, up_proj, down_proj` — MLP слои
- Можно применять ко всем или выборочно

---

## 7. Реализация в PyTorch

### 7.1. Базовая реализация LoRA слоя

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    """
    LoRA адаптация для линейного слоя.
    
    Args:
        in_features: размерность входа
        out_features: размерность выхода
        rank: ранг адаптации (r)
        alpha: коэффициент масштабирования
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA матрицы
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x, base_weight, base_bias=None):
        """
        Args:
            x: входной тензор [batch_size, ..., in_features]
            base_weight: базовые веса [out_features, in_features]
            base_bias: базовый bias [out_features] (опционально)
        """
        # Базовый выход
        base_output = F.linear(x, base_weight, base_bias)
        
        # LoRA адаптация: x @ A^T @ B^T  (F.linear(x, W) = x @ W^T)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output


class LinearWithLoRA(nn.Module):
    """
    Линейный слой с LoRA адаптацией.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        # Базовый слой (замороженный)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        # Замораживаем базовые веса
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # LoRA адаптация
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        
    def forward(self, x):
        return self.lora(x, self.base_layer.weight, self.base_layer.bias)
```

### 7.2. Интеграция с Hugging Face Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Загрузка модели
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Конфигурация LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # rank
    lora_alpha=16,  # alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,  # dropout для LoRA слоёв
    bias="none",  # не обучаем bias
)

# Применение LoRA
model = get_peft_model(model, lora_config)

# Теперь обучаются только LoRA параметры
model.print_trainable_parameters()  # trainable params: ~20M || all params: ~6.7B || trainable%: ~0.3

# Обучение
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,  # mixed precision
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 7.3. Сохранение и загрузка LoRA весов

```python
# Сохранение только LoRA весов (очень маленький файл)
model.save_pretrained("./lora_weights")

# Загрузка базовой модели + LoRA весов
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Применение LoRA весов
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# Инференс
model.eval()
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=100)
```

### 7.4. Мульти-таск обучение с LoRA

```python
# Можно создать несколько LoRA адаптаций для разных задач
task1_lora = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], ...)
task2_lora = LoraConfig(r=16, target_modules=["gate_proj", "up_proj"], ...)

# Несколько адаптеров на одной базовой модели (get_peft_model нельзя вызывать дважды на одной модели)
model = get_peft_model(base_model, task1_lora, adapter_name="task1")
model.add_adapter("task2", task2_lora)

# Переключаемся между адаптерами: обучаем/инферим по очереди, база общая
model.set_adapter("task1")
model.set_adapter("task2")
```

---

## 8. Связанные темы и источники

### 8.1. Связанные техники

- **QLoRA**: LoRA + квантизация для ещё большей экономии памяти
- **PEFT (Parameter-Efficient Fine-Tuning)**: общий фреймворк для эффективной настройки
- **Adapter Layers**: альтернативный подход (добавление маленьких слоёв)
- **Prompt Tuning / Prefix Tuning**: обучение промптов вместо весов
- **IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)**: адаптация через масштабирование активаций

### 8.2. Связанные документы

- **[Transformers, Attention and Vision Transformers](../transformers-attention-and-vision-transformers-vit/README.md)**: понимание архитектуры Transformer, к которой применяется LoRA
- **[Retrieval-Augmented Generation (RAG)](../retrieval-augmented-generation-rag/README.md)**: LoRA часто используется для fine-tuning моделей в RAG системах

### 8.3. Ключевые статьи

1. **LoRA: Low-Rank Adaptation of Large Language Models** (2021)
   - Edward J. Hu, Yelong Shen, Phillip Wallis, et al.
   - [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

2. **QLoRA: Efficient Finetuning of Quantized LLMs** (2023)
   - Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
   - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

3. **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning** (2023)
   - Qingru Zhang, Minshuo Chen, Alexander Bukharin, et al.
   - [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)

4. **DoRA: Weight-Decomposed Low-Rank Adaptation** (2024)
   - Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, et al.
   - [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)

### 8.4. Библиотеки и инструменты

- **PEFT (Parameter-Efficient Fine-Tuning)** от Hugging Face: [GitHub](https://github.com/huggingface/peft)
- **bitsandbytes**: для квантизации в QLoRA
- **Axolotl**: фреймворк для обучения LLM с LoRA

---

## 9. Как объяснить это 5‑летнему ребёнку

**Представь, что у тебя есть огромная книга с миллионами страниц (это большая AI модель).**

**Полная тонкая настройка** — это как переписать всю книгу заново, чтобы она лучше подходила для новой задачи. Это очень долго и дорого!

**LoRA** — это как добавить маленькие стикеры-заметки на некоторые страницы книги. Вместо того чтобы переписывать всю книгу, мы просто добавляем маленькие заметки, которые говорят: "здесь нужно думать немного по-другому". 

Эти заметки очень маленькие (всего несколько штук), но они делают книгу полезной для новой задачи! И самое главное — мы можем использовать одну и ту же книгу для разных задач, просто меняя стикеры.

**Почему это работает?** Потому что для новой задачи обычно не нужно менять всё в книге — достаточно изменить несколько важных мест, и LoRA как раз находит эти места и добавляет туда маленькие адаптивные заметки!

**Визуализация (HyperFrames, 43 с):** сцена 1 — full fine-tuning обновляет всю W (d × k), LoRA замораживает W и учит ΔW = B·A с рангом r ≪ min(d, k); сцена 2 — экономия параметров и памяти (слой 4096×4096: 16 777 216 → 65 536, LLaMA-7B: ≈ 112 GB → ≈ 28.1 GB); сцена 3 — прямой проход h = Wx + (α/r)·BAx со стартом B = 0, слияние W + B·A на инференсе и сменные адаптеры под задачи.

![LoRA: заморозка W, ΔW = B·A, слияние на инференсе и сменные адаптеры](./assets/visualizations/lora-low-rank-delta-w.gif)

*Полная версия: [MP4 1080p](./assets/visualizations/lora-low-rank-delta-w.mp4) · сториборд и исходники сцен: [`visualizations/hyperframes/`](./visualizations/hyperframes/storyboard.md).*

---
