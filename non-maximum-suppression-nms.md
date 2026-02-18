# Non-Maximum Suppression (NMS) и Современные End-to-End Детекторы

## Table of Contents

1. [Введение](#введение)
2. [Что такое Non-Maximum Suppression (NMS)?](#что-такое-non-maximum-suppression-nms)
3. [Алгоритм NMS](#алгоритм-nms)
4. [Agnostic NMS (Class-Agnostic NMS)](#agnostic-nms-class-agnostic-nms)
5. [Проблемы NMS в Production](#проблемы-nms-в-production)
6. [End-to-End Детекция без NMS](#end-to-end-детекция-без-nms)
7. [Distribution Focal Loss (DFL): Что Это и Зачем Нужен](#distribution-focal-loss-dfl-что-это-и-зачем-нужен)
8. [YOLO26: Удаление NMS](#yolo26-удаление-nms)
9. [Transformer-based Детекторы: DETR и его потомки](#transformer-based-детекторы-detr-и-его-потомки)
10. [Сравнение Подходов](#сравнение-подходов)
11. [Практические Рекомендации](#практические-рекомендации)
12. [Текущее Состояние (2023-2026)](#текущее-состояние-2023-2026)
13. [References](#references)

---

## Введение

**Non-Maximum Suppression (NMS)** — это классический алгоритм постобработки в задачах детекции объектов, который использовался десятилетиями для удаления дублирующихся детекций. Однако в последние годы произошла революция в архитектуре детекторов: современные модели, такие как **YOLO26** (2026) и **DETR** (2020), полностью отказались от NMS, используя end-to-end подходы, где модель сама учится выдавать финальные детекции без постобработки.

### Ключевые Характеристики

- **Традиционный подход**: Модель генерирует множество перекрывающихся боксов → NMS удаляет дубликаты
- **Современный подход**: Модель учится выдавать финальные детекции напрямую → NMS не нужен
- **Преимущества end-to-end**: Детерминированное время инференса, упрощенный деплой, лучшая производительность на edge-устройствах

### Исторический Контекст

- **2000-е**: NMS становится стандартом для всех детекторов (R-CNN, YOLO, SSD)
- **2020**: DETR (DEtection TRansformer) — первый end-to-end детектор без NMS
- **2021-2025**: Развитие DETR-подобных архитектур (Deformable DETR, DINO, RT-DETR)
- **2026**: YOLO26 полностью удаляет NMS, переходя на end-to-end архитектуру

---

## Что такое Non-Maximum Suppression (NMS)?

**Non-Maximum Suppression (NMS)** — это алгоритм постобработки, который удаляет дублирующиеся детекции объектов, оставляя только один наиболее уверенный бокс для каждого объекта.

### Проблема, которую решает NMS

Традиционные детекторы объектов (YOLO, SSD, R-CNN) генерируют **множество перекрывающихся боксов** для одного и того же объекта. Это происходит по нескольким причинам:

1. **Многоуровневая детекция**: Модель делает предсказания на разных масштабах и пространственных локациях одновременно
2. **Anchor-based подходы**: Генерируется большое количество кандидатных боксов вокруг каждой локации
3. **Grid-based детекция**: Когда объект находится на границе нескольких grid-ячеек, несколько ячеек могут предсказать бокс для этого объекта

**Результат**: Сырой вывод модели содержит несколько боксов для одного объекта, каждый со своим confidence score.

### Простая Аналогия

**Как бы я объяснил это 5-летнему ребенку?**

Представь, что ты ищешь игрушку в комнате. Ты можешь указать на нее несколько раз разными способами: "Вот она!", "Там!", "Смотри, игрушка!". Все эти указания говорят об одной и той же игрушке. NMS — это как умный помощник, который говорит: "Хорошо, я понял, это одна игрушка. Давай оставим только одно самое лучшее указание и забудем про остальные."

---

## Алгоритм NMS

### Математическая Формулировка

Пусть у нас есть множество детекций $\mathcal{D} = \{d_1, d_2, ..., d_N\}$, где каждая детекция $d_i$ содержит:
- Координаты бокса: $b_i = [x_1, y_1, x_2, y_2]$
- Confidence score: $s_i \in [0, 1]$
- Класс объекта: $c_i$

**Алгоритм NMS:**

1. **Thresholding**: Отфильтровать боксы с низким confidence:
   $$\mathcal{D}_{filtered} = \{d_i \in \mathcal{D} : s_i > \tau_{conf}\}$$

2. **Sorting**: Отсортировать по убыванию confidence:
   $$\mathcal{D}_{sorted} = \text{sort}(\mathcal{D}_{filtered}, \text{key}=s_i, \text{reverse}=\text{True})$$

3. **Iterative Suppression**: Пока $\mathcal{D}_{sorted}$ не пусто:
   - Выбрать бокс $d_{max}$ с максимальным confidence
   - Добавить $d_{max}$ в финальный результат $\mathcal{D}_{final}$
   - Удалить $d_{max}$ из $\mathcal{D}_{sorted}$
   - Удалить все боксы $d_j$ из $\mathcal{D}_{sorted}$, для которых:
     $$\text{IoU}(b_{max}, b_j) > \tau_{nms}$$
   
   где $\text{IoU}(b_i, b_j)$ — Intersection over Union между двумя боксами.

### Intersection over Union (IoU)

IoU измеряет перекрытие между двумя боксами:

$$\text{IoU}(b_i, b_j) = \frac{\text{Area}(b_i \cap b_j)}{\text{Area}(b_i \cup b_j)}$$

где:
- $b_i \cap b_j$ — область пересечения двух боксов
- $b_i \cup b_j$ — область объединения двух боксов

IoU принимает значения от 0 (нет перекрытия) до 1 (полное перекрытие).

### Реализация NMS

```python
import numpy as np
from typing import List, Tuple

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Вычисляет IoU между двумя боксами.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU значение
    """
    # Вычисляем координаты пересечения
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Площадь пересечения
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        inter_area = 0
    else:
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Площади боксов
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Площадь объединения
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def nms(boxes: np.ndarray, 
        scores: np.ndarray, 
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0) -> np.ndarray:
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: Массив боксов [N, 4] в формате [x1, y1, x2, y2]
        scores: Массив confidence scores [N]
        iou_threshold: Порог IoU для подавления (обычно 0.5)
        score_threshold: Минимальный confidence score
    
    Returns:
        Индексы боксов, которые нужно оставить
    """
    # Шаг 1: Фильтрация по confidence
    valid_indices = scores > score_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    original_indices = np.where(valid_indices)[0]
    
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Шаг 2: Сортировка по confidence
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        # Выбираем бокс с максимальным confidence
        current_idx = sorted_indices[0]
        keep.append(original_indices[current_idx])
        
        if len(sorted_indices) == 1:
            break
        
        # Вычисляем IoU с остальными боксами
        current_box = boxes[current_idx]
        other_boxes = boxes[sorted_indices[1:]]
        
        ious = np.array([
            calculate_iou(current_box, other_box) 
            for other_box in other_boxes
        ])
        
        # Оставляем только боксы с низким перекрытием
        low_overlap = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][low_overlap]
    
    return np.array(keep, dtype=np.int32)

# Пример использования
boxes = np.array([
    [100, 100, 200, 200],  # Бокс 1
    [110, 110, 210, 210],  # Бокс 2 (перекрывается с боксом 1)
    [300, 300, 400, 400],  # Бокс 3 (отдельный объект)
    [105, 105, 205, 205],  # Бокс 4 (перекрывается с боксом 1)
])

scores = np.array([0.9, 0.7, 0.8, 0.6])

# Применяем NMS
keep_indices = nms(boxes, scores, iou_threshold=0.5)
print(f"Оставлены боксы: {keep_indices}")  # [0, 2] - боксы 1 и 3
```

### Визуализация Работы NMS

```
До NMS:
┌─────────────────────────────────────┐
│  [Box 1: conf=0.9]                  │
│    [Box 2: conf=0.7] ──┐           │
│      [Box 4: conf=0.6]  │           │
│                         │           │
│                         └─→ Все     │
│                            перекры-  │
│                            ваются    │
│                                      │
│  [Box 3: conf=0.8]                  │
│  (отдельный объект)                 │
└─────────────────────────────────────┘

После NMS (iou_threshold=0.5):
┌─────────────────────────────────────┐
│  [Box 1: conf=0.9] ✓                │
│  (Box 2 и Box 4 подавлены)          │
│                                      │
│  [Box 3: conf=0.8] ✓                │
└─────────────────────────────────────┘
```

---

## Agnostic NMS (Class-Agnostic NMS)

### Определение

**Agnostic NMS** (также называемый **Class-Agnostic NMS**) — это вариант NMS, который подавляет дублирующиеся детекции **независимо от класса объекта**. В отличие от стандартного NMS, который подавляет дубликаты только внутри одного класса, Agnostic NMS рассматривает все детекции одинаково, независимо от их предсказанного класса.

### Разница между Standard NMS и Agnostic NMS

**Standard NMS (Class-Specific NMS):**

```python
def class_specific_nms(boxes, scores, classes, iou_threshold=0.5):
    """
    NMS применяется отдельно для каждого класса.
    """
    unique_classes = np.unique(classes)
    keep = []
    
    for cls in unique_classes:
        # Фильтруем боксы только этого класса
        class_mask = classes == cls
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # Применяем NMS только внутри класса
        class_keep = nms(class_boxes, class_scores, iou_threshold)
        keep.extend(class_indices[class_keep])
    
    return np.array(keep)
```

**Agnostic NMS:**

```python
def agnostic_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS применяется ко всем боксам независимо от класса.
    """
    # Просто применяем NMS ко всем боксам
    return nms(boxes, scores, iou_threshold)
```

### Когда Использовать Agnostic NMS?

**Agnostic NMS полезен когда:**

1. **Перекрывающиеся объекты разных классов**: Если объекты разных классов могут перекрываться (например, человек держит сумку), Agnostic NMS может подавить один из них
2. **Мультиклассовая детекция с неопределенностью**: Когда модель не уверена в классе, но уверена в наличии объекта
3. **Упрощение постобработки**: Меньше кода, проще реализация

**Standard NMS предпочтительнее когда:**

1. **Четкое разделение классов**: Объекты разных классов не должны подавляться друг другом
2. **Точная классификация важна**: Нужно сохранить все классы, даже если они перекрываются

### Пример Различия

```python
# Ситуация: человек держит сумку (перекрываются)
boxes = np.array([
    [100, 100, 200, 300],  # Человек (класс 0), conf=0.9
    [150, 150, 250, 250],  # Сумка (класс 1), conf=0.8
])

scores = np.array([0.9, 0.8])
classes = np.array([0, 1])

# Standard NMS: Оба бокса останутся (разные классы)
keep_standard = class_specific_nms(boxes, scores, classes, iou_threshold=0.5)
# Результат: [0, 1] - оба бокса сохранены

# Agnostic NMS: Бокс сумки может быть подавлен (ниже confidence)
keep_agnostic = agnostic_nms(boxes, scores, iou_threshold=0.5)
# Результат: [0] - только человек сохранен
```

---

## Проблемы NMS в Production

### 1. Вычислительная Сложность

NMS требует сравнения всех пар боксов, что имеет сложность $O(N^2)$ в худшем случае:

```python
# Для N боксов нужно вычислить N*(N-1)/2 пар IoU
# Это становится проблемой при большом количестве детекций
```

**Проблемы:**
- **Непараллелизуемость**: NMS итеративен и сложно распараллелить
- **Непредсказуемое время**: Время выполнения зависит от количества объектов в сцене
- **Edge-устройства**: На CPU и edge-устройствах NMS может стать узким местом

### 2. Сложность Деплоя

NMS не является частью нейронной сети, поэтому:

- **Разные реализации**: Каждая платформа (TensorRT, CoreML, ONNX) требует своей реализации NMS
- **Несовместимость**: Экспорт модели часто ломается из-за необходимости добавления NMS отдельно
- **Ручная настройка**: Параметры NMS (thresholds) нужно настраивать вручную для каждого деплоя

### 3. Проблемы с Оптимизацией

- **AI-ускорители**: Специализированные чипы (TPU, NPU) оптимизированы для нейронных сетей, но не для NMS
- **Квантизация**: NMS сложно квантизовать, что усложняет оптимизацию модели
- **Градиенты**: NMS недифференцируем, что мешает end-to-end обучению

### 4. Непредсказуемость

- **Параметры**: Confidence threshold и IoU threshold нужно настраивать эмпирически
- **Разные датасеты**: Параметры, работающие на одном датасете, могут не работать на другом
- **Разные сцены**: Количество объектов в сцене влияет на поведение NMS

---

## End-to-End Детекция без NMS

### Философия End-to-End Подхода

**Вместо**: Модель генерирует много кандидатов → NMS очищает результат

**Теперь**: Модель учится генерировать финальные детекции напрямую

### Ключевые Принципы

1. **One-to-One Matching**: Каждый объект ассоциируется с одним предсказанием
2. **Learnable Queries**: Модель использует обучаемые запросы для генерации детекций
3. **Set Prediction**: Модель предсказывает множество детекций фиксированного размера
4. **Hungarian Matching**: Во время обучения используется Hungarian algorithm для сопоставления предсказаний с ground truth

### Преимущества End-to-End Подхода

1. **Детерминированное время**: Время инференса не зависит от количества объектов
2. **Упрощенный деплой**: Модель полностью самодостаточна
3. **Лучшая производительность**: Нет постобработки, которая может стать узким местом
4. **Дифференцируемость**: Весь pipeline дифференцируем, что улучшает обучение

---

## Distribution Focal Loss (DFL): Что Это и Зачем Нужен

### Что такое DFL?

**Distribution Focal Loss (DFL)** — это функция потерь, используемая в YOLOv8 и YOLOv9 для улучшения регрессии bounding boxes через предсказание распределения вероятностей вместо прямого предсказания координат.

### Зачем Нужен DFL?

**Проблема прямой регрессии:**

Традиционные детекторы предсказывают координаты бокса напрямую:
- Вход: Features из backbone
- Выход: $[x_1, y_1, x_2, y_2]$ — координаты бокса

**Проблема**: Прямая регрессия не может выразить **неопределенность** в локализации. Модель вынуждена выбрать одно значение, даже если она не уверена.

**Решение DFL:**

Вместо предсказания одной координаты, DFL предсказывает **распределение вероятностей** возможных значений координаты.

### Как Работает DFL

#### Шаг 1: Дискретизация Диапазона

Для каждой координаты (например, $x_1$) определяем диапазон возможных значений и разбиваем его на дискретные точки (bins):

$$x_1 \in \{x_{min}, x_{min}+\Delta, x_{min}+2\Delta, ..., x_{max}\}$$

где $\Delta = \frac{x_{max} - x_{min}}{n-1}$ — шаг дискретизации, а $n$ — количество bins.

**Типичное количество bins:**

В **YOLOv8** и **YOLOv9** обычно используется **16 bins** для DFL. Это значение является компромиссом между:
- **Точностью**: Больше bins → более точная локализация, но больше параметров
- **Вычислительной эффективностью**: Меньше bins → быстрее обучение и инференс

**Типичные значения:**
- **16 bins** — стандартное значение в YOLOv8/YOLOv9 (наиболее распространенное)
- **8 bins** — для более легких моделей или edge-устройств
- **32 bins** — для более точных моделей (редко используется из-за вычислительной стоимости)

**Пример**: Для координаты в диапазоне [0, 640] с 16 bins:
- Bin 0: [0, 40)
- Bin 1: [40, 80)
- ...
- Bin 15: [600, 640]

Каждый bin представляет диапазон значений, а модель предсказывает вероятность того, что истинная координата попадает в каждый bin.

#### Шаг 2: Предсказание Распределения

Модель предсказывает вероятности для каждого дискретного значения:

$$P(x_1) = [p_1, p_2, ..., p_n]$$

где:
- $p_i \geq 0$ — вероятность того, что $x_1 = x_i$
- $\sum_{i=1}^{n} p_i = 1$ — распределение нормализовано

#### Шаг 3: Восстановление Координаты

Финальная координата вычисляется как математическое ожидание:

$$x_1 = \sum_{i=1}^{n} p_i \cdot x_i = \mathbb{E}[P(x_1)]$$

#### Шаг 4: Focal Loss

Используется Focal Loss для фокусировки на сложных примерах:

$$\text{DFL}(p, y) = -\sum_{i=1}^{n} \left((1-p_i)^\gamma \log(p_i)\right) \cdot \mathbf{1}[y_i = 1]$$

где:
- $p_i$ — предсказанная вероятность для значения $x_i$
- $y_i$ — ground truth (1 если $x_i$ близко к истинному значению, 0 иначе)
- $\gamma$ — фокусный параметр (обычно 2.0)
- $(1-p_i)^\gamma$ — фокусный вес, который увеличивает важность сложных примеров

### Преимущества DFL

1. **Выражение неопределенности**: Модель может выразить неопределенность через распределение
2. **Более точная локализация**: Распределение позволяет модели предсказывать более точные координаты
3. **Лучшая регрессия**: Focal Loss фокусируется на сложных примерах

### Недостатки DFL

1. **Сложность**: Добавляет дополнительную сложность в pipeline
2. **Зависимость от постобработки**: Увеличивает зависимость от NMS и других постобработок
3. **Усложнение обучения**: Усложняет обучение четких one-to-one соответствий
4. **Вычислительная стоимость**: Требует больше вычислений

### Почему YOLO26 Удалил DFL?

YOLO26 удалил DFL для поддержки **end-to-end подхода**:

- **Упрощение**: Прямая регрессия проще и быстрее
- **End-to-end обучение**: Прямая регрессия лучше подходит для обучения one-to-one соответствий
- **Edge-оптимизация**: Для edge-устройств проще предсказывать координаты напрямую
- **Меньше зависимостей**: Не требует сложной постобработки

### Пример Кода DFL

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss для регрессии координат
    
    Args:
        num_bins: Количество bins для дискретизации (обычно 16)
        gamma: Фокусный параметр для Focal Loss (обычно 2.0)
    """
    def __init__(self, num_bins=16, gamma=2.0):
        super().__init__()
        self.num_bins = num_bins
        self.gamma = gamma
    
    def forward(self, pred_distribution, target_value, value_range):
        """
        Args:
            pred_distribution: [B, num_bins] - предсказанное распределение
            target_value: [B] - истинное значение координаты
            value_range: (min_val, max_val) - диапазон значений
        """
        # Дискретизация диапазона
        min_val, max_val = value_range
        bin_centers = torch.linspace(min_val, max_val, self.num_bins)
        
        # Создаем ground truth распределение (one-hot или soft)
        target_dist = self._create_target_distribution(
            target_value, bin_centers
        )
        
        # Focal Loss
        focal_weight = (1 - pred_distribution) ** self.gamma
        loss = -focal_weight * target_dist * torch.log(pred_distribution + 1e-8)
        
        return loss.sum(dim=1).mean()
    
    def _create_target_distribution(self, target_value, bin_centers):
        """
        Создает target распределение (можно использовать soft assignment)
        """
        # Простой вариант: one-hot encoding для ближайшего bin
        distances = torch.abs(bin_centers.unsqueeze(0) - target_value.unsqueeze(1))
        nearest_bin = distances.argmin(dim=1)
        
        target_dist = torch.zeros_like(distances)
        target_dist.scatter_(1, nearest_bin.unsqueeze(1), 1.0)
        
        return target_dist
    
    def decode_coordinate(self, pred_distribution, value_range):
        """
        Декодирует координату из распределения
        """
        min_val, max_val = value_range
        bin_centers = torch.linspace(min_val, max_val, self.num_bins)
        
        # Математическое ожидание
        coordinate = (pred_distribution * bin_centers.unsqueeze(0)).sum(dim=1)
        
        return coordinate

# Пример использования
dfl = DistributionFocalLoss(num_bins=16, gamma=2.0)

# Предсказание: распределение для координаты x1
pred_dist = torch.softmax(torch.randn(32, 16), dim=1)  # [batch, num_bins]
target_x1 = torch.rand(32) * 640  # истинные значения x1

# Loss
loss = dfl(pred_dist, target_x1, (0, 640))

# Декодирование координаты
decoded_x1 = dfl.decode_coordinate(pred_dist, (0, 640))
```

---

## YOLO26: Удаление NMS

### Архитектурные Изменения

**YOLO26** (январь 2026) полностью переработал архитектуру для устранения необходимости в NMS:

### Краткое Резюме Ключевых Инноваций

**Как бы я объяснил это 5-летнему ребенку?**

YOLO26 — это как умный ученик, который учится правильно рисовать картинки с самого начала, а не рисует много неправильных и потом исправляет их. Он использует специальные "трюки" для обучения:

1. **DFL (удален)**: Раньше модель пыталась угадать, где находится объект, предсказывая много возможных мест. Теперь она просто говорит точное место.

2. **MuSGD**: Это как умный учитель, который знает, как лучше объяснить материал — иногда быстро, иногда медленно, в зависимости от того, что нужно ученику.

3. **ProgLoss**: Сначала модель учится видеть все объекты (даже если не очень точно), а потом постепенно учится быть более точной.

4. **STAL**: Модель специально учится видеть даже очень маленькие объекты, которые раньше часто пропускала.

#### 1. Learnable Query-Based Detection

YOLO26 использует обучаемые запросы (learnable queries), которые помогают модели фокусироваться на генерации одного уверенного предсказания для каждого объекта:

```python
# Концептуальная архитектура
class YOLO26DetectionHead(nn.Module):
    def __init__(self, num_queries=300):
        super().__init__()
        # Обучаемые запросы для детекции
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.decoder = TransformerDecoder(...)
    
    def forward(self, features):
        # Каждый query генерирует одну детекцию
        detections = self.decoder(self.queries, features)
        return detections  # Уже финальные детекции, без NMS
```

#### 2. STAL (Small-Target-Aware Label Assignment)

**STAL (Small-Target-Aware Label Assignment)** — это улучшенная стратегия назначения меток (label assignment) во время обучения, которая специально учитывает маленькие объекты.

**Проблема, которую решает STAL:**

В традиционных методах назначения меток (например, TAL - Task Alignment Learning) очень маленькие объекты (меньше ~8 пикселей на изображении 640×640) часто не получают ни одного anchor assignment. Это означает, что модель не получает supervision для этих объектов и не может научиться их детектировать.

**Решение STAL:**

STAL гарантирует, что объекты меньше 8 пикселей получают минимум 4 anchor assignments, обеспечивая достаточный supervision даже для самых маленьких объектов.

```python
# Концептуальная реализация STAL
def stal_label_assignment(predictions, ground_truth, min_size=8):
    """
    Small-Target-Aware Label Assignment
    """
    assignments = []
    
    for gt_box in ground_truth:
        box_size = calculate_size(gt_box)
        
        if box_size < min_size:
            # Для маленьких объектов: минимум 4 assignments
            min_assignments = 4
            assignments.extend(
                find_top_k_anchors(gt_box, predictions, k=min_assignments)
            )
        else:
            # Для больших объектов: стандартное назначение
            assignments.extend(
                standard_assignment(gt_box, predictions)
            )
    
    return assignments
```

**Преимущества STAL:**

- Улучшает детекцию маленьких объектов
- Стабилизирует обучение
- Особенно важно для edge-приложений: аэрофотосъемка, робототехника, IoT, где объекты часто маленькие или далекие

#### 3. Удаление DFL (Distribution Focal Loss)

**DFL (Distribution Focal Loss)** использовался в предыдущих версиях YOLO (YOLOv8, YOLOv9) для улучшения регрессии bounding boxes.

**Что такое DFL?**

Вместо прямого предсказания координат бокса (например, $x_1, y_1, x_2, y_2$), DFL предсказывает **распределение вероятностей** возможных локаций для каждой координаты.

**Как работает DFL:**

1. **Дискретизация диапазона**: Каждая координата (например, $x_1$) разбивается на дискретные значения: $[x_{min}, x_{min}+1, ..., x_{max}]$

2. **Предсказание распределения**: Модель предсказывает вероятности $p_i$ для каждого дискретного значения:
   $$P(x_1) = [p_1, p_2, ..., p_n]$$
   где $\sum_{i=1}^{n} p_i = 1$

3. **Восстановление координаты**: Финальная координата вычисляется как взвешенная сумма:
   $$x_1 = \sum_{i=1}^{n} p_i \cdot x_i$$

4. **Focal Loss**: Используется Focal Loss для фокусировки на сложных примерах:
   $$\text{DFL}(p, y) = -\sum_{i} \left((1-p_i)^\gamma \log(p_i)\right) \cdot \mathbf{1}[y_i = 1]$$
   где $\gamma$ — фокусный параметр (обычно 2.0)

**Зачем нужен DFL?**

- **Более точная локализация**: Распределение позволяет модели выражать неопределенность в локализации
- **Лучшая регрессия**: Модель может предсказывать более точные координаты через распределение

**Почему YOLO26 удалил DFL?**

1. **Сложность**: DFL добавляет дополнительную сложность в pipeline
2. **Зависимость от постобработки**: DFL увеличивает зависимость от постобработки (например, NMS)
3. **Усложнение one-to-one соответствий**: DFL усложняет обучение четких one-to-one соответствий между объектами и предсказаниями, необходимых для end-to-end подхода
4. **Упрощение для edge**: Для edge-устройств проще предсказывать координаты напрямую

**Альтернатива в YOLO26:**

YOLO26 использует прямую регрессию координат, что проще и лучше подходит для end-to-end обучения.

#### 4. ProgLoss (Progressive Loss Balancing)

**ProgLoss (Progressive Loss Balancing)** — это прогрессивная балансировка потерь, которая динамически изменяет веса разных компонентов loss function в процессе обучения.

**Проблема, которую решает ProgLoss:**

YOLO26 использует **два detection head** во время обучения:

1. **One-to-One Head**: Используется на inference, учит модель ассоциировать каждый объект с одним предсказанием
2. **One-to-Many Head**: Используется только во время обучения, позволяет нескольким предсказаниям ассоциироваться с одним объектом (более плотный supervision)

**Проблема**: Важность каждого head меняется в процессе обучения:
- **Ранние стадии**: One-to-many head более полезен (стабилизирует обучение, улучшает recall)
- **Поздние стадии**: One-to-one head более важен (выравнивает обучение с inference поведением)

**Решение ProgLoss:**

ProgLoss динамически изменяет веса каждого head в процессе обучения:

```python
# Концептуальная реализация ProgLoss
def progressive_loss_balancing(epoch, total_epochs, 
                               one_to_one_weight_init=0.0,
                               one_to_many_weight_init=1.0):
    """
    Прогрессивная балансировка весов loss
    """
    # Прогрессия от one-to-many к one-to-one
    progress = epoch / total_epochs  # от 0 до 1
    
    # Веса меняются линейно (можно использовать другие функции)
    one_to_one_weight = progress * 1.0  # от 0 до 1
    one_to_many_weight = (1 - progress) * 1.0  # от 1 до 0
    
    return {
        'one_to_one_weight': one_to_one_weight,
        'one_to_many_weight': one_to_many_weight
    }

# Пример использования
total_loss = (
    prog_weights['one_to_one_weight'] * one_to_one_loss +
    prog_weights['one_to_many_weight'] * one_to_many_loss
)
```

**Математическая формулировка:**

$$\mathcal{L}_{total}(t) = w_{1:1}(t) \cdot \mathcal{L}_{1:1} + w_{1:M}(t) \cdot \mathcal{L}_{1:M}$$

где:
- $t$ — текущая эпоха (или шаг обучения)
- $w_{1:1}(t)$ — вес one-to-one head (увеличивается со временем)
- $w_{1:M}(t)$ — вес one-to-many head (уменьшается со временем)
- $\mathcal{L}_{1:1}$ — loss от one-to-one head
- $\mathcal{L}_{1:M}$ — loss от one-to-many head

**Преимущества ProgLoss:**

- **Плавная сходимость**: Модель учится в правильном порядке
- **Стабильность**: Меньше нестабильных training runs
- **Консистентность**: Более предсказуемое финальное поведение
- **Выравнивание**: Обучение лучше выравнивается с inference поведением

#### 5. MuSGD (Momentum Update SGD)

**MuSGD** — это новый оптимизатор, разработанный специально для YOLO26, который комбинирует идеи из **Muon** (метод для обучения больших языковых моделей) с классическим **SGD**.

**Проблема стандартного SGD:**

Стандартный SGD может иметь проблемы со стабильностью и сходимостью, особенно для:
- Больших моделей
- End-to-end детекторов
- Сложных loss functions

**Решение MuSGD:**

MuSGD использует **гибридную стратегию обновления**:

1. **Некоторые параметры** обновляются с использованием комбинации Muon-inspired updates и SGD
2. **Другие параметры** обновляются только с SGD

Это позволяет добавить структуру в процесс оптимизации, сохраняя при этом robustness и generalization свойства SGD.

**Концептуальная реализация:**

```python
# Концептуальная реализация MuSGD
class MuSGD:
    def __init__(self, params, lr=0.01, momentum=0.9, muon_weight=0.5):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.muon_weight = muon_weight
        self.velocity = {}  # Momentum buffer
    
    def step(self, gradients):
        for param, grad in zip(self.params, gradients):
            # Определяем, использовать ли Muon-inspired update
            use_muon = should_use_muon(param)
            
            if use_muon:
                # Muon-inspired update с momentum
                if param not in self.velocity:
                    self.velocity[param] = torch.zeros_like(param)
                
                self.velocity[param] = (
                    self.momentum * self.velocity[param] + 
                    self.muon_weight * grad
                )
                param.data -= self.lr * self.velocity[param]
            else:
                # Стандартный SGD update
                param.data -= self.lr * grad
```

**Вдохновение от Muon:**

MuSGD вдохновлен методом **Muon**, который использовался в обучении больших языковых моделей (например, Moonshot AI's Kimi K2), демонстрируя улучшенное поведение обучения через более структурированные обновления параметров.

**Преимущества MuSGD:**

- **Более плавная оптимизация**: Меньше колебаний в процессе обучения
- **Быстрая сходимость**: Модель сходится быстрее
- **Предсказуемость**: Более предсказуемое поведение обучения
- **Масштабируемость**: Работает хорошо для разных размеров моделей (от Nano до Extra Large)

**Результаты:**

MuSGD является ключевой частью того, почему YOLO26 легче обучать и более надежен в масштабе.

### Результаты

**Производительность YOLO26:**

- **43% ускорение на CPU** по сравнению с YOLO11
- **Детерминированное время инференса**: не зависит от количества объектов
- **Упрощенный экспорт**: модель полностью самодостаточна
- **Лучшая производительность на edge**: оптимизирован для ограниченных ресурсов

### Код Экспорта YOLO26

```python
from ultralytics import YOLO

# Загрузка модели
model = YOLO('yolo26n.pt')  # nano версия

# Экспорт - NMS уже включен в модель!
model.export(format='onnx')  # Экспортируется без отдельного NMS
model.export(format='tensorrt')  # Работает из коробки
model.export(format='coreml')  # Полностью самодостаточная модель
```

---

## Transformer-based Детекторы: DETR и его потомки

### DETR (DEtection TRansformer)

**DETR** (2020) — первый полностью end-to-end детектор без NMS, основанный на архитектуре Transformer.

#### Архитектура DETR

```
Изображение → CNN Backbone → Transformer Encoder → Transformer Decoder → Предсказания
                                                      ↑
                                              Обучаемые Object Queries
```

**Ключевые Компоненты:**

1. **CNN Backbone**: Извлекает features из изображения (ResNet-50)
2. **Transformer Encoder**: Обрабатывает features
3. **Transformer Decoder**: Использует обучаемые object queries для генерации детекций
4. **Set Prediction Head**: Предсказывает фиксированное количество детекций (обычно 100)

#### Object Queries

DETR использует фиксированное количество (например, 100) обучаемых запросов:

```python
# Концептуальная реализация
class DETR(nn.Module):
    def __init__(self, num_queries=100):
        super().__init__()
        # Обучаемые queries - каждый будет генерировать одну детекцию
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = Transformer(...)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 для "no object"
        self.bbox_embed = nn.Linear(hidden_dim, 4)  # [x, y, w, h]
    
    def forward(self, image_features):
        # Object queries
        queries = self.query_embed.weight  # [100, hidden_dim]
        
        # Transformer decoder
        decoder_output = self.transformer.decoder(queries, image_features)
        
        # Предсказания
        class_logits = self.class_embed(decoder_output)  # [100, num_classes+1]
        bbox_coords = self.bbox_embed(decoder_output)  # [100, 4]
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_coords
        }
```

#### Hungarian Matching

Во время обучения DETR использует **Hungarian algorithm** для оптимального сопоставления предсказаний с ground truth:

```python
def hungarian_matching(pred_logits, pred_boxes, target_classes, target_boxes):
    """
    Находит оптимальное сопоставление между предсказаниями и ground truth.
    """
    # Вычисляем стоимость для каждой пары (prediction, target)
    cost_matrix = compute_cost_matrix(pred_logits, pred_boxes, 
                                      target_classes, target_boxes)
    
    # Hungarian algorithm находит минимальную стоимость
    matched_indices = hungarian_algorithm(cost_matrix)
    
    return matched_indices
```

**Преимущества:**
- Каждое предсказание сопоставляется с максимум одним объектом
- Модель учится генерировать финальные детекции
- Нет необходимости в NMS

#### Проблемы DETR и Решения

**Проблемы оригинального DETR:**

1. **Медленная сходимость**: Требуется много эпох для обучения
2. **Плохая работа с маленькими объектами**: Transformer плохо работает с маленькими объектами
3. **Высокая вычислительная сложность**: $O(N^2)$ для attention

**Решения в последующих версиях:**

1. **Deformable DETR (2020)**: Использует deformable attention для ускорения
2. **DINO (2022)**: Улучшенная архитектура с better initialization
3. **RT-DETR (2023)**: Real-time версия для практического использования

### Современные Transformer-based Детекторы

#### RT-DETR (Real-Time DETR)

RT-DETR оптимизирован для real-time детекции:

- **Hybrid Encoder**: Комбинация CNN и Transformer
- **IoU-aware Query Selection**: Улучшенный выбор queries
- **Real-time производительность**: ~30 FPS на GPU

#### DINO (DETR with Improved Denoising Anchor Boxes)

DINO улучшает DETR через:

- **Denoising Training**: Использует noisy ground truth для стабилизации обучения
- **Mixed Query Selection**: Комбинация content и positional queries
- **Look Forward Twice**: Улучшенная регрессия боксов

---

## Сравнение Подходов

### Традиционный Pipeline (с NMS)

```
Изображение → Backbone → Detection Head → Множество боксов → NMS → Финальные детекции
                                                                    ↑
                                                          Постобработка (недифференцируема)
```

**Характеристики:**
- ✅ Простая архитектура
- ✅ Быстрое обучение
- ❌ Требует NMS
- ❌ Непредсказуемое время инференса
- ❌ Сложный деплой

### End-to-End Pipeline (без NMS)

```
Изображение → Backbone → Transformer/Query-based Head → Финальные детекции
                                                          ↑
                                              Уже финальные, без постобработки
```

**Характеристики:**
- ✅ Нет NMS
- ✅ Детерминированное время
- ✅ Упрощенный деплой
- ✅ Дифференцируемый pipeline
- ❌ Более сложная архитектура
- ❌ Может требовать больше эпох обучения

### Таблица Сравнения

| Характеристика | Традиционный (с NMS) | End-to-End (без NMS) |
|----------------|---------------------|----------------------|
| **Время инференса** | Зависит от количества объектов | Детерминированное |
| **Деплой** | Сложный (нужна отдельная реализация NMS) | Простой (модель самодостаточна) |
| **Дифференцируемость** | NMS недифференцируем | Полностью дифференцируем |
| **Edge-устройства** | NMS может быть узким местом | Оптимизирован для edge |
| **Параметры** | Нужно настраивать thresholds | Меньше гиперпараметров |
| **Архитектура** | Простая | Более сложная |
| **Обучение** | Быстрое | Может требовать больше эпох |

---

## Практические Рекомендации

### Когда Использовать Традиционный Подход (с NMS)?

1. **Legacy системы**: Если у вас уже есть рабочая система с NMS
2. **Простота**: Если нужна простая архитектура для быстрого прототипирования
3. **Ограниченные ресурсы**: Если нет ресурсов для переобучения модели

### Когда Использовать End-to-End Подход (без NMS)?

1. **Production деплой**: Когда важна надежность и простота деплоя
2. **Edge-устройства**: Когда нужна оптимизация для ограниченных ресурсов
3. **Real-time приложения**: Когда важна предсказуемость времени инференса
4. **Новые проекты**: Когда можно выбрать архитектуру с нуля

### Миграция с NMS на End-to-End

Если вы хотите мигрировать с традиционного подхода на end-to-end:

1. **Выберите архитектуру**: DETR, RT-DETR, или YOLO26
2. **Переобучите модель**: End-to-end модели требуют специального обучения
3. **Обновите pipeline**: Уберите NMS из постобработки
4. **Протестируйте**: Убедитесь, что качество не ухудшилось

### Настройка Параметров (для традиционного подхода)

Если вы все еще используете NMS:

```python
# Рекомендуемые параметры для разных сценариев

# Высокая точность (меньше ложных срабатываний)
nms_iou_threshold = 0.3
confidence_threshold = 0.5

# Баланс точности и recall
nms_iou_threshold = 0.5
confidence_threshold = 0.25

# Высокий recall (больше детекций)
nms_iou_threshold = 0.7
confidence_threshold = 0.1
```

---

## Текущее Состояние (2023-2026)

### Тренды

1. **Отказ от NMS**: Все больше моделей переходят на end-to-end подходы
2. **Query-based детекция**: Становится стандартом для новых архитектур
3. **Edge-оптимизация**: Фокус на производительности на edge-устройствах
4. **Упрощение деплоя**: Модели становятся более самодостаточными

### Современные Модели (2023-2026)

#### 2023
- **RT-DETR**: Real-time transformer-based детектор
- **YOLOv8**: Еще использует NMS, но улучшенная архитектура

#### 2024
- **YOLO11**: Улучшения в архитектуре, но все еще использует NMS
- **DINOv2**: Улучшенная версия DINO

#### 2025-2026
- **YOLO26**: Полностью end-to-end, без NMS
- **Новые архитектуры**: Продолжается развитие query-based подходов

### Будущее

**Ожидаемые направления:**

1. **Полный переход на end-to-end**: NMS станет устаревшим
2. **Улучшение query-based подходов**: Более эффективные архитектуры
3. **Специализированные модели**: Модели для конкретных задач (edge, real-time, high-accuracy)
4. **Автоматическая оптимизация**: Автоматический выбор архитектуры под задачу

---

## References

### Связанные Документы

- **[Unscented Kalman Filter and Tracking](./unscented-kalman-filter-and-tracking.md)**: Методы отслеживания объектов, которые могут использоваться вместе с детекцией
- **[Deep Reinforcement Learning](./deep-reinforcement-learning.md)**: End-to-end обучение в RL, похожие принципы

### Ключевые Статьи

1. **DETR (2020)**: Carion, N., et al. "End-to-End Object Detection with Transformers." ECCV 2020.
2. **Deformable DETR (2020)**: Zhu, X., et al. "Deformable DETR: Deformable Transformers for End-to-End Object Detection." ICLR 2021.
3. **DINO (2022)**: Zhang, H., et al. "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection." ICLR 2023.
4. **RT-DETR (2023)**: Yao, Z., et al. "RT-DETR: Real-Time Detection Transformer." CVPR 2023.
5. **YOLO26 (2026)**: Ultralytics. "YOLO26: End-to-End Object Detection." 2026.

### Полезные Ресурсы

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [Non-Maximum Suppression Explained](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)

---

## Заключение

**Non-Maximum Suppression (NMS)** был стандартом для детекции объектов на протяжении десятилетий, но современные end-to-end архитектуры показывают, что можно обойтись без него. **YOLO26** и **DETR**-подобные модели демонстрируют, что модель может научиться генерировать финальные детекции напрямую, что упрощает деплой и улучшает производительность, особенно на edge-устройствах.

Переход от NMS к end-to-end подходам представляет собой фундаментальный сдвиг в том, как мы думаем о детекции объектов: вместо постобработки для очистки результатов, модель сама учится выдавать чистые результаты.

---

**Как бы я объяснил это 5-летнему ребенку?**

Раньше модель была как художник, который рисовал много похожих картинок одного и того же объекта, а потом помощник (NMS) выбирал лучшую и выбрасывал остальные. Теперь модель стала умнее — она сразу рисует одну правильную картинку, и помощник больше не нужен!
