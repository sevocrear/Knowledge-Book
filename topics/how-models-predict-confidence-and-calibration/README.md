---
title: "Уверенность, калибровка и неопределённость"
description: "Logits→softmax/sigmoid, reliability diagrams, ECE/Brier, temperature scaling и aleatoric/epistemic uncertainty (ensembles, MC Dropout)."
tags:
  - kb/topic
  - domain/ml-foundations
  - concept/calibration
  - concept/uncertainty
  - concept/confidence
aliases:
  - calibration
  - ECE
  - temperature scaling
  - model confidence
  - uncertainty estimation
related:
  - roc-curve-and-roc-auc
  - classification-losses-cross-entropy-focal-loss
  - ensemble-methods-model-combination
  - ml-system-design-for-cv-and-nlp
status: canonical
lang: ru
type: topic
slug: how-models-predict-confidence-and-calibration
updated: 2026-09-18
---
# Уверенность, калибровка и неопределённость в классификации: как модели предсказывают уверенность

## Оглавление
1. [Введение](#введение)
2. [Откуда берётся “уверенность”: logits -> вероятности](#откуда-берётся-уверенность-logits---вероятности)
3. [Почему это называют вероятностью](#почему-это-называют-вероятностью)
4. [Когда вероятности “врут”: проблема калибровки](#когда-вероятности-врут-проблема-калибровки)
5. [Как измеряют калибровку](#как-измеряют-калибровку)
6. [Temperature Scaling: базовая калибровка](#temperature-scaling-базовая-калибровка)
7. [Уверенность vs неопределённость (uncertainty)](#уверенность-vs-неопределённость-uncertainty)
8. [Интуитивное объяснение “для 5-летнего”](#интуитивное-объяснение-для-5-летнего)
9. [Примеры кода (PyTorch)](#примеры-кода-pytorch)
10. [Источники](#источники)

---

## Как объяснить 5-летнему ребёнку
Представь, что модель — это учитель, который смотрит на картинку и говорит: “это котик, потому что я думаю на `0.8`”. Эта цифра получается из того, как уверенно “сердце” учителя отвечает на вопрос. Но иногда учитель может быть слишком уверенным или слишком осторожным — тогда мы можем подправить его “шкалу уверенности”, чтобы `0.8` означало “примерно так и получается в реальности”.


**Визуализация (HyperFrames, 42 с):** сцена 1 — логиты → softmax и overconfident-уверенность 0.95 при 70% верных ответов; сцена 2 — reliability diagram (conf vs acc по бинам) и ECE; сцена 3 — temperature scaling softmax(z/T): T меняет остроту распределения, но не argmax.

![Калибровка: softmax, reliability diagram, temperature scaling](./assets/visualizations/softmax-calibration-temperature-scaling.gif)

*Полная версия: [MP4 1080p](./assets/visualizations/softmax-calibration-temperature-scaling.mp4) · сториборд и исходники сцен: [`visualizations/hyperframes/`](./visualizations/hyperframes/storyboard.md).*

---

## Введение

В классификации “уверенность” обычно означает число из диапазона $[0,1]$, которое модель сопоставляет одному классу (или “вероятность класса”). В практических системах это число используют для:

- порогового решения (например, “показывать метку только если уверенность > 0.7”);
- ранжирования (например, топ-k объектов);
- управления рисками (например, “если уверенность низкая — отправить на человека”);
- оценки неопределенности (uncertainty estimation).

Важно: “уверенность” и “правильность этой уверенности” — не одно и то же. Чаще всего число получается из `logits` и функции активации, но эти “вероятности” могут быть некалиброваны.

---

## Откуда берётся “уверенность”: logits -> вероятности

Почти все современные классификаторы устроены так:

1. Сеть (CNN/Transformer/и т.д.) извлекает признаки из входа: $h(x)$.
2. “Головка” классификации выдаёт для каждого класса число $z_k(x)$ — это **logits**.
3. Дальше logits преобразуются в “оценки вероятностей”.

### Многоклассовая классификация: softmax

Если классов $K$, то модель обычно делает:

$$
p_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
$$

Тогда “уверенность в классе $k$” — это $p_k$.

### Бинарная классификация: sigmoid

Для бинарного случая часто один логит $z(x)$ (или эквивалентно два logits), и делают:

$$
p = \sigma(z) = \frac{1}{1+e^{-z}}
$$

Здесь $p$ — вероятность класса “1” (часто используется для positive класса).

### Что такое logits интуитивно: log-odds

В бинарном случае logits напрямую связаны с **log-odds**:

$$
z = \log\frac{p}{1-p}
$$

То есть logits — это “логарифм отношения шансов”, а sigmoid просто переводит это отношение обратно в вероятность.

---

## Почему это называют вероятностью

Чаще всего модель обучают так, чтобы $p(y|x)$ была хорошим приближением к истинной условной вероятности.

### Обучение с Cross-Entropy

Для многоклассовой классификации кросс-энтропия:

$$
\mathcal{L} = -\log p_{y}(x)
$$

где $p_y(x)$ — вероятность истинного класса (полученная через softmax).

Если модель достаточно выразительная и обучение стабильное, то минимизация `cross-entropy` подтягивает “оценки” в сторону правдоподобных вероятностей.

Но это не гарантирует идеальную калибровку (см. ниже).

---

## Когда вероятности “врут”: проблема калибровки

Даже если модель оптимизировала cross-entropy, её $p_k$ могут быть:

- **слишком уверенными** (overconfident): например, при предсказанной уверенности 0.95 на деле правильными оказываются только 70% ответов;
- **слишком осторожными** (underconfident).

Типичные причины:

1. Несовпадение train/test (domain shift, OOD).
2. Недостаточная выразительность модели или плохая регуляризация.
3. Расхождение между “математической вероятностью” модели и реальными частотами (например, из-за переобучения: NLL на train → 0, уверенность → 1).
4. Архитектурные/тренировочные эффекты (например, heavy class imbalance, label smoothing и т.п.).

Это и называется **калибровкой**: насколько хорошо “предсказанная вероятность” совпадает с наблюдаемой частотой.

---

## Как измеряют калибровку

Самый практичный подход — смотреть соответствие “вероятность -> точность”.

### Reliability diagram (диаграмма калибровки)

1. Разбиваем предсказанные вероятности $p$ на bins (например, 10 бинов по 0.0..1.0).
2. Для каждого bin считаем:
   - среднее предсказанное $\mathbb{E}[p]$,
   - эмпирическую точность: долю объектов, где класс предсказан верно.
3. Идеально — точки лежат на диагонали $\text{accuracy}=p$.

### Expected Calibration Error (ECE)

Одна из популярных численных метрик:

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|
$$

где $B_m$ — bin, $\text{acc}(B_m)$ — точность внутри него, а $\text{conf}(B_m)$ — средняя уверенность.

### Brier score

Для бинарного случая:

$$
\text{Brier} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2
$$

Чем меньше — тем ближе вероятности к фактам.

---

## Temperature Scaling: базовая калибровка

Базовая и очень популярная пост-обработка: **Temperature Scaling**.

Идея: logits $z$ делятся на константу $T>0$, и только потом применяется softmax:

$$
p_k(T) = \text{softmax}\left(\frac{z}{T}\right)_k
$$

- если $T>1$ → вероятности становятся “мягче” (менее уверенные);
- если $T<1$ → модель становится “жёстче” (более уверенная).

Параметр $T$ подбирают на **валидационном** наборе, обычно минимизируя:

- NLL: $-\log p_y(T)$,
- либо напрямую ECE (реже из-за негладкости),
- либо при помощи калибровочных критериев.

Плюс: это быстро, не меняет ранжирование классов (argmax и порядок $p_k$ сохраняются, так как деление на $T>0$ монотонно), и часто заметно улучшает калибровку (Guo et al., 2017).

---

## Уверенность vs неопределённость (uncertainty)

Полезно различать:

1. **Aleatoric uncertainty** (шум в данных): даже человек может не знать ответ (размытость, неоднозначность).
2. **Epistemic uncertainty** (нехватка знания модели): модель не видела похожих примеров.

Обычный softmax-peak (“максимальная вероятность”) измеряет только один суррогат уверенности и плохо отделяет причины.

### Ансамбли и MC Dropout

Практические подходы:

- **Ensembles** (Deep Ensembles, Lakshminarayanan et al., 2017): усредняем предсказания нескольких моделей (разные инициализации/обучающие прогоны).
- **MC Dropout** (Gal & Ghahramani, 2016): включаем dropout на инференсе, делаем $S$ стохастических проходов и усредняем.

Тогда можно измерять вариативность предсказаний (и делать более осмысленную оценку uncertainty).

Классическая идея для энтропии:

$$
H(p) = -\sum_k p_k \log p_k
$$

Большая энтропия обычно означает “неуверенность” (но степень калибровки всё равно нужно проверять).

---

## Интуитивное объяснение “для 5-летнего”

Представь, что модель — это фонарик, который светит на картинку и выбирает, что там.

Он говорит: “я думаю это котик, потому что мой фонарик светит ярко”. Но “яркость” не всегда совпадает с тем, насколько часто это действительно котик. Если фонарик светит слишком ярко всегда — он будет переоценивать уверенность. Поэтому мы можем настроить шкалу, чтобы “0.8” означало “примерно в 8 из 10 случаев так и будет”.

---

## Примеры кода (PyTorch)

### Получение вероятностей из logits

```python
import torch
import torch.nn.functional as F

logits = torch.randn(8, 5)  # batch=8, num_classes=5
probs = F.softmax(logits, dim=-1)
confidence, pred_class = probs.max(dim=-1)

print(confidence.shape, pred_class.shape)
>>> torch.Size([8]) torch.Size([8])
```

### Temperature Scaling (минимально)

Ниже — каркас: `T` подбирают на валидации.

```python
import torch
import torch.nn.functional as F
from torch import optim

class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_t = torch.nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, logits):
        T = torch.exp(self.log_t)
        return logits / T

scaler = TemperatureScaler(init_temp=1.5)
optimizer = optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)

# logits_val: [N, K], y_val: [N] (истинные классы)
# logits_train/val нужно подготовить заранее
def closure():
    optimizer.zero_grad()
    scaled_logits = scaler(logits_val)
    loss = F.cross_entropy(scaled_logits, y_val)
    loss.backward()
    return loss

optimizer.step(closure)

scaled_logits_test = scaler(logits_test)
probs_test = F.softmax(scaled_logits_test, dim=-1)
```

---

## Источники

### Related Documents
- **[ROC-кривые и ROC AUC](../roc-curve-and-roc-auc/README.md)** — выбор порога и оценка качества при ранжировании.
- **[Cross Entropy and Focal Loss](../classification-losses-cross-entropy-focal-loss/README.md)** — связь loss с вероятностями классов.
- **[Bayes' Theorem and Probability Foundations](../bayes-theorem-and-probability-foundations/README.md)** — вероятностная база, на которой живёт калибровка.
- **[Ensemble Methods & Model Combination](../ensemble-methods-model-combination/README.md)** — ансамбли как источник улучшения неопределенности.
- **[System Design для CV и NLP](../ml-system-design-for-cv-and-nlp/README.md)** — калибровка и пороги как часть canary/мониторинга в serving.

### Статьи
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks*. ICML — temperature scaling, ECE, reliability diagrams.
- Lakshminarayanan, B. et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS.
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML — MC Dropout.

### Key Concepts
- logits, softmax, sigmoid
- calibration vs ranking metrics
- temperature scaling
- aleatoric vs epistemic uncertainty

