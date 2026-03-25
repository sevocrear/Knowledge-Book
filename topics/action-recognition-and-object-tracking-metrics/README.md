# Метрики оценки Action Recognition и Object Tracking

## Table of Contents
1. [Введение](#введение)
2. [How would I describe it to a person who is 5 years old](#how-would-i-describe-it-to-a-person-who-is-5-years-old)
3. [Какие подзадачи мы оцениваем](#какие-подзадачи-мы-оцениваем)
4. [Метрики для Action Recognition](#метрики-для-action-recognition)
5. [Метрики для Object Tracking](#метрики-для-object-tracking)
6. [Как выбирать метрики под задачу](#как-выбирать-метрики-под-задачу)
7. [Типичные ошибки при интерпретации](#типичные-ошибки-при-интерпретации)
8. [Краткий cheat sheet](#краткий-cheat-sheet)
9. [References](#references)

---

## Введение

Вопрос "какая метрика лучше?" для `Action Recognition` и `Object Tracking` почти всегда зависит от формулировки задачи и бенчмарка.

- Для распознавания действий важны классификация, временная локализация или spatio-temporal detection.
- Для трекинга важны не только точность детекции, но и сохранение identity объекта во времени.

Поэтому в реальных статьях обычно показывают набор метрик, а не одну.

---

## How would I describe it to a person who is 5 years old

Представь, что робот смотрит мультик.

- В `Action Recognition` мы проверяем: правильно ли робот понял, что персонаж "бежит", "прыгает" или "машет рукой".
- В `Object Tracking` мы проверяем: не перепутал ли робот одного и того же персонажа между кадрами.

Метрики - это "оценки в дневнике" за разные навыки: кто он, где он, и не перепутал ли его с другим.

---

## Какие подзадачи мы оцениваем

Перед выбором метрики зафиксируй постановку:

1. **Action classification (clip-level/video-level)**  
   Один label на клип/видео.
2. **Temporal action localization**  
   Нужно предсказать класс и интервал времени действия.
3. **Spatio-temporal action detection**  
   Нужно предсказать класс + бокс/трубу в кадрах + время.
4. **Single Object Tracking (SOT)**  
   Один объект, трек во времени.
5. **Multiple Object Tracking (MOT)**  
   Много объектов, плюс задача сохранения identity.

---

## Метрики для Action Recognition

### 1) Классификация действий (video-level)

Самые частые:

- **Top-1 Accuracy**  
  Доля видео, где первый предсказанный класс верный.
- **Top-k Accuracy**  
  Истинный класс входит в top-k (часто `k=5`).
- **Mean Class Accuracy (mAcc)**  
  Средняя accuracy по классам, важна при дисбалансе.
- **Precision / Recall / F1 (macro, micro, weighted)**  
  Часто нужны для long-tail и несимметричных ошибок.

Если важны вероятности, а не только ранжирование:

- **NLL (log loss)**  
  $$\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log p(y_i|x_i)$$
- **ECE (Expected Calibration Error)** для калибровки confidence.

### 2) Temporal Action Localization

Здесь ключевая идея: интервал действия должен совпасть по времени.

Используют **tIoU** (temporal IoU):

$$
\text{tIoU} = \frac{|[t_s, t_e] \cap [\hat t_s, \hat t_e]|}{|[t_s, t_e] \cup [\hat t_s, \hat t_e]|}
$$

На базе tIoU считают:

- **mAP@tIoU=0.5** (или 0.3/0.5/0.7)
- **Average mAP** по набору порогов (например, `0.5:0.05:0.95`)

Также для temporal segmentation встречаются:

- **Frame-wise Accuracy**
- **Segmental Edit Score**
- **F1@{10,25,50}** по overlap-порогам

### 3) Spatio-temporal Action Detection

Оценивается пересечение по пространству и времени:

- **frame-mAP** (на уровне отдельных кадров)
- **video-mAP / tube-mAP** с порогом по vIoU

где `vIoU` учитывает согласование траектории/трубы во времени.

---

## Метрики для Object Tracking

Важно разделять `SOT` и `MOT`: у них разные традиции оценки.

### 1) Single Object Tracking (SOT)

Классические метрики (OTB, LaSOT, TrackingNet, VOT):

- **Precision**: доля кадров, где ошибка центра меньше порога (например, 20 px).
- **Normalized Precision**: то же, но с нормализацией по размеру бокса/кадра.
- **Success**: доля кадров с IoU выше порога.
- **Success AUC**: площадь под curve Success-vs-IoU-threshold.
- **EAO (Expected Average Overlap)** в VOT.
- **Robustness**: как часто трекер теряет цель (failures).

### 2) Multiple Object Tracking (MOT)

Для MOT одной метрики почти никогда недостаточно.

#### CLEAR MOT семейство

- **MOTA**
  $$\text{MOTA}=1-\frac{\text{FN}+\text{FP}+\text{IDSW}}{\text{GT}}$$
  Где `FN` - пропуски, `FP` - ложные, `IDSW` - identity switches.
- **MOTP**: точность локализации сопоставленных боксов (исторически используется реже, чем MOTA/IDF1/HOTA в новых работах).

Минус MOTA: может выглядеть "неплохо" при плохом сохранении identity.

#### Identity-aware метрики

- **IDP / IDR / IDF1**
  - `IDP` - precision по identity-сопоставлениям
  - `IDR` - recall по identity-сопоставлениям
  - `IDF1` - гармоническое среднее между IDP и IDR

`IDF1` гораздо лучше отражает стабильность ID-трека, чем голая MOTA.

#### Современный стандарт: HOTA

- **HOTA (Higher Order Tracking Accuracy)** объединяет качество:
  - детекции (`DetA`)
  - ассоциации (`AssA`)
  - локализации (`LocA`)

Точная формула задаётся через порог overlap $\alpha$ (обычно IoU-порог для матчинга):

$$
\text{HOTA}_{\alpha}=\sqrt{\text{DetA}_{\alpha}\cdot\text{AssA}_{\alpha}}
$$

где

$$
\text{DetA}_{\alpha}=\frac{|\mathrm{TP}_{\alpha}|}{|\mathrm{TP}_{\alpha}|+|\mathrm{FN}_{\alpha}|+|\mathrm{FP}_{\alpha}|}
$$

и

$$
\text{AssA}_{\alpha}=\frac{1}{|\mathrm{TP}_{\alpha}|}\sum_{c\in \mathrm{TP}_{\alpha}}
\frac{\mathrm{TPA}_{\alpha}(c)}
{\mathrm{TPA}_{\alpha}(c)+\mathrm{FNA}_{\alpha}(c)+\mathrm{FPA}_{\alpha}(c)}
$$

Здесь для каждого корректного match-а $c$:
- $\mathrm{TPA}_{\alpha}(c)$: сколько раз этот GT-трек и предсказанный трек корректно совпали между собой во времени;
- $\mathrm{FNA}_{\alpha}(c)$: сколько GT-элементов этого identity не было сопоставлено с данным предикт-треком;
- $\mathrm{FPA}_{\alpha}(c)$: сколько лишних элементов у предикт-трека (чужие identity относительно данного GT).

Итоговая метрика усредняется по набору порогов $\alpha$:

$$
\text{HOTA}=\frac{1}{|\mathcal{A}|}\sum_{\alpha\in\mathcal{A}}\text{HOTA}_{\alpha},
\quad
\mathcal{A}=\{0.05,0.10,\dots,0.95\}
$$

Отдельно обычно репортят локализацию:

$$
\text{LocA}_{\alpha}=\frac{1}{|\mathrm{TP}_{\alpha}|}\sum_{c\in \mathrm{TP}_{\alpha}}\mathrm{IoU}(c)
$$

`LocA` не входит напрямую в базовую формулу `HOTA_alpha`, но даёт важную диагностику качества боксов.

Практика 2023-2026: на MOTChallenge обычно смотрят вместе `HOTA`, `IDF1`, `MOTA`.

### 3) Вспомогательные MOT-метрики

- **MT / PT / ML**: Mostly Tracked / Partially Tracked / Mostly Lost.
- **Frag**: число фрагментаций треков.
- **IDs**: число переключений identity.
- **FP / FN**: ложные и пропущенные detections.

Эти метрики помогают диагностировать, где именно система ломается.

---

## Как выбирать метрики под задачу

Простой рабочий принцип:

1. **Action classification**: `Top-1`, `Top-5`, `macro-F1` (если дисбаланс), + калибровка (`NLL/ECE`) при риск-чувствительных применениях.
2. **Temporal localization**: `mAP@tIoU` и `average mAP` по нескольким tIoU.
3. **SOT**: `Success AUC` + `Normalized Precision` + `Robustness`.
4. **MOT**: минимум `HOTA + IDF1 + MOTA`, и отдельно смотреть `IDs/Frag/FN/FP`.

---

## Типичные ошибки при интерпретации

1. Сравнивать результаты на разных `IoU/tIoU` порогах как будто это одно и то же.
2. Опираться только на `MOTA` и игнорировать `IDF1/HOTA`.
3. Не проверять дисбаланс классов в Action Recognition (Top-1 может обманывать).
4. Смешивать offline и online трекеры без явного уточнения протокола.
5. Не фиксировать split, FPS режим, post-processing и tracker confidence thresholds.

---

## Краткий cheat sheet

| Задача | Базовые метрики | Что добавить |
|---|---|---|
| Action classification | Top-1, Top-5 | macro-F1, mAcc, NLL/ECE |
| Temporal action localization | mAP@tIoU | average mAP по нескольким tIoU |
| SOT | Success AUC, Precision | Normalized Precision, Robustness, EAO |
| MOT | HOTA, IDF1, MOTA | IDs, Frag, FP, FN, MT/ML |

---

## References

### Related Documents
- **[Confidence, Calibration and Uncertainty](../how-models-predict-confidence-and-calibration/README.md)** - калибровка confidence (ECE/NLL) для классификации.
- **[ROC Curves and ROC AUC](../roc-curve-and-roc-auc/README.md)** - ranking-метрики и пороговые trade-offs.
- **[Non-Maximum Suppression (NMS)](../non-maximum-suppression-nms/README.md)** - post-processing в детекции перед трекингом.
- **[Losses for Detection, Segmentation, and 3D Detection](../detection-segmentation-3d-losses/README.md)** - связь loss-функций с downstream метриками.
- **[Unscented Kalman Filter and Modern Tracking Methods](../unscented-kalman-filter-and-tracking/README.md)** - алгоритмическая база tracking pipeline.

### External Benchmarks and Protocols
- **MOTChallenge**: HOTA, IDF1, MOTA и диагностические метрики.
- **KITTI Tracking / BDD100K / DanceTrack**: разные акценты на motion, occlusion и ID consistency.
- **Kinetics / Something-Something / UCF101 / HMDB51**: action classification.
- **ActivityNet / THUMOS**: temporal action localization.
