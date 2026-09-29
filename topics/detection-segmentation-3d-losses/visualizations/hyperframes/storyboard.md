---
title: "Сториборд клипа: loss детекции и сегментации — IoU, GIoU, Dice"
description: "Сториборд и команды сборки HyperFrames-клипа detection-losses-iou-giou-dice: составной loss детектора (CE/Focal + регрессия бокса), почему IoU-loss не даёт градиента без перекрытия и как его чинят GIoU/DIoU/CIoU, Dice и Tversky для масок сегментации."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/loss
  - concept/object-detection
aliases:
  - detection-losses-iou-giou-dice
  - Detection losses storyboard
related:
  - detection-segmentation-3d-losses
  - classification-losses-cross-entropy-focal-loss
status: notes
lang: ru
type: note
slug: detection-segmentation-3d-losses/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Loss детекции и сегментации: IoU, GIoU, Dice — сториборд

Клип `assets/visualizations/detection-losses-iou-giou-dice.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы 1 «Обзор», 2.1 «Классификационные loss’ы», 2.2 «Loss’ы для регрессии боксов»,
3 «Loss функции для сегментации», 5 «Focal Loss»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Из чего состоит loss детектора? | «Изображение» с абстрактным объектом, gt-бокс (зелёный пунктир) и предсказанный бокс (синий) с тегом класса; карточки: L_total = λ_cls·L_cls + λ_reg·L_reg + λ_aux·L_aux; L_FL(p) = −α(1−p)^γ log(p); L1 / Smooth L1 по (x, y, w, h) или IoU-loss; в конце pred-бокс подтягивается к gt | loss составной: класс + геометрия бокса + вспомогательные термы (objectness, centerness) · классификация: CE или Focal, (1−p)^γ гасит лёгкие примеры · регрессия: L1 / Smooth L1 по координатам или loss прямо по IoU, коррелирующему с метрикой |
| 2 | 13–29 с | Что делать, если боксы не пересекаются? | Сетка 12×4 клеток, B = gt (зелёный, 4×3) и A = pred (синий, 4×3); пересечение A∩B = 6, IoU = 6/18 = 0.33; A уезжает вправо → IoU = 0/24 = 0, покачивание вдали — IoU не меняется; рисуется охватывающий бокс C (оранжевый): GIoU = 0 − 24/48 = −0.50; A приближается, C сужается: GIoU = −0.40; фиолетовая линия ρ между центрами (DIoU), карточка DIoU / CIoU | IoU = пересечение / объединение, IoU-loss оптимизирует метрику напрямую · боксы разошлись: IoU = 0 при любом расстоянии, градиент нулевой · GIoU вычитает долю пустоты в C — чем дальше, тем ниже, градиент есть · DIoU добавляет расстояние центров, CIoU — соотношение сторон; так учат современные YOLO |
| 3 | 29–42 с | Как считать loss по маске сегментации? | Сетка 10×8 пикселей: G = gt (зелёный, 20 клеток), P = pred (синий контур, 16), P∩G (бирюзовый, 12); затем FP = 4 (красный) и FN = 8 (оранжевый); карточки Dice = 2|P∩G| / (|P|+|G|) = 24/36 = 0.67, L_Dice = 1 − Dice_soft = 0.33; Tversky T = |P∩G| / (|P∩G| + α·FP + β·FN); «на практике: CE + λ·Dice, Focal + Dice/Tversky» | маска — множество пикселей: G = 20, P = 16, P∩G = 12 (иллюстративно) · Dice = 2|P∩G| / (|P|+|G|), L_Dice = 1 − Dice_soft: фон в формулу не входит · **ключевая идея**: loss = класс (CE/Focal) + геометрия: IoU-семейство для боксов, Dice/Tversky для масок — оптимизируем то, что измеряем |

Числа и формулы:

- Из README: L_total = λ_cls·L_cls + λ_reg·L_reg + λ_aux·L_aux (разд. 1); L_FL(p) = −α(1−p)^γ log(p), γ = 0 → CE (разд. 2.1, 5);
  L1 / Smooth L1 по координатам, IoU / GIoU («штраф за отсутствие перекрытия») / DIoU («расстояние между центрами») / CIoU («соотношение сторон»),
  YOLO (разд. 2.2); Dice = 2|P∩G| / (|P|+|G|), L_Dice = 1 − Dice_soft, Tversky — разные веса FN и FP, Focal Tversky, комбинации CE + λ·Dice и
  Focal + Dice/Tversky (разд. 3).
- Из первоисточников с пометкой на экране «в статье …»: GIoU = IoU − |C ∖ (A∪B)| / |C| (Rezatofighi et al., 2019); DIoU-штраф ρ²(центры)/c²,
  CIoU + α·v (Zheng et al., 2020); Tversky T = |P∩G| / (|P∩G| + α·FP + β·FN), α = β = 0.5 → Dice (Salehi et al., 2017).
- Иллюстративные (помечены «иллюстративно»): боксы 4×3 клетки, |A| = |B| = 12; перекрытие A∩B = 6, A∪B = 18, IoU = 0.33;
  без перекрытия A∪B = 24, |C| = 48 → GIoU = −0.50, ближе |C| = 40 → GIoU = −0.40; маска 10×8: |G| = 20, |P| = 16, |P∩G| = 12,
  FP = 4, FN = 8, Dice = 24/36 = 0.67, L_Dice = 0.33, IoU маски = 12/24 = 0.50; p = 0.9 у тега класса.

## Сборка

```bash
cd topics/detection-segmentation-3d-losses/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7,11.5,16,20,24,27.5,32,36,40
npx -y hyperframes@0.8.81 render --quality looks --output renders/detection-losses-iou-giou-dice.mp4
ffmpeg -i renders/detection-losses-iou-giou-dice.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/detection-losses-iou-giou-dice.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/detection-losses-iou-giou-dice.mp4 -o ../../assets/visualizations/detection-losses-iou-giou-dice.gif
```
