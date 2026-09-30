---
title: "Тег: concept/loss"
description: Заметки с тегом concept/loss в книге знаний.
tags:
  - concept/loss
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `concept/loss`

## Заметки

- [ArcFace и angular-margin losses для идентификации](../../topics/arcface-and-angular-margin-losses-for-identification/README.md) — Additive angular margin loss для идентификации: геометрия на гиперсфере, сравнение с CosFace/SphereFace, face/SKU/re-ID и open-set пороги.
- [Cross Entropy и Focal Loss](../../topics/classification-losses-cross-entropy-focal-loss/README.md) — Бинарная и многоклассовая кросс-энтропия, Focal Loss (α, γ) для дисбаланса и детекции (RetinaNet), когда выбирать CE vs Focal.
- [Сториборд клипа: Cross Entropy vs Focal Loss](../../topics/classification-losses-cross-entropy-focal-loss/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа cross-entropy-vs-focal-loss: логиты → softmax → −log p, кривые Focal Loss для γ = 0, 1, 2, 5 и α, когда выбирать CE, а когда Focal (RetinaNet, dense prediction).
- [Loss функции для детекции, сегментации и 3D-детекции](../../topics/detection-segmentation-3d-losses/README.md) — Составные loss'ы детекторов: CE/Focal/QFL, L1/IoU/GIoU/DIoU/CIoU, Dice/Tversky для сегментации и 3D/BEV losses.
- [Сториборд клипа: loss детекции и сегментации — IoU, GIoU, Dice](../../topics/detection-segmentation-3d-losses/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа detection-losses-iou-giou-dice: составной loss детектора (CE/Focal + регрессия бокса), почему IoU-loss не даёт градиента без перекрытия и как его чинят GIoU/DIoU/CIoU, Dice и Tversky для масок сегментации.
- [Лоссы metric learning и подбор майнеров](../../topics/metric-learning-losses-and-miners/README.md) — Каталог лоссов metric learning (contrastive, triplet, N-pair, Multi-Similarity, Circle, InfoNCE/SupCon, Proxy-NCA/Anchor, SoftTriple, ArcFace/CosFace/AdaFace): формулы, интуиция, какие майнеры к какому лоссу и когда что выбирать.
- [Сториборд клипа: triplet loss и майнеры](../../topics/metric-learning-losses-and-miners/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа metric-learning-losses-and-miners: якорь/positive/negative и triplet loss с margin, зоны easy / semi-hard / hard негативов, конвейер сэмплер P×K → майнер → лосс.

