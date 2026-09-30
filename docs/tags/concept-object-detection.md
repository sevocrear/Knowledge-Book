---
title: "Тег: concept/object-detection"
description: Заметки с тегом concept/object-detection в книге знаний.
tags:
  - concept/object-detection
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `concept/object-detection`

## Заметки

- [Loss функции для детекции, сегментации и 3D-детекции](../../topics/detection-segmentation-3d-losses/README.md) — Составные loss'ы детекторов: CE/Focal/QFL, L1/IoU/GIoU/DIoU/CIoU, Dice/Tversky для сегментации и 3D/BEV losses.
- [Сториборд клипа: loss детекции и сегментации — IoU, GIoU, Dice](../../topics/detection-segmentation-3d-losses/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа detection-losses-iou-giou-dice: составной loss детектора (CE/Focal + регрессия бокса), почему IoU-loss не даёт градиента без перекрытия и как его чинят GIoU/DIoU/CIoU, Dice и Tversky для масок сегментации.
- [Non-Maximum Suppression (NMS) и современные end-to-end детекторы](../../topics/non-maximum-suppression-nms/README.md) — Классический NMS/Soft-NMS, проблемы в production и переход к NMS-free детекторам: DETR, RT-DETR, YOLO26 (dual-head).
- [Сториборд клипа: NMS → Soft-NMS → NMS-free детекторы](../../topics/non-maximum-suppression-nms/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа nms-soft-nms-nms-free: greedy NMS на примере из README (сортировка по score, IoU > τ → удалить), Soft-NMS с линейным и гауссовым понижением score и NMS-free детекторы (DETR one-to-one matching, dual-head YOLOv10 → YOLO26).
- [SOTA-метрики для детекции, сегментации и мультиклассовой классификации](../../topics/sota-metrics-for-detection-segmentation-multiclass-classification/README.md) — COCO AP/AR, mIoU/Mask AP/PQ, Top-1/Top-5, macro/micro F1 и калибровочные метрики как стандартные протоколы сравнения.
- [Сториборд клипа: COCO AP, mIoU, Top-1 и F1](../../topics/sota-metrics-for-detection-segmentation-multiclass-classification/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа coco-ap-miou-top1-f1: IoU-матчинг, PR-кривая и усреднение COCO AP по порогам IoU 0.50:0.95; per-class IoU и mIoU против pixel accuracy; матрица ошибок, Top-1/Top-5 и macro vs micro F1.

