---
title: "Тег: concept/metrics"
description: Заметки с тегом concept/metrics в книге знаний.
tags:
  - concept/metrics
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `concept/metrics`

## Заметки

- [Метрики оценки Action Recognition и Object Tracking](../../topics/action-recognition-and-object-tracking-metrics/README.md) — Протоколы и метрики для video action recognition, temporal localization, SOT и MOT: Top-1/Top-5, mAP@tIoU, Success AUC, IDF1, MOTA, HOTA.
- [Сториборд клипа: метрики MOT — MOTA, IDF1, HOTA](../../topics/action-recognition-and-object-tracking-metrics/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа tracking-metrics-mota-idf1-hota: покадровое сопоставление детекций с GT-треками по IoU (TP / FN / FP / IDSW), чем MOTA отличается от IDF1 и как HOTA объединяет DetA и AssA с усреднением по порогам α.
- [ROC-кривые и ROC AUC](../../topics/roc-curve-and-roc-auc/README.md) — TPR/FPR, построение ROC, AUC как метрика ранжирования, выбор порога (Youden’s J) и связь с PR-кривыми.
- [Сториборд клипа: ROC-кривая и ROC AUC — порог, кривая, площадь](../../topics/roc-curve-and-roc-auc/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа roc-curve-threshold-sweep-auc: два холма скоров и движущийся порог t (TPR/FPR), прогон порогов рисует ROC-кривую с точкой Youden’s J, AUC как площадь и вероятность P(s(x⁺) > s(x⁻)), контраст с PR-кривой.
- [SOTA-метрики для детекции, сегментации и мультиклассовой классификации](../../topics/sota-metrics-for-detection-segmentation-multiclass-classification/README.md) — COCO AP/AR, mIoU/Mask AP/PQ, Top-1/Top-5, macro/micro F1 и калибровочные метрики как стандартные протоколы сравнения.
- [Сториборд клипа: COCO AP, mIoU, Top-1 и F1](../../topics/sota-metrics-for-detection-segmentation-multiclass-classification/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа coco-ap-miou-top1-f1: IoU-матчинг, PR-кривая и усреднение COCO AP по порогам IoU 0.50:0.95; per-class IoU и mIoU против pixel accuracy; матрица ошибок, Top-1/Top-5 и macro vs micro F1.

