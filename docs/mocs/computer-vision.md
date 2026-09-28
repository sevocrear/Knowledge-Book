---
title: "MOC: компьютерное зрение"
description: "CNN, detection/segmentation, video codecs, serving камер, metric learning, SSL и tracking metrics."
tags:
  - kb/moc
  - domain/cv
type: moc
status: canonical
updated: 2026-09-18
---

# MOC: компьютерное зрение

CNN, detection/segmentation, video codecs, serving камер, metric learning, SSL и tracking metrics.

## Темы

- [Свёртки в CNN, размеры карт признаков и число параметров](../../topics/convolutions-and-parameters-in-cnn/README.md) — Почему популярны ядра 3×3, формулы размера feature map, transposed conv и подсчёт параметров Conv/Linear/BatchNorm/depthwise.
- [Non-Maximum Suppression (NMS) и современные end-to-end детекторы](../../topics/non-maximum-suppression-nms/README.md) — Классический NMS/Soft-NMS, проблемы в production и переход к NMS-free детекторам: DETR, RT-DETR, YOLO26 (dual-head).
- [Loss функции для детекции, сегментации и 3D-детекции](../../topics/detection-segmentation-3d-losses/README.md) — Составные loss'ы детекторов: CE/Focal/QFL, L1/IoU/GIoU/DIoU/CIoU, Dice/Tversky для сегментации и 3D/BEV losses.
- [Cross Entropy и Focal Loss](../../topics/classification-losses-cross-entropy-focal-loss/README.md) — Бинарная и многоклассовая кросс-энтропия, Focal Loss (α, γ) для дисбаланса и детекции (RetinaNet), когда выбирать CE vs Focal.
- [SOTA-метрики для детекции, сегментации и мультиклассовой классификации](../../topics/sota-metrics-for-detection-segmentation-multiclass-classification/README.md) — COCO AP/AR, mIoU/Mask AP/PQ, Top-1/Top-5, macro/micro F1 и калибровочные метрики как стандартные протоколы сравнения.
- [Transformers, Attention и Vision Transformers (ViT)](../../topics/transformers-attention-and-vision-transformers-vit/README.md) — Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.
- [DINOv3: Self-Supervised Vision Transformer и 2D RoPE](../../topics/dinov3-self-supervised-vision-transformer-and-2d-rope/README.md) — Self-supervised ViT (student–teacher), 2D RoPE для патчей, глобальные и dense-фичи для классификации, детекции и сегментации.
- [Few-Shot Anomaly Detection: AnomalyDINO](../../topics/few-shot-anomaly-detection-anomalydino/README.md) — Patch-level nearest neighbor на DINOv2 без обучения: memory bank, косинусное расстояние и pixel-level anomaly maps для industrial QC.
- [Contrastive и metric learning для fine-grained распознавания](../../topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md) — Contrastive/triplet/InfoNCE и proxy losses для fine-grained retrieval: mining, Recall@K, ANN-индексы и continual learning новых классов.
- [ArcFace и angular-margin losses для идентификации](../../topics/arcface-and-angular-margin-losses-for-identification/README.md) — Additive angular margin loss для идентификации: геометрия на гиперсфере, сравнение с CosFace/SphereFace, face/SKU/re-ID и open-set пороги.
- [Лоссы metric learning и подбор майнеров](../../topics/metric-learning-losses-and-miners/README.md) — Каталог лоссов metric learning (contrastive, triplet, N-pair, Multi-Similarity, Circle, InfoNCE/SupCon, Proxy-NCA/Anchor, SoftTriple, ArcFace/CosFace/AdaFace): формулы, интуиция, какие майнеры к какому лоссу и когда что выбирать.
- [Метрики оценки Action Recognition и Object Tracking](../../topics/action-recognition-and-object-tracking-metrics/README.md) — Протоколы и метрики для video action recognition, temporal localization, SOT и MOT: Top-1/Top-5, mAP@tIoU, Success AUC, IDF1, MOTA, HOTA.
- [Unscented Kalman Filter и современные методы отслеживания](../../topics/unscented-kalman-filter-and-tracking/README.md) — UKF vs KF/EKF/PF, sigma-points, DeepSORT/ByteTrack/Transformer tracking и χ²-тест выбросов в трекинге.
- [Видеокодеки H.264/H.265 и GPU-декодирование](../../topics/video-codecs-h264-h265-and-gpu-decode/README.md) — Intra/inter сжатие, GOP, H.264 vs H.265, типичный битрейт и hardware decode IP-камер (NVDEC, VAAPI, Quick Sync).
- [System Design для Computer Vision и NLP](../../topics/ml-system-design-for-cv-and-nlp/README.md) — Ёмкость, балансировка, serving на 100 vs 1000 клиентов, dynamic batching, KV cache и обработка 10–50 видеопотоков.

## См. также

- [Все темы](../index.md)
- [Теги](../tags/README.md)
