---
title: "Тег: domain/cv"
description: Заметки с тегом domain/cv в книге знаний.
tags:
  - domain/cv
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `domain/cv`

## Заметки

- [Метрики оценки Action Recognition и Object Tracking](../../topics/action-recognition-and-object-tracking-metrics/README.md) — Протоколы и метрики для video action recognition, temporal localization, SOT и MOT: Top-1/Top-5, mAP@tIoU, Success AUC, IDF1, MOTA, HOTA.
- [ArcFace и angular-margin losses для идентификации](../../topics/arcface-and-angular-margin-losses-for-identification/README.md) — Additive angular margin loss для идентификации: геометрия на гиперсфере, сравнение с CosFace/SphereFace, face/SKU/re-ID и open-set пороги.
- [Сториборд клипа: ArcFace — угловой margin на гиперсфере](../../topics/arcface-and-angular-margin-losses-for-identification/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа arcface_angular_margin: единичная сфера и угол θ, что делает additive angular margin m, компактные классы с зазором и сравнение SphereFace / CosFace / ArcFace.
- [Cross Entropy и Focal Loss](../../topics/classification-losses-cross-entropy-focal-loss/README.md) — Бинарная и многоклассовая кросс-энтропия, Focal Loss (α, γ) для дисбаланса и детекции (RetinaNet), когда выбирать CE vs Focal.
- [Contrastive и metric learning для fine-grained распознавания](../../topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md) — Contrastive/triplet/InfoNCE и proxy losses для fine-grained retrieval: mining, Recall@K, ANN-индексы и continual learning новых классов.
- [Сториборд клипа: contrastive / metric learning и triplet loss](../../topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа contrastive_embedding_space: зачем эмбеддинги вместо классификатора, triplet loss с margin и semi-hard mining, кластеры, поиск ближайших соседей и порог τ в проде.
- [Свёртки в CNN, размеры карт признаков и число параметров](../../topics/convolutions-and-parameters-in-cnn/README.md) — Почему популярны ядра 3×3, формулы размера feature map, transposed conv и подсчёт параметров Conv/Linear/BatchNorm/depthwise.
- [Loss функции для детекции, сегментации и 3D-детекции](../../topics/detection-segmentation-3d-losses/README.md) — Составные loss'ы детекторов: CE/Focal/QFL, L1/IoU/GIoU/DIoU/CIoU, Dice/Tversky для сегментации и 3D/BEV losses.
- [DINOv3: Self-Supervised Vision Transformer и 2D RoPE](../../topics/dinov3-self-supervised-vision-transformer-and-2d-rope/README.md) — Self-supervised ViT (student–teacher), 2D RoPE для патчей, глобальные и dense-фичи для классификации, детекции и сегментации.
- [Few-Shot Anomaly Detection: AnomalyDINO](../../topics/few-shot-anomaly-detection-anomalydino/README.md) — Patch-level nearest neighbor на DINOv2 без обучения: memory bank, косинусное расстояние и pixel-level anomaly maps для industrial QC.
- [Лоссы metric learning и подбор майнеров](../../topics/metric-learning-losses-and-miners/README.md) — Каталог лоссов metric learning (contrastive, triplet, N-pair, Multi-Similarity, Circle, InfoNCE/SupCon, Proxy-NCA/Anchor, SoftTriple, ArcFace/CosFace/AdaFace): формулы, интуиция, какие майнеры к какому лоссу и когда что выбирать.
- [System Design для Computer Vision и NLP](../../topics/ml-system-design-for-cv-and-nlp/README.md) — Ёмкость, балансировка, serving на 100 vs 1000 клиентов, dynamic batching, KV cache и обработка 10–50 видеопотоков.
- [Non-Maximum Suppression (NMS) и современные end-to-end детекторы](../../topics/non-maximum-suppression-nms/README.md) — Классический NMS/Soft-NMS, проблемы в production и переход к NMS-free детекторам: DETR, RT-DETR, YOLO26 (dual-head).
- [SOTA-метрики для детекции, сегментации и мультиклассовой классификации](../../topics/sota-metrics-for-detection-segmentation-multiclass-classification/README.md) — COCO AP/AR, mIoU/Mask AP/PQ, Top-1/Top-5, macro/micro F1 и калибровочные метрики как стандартные протоколы сравнения.
- [Transformers, Attention и Vision Transformers (ViT)](../../topics/transformers-attention-and-vision-transformers-vit/README.md) — Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.
- [Сториборд клипа: ViT — патчи, self-attention и CLS](../../topics/transformers-attention-and-vision-transformers-vit/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа vit_patches_attention: картинка → патчи и токены с CLS, scaled dot-product attention для строки CLS, энкодер из L блоков и голова классификации.
- [Triton Inference Server и развёртывание моделей на 1–N GPU](../../topics/triton-inference-server-and-gpu-model-serving/README.md) — NVIDIA Triton: dynamic batching, concurrent execution, ensembles; полезен ли на 1 GPU; SOTA serving (vLLM, TensorRT-LLM, SGLang) для 100× пользователей.
- [Unscented Kalman Filter и современные методы отслеживания](../../topics/unscented-kalman-filter-and-tracking/README.md) — UKF vs KF/EKF/PF, sigma-points, DeepSORT/ByteTrack/Transformer tracking и χ²-тест выбросов в трекинге.
- [Видеокодеки H.264/H.265 и GPU-декодирование](../../topics/video-codecs-h264-h265-and-gpu-decode/README.md) — Intra/inter сжатие, GOP, H.264 vs H.265, типичный битрейт и hardware decode IP-камер (NVDEC, VAAPI, Quick Sync).
- [Сториборд клипа: видеокодеки — intra, inter и GOP из I/P/B](../../topics/video-codecs-h264-h265-and-gpu-decode/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа gop-i-p-b-prediction: блок → DCT → квантование, вектор движения и остаток, GOP с I/P/B-кадрами, ссылками между ними и бюджетом бит.

