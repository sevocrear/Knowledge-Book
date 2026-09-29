---
title: "Тег: kb/topic"
description: Заметки с тегом kb/topic в книге знаний.
tags:
  - kb/topic
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `kb/topic`

## Заметки

- [Метрики оценки Action Recognition и Object Tracking](../../topics/action-recognition-and-object-tracking-metrics/README.md) — Протоколы и метрики для video action recognition, temporal localization, SOT и MOT: Top-1/Top-5, mAP@tIoU, Success AUC, IDF1, MOTA, HOTA.
- [MCP, ACP, UCP и Agent Harness](../../topics/agent-protocols-mcp-acp-ucp-and-harness/README.md) — Слои агентных протоколов (MCP, ACP, UCP, A2A) и agent harness: что к чему подключается, чем не путать аббревиатуры и как собрать эффективный harness в Cursor.
- [ArcFace и angular-margin losses для идентификации](../../topics/arcface-and-angular-margin-losses-for-identification/README.md) — Additive angular margin loss для идентификации: геометрия на гиперсфере, сравнение с CosFace/SphereFace, face/SKU/re-ID и open-set пороги.
- [Теорема Байеса и основы теории вероятностей](../../topics/bayes-theorem-and-probability-foundations/README.md) — Аксиомы Колмогорова, условная вероятность, формула полной вероятности, теорема Байеса, MAP/MLE и наивный Байес в ML.
- [Cross Entropy и Focal Loss](../../topics/classification-losses-cross-entropy-focal-loss/README.md) — Бинарная и многоклассовая кросс-энтропия, Focal Loss (α, γ) для дисбаланса и детекции (RetinaNet), когда выбирать CE vs Focal.
- [Code Agents, AutoResearch и Loopy Era](../../topics/code-agents-autoresearch-and-loopy-era/README.md) — Оркестрация code agents, AutoResearch loops, verification gates, harness engineering и переход от ручного кода к управлению агентными циклами.
- [Contrastive и metric learning для fine-grained распознавания](../../topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md) — Contrastive/triplet/InfoNCE и proxy losses для fine-grained retrieval: mining, Recall@K, ANN-индексы и continual learning новых классов.
- [Свёртки в CNN, размеры карт признаков и число параметров](../../topics/convolutions-and-parameters-in-cnn/README.md) — Почему популярны ядра 3×3, формулы размера feature map, transposed conv и подсчёт параметров Conv/Linear/BatchNorm/depthwise.
- [Деревья решений (Decision Trees)](../../topics/decision-trees/README.md) — Структура дерева, Gini/энтропия/Information Gain, ID3/C4.5/CART, переобучение и связь с ансамблями Random Forest/XGBoost.
- [Deep Reinforcement Learning (глубокое RL)](../../topics/deep-reinforcement-learning/README.md) — MDP, DQN, policy gradient, actor-critic (PPO/SAC/TD3), sim-to-real и применения в робототехнике и автономном вождении.
- [Loss функции для детекции, сегментации и 3D-детекции](../../topics/detection-segmentation-3d-losses/README.md) — Составные loss'ы детекторов: CE/Focal/QFL, L1/IoU/GIoU/DIoU/CIoU, Dice/Tversky для сегментации и 3D/BEV losses.
- [Diffusion Models (диффузионные модели)](../../topics/diffusion-models/README.md) — Прямой и обратный процесс диффузии, DDPM/DDIM, latent diffusion (Stable Diffusion), Consistency Models, Flow Matching и DiT.
- [DINOv3: Self-Supervised Vision Transformer и 2D RoPE](../../topics/dinov3-self-supervised-vision-transformer-and-2d-rope/README.md) — Self-supervised ViT (student–teacher), 2D RoPE для патчей, глобальные и dense-фичи для классификации, детекции и сегментации.
- [Эмбеддинги и матрица эмбеддингов](../../topics/embeddings-and-embedding-matrix/README.md) — Векторные представления токенов, матрица эмбеддингов V×d, lookup по ID и роль эмбеддингов в Transformer и RAG.
- [Методы комбинирования моделей (Ensemble Methods)](../../topics/ensemble-methods-model-combination/README.md) — Bagging/boosting/stacking, XGBoost/LightGBM/CatBoost, MoE, distillation и model merging (TIES/DARE/SLERP) для LLM.
- [Few-Shot Anomaly Detection: AnomalyDINO](../../topics/few-shot-anomaly-detection-anomalydino/README.md) — Patch-level nearest neighbor на DINOv2 без обучения: memory bank, косинусное расстояние и pixel-level anomaly maps для industrial QC.
- [Гауссово распределение (Normal Distribution)](../../topics/gaussian-distribution/README.md) — Одномерное и многомерное нормальное распределение, PDF/CDF и роль гауссианы в VAE, diffusion и Kalman filtering.
- [Generative Adversarial Networks (GAN)](../../topics/generative-adversarial-networks-gans/README.md) — Состязательное обучение generator/discriminator, mode collapse, современные варианты GAN и сравнение с VAE и diffusion.
- [Уверенность, калибровка и неопределённость](../../topics/how-models-predict-confidence-and-calibration/README.md) — Logits→softmax/sigmoid, reliability diagrams, ECE/Brier, temperature scaling и aleatoric/epistemic uncertainty (ensembles, MC Dropout).
- [Настройка гиперпараметров (Hyperparameter Tuning)](../../topics/hyperparameter-tuning/README.md) — Grid/Random search, Bayesian Optimization (Optuna/TPE), Hyperband/BOHB, PBT, CMA-ES, NAS и LR schedules.
- [Low-Rank Adaptation (LoRA)](../../topics/low-rank-adaptation-lora/README.md) — PEFT через низкоранговые адаптеры ΔW≈BA: математика, QLoRA/AdaLoRA/DoRA, эффективность памяти и практика в Hugging Face PEFT.
- [Лоссы metric learning и подбор майнеров](../../topics/metric-learning-losses-and-miners/README.md) — Каталог лоссов metric learning (contrastive, triplet, N-pair, Multi-Similarity, Circle, InfoNCE/SupCon, Proxy-NCA/Anchor, SoftTriple, ArcFace/CosFace/AdaFace): формулы, интуиция, какие майнеры к какому лоссу и когда что выбирать.
- [System Design для Computer Vision и NLP](../../topics/ml-system-design-for-cv-and-nlp/README.md) — Ёмкость, балансировка, serving на 100 vs 1000 клиентов, dynamic batching, KV cache и обработка 10–50 видеопотоков.
- [Non-Maximum Suppression (NMS) и современные end-to-end детекторы](../../topics/non-maximum-suppression-nms/README.md) — Классический NMS/Soft-NMS, проблемы в production и переход к NMS-free детекторам: DETR, RT-DETR, YOLO26 (dual-head).
- [Batch Normalization и Layer Normalization](../../topics/normalization-layers-batchnorm-layernorm/README.md) — Нормализация активаций: формулы BatchNorm vs LayerNorm, влияние на обучение, выбор для CNN и Transformer.
- [Retrieval-Augmented Generation (RAG)](../../topics/retrieval-augmented-generation-rag/README.md) — Архитектуры RAG (Naive/Advanced/Modular/Self-RAG/Corrective/LightRAG), retriever/reranker, chunking, оценка и production-практики.
- [ROC-кривые и ROC AUC](../../topics/roc-curve-and-roc-auc/README.md) — TPR/FPR, построение ROC, AUC как метрика ранжирования, выбор порога (Youden’s J) и связь с PR-кривыми.
- [SOTA-метрики для детекции, сегментации и мультиклассовой классификации](../../topics/sota-metrics-for-detection-segmentation-multiclass-classification/README.md) — COCO AP/AR, mIoU/Mask AP/PQ, Top-1/Top-5, macro/micro F1 и калибровочные метрики как стандартные протоколы сравнения.
- [Support Vector Machines (SVM) и Kernel Trick](../../topics/support-vector-machines-svm-and-kernel-trick/README.md) — Max-margin классификация, soft-margin C, dual formulation и kernel trick (linear/poly/RBF) без явного φ(x).
- [Токенизация и сжатие текста в LLM](../../topics/tokenization-and-text-compression-in-llms/README.md) — Word/char/BPE/WordPiece/Unigram токенизация как сжатие текста перед LLM и влияние на стоимость attention.
- [Transformers, Attention и Vision Transformers (ViT)](../../topics/transformers-attention-and-vision-transformers-vit/README.md) — Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.
- [Triton Inference Server и развёртывание моделей на 1–N GPU](../../topics/triton-inference-server-and-gpu-model-serving/README.md) — NVIDIA Triton: dynamic batching, concurrent execution, ensembles; полезен ли на 1 GPU; SOTA serving (vLLM, TensorRT-LLM, SGLang) для 100× пользователей.
- [Unscented Kalman Filter и современные методы отслеживания](../../topics/unscented-kalman-filter-and-tracking/README.md) — UKF vs KF/EKF/PF, sigma-points, DeepSORT/ByteTrack/Transformer tracking и χ²-тест выбросов в трекинге.
- [Variational Autoencoders (VAE)](../../topics/variational-autoencoders-vaes/README.md) — ELBO, encoder/decoder, reparameterization trick, латентное пространство и роль VAE в современных generative pipelines.
- [Видеокодеки H.264/H.265 и GPU-декодирование](../../topics/video-codecs-h264-h265-and-gpu-decode/README.md) — Intra/inter сжатие, GOP, H.264 vs H.265, типичный битрейт и hardware decode IP-камер (NVDEC, VAAPI, Quick Sync).
- [Vision-based обучение роботов](../../topics/vision-based-robot-training-methods/README.md) — Imitation learning, RL и VLA для визуального обучения роботов: OpenVLA, Octo, RT-1/RT-2, Open X-Embodiment и sim-to-real.
- [Модели Vision-Language-Action (VLA)](../../topics/vision-language-action-models-vla/README.md) — Объединение vision/language/action: RT-1/RT-2, OpenVLA, Octo, SmolVLA — архитектуры, данные, fine-tuning и сравнение с RL/IL.

