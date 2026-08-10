---
title: Topic Catalog
description: Annotated catalog of all knowledge-book topics with tags and descriptions for RAG.
tags:
  - kb/index
status: canonical
lang: en
type: index
updated: 2026-08-10
---

# Topic Catalog

Canonical notes live under `topics/<slug>/README.md`. Descriptions below are the same strings stored in frontmatter for retrieval.

## [Метрики оценки Action Recognition и Object Tracking](../topics/action-recognition-and-object-tracking-metrics/README.md)

- **slug:** `action-recognition-and-object-tracking-metrics`
- **description:** Протоколы и метрики для video action recognition, temporal localization, SOT и MOT: Top-1/Top-5, mAP@tIoU, Success AUC, IDF1, MOTA, HOTA.
- **tags:** `domain/cv`, `domain/video`, `concept/metrics`, `concept/tracking`, `concept/action-recognition`
- **aliases:** action recognition metrics, object tracking metrics, MOTA, HOTA, IDF1

## [ArcFace and Angular-Margin Losses for Identification](../topics/arcface-and-angular-margin-losses-for-identification/README.md)

- **slug:** `arcface-and-angular-margin-losses-for-identification`
- **description:** Additive angular margin loss для идентификации: геометрия на гиперсфере, сравнение с CosFace/SphereFace, face/SKU/re-ID и open-set пороги.
- **tags:** `domain/cv`, `concept/metric-learning`, `concept/loss`, `concept/embeddings`, `concept/face-recognition`
- **aliases:** ArcFace, CosFace, SphereFace, angular margin loss

## [Теорема Байеса и основы теории вероятностей](../topics/bayes-theorem-and-probability-foundations/README.md)

- **slug:** `bayes-theorem-and-probability-foundations`
- **description:** Аксиомы Колмогорова, условная вероятность, формула полной вероятности, теорема Байеса, MAP/MLE и наивный Байес в ML.
- **tags:** `domain/math`, `domain/ml-foundations`, `concept/probability`, `concept/bayes`
- **aliases:** Bayes theorem, теорема Байеса, conditional probability, MAP, MLE

## [Cross Entropy и Focal Loss](../topics/classification-losses-cross-entropy-focal-loss/README.md)

- **slug:** `classification-losses-cross-entropy-focal-loss`
- **description:** Бинарная и многоклассовая кросс-энтропия, Focal Loss (α, γ) для дисбаланса и детекции (RetinaNet), когда выбирать CE vs Focal.
- **tags:** `domain/cv`, `domain/ml-foundations`, `concept/loss`, `concept/classification`
- **aliases:** cross entropy, focal loss, кросс-энтропия, RetinaNet loss

## [Code Agents, AutoResearch и Loopy Era](../topics/code-agents-autoresearch-and-loopy-era/README.md)

- **slug:** `code-agents-autoresearch-and-loopy-era`
- **description:** Оркестрация code agents, AutoResearch loops, verification gates, harness engineering и переход от ручного кода к управлению агентными циклами.
- **tags:** `domain/agents`, `domain/llm`, `concept/orchestration`, `concept/verification`, `concept/autoresearch`
- **aliases:** code agents, AutoResearch, loopy era, agent harness, Karpathy agents

## [Contrastive & Metric Learning for Fine-Grained Visual Recognition](../topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md)

- **slug:** `contrastive-and-metric-learning-for-fine-grained-visual-recognition`
- **description:** Contrastive/triplet/InfoNCE и proxy losses для fine-grained retrieval: mining, Recall@K, ANN-индексы и continual learning новых классов.
- **tags:** `domain/cv`, `concept/metric-learning`, `concept/contrastive-learning`, `concept/embeddings`, `concept/retrieval`
- **aliases:** contrastive learning, metric learning, triplet loss, InfoNCE, fine-grained recognition

## [Свёртки в CNN, размеры карт признаков и число параметров](../topics/convolutions-and-parameters-in-cnn/README.md)

- **slug:** `convolutions-and-parameters-in-cnn`
- **description:** Почему популярны ядра 3×3, формулы размера feature map, transposed conv и подсчёт параметров Conv/Linear/BatchNorm/depthwise.
- **tags:** `domain/cv`, `domain/dl-foundations`, `concept/cnn`, `concept/convolution`
- **aliases:** CNN convolutions, feature map size, transposed convolution, DeConv

## [Деревья решений (Decision Trees)](../topics/decision-trees/README.md)

- **slug:** `decision-trees`
- **description:** Структура дерева, Gini/энтропия/Information Gain, ID3/C4.5/CART, переобучение и связь с ансамблями Random Forest/XGBoost.
- **tags:** `domain/classical-ml`, `concept/decision-trees`, `concept/ensemble`
- **aliases:** decision trees, деревья решений, Gini, CART, Information Gain

## [Deep Reinforcement Learning](../topics/deep-reinforcement-learning/README.md)

- **slug:** `deep-reinforcement-learning`
- **description:** MDP, DQN, policy gradient, actor-critic (PPO/SAC/TD3), sim-to-real и применения в робототехнике и автономном вождении.
- **tags:** `domain/rl`, `domain/robotics`, `concept/rl`, `concept/ppo`, `concept/sac`
- **aliases:** Deep RL, reinforcement learning, DQN, PPO, SAC, TD3

## [Loss функции для детекции, сегментации и 3D-детекции](../topics/detection-segmentation-3d-losses/README.md)

- **slug:** `detection-segmentation-3d-losses`
- **description:** Составные loss'ы детекторов: CE/Focal/QFL, L1/IoU/GIoU/DIoU/CIoU, Dice/Tversky для сегментации и 3D/BEV losses.
- **tags:** `domain/cv`, `concept/loss`, `concept/object-detection`, `concept/segmentation`, `concept/3d-detection`
- **aliases:** detection losses, IoU loss, Dice loss, GIoU, 3D detection loss

## [Diffusion Models](../topics/diffusion-models/README.md)

- **slug:** `diffusion-models`
- **description:** Forward/reverse diffusion, DDPM/DDIM, latent diffusion (Stable Diffusion), Consistency Models, Flow Matching и DiT.
- **tags:** `domain/generative`, `concept/diffusion`, `concept/score-matching`, `concept/latent-diffusion`
- **aliases:** diffusion models, DDPM, DDIM, Stable Diffusion, Flow Matching, DiT

## [DINOv3: Self-Supervised Vision Transformer и 2D RoPE](../topics/dinov3-self-supervised-vision-transformer-and-2d-rope/README.md)

- **slug:** `dinov3-self-supervised-vision-transformer-and-2d-rope`
- **description:** Self-supervised ViT (student–teacher), 2D RoPE для патчей, глобальные и dense-фичи для классификации, детекции и сегментации.
- **tags:** `domain/cv`, `concept/self-supervised`, `concept/vit`, `concept/rope`, `concept/dino`
- **aliases:** DINOv3, DINOv2, 2D RoPE, self-supervised ViT

## [Embeddings and Embedding Matrix](../topics/embeddings-and-embedding-matrix/README.md)

- **slug:** `embeddings-and-embedding-matrix`
- **description:** Векторные представления токенов, матрица эмбеддингов V×d, lookup по ID и роль эмбеддингов в Transformer и RAG.
- **tags:** `domain/nlp`, `domain/llm`, `concept/embeddings`, `concept/tokenization`
- **aliases:** embedding matrix, token embeddings, эмбеддинги, word embeddings

## [Методы комбинирования моделей (Ensemble Methods)](../topics/ensemble-methods-model-combination/README.md)

- **slug:** `ensemble-methods-model-combination`
- **description:** Bagging/boosting/stacking, XGBoost/LightGBM/CatBoost, MoE, distillation и model merging (TIES/DARE/SLERP) для LLM.
- **tags:** `domain/classical-ml`, `domain/llm`, `concept/ensemble`, `concept/boosting`, `concept/moe`, `concept/model-merging`
- **aliases:** ensemble methods, Random Forest, XGBoost, LightGBM, Mixture of Experts, model merging

## [Few-Shot Anomaly Detection: AnomalyDINO](../topics/few-shot-anomaly-detection-anomalydino/README.md)

- **slug:** `few-shot-anomaly-detection-anomalydino`
- **description:** Patch-level nearest neighbor на DINOv2 без обучения: memory bank, косинусное расстояние и pixel-level anomaly maps для industrial QC.
- **tags:** `domain/cv`, `concept/anomaly-detection`, `concept/few-shot`, `concept/dino`
- **aliases:** AnomalyDINO, few-shot anomaly detection, patch nearest neighbor

## [Гауссово распределение (Normal Distribution)](../topics/gaussian-distribution/README.md)

- **slug:** `gaussian-distribution`
- **description:** Одномерное и многомерное нормальное распределение, PDF/CDF и роль гауссианы в VAE, diffusion и Kalman filtering.
- **tags:** `domain/math`, `domain/ml-foundations`, `concept/probability`, `concept/gaussian`
- **aliases:** Gaussian distribution, normal distribution, гауссово распределение

## [Generative Adversarial Networks (GANs)](../topics/generative-adversarial-networks-gans/README.md)

- **slug:** `generative-adversarial-networks-gans`
- **description:** Adversarial training generator/discriminator, mode collapse, современные варианты GAN и сравнение с VAE и diffusion.
- **tags:** `domain/generative`, `concept/gan`, `concept/adversarial-training`
- **aliases:** GAN, Generative Adversarial Networks, StyleGAN, mode collapse

## [Confidence, Calibration and Uncertainty](../topics/how-models-predict-confidence-and-calibration/README.md)

- **slug:** `how-models-predict-confidence-and-calibration`
- **description:** Logits→softmax/sigmoid, reliability diagrams, ECE/Brier, temperature scaling и aleatoric/epistemic uncertainty (ensembles, MC Dropout).
- **tags:** `domain/ml-foundations`, `concept/calibration`, `concept/uncertainty`, `concept/confidence`
- **aliases:** calibration, ECE, temperature scaling, model confidence, uncertainty estimation

## [Настройка гиперпараметров (Hyperparameter Tuning)](../topics/hyperparameter-tuning/README.md)

- **slug:** `hyperparameter-tuning`
- **description:** Grid/Random search, Bayesian Optimization (Optuna/TPE), Hyperband/BOHB, PBT, CMA-ES, NAS и LR schedules.
- **tags:** `domain/mlops`, `domain/ml-foundations`, `concept/hyperparameter-tuning`, `concept/bayesian-optimization`, `concept/nas`
- **aliases:** hyperparameter tuning, Optuna, Hyperband, Bayesian Optimization, NAS

## [Low-Rank Adaptation (LoRA)](../topics/low-rank-adaptation-lora/README.md)

- **slug:** `low-rank-adaptation-lora`
- **description:** PEFT через низкоранговые адаптеры ΔW≈BA: математика, QLoRA/AdaLoRA/DoRA, эффективность памяти и практика в Hugging Face PEFT.
- **tags:** `domain/llm`, `concept/peft`, `concept/lora`, `concept/fine-tuning`
- **aliases:** LoRA, QLoRA, AdaLoRA, DoRA, PEFT

## [Non-Maximum Suppression (NMS) и современные end-to-end детекторы](../topics/non-maximum-suppression-nms/README.md)

- **slug:** `non-maximum-suppression-nms`
- **description:** Классический NMS/Soft-NMS, проблемы в production и переход к NMS-free детекторам: DETR, RT-DETR, YOLO26 (dual-head).
- **tags:** `domain/cv`, `concept/nms`, `concept/object-detection`, `concept/end-to-end-detection`
- **aliases:** NMS, Non-Maximum Suppression, YOLO26, DETR, RT-DETR, Soft-NMS

## [Batch Normalization и Layer Normalization](../topics/normalization-layers-batchnorm-layernorm/README.md)

- **slug:** `normalization-layers-batchnorm-layernorm`
- **description:** Нормализация активаций: формулы BatchNorm vs LayerNorm, влияние на обучение, выбор для CNN и Transformer.
- **tags:** `domain/dl-foundations`, `concept/normalization`, `concept/batchnorm`, `concept/layernorm`
- **aliases:** BatchNorm, LayerNorm, Batch Normalization, Layer Normalization

## [Retrieval-Augmented Generation (RAG)](../topics/retrieval-augmented-generation-rag/README.md)

- **slug:** `retrieval-augmented-generation-rag`
- **description:** Архитектуры RAG (Naive/Advanced/Modular/Self-RAG/Corrective/LightRAG), retriever/reranker, chunking, оценка и production-практики.
- **tags:** `domain/nlp`, `domain/llm`, `concept/rag`, `concept/retrieval`, `concept/vector-search`
- **aliases:** RAG, Retrieval-Augmented Generation, Self-RAG, Corrective RAG, LightRAG, vector database

## [ROC-кривые и ROC AUC](../topics/roc-curve-and-roc-auc/README.md)

- **slug:** `roc-curve-and-roc-auc`
- **description:** TPR/FPR, построение ROC, AUC как метрика ранжирования, выбор порога (Youden’s J) и связь с PR-кривыми.
- **tags:** `domain/ml-foundations`, `concept/metrics`, `concept/roc`, `concept/classification`
- **aliases:** ROC, ROC AUC, TPR, FPR, Youden J, PR curve

## [SOTA-метрики для детекции, сегментации и мультиклассовой классификации](../topics/sota-metrics-for-detection-segmentation-multiclass-classification/README.md)

- **slug:** `sota-metrics-for-detection-segmentation-multiclass-classification`
- **description:** COCO AP/AR, mIoU/Mask AP/PQ, Top-1/Top-5, macro/micro F1 и калибровочные метрики как стандартные протоколы сравнения.
- **tags:** `domain/cv`, `concept/metrics`, `concept/object-detection`, `concept/segmentation`, `concept/classification`
- **aliases:** COCO AP, mIoU, panoptic PQ, Mask AP, Top-1 accuracy

## [Support Vector Machines (SVM) и Kernel Trick](../topics/support-vector-machines-svm-and-kernel-trick/README.md)

- **slug:** `support-vector-machines-svm-and-kernel-trick`
- **description:** Max-margin классификация, soft-margin C, dual formulation и kernel trick (linear/poly/RBF) без явного φ(x).
- **tags:** `domain/classical-ml`, `concept/svm`, `concept/kernel-methods`
- **aliases:** SVM, support vector machine, kernel trick, RBF kernel, margin

## [Tokenization and Text Compression in LLMs](../topics/tokenization-and-text-compression-in-llms/README.md)

- **slug:** `tokenization-and-text-compression-in-llms`
- **description:** Word/char/BPE/WordPiece/Unigram токенизация как сжатие текста перед LLM и влияние на стоимость attention.
- **tags:** `domain/nlp`, `domain/llm`, `concept/tokenization`, `concept/bpe`, `concept/compression`
- **aliases:** tokenization, BPE, WordPiece, Unigram LM, byte-level BPE

## [Transformers, Attention и Vision Transformers (ViT)](../topics/transformers-attention-and-vision-transformers-vit/README.md)

- **slug:** `transformers-attention-and-vision-transformers-vit`
- **description:** Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.
- **tags:** `domain/nlp`, `domain/cv`, `domain/llm`, `concept/attention`, `concept/transformer`, `concept/vit`
- **aliases:** Transformer, self-attention, ViT, Vision Transformer, KV cache, RoPE

## [Unscented Kalman Filter и современные методы отслеживания](../topics/unscented-kalman-filter-and-tracking/README.md)

- **slug:** `unscented-kalman-filter-and-tracking`
- **description:** UKF vs KF/EKF/PF, sigma-points, DeepSORT/ByteTrack/Transformer tracking и χ²-тест выбросов в трекинге.
- **tags:** `domain/cv`, `domain/robotics`, `concept/kalman-filter`, `concept/tracking`, `concept/ukf`
- **aliases:** UKF, Unscented Kalman Filter, Kalman Filter, ByteTrack, DeepSORT, object tracking

## [Variational Autoencoders (VAEs)](../topics/variational-autoencoders-vaes/README.md)

- **slug:** `variational-autoencoders-vaes`
- **description:** ELBO, encoder/decoder, reparameterization trick, латентное пространство и роль VAE в современных generative pipelines.
- **tags:** `domain/generative`, `concept/vae`, `concept/latent-variable`, `concept/elbo`
- **aliases:** VAE, Variational Autoencoder, ELBO, reparameterization trick

## [Vision-Based Robot Training Methods](../topics/vision-based-robot-training-methods/README.md)

- **slug:** `vision-based-robot-training-methods`
- **description:** Imitation learning, RL и VLA для визуального обучения роботов: OpenVLA, Octo, RT-1/RT-2, Open X-Embodiment и sim-to-real.
- **tags:** `domain/robotics`, `domain/embodied-ai`, `concept/imitation-learning`, `concept/vla`, `concept/sim-to-real`
- **aliases:** robot training, OpenVLA, Octo, Open X-Embodiment, vision-based robotics

## [Vision-Language-Action (VLA) Models](../topics/vision-language-action-models-vla/README.md)

- **slug:** `vision-language-action-models-vla`
- **description:** Объединение vision/language/action: RT-1/RT-2, OpenVLA, Octo, SmolVLA — архитектуры, данные, fine-tuning и сравнение с RL/IL.
- **tags:** `domain/robotics`, `domain/embodied-ai`, `domain/multimodal`, `concept/vla`, `concept/foundation-models`
- **aliases:** VLA, Vision-Language-Action, RT-1, RT-2, OpenVLA, SmolVLA

## Nested notes

### [AI Harness Engineering (Tejas, IBM)](../topics/code-agents-autoresearch-and-loopy-era/ai-harness-engineering-tejas-ibm.md)

- **description:** Конспект про harness engineering: guardrails, verify step и почему обвязка агента важнее одного удачного промпта.
- **parent:** `code-agents-autoresearch-and-loopy-era`

### [Stop Babysitting Your Agents (Claude Code)](../topics/code-agents-autoresearch-and-loopy-era/stop-babysitting-your-agents-claude-code.md)

- **description:** Конспект про verification skills, /loop и Routines в Claude Code: как меньше babysit'ить агентов и больше опираться на verify-циклы.
- **parent:** `code-agents-autoresearch-and-loopy-era`

