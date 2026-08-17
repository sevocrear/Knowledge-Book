"""Canonical Obsidian-style metadata for knowledge-book topics.

Used by apply/validate scripts and for generating docs/ indexes for RAG.
"""

from __future__ import annotations

from typing import TypedDict


class TopicMeta(TypedDict):
    title: str
    description: str
    tags: list[str]
    aliases: list[str]
    related: list[str]
    status: str
    lang: str
    type: str  # topic | note


# status: canonical | notes | draft
TOPIC_METADATA: dict[str, TopicMeta] = {
    "action-recognition-and-object-tracking-metrics": {
        "title": "Метрики оценки Action Recognition и Object Tracking",
        "description": (
            "Протоколы и метрики для video action recognition, temporal localization, "
            "SOT и MOT: Top-1/Top-5, mAP@tIoU, Success AUC, IDF1, MOTA, HOTA."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "domain/video",
            "concept/metrics",
            "concept/tracking",
            "concept/action-recognition",
        ],
        "aliases": [
            "action recognition metrics",
            "object tracking metrics",
            "MOTA",
            "HOTA",
            "IDF1",
        ],
        "related": [
            "how-models-predict-confidence-and-calibration",
            "non-maximum-suppression-nms",
            "unscented-kalman-filter-and-tracking",
            "sota-metrics-for-detection-segmentation-multiclass-classification",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "arcface-and-angular-margin-losses-for-identification": {
        "title": "ArcFace and Angular-Margin Losses for Identification",
        "description": (
            "Additive angular margin loss для идентификации: геометрия на гиперсфере, "
            "сравнение с CosFace/SphereFace, face/SKU/re-ID и open-set пороги."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/metric-learning",
            "concept/loss",
            "concept/embeddings",
            "concept/face-recognition",
        ],
        "aliases": ["ArcFace", "CosFace", "SphereFace", "angular margin loss"],
        "related": [
            "contrastive-and-metric-learning-for-fine-grained-visual-recognition",
            "embeddings-and-embedding-matrix",
            "roc-curve-and-roc-auc",
            "how-models-predict-confidence-and-calibration",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "bayes-theorem-and-probability-foundations": {
        "title": "Теорема Байеса и основы теории вероятностей",
        "description": (
            "Аксиомы Колмогорова, условная вероятность, формула полной вероятности, "
            "теорема Байеса, MAP/MLE и наивный Байес в ML."
        ),
        "tags": [
            "kb/topic",
            "domain/math",
            "domain/ml-foundations",
            "concept/probability",
            "concept/bayes",
        ],
        "aliases": ["Bayes theorem", "теорема Байеса", "conditional probability", "MAP", "MLE"],
        "related": [
            "gaussian-distribution",
            "variational-autoencoders-vaes",
            "unscented-kalman-filter-and-tracking",
            "retrieval-augmented-generation-rag",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "classification-losses-cross-entropy-focal-loss": {
        "title": "Cross Entropy и Focal Loss",
        "description": (
            "Бинарная и многоклассовая кросс-энтропия, Focal Loss (α, γ) для дисбаланса "
            "и детекции (RetinaNet), когда выбирать CE vs Focal."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "domain/ml-foundations",
            "concept/loss",
            "concept/classification",
        ],
        "aliases": ["cross entropy", "focal loss", "кросс-энтропия", "RetinaNet loss"],
        "related": [
            "detection-segmentation-3d-losses",
            "non-maximum-suppression-nms",
            "convolutions-and-parameters-in-cnn",
            "how-models-predict-confidence-and-calibration",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "code-agents-autoresearch-and-loopy-era": {
        "title": "Code Agents, AutoResearch и Loopy Era",
        "description": (
            "Оркестрация code agents, AutoResearch loops, verification gates, harness "
            "engineering и переход от ручного кода к управлению агентными циклами."
        ),
        "tags": [
            "kb/topic",
            "domain/agents",
            "domain/llm",
            "concept/orchestration",
            "concept/verification",
            "concept/autoresearch",
        ],
        "aliases": [
            "code agents",
            "AutoResearch",
            "loopy era",
            "agent harness",
            "Karpathy agents",
        ],
        "related": [
            "retrieval-augmented-generation-rag",
            "hyperparameter-tuning",
            "low-rank-adaptation-lora",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "contrastive-and-metric-learning-for-fine-grained-visual-recognition": {
        "title": "Contrastive & Metric Learning for Fine-Grained Visual Recognition",
        "description": (
            "Contrastive/triplet/InfoNCE и proxy losses для fine-grained retrieval: "
            "mining, Recall@K, ANN-индексы и continual learning новых классов."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/metric-learning",
            "concept/contrastive-learning",
            "concept/embeddings",
            "concept/retrieval",
        ],
        "aliases": [
            "contrastive learning",
            "metric learning",
            "triplet loss",
            "InfoNCE",
            "fine-grained recognition",
        ],
        "related": [
            "arcface-and-angular-margin-losses-for-identification",
            "embeddings-and-embedding-matrix",
            "roc-curve-and-roc-auc",
            "dinov3-self-supervised-vision-transformer-and-2d-rope",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "convolutions-and-parameters-in-cnn": {
        "title": "Свёртки в CNN, размеры карт признаков и число параметров",
        "description": (
            "Почему популярны ядра 3×3, формулы размера feature map, transposed conv "
            "и подсчёт параметров Conv/Linear/BatchNorm/depthwise."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "domain/dl-foundations",
            "concept/cnn",
            "concept/convolution",
        ],
        "aliases": ["CNN convolutions", "feature map size", "transposed convolution", "DeConv"],
        "related": [
            "non-maximum-suppression-nms",
            "normalization-layers-batchnorm-layernorm",
            "transformers-attention-and-vision-transformers-vit",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "decision-trees": {
        "title": "Деревья решений (Decision Trees)",
        "description": (
            "Структура дерева, Gini/энтропия/Information Gain, ID3/C4.5/CART, "
            "переобучение и связь с ансамблями Random Forest/XGBoost."
        ),
        "tags": [
            "kb/topic",
            "domain/classical-ml",
            "concept/decision-trees",
            "concept/ensemble",
        ],
        "aliases": ["decision trees", "деревья решений", "Gini", "CART", "Information Gain"],
        "related": [
            "ensemble-methods-model-combination",
            "roc-curve-and-roc-auc",
            "classification-losses-cross-entropy-focal-loss",
            "hyperparameter-tuning",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "deep-reinforcement-learning": {
        "title": "Deep Reinforcement Learning",
        "description": (
            "MDP, DQN, policy gradient, actor-critic (PPO/SAC/TD3), sim-to-real "
            "и применения в робототехнике и автономном вождении."
        ),
        "tags": [
            "kb/topic",
            "domain/rl",
            "domain/robotics",
            "concept/rl",
            "concept/ppo",
            "concept/sac",
        ],
        "aliases": ["Deep RL", "reinforcement learning", "DQN", "PPO", "SAC", "TD3"],
        "related": [
            "vision-language-action-models-vla",
            "vision-based-robot-training-methods",
            "unscented-kalman-filter-and-tracking",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "detection-segmentation-3d-losses": {
        "title": "Loss функции для детекции, сегментации и 3D-детекции",
        "description": (
            "Составные loss'ы детекторов: CE/Focal/QFL, L1/IoU/GIoU/DIoU/CIoU, "
            "Dice/Tversky для сегментации и 3D/BEV losses."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/loss",
            "concept/object-detection",
            "concept/segmentation",
            "concept/3d-detection",
        ],
        "aliases": ["detection losses", "IoU loss", "Dice loss", "GIoU", "3D detection loss"],
        "related": [
            "classification-losses-cross-entropy-focal-loss",
            "non-maximum-suppression-nms",
            "sota-metrics-for-detection-segmentation-multiclass-classification",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "diffusion-models": {
        "title": "Diffusion Models",
        "description": (
            "Forward/reverse diffusion, DDPM/DDIM, latent diffusion (Stable Diffusion), "
            "Consistency Models, Flow Matching и DiT."
        ),
        "tags": [
            "kb/topic",
            "domain/generative",
            "concept/diffusion",
            "concept/score-matching",
            "concept/latent-diffusion",
        ],
        "aliases": [
            "diffusion models",
            "DDPM",
            "DDIM",
            "Stable Diffusion",
            "Flow Matching",
            "DiT",
        ],
        "related": [
            "variational-autoencoders-vaes",
            "generative-adversarial-networks-gans",
            "gaussian-distribution",
        ],
        "status": "canonical",
        "lang": "en",
        "type": "topic",
    },
    "dinov3-self-supervised-vision-transformer-and-2d-rope": {
        "title": "DINOv3: Self-Supervised Vision Transformer и 2D RoPE",
        "description": (
            "Self-supervised ViT (student–teacher), 2D RoPE для патчей, глобальные "
            "и dense-фичи для классификации, детекции и сегментации."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/self-supervised",
            "concept/vit",
            "concept/rope",
            "concept/dino",
        ],
        "aliases": ["DINOv3", "DINOv2", "2D RoPE", "self-supervised ViT"],
        "related": [
            "transformers-attention-and-vision-transformers-vit",
            "few-shot-anomaly-detection-anomalydino",
            "contrastive-and-metric-learning-for-fine-grained-visual-recognition",
            "non-maximum-suppression-nms",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "embeddings-and-embedding-matrix": {
        "title": "Embeddings and Embedding Matrix",
        "description": (
            "Векторные представления токенов, матрица эмбеддингов V×d, lookup по ID "
            "и роль эмбеддингов в Transformer и RAG."
        ),
        "tags": [
            "kb/topic",
            "domain/nlp",
            "domain/llm",
            "concept/embeddings",
            "concept/tokenization",
        ],
        "aliases": ["embedding matrix", "token embeddings", "эмбеддинги", "word embeddings"],
        "related": [
            "tokenization-and-text-compression-in-llms",
            "transformers-attention-and-vision-transformers-vit",
            "retrieval-augmented-generation-rag",
            "contrastive-and-metric-learning-for-fine-grained-visual-recognition",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "ensemble-methods-model-combination": {
        "title": "Методы комбинирования моделей (Ensemble Methods)",
        "description": (
            "Bagging/boosting/stacking, XGBoost/LightGBM/CatBoost, MoE, distillation "
            "и model merging (TIES/DARE/SLERP) для LLM."
        ),
        "tags": [
            "kb/topic",
            "domain/classical-ml",
            "domain/llm",
            "concept/ensemble",
            "concept/boosting",
            "concept/moe",
            "concept/model-merging",
        ],
        "aliases": [
            "ensemble methods",
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "Mixture of Experts",
            "model merging",
        ],
        "related": [
            "decision-trees",
            "hyperparameter-tuning",
            "roc-curve-and-roc-auc",
            "low-rank-adaptation-lora",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "few-shot-anomaly-detection-anomalydino": {
        "title": "Few-Shot Anomaly Detection: AnomalyDINO",
        "description": (
            "Patch-level nearest neighbor на DINOv2 без обучения: memory bank, "
            "косинусное расстояние и pixel-level anomaly maps для industrial QC."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/anomaly-detection",
            "concept/few-shot",
            "concept/dino",
        ],
        "aliases": ["AnomalyDINO", "few-shot anomaly detection", "patch nearest neighbor"],
        "related": [
            "dinov3-self-supervised-vision-transformer-and-2d-rope",
            "roc-curve-and-roc-auc",
            "sota-metrics-for-detection-segmentation-multiclass-classification",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "gaussian-distribution": {
        "title": "Гауссово распределение (Normal Distribution)",
        "description": (
            "Одномерное и многомерное нормальное распределение, PDF/CDF и роль "
            "гауссианы в VAE, diffusion и Kalman filtering."
        ),
        "tags": [
            "kb/topic",
            "domain/math",
            "domain/ml-foundations",
            "concept/probability",
            "concept/gaussian",
        ],
        "aliases": ["Gaussian distribution", "normal distribution", "гауссово распределение"],
        "related": [
            "bayes-theorem-and-probability-foundations",
            "variational-autoencoders-vaes",
            "diffusion-models",
            "unscented-kalman-filter-and-tracking",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "generative-adversarial-networks-gans": {
        "title": "Generative Adversarial Networks (GANs)",
        "description": (
            "Adversarial training generator/discriminator, mode collapse, современные "
            "варианты GAN и сравнение с VAE и diffusion."
        ),
        "tags": [
            "kb/topic",
            "domain/generative",
            "concept/gan",
            "concept/adversarial-training",
        ],
        "aliases": ["GAN", "Generative Adversarial Networks", "StyleGAN", "mode collapse"],
        "related": [
            "variational-autoencoders-vaes",
            "diffusion-models",
            "gaussian-distribution",
        ],
        "status": "canonical",
        "lang": "en",
        "type": "topic",
    },
    "how-models-predict-confidence-and-calibration": {
        "title": "Confidence, Calibration and Uncertainty",
        "description": (
            "Logits→softmax/sigmoid, reliability diagrams, ECE/Brier, temperature "
            "scaling и aleatoric/epistemic uncertainty (ensembles, MC Dropout)."
        ),
        "tags": [
            "kb/topic",
            "domain/ml-foundations",
            "concept/calibration",
            "concept/uncertainty",
            "concept/confidence",
        ],
        "aliases": [
            "calibration",
            "ECE",
            "temperature scaling",
            "model confidence",
            "uncertainty estimation",
        ],
        "related": [
            "roc-curve-and-roc-auc",
            "classification-losses-cross-entropy-focal-loss",
            "ensemble-methods-model-combination",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "hyperparameter-tuning": {
        "title": "Настройка гиперпараметров (Hyperparameter Tuning)",
        "description": (
            "Grid/Random search, Bayesian Optimization (Optuna/TPE), Hyperband/BOHB, "
            "PBT, CMA-ES, NAS и LR schedules."
        ),
        "tags": [
            "kb/topic",
            "domain/mlops",
            "domain/ml-foundations",
            "concept/hyperparameter-tuning",
            "concept/bayesian-optimization",
            "concept/nas",
        ],
        "aliases": [
            "hyperparameter tuning",
            "Optuna",
            "Hyperband",
            "Bayesian Optimization",
            "NAS",
        ],
        "related": [
            "ensemble-methods-model-combination",
            "decision-trees",
            "low-rank-adaptation-lora",
            "bayes-theorem-and-probability-foundations",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "low-rank-adaptation-lora": {
        "title": "Low-Rank Adaptation (LoRA)",
        "description": (
            "PEFT через низкоранговые адаптеры ΔW≈BA: математика, QLoRA/AdaLoRA/DoRA, "
            "эффективность памяти и практика в Hugging Face PEFT."
        ),
        "tags": [
            "kb/topic",
            "domain/llm",
            "concept/peft",
            "concept/lora",
            "concept/fine-tuning",
        ],
        "aliases": ["LoRA", "QLoRA", "AdaLoRA", "DoRA", "PEFT"],
        "related": [
            "transformers-attention-and-vision-transformers-vit",
            "retrieval-augmented-generation-rag",
            "vision-language-action-models-vla",
            "ensemble-methods-model-combination",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "non-maximum-suppression-nms": {
        "title": "Non-Maximum Suppression (NMS) и современные end-to-end детекторы",
        "description": (
            "Классический NMS/Soft-NMS, проблемы в production и переход к NMS-free "
            "детекторам: DETR, RT-DETR, YOLO26 (dual-head)."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/nms",
            "concept/object-detection",
            "concept/end-to-end-detection",
        ],
        "aliases": ["NMS", "Non-Maximum Suppression", "YOLO26", "DETR", "RT-DETR", "Soft-NMS"],
        "related": [
            "unscented-kalman-filter-and-tracking",
            "transformers-attention-and-vision-transformers-vit",
            "detection-segmentation-3d-losses",
            "sota-metrics-for-detection-segmentation-multiclass-classification",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "normalization-layers-batchnorm-layernorm": {
        "title": "Batch Normalization и Layer Normalization",
        "description": (
            "Нормализация активаций: формулы BatchNorm vs LayerNorm, влияние на "
            "обучение, выбор для CNN и Transformer."
        ),
        "tags": [
            "kb/topic",
            "domain/dl-foundations",
            "concept/normalization",
            "concept/batchnorm",
            "concept/layernorm",
        ],
        "aliases": ["BatchNorm", "LayerNorm", "Batch Normalization", "Layer Normalization"],
        "related": [
            "convolutions-and-parameters-in-cnn",
            "transformers-attention-and-vision-transformers-vit",
            "deep-reinforcement-learning",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "retrieval-augmented-generation-rag": {
        "title": "Retrieval-Augmented Generation (RAG)",
        "description": (
            "Архитектуры RAG (Naive/Advanced/Modular/Self-RAG/Corrective/LightRAG), "
            "retriever/reranker, chunking, оценка и production-практики."
        ),
        "tags": [
            "kb/topic",
            "domain/nlp",
            "domain/llm",
            "concept/rag",
            "concept/retrieval",
            "concept/vector-search",
        ],
        "aliases": [
            "RAG",
            "Retrieval-Augmented Generation",
            "Self-RAG",
            "Corrective RAG",
            "LightRAG",
            "vector database",
        ],
        "related": [
            "embeddings-and-embedding-matrix",
            "tokenization-and-text-compression-in-llms",
            "transformers-attention-and-vision-transformers-vit",
            "code-agents-autoresearch-and-loopy-era",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "roc-curve-and-roc-auc": {
        "title": "ROC-кривые и ROC AUC",
        "description": (
            "TPR/FPR, построение ROC, AUC как метрика ранжирования, выбор порога "
            "(Youden’s J) и связь с PR-кривыми."
        ),
        "tags": [
            "kb/topic",
            "domain/ml-foundations",
            "concept/metrics",
            "concept/roc",
            "concept/classification",
        ],
        "aliases": ["ROC", "ROC AUC", "TPR", "FPR", "Youden J", "PR curve"],
        "related": [
            "how-models-predict-confidence-and-calibration",
            "sota-metrics-for-detection-segmentation-multiclass-classification",
            "decision-trees",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "sota-metrics-for-detection-segmentation-multiclass-classification": {
        "title": "SOTA-метрики для детекции, сегментации и мультиклассовой классификации",
        "description": (
            "COCO AP/AR, mIoU/Mask AP/PQ, Top-1/Top-5, macro/micro F1 и калибровочные "
            "метрики как стандартные протоколы сравнения."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/metrics",
            "concept/object-detection",
            "concept/segmentation",
            "concept/classification",
        ],
        "aliases": ["COCO AP", "mIoU", "panoptic PQ", "Mask AP", "Top-1 accuracy"],
        "related": [
            "roc-curve-and-roc-auc",
            "how-models-predict-confidence-and-calibration",
            "non-maximum-suppression-nms",
            "action-recognition-and-object-tracking-metrics",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "support-vector-machines-svm-and-kernel-trick": {
        "title": "Support Vector Machines (SVM) и Kernel Trick",
        "description": (
            "Max-margin классификация, soft-margin C, dual formulation и kernel trick "
            "(linear/poly/RBF) без явного φ(x)."
        ),
        "tags": [
            "kb/topic",
            "domain/classical-ml",
            "concept/svm",
            "concept/kernel-methods",
        ],
        "aliases": ["SVM", "support vector machine", "kernel trick", "RBF kernel", "margin"],
        "related": [
            "decision-trees",
            "roc-curve-and-roc-auc",
            "bayes-theorem-and-probability-foundations",
            "retrieval-augmented-generation-rag",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "tokenization-and-text-compression-in-llms": {
        "title": "Tokenization and Text Compression in LLMs",
        "description": (
            "Word/char/BPE/WordPiece/Unigram токенизация как сжатие текста перед LLM "
            "и влияние на стоимость attention."
        ),
        "tags": [
            "kb/topic",
            "domain/nlp",
            "domain/llm",
            "concept/tokenization",
            "concept/bpe",
            "concept/compression",
        ],
        "aliases": ["tokenization", "BPE", "WordPiece", "Unigram LM", "byte-level BPE"],
        "related": [
            "embeddings-and-embedding-matrix",
            "transformers-attention-and-vision-transformers-vit",
            "retrieval-augmented-generation-rag",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "transformers-attention-and-vision-transformers-vit": {
        "title": "Transformers, Attention и Vision Transformers (ViT)",
        "description": (
            "Scaled dot-product attention, QKV, KV cache, positional encodings "
            "(в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация."
        ),
        "tags": [
            "kb/topic",
            "domain/nlp",
            "domain/cv",
            "domain/llm",
            "concept/attention",
            "concept/transformer",
            "concept/vit",
        ],
        "aliases": [
            "Transformer",
            "self-attention",
            "ViT",
            "Vision Transformer",
            "KV cache",
            "RoPE",
            "DETR",
        ],
        "related": [
            "normalization-layers-batchnorm-layernorm",
            "dinov3-self-supervised-vision-transformer-and-2d-rope",
            "non-maximum-suppression-nms",
            "low-rank-adaptation-lora",
            "retrieval-augmented-generation-rag",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "unscented-kalman-filter-and-tracking": {
        "title": "Unscented Kalman Filter и современные методы отслеживания",
        "description": (
            "UKF vs KF/EKF/PF, sigma-points, DeepSORT/ByteTrack/Transformer tracking "
            "и χ²-тест выбросов в трекинге."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "domain/robotics",
            "concept/kalman-filter",
            "concept/tracking",
            "concept/ukf",
        ],
        "aliases": [
            "UKF",
            "Unscented Kalman Filter",
            "Kalman Filter",
            "ByteTrack",
            "DeepSORT",
            "object tracking",
        ],
        "related": [
            "gaussian-distribution",
            "non-maximum-suppression-nms",
            "action-recognition-and-object-tracking-metrics",
            "bayes-theorem-and-probability-foundations",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "variational-autoencoders-vaes": {
        "title": "Variational Autoencoders (VAEs)",
        "description": (
            "ELBO, encoder/decoder, reparameterization trick, латентное пространство "
            "и роль VAE в современных generative pipelines."
        ),
        "tags": [
            "kb/topic",
            "domain/generative",
            "concept/vae",
            "concept/latent-variable",
            "concept/elbo",
        ],
        "aliases": ["VAE", "Variational Autoencoder", "ELBO", "reparameterization trick"],
        "related": [
            "gaussian-distribution",
            "generative-adversarial-networks-gans",
            "diffusion-models",
            "bayes-theorem-and-probability-foundations",
        ],
        "status": "canonical",
        "lang": "en",
        "type": "topic",
    },
    "vision-based-robot-training-methods": {
        "title": "Vision-Based Robot Training Methods",
        "description": (
            "Imitation learning, RL и VLA для визуального обучения роботов: OpenVLA, "
            "Octo, RT-1/RT-2, Open X-Embodiment и sim-to-real."
        ),
        "tags": [
            "kb/topic",
            "domain/robotics",
            "domain/embodied-ai",
            "concept/imitation-learning",
            "concept/vla",
            "concept/sim-to-real",
        ],
        "aliases": [
            "robot training",
            "OpenVLA",
            "Octo",
            "Open X-Embodiment",
            "vision-based robotics",
        ],
        "related": [
            "vision-language-action-models-vla",
            "deep-reinforcement-learning",
            "low-rank-adaptation-lora",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "vision-language-action-models-vla": {
        "title": "Vision-Language-Action (VLA) Models",
        "description": (
            "Объединение vision/language/action: RT-1/RT-2, OpenVLA, Octo, SmolVLA — "
            "архитектуры, данные, fine-tuning и сравнение с RL/IL."
        ),
        "tags": [
            "kb/topic",
            "domain/robotics",
            "domain/embodied-ai",
            "domain/multimodal",
            "concept/vla",
            "concept/foundation-models",
        ],
        "aliases": ["VLA", "Vision-Language-Action", "RT-1", "RT-2", "OpenVLA", "SmolVLA"],
        "related": [
            "vision-based-robot-training-methods",
            "deep-reinforcement-learning",
            "transformers-attention-and-vision-transformers-vit",
            "low-rank-adaptation-lora",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "computer-vision-business-guide": {
        "title": "Компьютерное зрение: вводное руководство для бизнеса",
        "description": (
            "Бизнес-книга (~30+ стр.) на понятном русском: что такое CV, как работает, "
            "задачи, отрасли (промышленность, офис, город, retail), внедрение, этика. EPUB + HTML."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "domain/mlops",
            "concept/classification",
            "concept/object-detection",
            "concept/segmentation",
        ],
        "aliases": [
            "Computer Vision Business Guide",
            "CV для бизнеса",
            "вводное руководство по компьютерному зрению",
        ],
        "related": [
            "convolutions-and-parameters-in-cnn",
            "non-maximum-suppression-nms",
            "sota-metrics-for-detection-segmentation-multiclass-classification",
            "few-shot-anomaly-detection-anomalydino",
            "action-recognition-and-object-tracking-metrics",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
}

NOTE_METADATA: dict[str, TopicMeta] = {
    "code-agents-autoresearch-and-loopy-era/stop-babysitting-your-agents-claude-code": {
        "title": "Stop Babysitting Your Agents (Claude Code)",
        "description": (
            "Конспект про verification skills, /loop и Routines в Claude Code: "
            "как меньше babysit'ить агентов и больше опираться на verify-циклы."
        ),
        "tags": [
            "kb/note",
            "domain/agents",
            "concept/verification",
            "concept/claude-code",
            "source/youtube",
        ],
        "aliases": ["Stop Babysitting Your Agents", "Claude Code loops"],
        "related": ["code-agents-autoresearch-and-loopy-era"],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    "code-agents-autoresearch-and-loopy-era/ai-harness-engineering-tejas-ibm": {
        "title": "AI Harness Engineering (Tejas, IBM)",
        "description": (
            "Конспект про harness engineering: guardrails, verify step и почему "
            "обвязка агента важнее одного удачного промпта."
        ),
        "tags": [
            "kb/note",
            "domain/agents",
            "concept/harness",
            "concept/verification",
            "source/youtube",
        ],
        "aliases": ["AI Harness Engineering", "agent harness", "Tejas IBM"],
        "related": ["code-agents-autoresearch-and-loopy-era"],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
}

# Maps of Content: docs/mocs/<name>.md
MOCS: dict[str, dict[str, object]] = {
    "foundations": {
        "title": "MOC: Mathematical & ML Foundations",
        "description": "Вероятность, метрики, классический ML и базовые строительные блоки.",
        "tags": ["kb/moc", "domain/ml-foundations"],
        "topics": [
            "bayes-theorem-and-probability-foundations",
            "gaussian-distribution",
            "roc-curve-and-roc-auc",
            "how-models-predict-confidence-and-calibration",
            "decision-trees",
            "support-vector-machines-svm-and-kernel-trick",
            "hyperparameter-tuning",
            "ensemble-methods-model-combination",
            "normalization-layers-batchnorm-layernorm",
        ],
    },
    "generative-models": {
        "title": "MOC: Generative Models",
        "description": "VAE, GAN и diffusion — три основных семейства генеративных моделей.",
        "tags": ["kb/moc", "domain/generative"],
        "topics": [
            "variational-autoencoders-vaes",
            "generative-adversarial-networks-gans",
            "diffusion-models",
            "gaussian-distribution",
        ],
    },
    "nlp-llm": {
        "title": "MOC: NLP, LLM & RAG",
        "description": "Токенизация, эмбеддинги, transformers, LoRA, RAG и code agents.",
        "tags": ["kb/moc", "domain/nlp", "domain/llm"],
        "topics": [
            "tokenization-and-text-compression-in-llms",
            "embeddings-and-embedding-matrix",
            "transformers-attention-and-vision-transformers-vit",
            "low-rank-adaptation-lora",
            "retrieval-augmented-generation-rag",
            "code-agents-autoresearch-and-loopy-era",
        ],
    },
    "computer-vision": {
        "title": "MOC: Computer Vision",
        "description": "CNN, detection/segmentation, metric learning, SSL и tracking metrics.",
        "tags": ["kb/moc", "domain/cv"],
        "topics": [
            "computer-vision-business-guide",
            "convolutions-and-parameters-in-cnn",
            "non-maximum-suppression-nms",
            "detection-segmentation-3d-losses",
            "classification-losses-cross-entropy-focal-loss",
            "sota-metrics-for-detection-segmentation-multiclass-classification",
            "transformers-attention-and-vision-transformers-vit",
            "dinov3-self-supervised-vision-transformer-and-2d-rope",
            "few-shot-anomaly-detection-anomalydino",
            "contrastive-and-metric-learning-for-fine-grained-visual-recognition",
            "arcface-and-angular-margin-losses-for-identification",
            "action-recognition-and-object-tracking-metrics",
            "unscented-kalman-filter-and-tracking",
        ],
    },
    "robotics-embodied": {
        "title": "MOC: Robotics & Embodied AI",
        "description": "Deep RL, VLA и vision-based обучение роботов.",
        "tags": ["kb/moc", "domain/robotics", "domain/embodied-ai"],
        "topics": [
            "deep-reinforcement-learning",
            "vision-language-action-models-vla",
            "vision-based-robot-training-methods",
            "unscented-kalman-filter-and-tracking",
        ],
    },
}
