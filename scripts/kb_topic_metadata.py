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
    "agent-protocols-mcp-acp-ucp-and-harness": {
        "title": "MCP, ACP, UCP и Agent Harness",
        "description": (
            "Слои агентных протоколов (MCP, ACP, UCP, A2A) и agent harness: что к чему "
            "подключается, чем не путать аббревиатуры и как собрать эффективный harness в Cursor."
        ),
        "tags": [
            "kb/topic",
            "domain/agents",
            "domain/llm",
            "concept/mcp",
            "concept/harness",
            "concept/orchestration",
        ],
        "aliases": [
            "Model Context Protocol",
            "Agent Client Protocol",
            "Universal Commerce Protocol",
            "Agentic Commerce Protocol",
            "agent harness",
            "MCP ACP UCP",
        ],
        "related": [
            "code-agents-autoresearch-and-loopy-era",
            "retrieval-augmented-generation-rag",
            "ml-system-design-for-cv-and-nlp",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
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
            "video-codecs-h264-h265-and-gpu-decode",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "arcface-and-angular-margin-losses-for-identification": {
        "title": "ArcFace и angular-margin losses для идентификации",
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
            "agent-protocols-mcp-acp-ucp-and-harness",
            "retrieval-augmented-generation-rag",
            "hyperparameter-tuning",
            "low-rank-adaptation-lora",
            "ml-system-design-for-cv-and-nlp",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "contrastive-and-metric-learning-for-fine-grained-visual-recognition": {
        "title": "Contrastive и metric learning для fine-grained распознавания",
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
            "video-codecs-h264-h265-and-gpu-decode",
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
        "title": "Deep Reinforcement Learning (глубокое RL)",
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
        "title": "Diffusion Models (диффузионные модели)",
        "description": (
            "Прямой и обратный процесс диффузии, DDPM/DDIM, latent diffusion (Stable Diffusion), "
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
        "lang": "ru",
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
        "title": "Эмбеддинги и матрица эмбеддингов",
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
            "ml-system-design-for-cv-and-nlp",
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
        "title": "Generative Adversarial Networks (GAN)",
        "description": (
            "Состязательное обучение generator/discriminator, mode collapse, современные "
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
        "lang": "ru",
        "type": "topic",
    },
    "how-models-predict-confidence-and-calibration": {
        "title": "Уверенность, калибровка и неопределённость",
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
            "ml-system-design-for-cv-and-nlp",
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
            "ml-system-design-for-cv-and-nlp",
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
    "metric-learning-losses-and-miners": {
        "title": "Лоссы metric learning и подбор майнеров",
        "description": (
            "Каталог лоссов metric learning (contrastive, triplet, N-pair, Multi-Similarity, Circle, "
            "InfoNCE/SupCon, Proxy-NCA/Anchor, SoftTriple, ArcFace/CosFace/AdaFace): формулы, интуиция, "
            "какие майнеры к какому лоссу и когда что выбирать."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "concept/metric-learning",
            "concept/loss",
            "concept/embeddings",
            "concept/contrastive-learning",
        ],
        "aliases": [
            "triplet loss",
            "Multi-Similarity loss",
            "Circle loss",
            "Proxy-Anchor",
            "hard negative mining",
            "semi-hard mining",
            "miners",
        ],
        "related": [
            "arcface-and-angular-margin-losses-for-identification",
            "contrastive-and-metric-learning-for-fine-grained-visual-recognition",
            "classification-losses-cross-entropy-focal-loss",
            "embeddings-and-embedding-matrix",
            "roc-curve-and-roc-auc",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "ml-system-design-for-cv-and-nlp": {
        "title": "System Design для Computer Vision и NLP",
        "description": (
            "Ёмкость, балансировка, serving на 100 vs 1000 клиентов, dynamic batching, "
            "KV cache и обработка 10–50 видеопотоков."
        ),
        "tags": [
            "kb/topic",
            "domain/mlops",
            "domain/cv",
            "domain/nlp",
            "concept/system-design",
            "concept/model-serving",
            "concept/load-balancing",
        ],
        "aliases": [
            "System Design",
            "model serving",
            "inference serving",
            "load balancing",
            "multi-camera pipeline",
            "dynamic batching",
        ],
        "related": [
            "transformers-attention-and-vision-transformers-vit",
            "non-maximum-suppression-nms",
            "unscented-kalman-filter-and-tracking",
            "retrieval-augmented-generation-rag",
            "embeddings-and-embedding-matrix",
            "hyperparameter-tuning",
            "how-models-predict-confidence-and-calibration",
            "video-codecs-h264-h265-and-gpu-decode",
            "agent-protocols-mcp-acp-ucp-and-harness",
            "triton-inference-server-and-gpu-model-serving",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "triton-inference-server-and-gpu-model-serving": {
        "title": "Triton Inference Server и развёртывание моделей на 1–N GPU",
        "description": (
            "NVIDIA Triton: dynamic batching, concurrent execution, ensembles; "
            "полезен ли на 1 GPU; SOTA serving (vLLM, TensorRT-LLM, SGLang) для 100× пользователей."
        ),
        "tags": [
            "kb/topic",
            "domain/mlops",
            "domain/llm",
            "domain/cv",
            "concept/model-serving",
            "concept/triton",
            "concept/inference",
        ],
        "aliases": [
            "Triton Inference Server",
            "NVIDIA Triton",
            "GPU model serving",
            "inference serving stack",
            "vLLM vs Triton",
        ],
        "related": [
            "ml-system-design-for-cv-and-nlp",
            "transformers-attention-and-vision-transformers-vit",
            "tokenization-and-text-compression-in-llms",
            "retrieval-augmented-generation-rag",
            "video-codecs-h264-h265-and-gpu-decode",
            "how-models-predict-confidence-and-calibration",
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
            "ml-system-design-for-cv-and-nlp",
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
            "agent-protocols-mcp-acp-ucp-and-harness",
            "ml-system-design-for-cv-and-nlp",
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
        "title": "Токенизация и сжатие текста в LLM",
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
            "video-codecs-h264-h265-and-gpu-decode",
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
            "ml-system-design-for-cv-and-nlp",
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
            "video-codecs-h264-h265-and-gpu-decode",
            "ml-system-design-for-cv-and-nlp",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "variational-autoencoders-vaes": {
        "title": "Variational Autoencoders (VAE)",
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
        "lang": "ru",
        "type": "topic",
    },
    "video-codecs-h264-h265-and-gpu-decode": {
        "title": "Видеокодеки H.264/H.265 и GPU-декодирование",
        "description": (
            "Intra/inter сжатие, GOP, H.264 vs H.265, типичный битрейт и hardware "
            "decode IP-камер (NVDEC, VAAPI, Quick Sync)."
        ),
        "tags": [
            "kb/topic",
            "domain/cv",
            "domain/video",
            "concept/video-codec",
            "concept/h264",
            "concept/h265",
            "concept/hardware-decode",
        ],
        "aliases": ["H.264", "H.265", "HEVC", "AVC", "NVDEC", "видеокодек"],
        "related": [
            "action-recognition-and-object-tracking-metrics",
            "convolutions-and-parameters-in-cnn",
            "tokenization-and-text-compression-in-llms",
            "unscented-kalman-filter-and-tracking",
            "ml-system-design-for-cv-and-nlp",
        ],
        "status": "canonical",
        "lang": "ru",
        "type": "topic",
    },
    "vision-based-robot-training-methods": {
        "title": "Vision-based обучение роботов",
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
        "title": "Модели Vision-Language-Action (VLA)",
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
        "related": [
            "code-agents-autoresearch-and-loopy-era",
            "agent-protocols-mcp-acp-ucp-and-harness",
        ],
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
        "related": [
            "code-agents-autoresearch-and-loopy-era",
            "agent-protocols-mcp-acp-ucp-and-harness",
        ],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    # --- storyboards (auto) ---
    "arcface-and-angular-margin-losses-for-identification/visualizations/hyperframes/storyboard": {
        "title": "Сториборд клипа: ArcFace — угловой margin на гиперсфере",
        "description": (
            "Сториборд и команды сборки HyperFrames-клипа arcface_angular_margin: единичная сфера и угол θ, что делает additive angular margin m, компактные классы с зазором и сравнение SphereFace / CosFace / ArcFace."
        ),
        "tags": [
            "kb/note",
            "kb/visualization",
            "domain/cv",
            "concept/metric-learning",
            "concept/face-recognition",
        ],
        "aliases": ["ArcFace storyboard", "arcface_angular_margin"],
        "related": [
            "arcface-and-angular-margin-losses-for-identification",
            "contrastive-and-metric-learning-for-fine-grained-visual-recognition",
        ],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    "contrastive-and-metric-learning-for-fine-grained-visual-recognition/visualizations/hyperframes/storyboard": {
        "title": "Сториборд клипа: contrastive / metric learning и triplet loss",
        "description": (
            "Сториборд и команды сборки HyperFrames-клипа contrastive_embedding_space: зачем эмбеддинги вместо классификатора, triplet loss с margin и semi-hard mining, кластеры, поиск ближайших соседей и порог τ в проде."
        ),
        "tags": [
            "kb/note",
            "kb/visualization",
            "domain/cv",
            "concept/contrastive-learning",
            "concept/metric-learning",
        ],
        "aliases": ["metric learning storyboard", "contrastive_embedding_space"],
        "related": [
            "contrastive-and-metric-learning-for-fine-grained-visual-recognition",
            "arcface-and-angular-margin-losses-for-identification",
        ],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    "ml-system-design-for-cv-and-nlp/visualizations/hyperframes/storyboard": {
        "title": "Сториборд клипа: serving — закон очередей, балансировщик, dynamic batching",
        "description": (
            "Сториборд и команды сборки HyperFrames-клипа serving-load-balancer-batching: ρ = λ/μ на 100 и 1000 клиентов, реплики за балансировщиком с автоскейлом по очереди, dynamic batching на GPU-ноде."
        ),
        "tags": [
            "kb/note",
            "kb/visualization",
            "domain/mlops",
            "concept/system-design",
            "concept/model-serving",
        ],
        "aliases": ["serving storyboard", "serving-load-balancer-batching"],
        "related": [
            "ml-system-design-for-cv-and-nlp",
            "triton-inference-server-and-gpu-model-serving",
        ],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    "retrieval-augmented-generation-rag/visualizations/hyperframes/storyboard": {
        "title": "Сториборд клипа: RAG — индексация, retrieval, augmentation, generation",
        "description": (
            "Сториборд и команды сборки HyperFrames-клипа rag_pipeline: почему не просто LLM, offline-индексация документов в vector store, online-поиск Top-K чанков, промпт с контекстом и ответ с источниками."
        ),
        "tags": [
            "kb/note",
            "kb/visualization",
            "domain/nlp",
            "concept/rag",
            "concept/embeddings",
        ],
        "aliases": ["RAG storyboard", "rag_pipeline"],
        "related": [
            "retrieval-augmented-generation-rag",
            "embeddings-and-embedding-matrix",
        ],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    "transformers-attention-and-vision-transformers-vit/visualizations/hyperframes/storyboard": {
        "title": "Сториборд клипа: ViT — патчи, self-attention и CLS",
        "description": (
            "Сториборд и команды сборки HyperFrames-клипа vit_patches_attention: картинка → патчи и токены с CLS, scaled dot-product attention для строки CLS, энкодер из L блоков и голова классификации."
        ),
        "tags": [
            "kb/note",
            "kb/visualization",
            "domain/cv",
            "concept/attention",
            "concept/vit",
        ],
        "aliases": ["ViT storyboard", "vit_patches_attention"],
        "related": [
            "transformers-attention-and-vision-transformers-vit",
            "embeddings-and-embedding-matrix",
        ],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    "video-codecs-h264-h265-and-gpu-decode/visualizations/hyperframes/storyboard": {
        "title": "Сториборд клипа: видеокодеки — intra, inter и GOP из I/P/B",
        "description": (
            "Сториборд и команды сборки HyperFrames-клипа gop-i-p-b-prediction: блок → DCT → квантование, вектор движения и остаток, GOP с I/P/B-кадрами, ссылками между ними и бюджетом бит."
        ),
        "tags": [
            "kb/note",
            "kb/visualization",
            "domain/cv",
            "concept/compression",
        ],
        "aliases": ["codec storyboard", "gop-i-p-b-prediction"],
        "related": [
            "video-codecs-h264-h265-and-gpu-decode",
            "ml-system-design-for-cv-and-nlp",
        ],
        "status": "notes",
        "lang": "ru",
        "type": "note",
    },
    # --- end storyboards (auto) ---
}

# Maps of Content: docs/mocs/<name>.md
MOCS: dict[str, dict[str, object]] = {
    "foundations": {
        "title": "MOC: математика и основы ML",
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
        "title": "MOC: генеративные модели",
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
        "title": "MOC: NLP, LLM и RAG",
        "description": "Токенизация, эмбеддинги, transformers, LoRA, RAG, code agents, протоколы агентов и serving.",
        "tags": ["kb/moc", "domain/nlp", "domain/llm"],
        "topics": [
            "tokenization-and-text-compression-in-llms",
            "embeddings-and-embedding-matrix",
            "transformers-attention-and-vision-transformers-vit",
            "low-rank-adaptation-lora",
            "retrieval-augmented-generation-rag",
            "code-agents-autoresearch-and-loopy-era",
            "agent-protocols-mcp-acp-ucp-and-harness",
            "ml-system-design-for-cv-and-nlp",
        ],
    },
    "computer-vision": {
        "title": "MOC: компьютерное зрение",
        "description": "CNN, detection/segmentation, video codecs, serving камер, metric learning, SSL и tracking metrics.",
        "tags": ["kb/moc", "domain/cv"],
        "topics": [
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
            "metric-learning-losses-and-miners",
            "action-recognition-and-object-tracking-metrics",
            "unscented-kalman-filter-and-tracking",
            "video-codecs-h264-h265-and-gpu-decode",
            "ml-system-design-for-cv-and-nlp",
        ],
    },
    "systems-mlops": {
        "title": "MOC: системы, serving и MLOps",
        "description": "Ёмкость, балансировка нагрузки, serving моделей и соседний слой обучения (гиперпараметры).",
        "tags": ["kb/moc", "domain/mlops"],
        "topics": [
            "ml-system-design-for-cv-and-nlp",
            "hyperparameter-tuning",
            "video-codecs-h264-h265-and-gpu-decode",
        ],
    },
    "robotics-embodied": {
        "title": "MOC: робототехника и Embodied AI",
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
