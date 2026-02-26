# Knowledge Book: Deep Learning & AI Education
# Книга Знаний: Deep Learning & AI Education

This knowledge book contains comprehensive guides on various topics in Deep Learning, Computer Vision, NLP, LLMs, and cutting-edge AI techniques in Russian.

## Structure

Documents are organized by topic and include:
- Comprehensive explanations
- Mathematical formulations
- Code examples
- Current status and applications
- References to related topics

## Contents

### Generative Models

1. **[Variational Autoencoders (VAEs)](./variational-autoencoders-vaes.md)**
   - Core concepts, mathematical foundations, implementation
   - Current applications and status (2025-2026)
   - Related: GANs, Diffusion Models

2. **[Generative Adversarial Networks (GANs)](./generative-adversarial-networks-gans.md)**
   - Adversarial training, architecture, modern variants
   - Comparison with VAEs
   - Current applications and status (2025-2026)
   - Related: VAEs, Diffusion Models

3. **[Diffusion Models](./diffusion-models.md)**
   - Forward and reverse diffusion processes, mathematical foundations
   - DDPM, DDIM, Latent Diffusion Models (Stable Diffusion)
   - Modern variants: Consistency Models, Flow Matching, DiT
   - Applications: text-to-image, video generation, 3D generation
   - Current state-of-the-art (2023-2026)
   - Related: VAEs, GANs

### Mathematical Foundations

1. **[Bayes' Theorem and Probability Foundations](./bayes-theorem-and-probability-foundations.md)**
   - Теоретико-множественные основы: пространство исходов, события, операции
   - Аксиомы вероятности Колмогорова
   - Условная вероятность и правило умножения
   - Независимость событий и условная независимость
   - Формула полной вероятности
   - Теорема Байеса: вывод, терминология (prior, likelihood, posterior)
   - Байесовский вывод в ML: MAP, MLE, регуляризация
   - Наивный Байесовский классификатор
   - Связь с VAE, фильтром Калмана, RAG
   - Реализация на Python с нуля

2. **[Gaussian Distribution (Normal Distribution)](./gaussian-distribution.md)**
   - Definition, properties, PDF and CDF
   - Multivariate Gaussian distribution
   - Applications in machine learning
   - Connection to Diffusion Models and VAEs
   - Visualization and examples

3. **[ROC Curves and ROC AUC](./roc-curve-and-roc-auc.md)**
   - ROC-кривые, TPR/FPR и их интуиция
   - ROC AUC как метрика ранжирования и качество разделения классов
   - Выбор порога классификации, Youden’s J, контроль FPR
   - Сравнение моделей по ROC/ROC AUC и связь с PR-кривыми
   - Примеры кода на Python/sklearn

### Classical Machine Learning

1. **[Decision Trees (Деревья решений)](./decision-trees.md)**
   - Что такое деревья решений: структура, узлы, ветви, листья
   - Критерии выбора разбиения: Gini, энтропия, прирост информации (Information Gain)
   - Методы построения: ID3, C4.5, CART; Gain Ratio; техники против переобучения
   - Области применения: скоринг, медицина, маркетинг, ансамбли (Random Forest, XGBoost)
   - Связь с ROC AUC, Cross Entropy; примеры кода sklearn
   - Related: ROC Curves and ROC AUC, Cross Entropy and Focal Loss, Ensemble Methods

2. **[Support Vector Machines (SVM) и Kernel Trick](./support-vector-machines-svm-and-kernel-trick.md)**
   - SVM: максимальный запас (margin), опорные векторы, примарная и двойственная задачи
   - Soft margin и параметр C
   - Kernel trick: нелинейные границы через ядра без явного отображения $\phi$
   - Типичные ядра: Linear, Polynomial, RBF (Gaussian), Sigmoid
   - Примеры кода на sklearn (linear и RBF для линейно и нелинейно разделимых данных)
   - Related: Decision Trees, ROC AUC, RAG, Bayes

### Hyperparameter Tuning (Настройка гиперпараметров)

1. **[Hyperparameter Tuning](./hyperparameter-tuning.md)**
   - Параметры vs гиперпараметры, пространство поиска
   - Grid Search и Random Search: принципы и сравнение
   - Bayesian Optimization: GP, TPE, Acquisition Functions, Optuna
   - Bandit-based методы: Successive Halving, Hyperband, BOHB
   - Population-Based Training (PBT) для параллельного тюнинга
   - Эволюционные алгоритмы: CMA-ES
   - Neural Architecture Search (NAS): DARTS, OFA
   - Автоматический подбор LR: LR Finder, OneCycleLR, Cosine Annealing
   - Кросс-валидация: K-Fold, Stratified, Nested CV
   - Практические рекомендации и что популярно в 2024-2026
   - Related: Ensemble Methods, Decision Trees, ROC AUC, LoRA, Bayes' Theorem

### Ensemble Methods (Ансамблевые методы)

1. **[Ensemble Methods & Model Combination](./ensemble-methods-model-combination.md)**
   - Bias-Variance Decomposition: зачем комбинировать модели
   - Bagging и Random Forest: параллельные ансамбли
   - Boosting: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
   - Stacking, Voting, Blending — мета-обучение
   - Mixture of Experts (MoE): sparse activation в LLM (Mixtral, DeepSeek-V3)
   - Knowledge Distillation: teacher → student, soft labels
   - Model Merging для LLM: TIES, DARE, SLERP, Model Soups
   - Ансамбли в DL: TTA, SWA, Snapshot Ensembles, MC Dropout
   - Что используется больше всего в 2024-2026
   - Related: Decision Trees, ROC AUC, Transformers, LoRA

### NLP and LLM Systems

1. **[Retrieval-Augmented Generation (RAG)](./retrieval-augmented-generation-rag.md)**
   - Как работает RAG: архитектура и компоненты
   - Типы RAG систем: Naive, Advanced, Modular, Self-RAG, Corrective RAG, LightRAG, Agentic RAG
   - Техники улучшения: query rewriting, re-ranking, context compression
   - Векторные базы данных и модели эмбеддингов
   - Реализация и оценка качества
   - Текущее состояние и тренды (2023-2026)
   - Related: Large Language Models, Attention Mechanisms

2. **[Low-Rank Adaptation (LoRA)](./low-rank-adaptation-lora.md)**
   - Проблема тонкой настройки больших языковых моделей
   - Математическая основа: разложение матриц низкого ранга
   - Архитектура LoRA и применение к Transformer слоям
   - Параметры, эффективность и сравнение с полной настройкой
   - Варианты: QLoRA, AdaLoRA, DoRA
   - Практическое применение и реализация в PyTorch
   - Текущее состояние и тренды (2021-2026)
   - Related: Transformers, Attention Mechanisms, RAG

3. **[Tokenization and Text Compression in LLMs](./tokenization-and-text-compression-in-llms.md)**
   - Что такое токенизатор и зачем он нужен
   - Как текст превращается в последовательность токенов и ID
   - Основные техники токенизации: word‑level, char‑level, BPE, WordPiece, Unigram, byte‑level BPE
   - Почему токенизация — это по сути алгоритм сжатия текста перед входом в LLM
   - Связь с архитектурой трансформеров и стоимостью внимания
   - Related: Embeddings and Embedding Matrix

4. **[Embeddings and Embedding Matrix](./embeddings-and-embedding-matrix.md)**
   - Что такое эмбеддинги (векторное представление дискретных символов/токенов)
   - Матрица эмбеддингов: размер $V \times d$, lookup по ID токена
   - Связь с токенизатором: токенизатор даёт ID, матрица эмбеддингов даёт векторы для входа в Transformer
   - Обучение эмбеддингов в LLM, размерность и размер словаря
   - Эмбеддинги в RAG и семантическом поиске
   - Related: Tokenization, Transformers, RAG

### Transformers, Attention and Vision Transformers

1. **[Transformers, Attention and Vision Transformers (ViT)](./transformers-attention-and-vision-transformers-vit.md)**
   - Scaled Dot-Product Attention, Q/K/V и виды attention
   - KV Cache и оптимизация инференса LLM
   - Позиционное кодирование: абсолютное, относительное, rotary, 2D‑позиции
   - Архитектура ViT, CLS‑токен и классификация на трансформерах
   - Детекция и сегментация с помощью DETR‑подобных архитектур
   - Related: RAG, Non-Maximum Suppression (NMS), Convolutions and Parameters in CNN, Batch/Layer Normalization

2. **[DINOv3: Self-Supervised Vision Transformer и 2D RoPE](./dinov3-self-supervised-vision-transformer-and-2d-rope.md)**
   - Self-supervised pretraining ViT‑бэкбонов (student–teacher, multi‑view, multi‑loss)
   - 2D Rotary Positional Embeddings (2D RoPE) для кодирования координат патчей
   - Глобальные и dense‑фичи DINOv3 и их использование в классификации, детекции и сегментации
   - Related: Transformers, Non-Maximum Suppression (NMS), Detection/Segmentation Losses, Convolutions in CNN

### Computer Vision and Object Detection

1. **[Non-Maximum Suppression (NMS) and Modern End-to-End Detectors](./non-maximum-suppression-nms.md)**
   - Non-Maximum Suppression (NMS): алгоритм и реализация
   - Agnostic NMS (Class-Agnostic NMS)
   - Проблемы NMS в production и деплое
   - End-to-End детекция без NMS: YOLO26, DETR, RT-DETR
   - Transformer-based детекторы и query-based подходы
   - Сравнение традиционных и end-to-end подходов
   - Текущее состояние и тренды (2023-2026)
   - Related: Unscented Kalman Filter (для отслеживания объектов)

2. **[Convolutions and Parameters in CNN](./convolutions-and-parameters-in-cnn.md)**
   - Почему в CNN популярны свёртки `3×3` и нечётные ядра
   - Эффективность больших свёрток `5×5`, `7×7`
   - Формулы размеров feature map для Conv/Pooling и Transposed Conv
   - Подсчёт числа обучаемых параметров (Conv, Linear, BatchNorm, depthwise/pointwise)
   - Related: Non-Maximum Suppression (NMS), Deep Reinforcement Learning

### Normalization and Training Stabilization

1. **[Batch Normalization and Layer Normalization](./normalization-layers-batchnorm-layernorm.md)**
   - Зачем нужна нормализация активаций в глубоких сетях
   - Формулы и интуиция Batch Normalization
   - Формулы и интуиция Layer Normalization
   - Сравнение BatchNorm vs LayerNorm, влияние на обучение
   - Related: Convolutions and Parameters in CNN, Deep Reinforcement Learning, RAG/Transformers

### Loss Functions for Classification and Detection

1. **[Cross Entropy and Focal Loss](./classification-losses-cross-entropy-focal-loss.md)**
   - Кросс‑энтропия в бинарной и многоклассовой классификации
   - Интуиция: почему CE так популярна
   - Focal Loss: мотивация, формула, роль параметров α и γ
   - Применение в object detection и задачах с дисбалансом классов
   - Related: Non-Maximum Suppression (NMS), Convolutions and Parameters in CNN

2. **[Losses for Detection, Segmentation, and 3D Detection](./detection-segmentation-3d-losses.md)**
   - Составные loss’ы в современных детекторах и сегментаторах
   - Классификационные loss’ы (CE, Focal, Quality Focal, Varifocal)
   - Loss’ы для регрессии боксов (L1/Smooth L1, IoU, GIoU/DIoU/CIoU)
   - Loss’ы для сегментации (CE, Dice, IoU, Tversky, Lovász-Softmax)
   - Loss’ы для 3D‑детекции (3D/BEV IoU, L1 по центрам/размерам, heatmap‑based)
   - Related: Non-Maximum Suppression (NMS), Convolutions and Parameters in CNN

### Filtering and Object Tracking

1. **[Unscented Kalman Filter and Modern Tracking Methods](./unscented-kalman-filter-and-tracking.md)**
   - Unscented Kalman Filter (UKF): теория и алгоритм
   - Сравнение с Kalman Filter, Extended Kalman Filter, Particle Filter
   - Современные методы отслеживания объектов (DeepSORT, ByteTrack, Transformer-based)
   - Применения в компьютерном зрении, робототехнике, навигации
   - Реализация UKF и примеры использования
   - Статистика хи-квадрат для обнаружения выбросов
   - Текущее состояние (2023-2026)
   - Related: Gaussian Distribution, Non-Maximum Suppression

### Reinforcement Learning and Control

1. **[Deep Reinforcement Learning](./deep-reinforcement-learning.md)**
   - Основы Reinforcement Learning и MDP
   - Deep Q-Network (DQN), Policy Gradient, Actor-Critic методы
   - Современные методы: PPO, SAC, TD3
   - Применения в робототехнике: манипуляция, локомоция, управление
   - Применения в автономных автомобилях: end-to-end обучение, hierarchical RL
   - Sim-to-Real transfer и domain randomization
   - Текущее состояние и тренды (2024-2026)
   - Related: Unscented Kalman Filter (для фильтрации состояний)

### Robotics and Embodied AI

1. **[Vision-Language-Action (VLA) Models](./vision-language-action-models-vla.md)**
   - Что такое VLA модели и зачем они нужны
   - Архитектура: объединение Vision, Language и Action
   - Ключевые компоненты: энкодеры, проекторы, декодеры действий
   - Обучение VLA моделей: данные, loss функции, fine-tuning
   - Современные модели: RT-1, RT-2, OpenVLA, F1-VLA, Octo
   - Применения: манипуляция, навигация, автономные системы
   - Сравнение с RL и Imitation Learning
   - Реализация и примеры кода
   - Текущее состояние и тренды (2022-2026)
   - Related: Transformers, Deep Reinforcement Learning, Low-Rank Adaptation (LoRA)

2. **[Vision-Based Robot Training Methods](./vision-based-robot-training-methods.md)**
   - Обучение роботов с визуальным восприятием
   - Основные подходы: Imitation Learning, RL, VLA
   - Лучшие Open-Source методы (2024-2025): OpenVLA, Octo, RT-1/RT-2, AutoRT
   - Обучение для разных типов роботов: гуманоиды, четвероногие, колёсные, манипуляторы
   - Датасеты: Open X-Embodiment, RT-1 Dataset
   - Практические примеры: код и использование
   - Sim-to-Real transfer: от симуляции к реальности
   - Сравнение методов и выбор подхода
   - Текущее состояние и тренды (2024-2026)
   - Related: Vision-Language-Action (VLA) Models, Deep Reinforcement Learning

## Reading Order

### For Understanding Generative Models:
1. Start with **Gaussian Distribution** for fundamental probability concepts
2. Read **VAEs** for foundational probabilistic generative modeling
3. Then read **GANs** for adversarial training approach
4. Study **Diffusion Models** for state-of-the-art generation techniques
5. Compare the approaches using the comparison sections
6. Explore advanced topics and recent research

### For Mathematical Foundations:
1. Start with **Bayes' Theorem and Probability Foundations** for fundamental probability theory
2. Study **Gaussian Distribution** as a key building block for probabilistic models
3. Understand how both concepts combine in **VAEs** (latent space, ELBO) and **Diffusion Models** (noise)
4. Learn about **Bayesian inference** in ML: MAP, MLE, regularization
5. Apply knowledge to **Naive Bayes classifier** and **Kalman filters**

### For NLP and RAG Systems:
1. Start with **Retrieval-Augmented Generation (RAG)** for understanding how to enhance LLMs with external knowledge
2. Learn about different RAG architectures and when to use each
3. Explore advanced techniques for improving retrieval and generation quality
4. Understand evaluation metrics and best practices

### For Fine-Tuning Large Language Models:
1. Read **Transformers, Attention and Vision Transformers** to understand Transformer architecture
2. Study **Low-Rank Adaptation (LoRA)** for efficient fine-tuning techniques
3. Learn when to use LoRA vs full fine-tuning
4. Explore variants like QLoRA for memory-constrained scenarios
5. Apply LoRA in practice with Hugging Face PEFT library

### For Computer Vision and Object Detection:
1. Start with **Non-Maximum Suppression (NMS)** for understanding traditional object detection pipelines
2. Learn about end-to-end approaches (YOLO26, DETR) that eliminate NMS
3. Understand the evolution from NMS-based to query-based detection
4. Explore transformer-based detectors and their advantages

### For Filtering and Tracking:
1. Start with **Gaussian Distribution** for understanding probability distributions
2. Read **Unscented Kalman Filter** for non-linear filtering and object tracking
3. Understand the evolution from KF → EKF → UKF → Particle Filter
4. Explore modern deep learning approaches to tracking
5. Connect with **Non-Maximum Suppression** for object detection pipelines

### For Reinforcement Learning and Control:
1. Start with **Deep Reinforcement Learning** for understanding RL fundamentals
2. Learn about value-based (DQN), policy-based (PPO), and actor-critic (SAC) methods
3. Explore applications in robotics and autonomous vehicles
4. Understand sim-to-real transfer and safety considerations
5. Study modern approaches: foundation models, diffusion policies, hierarchical RL

### For Robotics and Vision-Based Robot Learning:
1. Read **Vision-Language-Action (VLA) Models** to understand how vision, language, and action are combined
2. Study modern VLA architectures: OpenVLA, RT-1, RT-2, Octo
3. Learn about **Vision-Based Robot Training Methods** for practical robot learning
4. Understand different approaches: Imitation Learning, RL, VLA
5. Explore open-source methods and datasets (Open X-Embodiment)
6. Study sim-to-real transfer techniques
7. Apply methods to different robot types: humanoids, quadrupeds, wheeled robots, manipulators

### For Hyperparameter Tuning:
1. Start with **Decision Trees** and **Ensemble Methods** to understand models with many hyperparameters
2. Read **Hyperparameter Tuning** for comprehensive coverage of all methods
3. Learn Grid Search → Random Search → Bayesian Optimization (Optuna)
4. Study advanced methods: Hyperband, BOHB, PBT
5. Apply LR Finder and schedules for neural networks
6. Explore NAS for architecture search

### For Ensemble Methods and Model Combination:
1. Start with **Decision Trees** as the building block for most ensembles
2. Read **Ensemble Methods & Model Combination** for comprehensive coverage
3. Understand Bagging (Random Forest) → Boosting (XGBoost/LightGBM/CatBoost)
4. Learn Stacking and Voting for combining diverse models
5. Study **Mixture of Experts** and **Model Merging** for LLM-scale approaches
6. Explore Knowledge Distillation and DL-specific ensembles (TTA, SWA)

## Contributing

When adding new documents:
- Follow the established structure (Table of Contents, sections, References)
- Include mathematical formulations with LaTeX
- Provide code examples in Python/PyTorch
- Update this README with new entries
- Add cross-references to related documents


## References
- https://github.com/Mathews-Tom/no-magic
