# Книга знаний: Deep Learning и AI

Подробные конспекты по Deep Learning, Computer Vision, NLP, LLM и современным методам AI. Связный текст — на русском; термины, имена моделей и код можно оставлять на английском.

## Структура

Материалы сгруппированы по темам. В каждом топике обычно есть:

- развёрнутое объяснение
- формулы
- примеры кода
- текущий статус и применения
- ссылки на соседние темы

Как устроен каталог:

- `topics/<topic-slug>/README.md` — основной материал темы (с **Obsidian YAML frontmatter**)
- `topics/<topic-slug>/scripts/` — нумерованные запускаемые примеры (`01_*.py`, `02_*.py`, ...)
- `topics/<topic-slug>/tests/` — тесты к скриптам темы (`test_*.py`)

Слой индекса для Obsidian / RAG:

- [`docs/README.md`](./docs/README.md) — обзор хранилища
- [`docs/index.md`](./docs/index.md) — аннотированный каталог (`description`, теги, aliases)
- [`docs/mocs/`](./docs/mocs/foundations.md) — Maps of Content (с них удобно начинать)
- [`docs/tags/`](./docs/tags/README.md) — страницы таксономии тегов
- [`docs/SCHEMA.md`](./docs/SCHEMA.md) — схема frontmatter для поиска

## Соглашения

- Одна тема — одна директория, один основной документ: `topics/<topic-slug>/README.md`
- У каждого топика/заметки в начале YAML frontmatter: `title`, `description`, `tags`, `aliases`, `related`, `status`, `lang`, `type`, `slug`, `updated`
- В каждом README темы держать явное `## Оглавление`, короткий абстракт «Как объяснить 5-летнему ребёнку» и секцию `Источники`
- В `Источниках` — рабочие markdown-ссылки (`../other-slug/README.md`), а не голые имена файлов
- Если в тексте есть код — выносить его в отдельные скрипты в `scripts/`
- У каждого скрипта должен быть автотест в `tests/`
- Окружение — корневой `uv`, зависимости по умолчанию CPU-first
- После правок метаданных: `uv run python scripts/kb_apply_obsidian_frontmatter.py`

## Разработка (uv)

- `uv sync` — установить зависимости проекта и dev
- `uv sync --group viz` — добавить Manim + imageio (локальный рендер `dl-viz`; включает smoke-тесты Manim)
- `uv run pytest` — все тесты; тесты с `@pytest.mark.manim` пропускаются, если Manim не установлен
- `uv run python scripts/kb_validate_links.py` — проверка ссылок, оглавлений и frontmatter
- GitHub Actions (`.github/workflows/ci.yml`) гоняет `pytest` и проверку ссылок; merge в `main` только при зелёном check **CI**
- `uv run python topics/<topic-slug>/scripts/01_<name>.py` — запустить пример темы
- `uv run python scripts/viz/mp4_to_gif.py <file.mp4> -o <out.gif>` — GIF из отрендеренного MP4

### Транскрипты YouTube

- `uv sync --group tools` — `youtube-transcript-api`, `yt-dlp`, `secretstorage` (cookies Chrome на Linux)
- `uv run python scripts/youtube_fetch_transcript.py "<url>" -o outputs/transcripts/<VIDEO_ID>.txt -v`
- `uv run python scripts/verify_youtube_transcript.py outputs/transcripts/<VIDEO_ID>.txt` — должно пройти до написания конспекта
- Проверенные файлы коммитить в `outputs/transcripts/` (см. [outputs/transcripts/README.md](./outputs/transcripts/README.md)); `.gitignore` разрешает эти `.txt`
- Workflow агента: `.cursor/rules/youtube-transcript-to-knowledge-book.mdc`

## Содержание

### Генеративные модели

1. **[Variational Autoencoders (VAE)](./topics/variational-autoencoders-vaes/README.md)**
   - Базовые идеи, математика, реализация
   - Текущие применения и статус (2025–2026)
   - Связано: GAN, Diffusion Models

2. **[Generative Adversarial Networks (GAN)](./topics/generative-adversarial-networks-gans/README.md)**
   - Adversarial training, архитектура, современные варианты
   - Сравнение с VAE
   - Текущие применения и статус (2025–2026)
   - Связано: VAE, Diffusion Models

3. **[Diffusion Models](./topics/diffusion-models/README.md)**
   - Прямой и обратный процессы диффузии, математические основы
   - DDPM, DDIM, Latent Diffusion Models (Stable Diffusion)
   - Современные варианты: Consistency Models, Flow Matching, DiT
   - Применения: text-to-image, генерация видео и 3D
   - Текущий state-of-the-art (2023–2026)
   - Связано: VAE, GAN

### Математические основы

1. **[Теорема Байеса и основы теории вероятностей](./topics/bayes-theorem-and-probability-foundations/README.md)**
   - Теоретико-множественные основы: пространство исходов, события, операции
   - Аксиомы вероятности Колмогорова
   - Условная вероятность и правило умножения
   - Независимость событий и условная независимость
   - Формула полной вероятности
   - Теорема Байеса: вывод, терминология (prior, likelihood, posterior)
   - Байесовский вывод в ML: MAP, MLE, регуляризация
   - Наивный байесовский классификатор
   - Связь с VAE, фильтром Калмана, RAG
   - Реализация на Python с нуля

2. **[Гауссово распределение (Normal Distribution)](./topics/gaussian-distribution/README.md)**
   - Определение, свойства, PDF и CDF
   - Многомерное гауссово распределение
   - Применения в машинном обучении
   - Связь с Diffusion Models и VAE
   - Визуализация и примеры

3. **[ROC-кривые и ROC AUC](./topics/roc-curve-and-roc-auc/README.md)**
   - ROC-кривые, TPR/FPR и их интуиция
   - ROC AUC как метрика ранжирования и качество разделения классов
   - Выбор порога классификации, Youden’s J, контроль FPR
   - Сравнение моделей по ROC/ROC AUC и связь с PR-кривыми
   - Примеры кода на Python/sklearn

4. **[Уверенность, калибровка и неопределённость](./topics/how-models-predict-confidence-and-calibration/README.md)**
   - Как модели получают `logits` и превращают их в «уверенность» через `softmax`/`sigmoid`
   - Калибровка вероятностей: reliability diagram, ECE, Brier
   - Temperature Scaling и зачем он нужен
   - Уверенность vs неопределённость: aleatoric/epistemic, ensembles/MC Dropout
   - Связано: ROC-кривые и ROC AUC, кросс-энтропия, ансамбли

### Классическое машинное обучение

1. **[Деревья решений (Decision Trees)](./topics/decision-trees/README.md)**
   - Что такое деревья решений: структура, узлы, ветви, листья
   - Критерии выбора разбиения: Gini, энтропия, прирост информации (Information Gain)
   - Методы построения: ID3, C4.5, CART; Gain Ratio; техники против переобучения
   - Области применения: скоринг, медицина, маркетинг, ансамбли (Random Forest, XGBoost)
   - Связь с ROC AUC, кросс-энтропией; примеры кода sklearn
   - Связано: ROC AUC, Cross Entropy и Focal Loss, ансамбли

2. **[Support Vector Machines (SVM) и Kernel Trick](./topics/support-vector-machines-svm-and-kernel-trick/README.md)**
   - SVM: максимальный запас (margin), опорные векторы, прямая и двойственная задачи
   - Soft margin и параметр C
   - Kernel trick: нелинейные границы через ядра без явного отображения $\phi$
   - Типичные ядра: Linear, Polynomial, RBF (Gaussian), Sigmoid
   - Примеры кода на sklearn (linear и RBF для линейно и нелинейно разделимых данных)
   - Связано: деревья решений, ROC AUC, RAG, Байес

### Настройка гиперпараметров (Hyperparameter Tuning)

1. **[Настройка гиперпараметров](./topics/hyperparameter-tuning/README.md)**
   - Параметры vs гиперпараметры, пространство поиска
   - Grid Search и Random Search: принципы и сравнение
   - Bayesian Optimization: GP, TPE, Acquisition Functions, Optuna
   - Bandit-методы: Successive Halving, Hyperband, BOHB
   - Population-Based Training (PBT) для параллельного тюнинга
   - Эволюционные алгоритмы: CMA-ES
   - Neural Architecture Search (NAS): DARTS, OFA
   - Автоматический подбор LR: LR Finder, OneCycleLR, Cosine Annealing
   - Кросс-валидация: K-Fold, Stratified, Nested CV
   - Практические рекомендации и что популярно в 2024–2026
   - Связано: ансамбли, деревья решений, ROC AUC, LoRA, теорема Байеса

### Ансамблевые методы (Ensemble Methods)

1. **[Методы комбинирования моделей](./topics/ensemble-methods-model-combination/README.md)**
   - Bias-Variance Decomposition: зачем комбинировать модели
   - Bagging и Random Forest: параллельные ансамбли
   - Boosting: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
   - Stacking, Voting, Blending — мета-обучение
   - Mixture of Experts (MoE): sparse activation в LLM (Mixtral, DeepSeek-V3)
   - Knowledge Distillation: teacher → student, soft labels
   - Model Merging для LLM: TIES, DARE, SLERP, Model Soups
   - Ансамбли в DL: TTA, SWA, Snapshot Ensembles, MC Dropout
   - Что используется больше всего в 2024–2026
   - Связано: деревья решений, ROC AUC, Transformers, LoRA

### NLP и LLM-системы

1. **[Retrieval-Augmented Generation (RAG)](./topics/retrieval-augmented-generation-rag/README.md)**
   - Как работает RAG: архитектура и компоненты
   - Типы RAG-систем: Naive, Advanced, Modular, Self-RAG, Corrective RAG, LightRAG, Agentic RAG
   - Техники улучшения: query rewriting, re-ranking, context compression
   - Векторные базы данных и модели эмбеддингов
   - Реализация и оценка качества
   - Текущее состояние и тренды (2023–2026)
   - Связано: LLM, механизмы внимания

2. **[Low-Rank Adaptation (LoRA)](./topics/low-rank-adaptation-lora/README.md)**
   - Проблема тонкой настройки больших языковых моделей
   - Математическая основа: разложение матриц низкого ранга
   - Архитектура LoRA и применение к слоям Transformer
   - Параметры, эффективность и сравнение с полной настройкой
   - Варианты: QLoRA, AdaLoRA, DoRA
   - Практическое применение и реализация в PyTorch
   - Текущее состояние и тренды (2021–2026)
   - Связано: Transformers, внимание, RAG

3. **[Токенизация и сжатие текста в LLM](./topics/tokenization-and-text-compression-in-llms/README.md)**
   - Что такое токенизатор и зачем он нужен
   - Как текст превращается в последовательность токенов и ID
   - Основные техники: word-level, char-level, BPE, WordPiece, Unigram, byte-level BPE
   - Почему токенизация — это по сути сжатие текста перед входом в LLM
   - Связь с архитектурой трансформеров и стоимостью внимания
   - Связано: эмбеддинги и матрица эмбеддингов

4. **[Эмбеддинги и матрица эмбеддингов](./topics/embeddings-and-embedding-matrix/README.md)**
   - Что такое эмбеддинги (векторное представление дискретных символов/токенов)
   - Матрица эмбеддингов: размер $V \times d$, lookup по ID токена
   - Связь с токенизатором: токенизатор даёт ID, матрица эмбеддингов — векторы для входа в Transformer
   - Обучение эмбеддингов в LLM, размерность и размер словаря
   - Эмбеддинги в RAG и семантическом поиске
   - Связано: токенизация, Transformers, RAG

5. **[Contrastive и metric learning для fine-grained распознавания](./topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md)**
   - Contrastive learning vs metric learning: что это и когда нужно
   - Fine-grained пример: «15 похожих сортов яблок» как retrieval/verification
   - Лоссы (contrastive/triplet/InfoNCE) и mining
   - Метрики: Recall@K, mAP, ROC/EER и деплой через embedding index
   - Связано: эмбеддинги, ROC AUC, self-supervised CV

6. **[ArcFace и angular-margin losses для идентификации](./topics/arcface-and-angular-margin-losses-for-identification/README.md)**
   - ArcFace: математическая интуиция и формула additive angular margin
   - Практика face identification (1:1 verification и 1:N identification)
   - Практика SKU/product identification в open-set режиме
   - Современные датасеты для metric learning и re-ID (face/person/product/vehicle)
   - Связано: contrastive и metric learning, ROC AUC, калибровка, лоссы и майнеры

7. **[Лоссы metric learning и подбор майнеров](./topics/metric-learning-losses-and-miners/README.md)**
   - Таксономия: pair-based, triplet/tuple, batch-contrastive (InfoNCE/SupCon), proxy-based (Proxy-Anchor, SoftTriple, ArcFace)
   - Формулы и интуиция: Contrastive, Triplet, N-pair, Multi-Similarity, Circle, CosFace/ArcFace/AdaFace
   - Майнеры: semi-hard / batch-hard / distance-weighted / MS-miner, P×K-сэмплер, XBM
   - Таблица «лосс ↔ майнер» и decision guide: когда что использовать
   - Связано: ArcFace, contrastive и metric learning, cross entropy / focal loss

8. **[Code Agents, AutoResearch и Loopy Era](./topics/code-agents-autoresearch-and-loopy-era/README.md)**
   - Что меняется в инженерии при переходе от ручного кода к orchestration
   - Multi-agent workflows: роли, параллелизм, quality gates
   - AutoResearch loops: objective, evaluator, verifier, метрики
   - [Stop Babysitting Your Agents (Claude Code)](./topics/code-agents-autoresearch-and-loopy-era/stop-babysitting-your-agents-claude-code.md): verification skills, `/loop`, Routines
   - [AI Harness Engineering (Tejas, IBM)](./topics/code-agents-autoresearch-and-loopy-era/ai-harness-engineering-tejas-ibm.md): guardrails, шаг verify, harness vs prompt
   - Правила большого пальца и практический checklist для команды
   - Связано: RAG, настройка гиперпараметров, LoRA, протоколы агентов

9. **[MCP, ACP, UCP и Agent Harness](./topics/agent-protocols-mcp-acp-ucp-and-harness/README.md)**
   - Слои: MCP (инструменты), ACP (агент в IDE / checkout — не путать), UCP (коммерция), A2A
   - Harness как обвязка вокруг модели: rules, skills, MCP, hooks, verify
   - Как собрать эффективный harness в Cursor без раздувания контекста
   - Связано: code agents, RAG, system design

### Transformers, внимание и Vision Transformers

1. **[Transformers, Attention и Vision Transformers (ViT)](./topics/transformers-attention-and-vision-transformers-vit/README.md)**
   - Scaled Dot-Product Attention, Q/K/V и виды attention
   - KV Cache и оптимизация инференса LLM
   - Позиционное кодирование: абсолютное, относительное, rotary, 2D-позиции
   - Архитектура ViT, CLS-токен и классификация на трансформерах
   - Детекция и сегментация с помощью DETR-подобных архитектур
   - Связано: RAG, NMS, свёртки в CNN, Batch/Layer Normalization

2. **[DINOv3: Self-Supervised Vision Transformer и 2D RoPE](./topics/dinov3-self-supervised-vision-transformer-and-2d-rope/README.md)**
   - Self-supervised pretraining ViT-бэкбонов (student–teacher, multi-view, multi-loss)
   - 2D Rotary Positional Embeddings (2D RoPE) для кодирования координат патчей
   - Глобальные и dense-фичи DINOv3 и их использование в классификации, детекции и сегментации
   - Связано: Transformers, NMS, loss'ы детекции/сегментации, свёртки в CNN

3. **[Few-shot anomaly detection: AnomalyDINO](./topics/few-shot-anomaly-detection-anomalydino/README.md)**
   - Patch-level deep nearest neighbor с DINOv2 без обучения (1–16 эталонов)
   - Memory bank, zero-shot masking, косинусное расстояние, агрегация top-1%
   - Pixel-level anomaly map и применение к industrial QC (в т.ч. уровень жидкости)
   - Связано: DINOv3, ROC AUC, SOTA-метрики детекции/сегментации

### Компьютерное зрение и детекция объектов

1. **[Non-Maximum Suppression (NMS) и современные end-to-end детекторы](./topics/non-maximum-suppression-nms/README.md)**
   - Non-Maximum Suppression (NMS): алгоритм и реализация
   - Agnostic NMS (Class-Agnostic NMS)
   - Проблемы NMS в production и деплое
   - End-to-end детекция без NMS: YOLO26, DETR, RT-DETR
   - Transformer-based детекторы и query-based подходы
   - Сравнение традиционных и end-to-end подходов
   - Текущее состояние и тренды (2023–2026)
   - Связано: Unscented Kalman Filter (для отслеживания объектов)

2. **[SOTA-метрики для детекции, сегментации и мультиклассовой классификации](./topics/sota-metrics-for-detection-segmentation-multiclass-classification/README.md)**
   - Детекция: COCO AP (AP@[.50:.95]), AP50/AP75, AP_S/M/L, AR
   - Сегментация: semantic mIoU, instance Mask AP, panoptic PQ
   - Мультикласс: Top-1/Top-5, macro/micro F1, NLL, калибровка (ECE/Brier)
   - Связано: ROC AUC, уверенность и калибровка, NMS

3. **[Свёртки и параметры в CNN](./topics/convolutions-and-parameters-in-cnn/README.md)**
   - Почему в CNN популярны свёртки `3×3` и нечётные ядра
   - Эффективность больших свёрток `5×5`, `7×7`
   - Формулы размеров feature map для Conv/Pooling и Transposed Conv
   - Подсчёт числа обучаемых параметров (Conv, Linear, BatchNorm, depthwise/pointwise)
   - Связано: NMS, Deep Reinforcement Learning, System Design

4. **[Видеокодеки H.264/H.265 и GPU-декодирование](./topics/video-codecs-h264-h265-and-gpu-decode/README.md)**
   - Intra vs inter, GOP (I/P/B), DCT и компенсация движения
   - H.264/AVC и H.265/HEVC: в чём разница и насколько сжимают поток
   - Типичный битрейт 1080p/4K для IP-камер
   - Hardware decode: NVDEC, Quick Sync, VAAPI; зачем это аналитике
   - Связано: метрики видео/трекинга, свёртки в CNN, сжатие текста в LLM, System Design

### Системы, serving и MLOps

1. **[System Design для Computer Vision и NLP](./topics/ml-system-design-for-cv-and-nlp/README.md)**
   - Что такое System Design: SLO, очереди, закон Литтла, утилизация $\rho$
   - Балансировка нагрузки для ML (не только round-robin)
   - Serving на 100 vs 1000 клиентов: реплики, dynamic batching, KV-кэш LLM
   - 10 vs 50 камер: latest-frame, пачка на GPU, когда нужен NVDEC
   - Связано: KV cache, NMS, трекинг, RAG, видеокодеки, гиперпараметры, Triton

2. **[Triton Inference Server и развёртывание на 1–N GPU](./topics/triton-inference-server-and-gpu-model-serving/README.md)**
   - Что такое NVIDIA Triton и какие задачи решает (batching, concurrent models, ensembles)
   - Полезен ли Triton на одной GPU; когда лучше vLLM / TensorRT-LLM / SGLang
   - SOTA-стек serving для «100× пользователей» на 1–N GPU
   - Связано: System Design, KV cache, токенизация, RAG, видеокодеки

### Нормализация и стабилизация обучения

1. **[Batch Normalization и Layer Normalization](./topics/normalization-layers-batchnorm-layernorm/README.md)**
   - Зачем нужна нормализация активаций в глубоких сетях
   - Формулы и интуиция Batch Normalization
   - Формулы и интуиция Layer Normalization
   - Сравнение BatchNorm vs LayerNorm, влияние на обучение
   - Связано: свёртки в CNN, Deep RL, RAG/Transformers

### Функции потерь для классификации и детекции

1. **[Cross Entropy и Focal Loss](./topics/classification-losses-cross-entropy-focal-loss/README.md)**
   - Кросс-энтропия в бинарной и многоклассовой классификации
   - Интуиция: почему CE так популярна
   - Focal Loss: мотивация, формула, роль параметров α и γ
   - Применение в object detection и задачах с дисбалансом классов
   - Связано: NMS, свёртки в CNN

2. **[Loss'ы для детекции, сегментации и 3D-детекции](./topics/detection-segmentation-3d-losses/README.md)**
   - Составные loss'ы в современных детекторах и сегментаторах
   - Классификационные loss'ы (CE, Focal, Quality Focal, Varifocal)
   - Loss'ы для регрессии боксов (L1/Smooth L1, IoU, GIoU/DIoU/CIoU)
   - Loss'ы для сегментации (CE, Dice, IoU, Tversky, Lovász-Softmax)
   - Loss'ы для 3D-детекции (3D/BEV IoU, L1 по центрам/размерам, heatmap-based)
   - Связано: NMS, свёртки в CNN

### Фильтрация и трекинг объектов

1. **[Unscented Kalman Filter и современные методы трекинга](./topics/unscented-kalman-filter-and-tracking/README.md)**
   - Unscented Kalman Filter (UKF): теория и алгоритм
   - Сравнение с Kalman Filter, Extended Kalman Filter, Particle Filter
   - Современные методы отслеживания объектов (DeepSORT, ByteTrack, Transformer-based)
   - Применения в компьютерном зрении, робототехнике, навигации
   - Реализация UKF и примеры использования
   - Статистика хи-квадрат для обнаружения выбросов
   - Текущее состояние (2023–2026)
   - Связано: гауссово распределение, NMS

2. **[Метрики action recognition и object tracking](./topics/action-recognition-and-object-tracking-metrics/README.md)**
   - Метрики для video-level action recognition: Top-1/Top-5, macro-F1, mAcc
   - Метрики для temporal localization: mAP@tIoU и average mAP
   - Метрики для SOT и MOT: Success AUC, IDF1, MOTA, HOTA
   - Практический гайд по выбору метрик под benchmark и постановку
   - Связано: калибровка, NMS, UKF-трекинг

### Обучение с подкреплением и управление

1. **[Deep Reinforcement Learning](./topics/deep-reinforcement-learning/README.md)**
   - Основы Reinforcement Learning и MDP
   - Deep Q-Network (DQN), Policy Gradient, Actor-Critic методы
   - Современные методы: PPO, SAC, TD3
   - Применения в робототехнике: манипуляция, локомоция, управление
   - Применения в автономных автомобилях: end-to-end обучение, hierarchical RL
   - Sim-to-real transfer и domain randomization
   - Текущее состояние и тренды (2024–2026)
   - Связано: Unscented Kalman Filter (для фильтрации состояний)

### Робототехника и Embodied AI

1. **[Модели Vision-Language-Action (VLA)](./topics/vision-language-action-models-vla/README.md)**
   - Что такое VLA-модели и зачем они нужны
   - Архитектура: объединение Vision, Language и Action
   - Ключевые компоненты: энкодеры, проекторы, декодеры действий
   - Обучение VLA-моделей: данные, loss-функции, fine-tuning
   - Современные модели: RT-1, RT-2, OpenVLA, F1-VLA, Octo
   - Применения: манипуляция, навигация, автономные системы
   - Сравнение с RL и Imitation Learning
   - Реализация и примеры кода
   - Текущее состояние и тренды (2022–2026)
   - Связано: Transformers, Deep RL, LoRA

2. **[Vision-based обучение роботов](./topics/vision-based-robot-training-methods/README.md)**
   - Обучение роботов с визуальным восприятием
   - Основные подходы: Imitation Learning, RL, VLA
   - Лучшие open-source методы (2024–2025): OpenVLA, Octo, RT-1/RT-2, AutoRT
   - Обучение для разных типов роботов: гуманоиды, четвероногие, колёсные, манипуляторы
   - Датасеты: Open X-Embodiment, RT-1 Dataset
   - Практические примеры: код и использование
   - Sim-to-real transfer: от симуляции к реальности
   - Сравнение методов и выбор подхода
   - Текущее состояние и тренды (2024–2026)
   - Связано: VLA-модели, Deep RL

## Порядок чтения

### Чтобы разобраться в генеративных моделях

1. Начните с **гауссова распределения** — базовые вероятностные понятия
2. Прочитайте **VAE** — вероятностное генеративное моделирование
3. Затем **GAN** — adversarial training
4. Изучите **Diffusion Models** — текущий state-of-the-art генерации
5. Сравните подходы по разделам сравнения в топиках
6. Дальше — продвинутые темы и свежие статьи

### Математические основы

1. **Теорема Байеса и основы вероятностей**
2. **Гауссово распределение** как строительный блок вероятностных моделей
3. Как оба понятия сходятся в **VAE** (латентное пространство, ELBO) и **Diffusion Models** (шум)
4. **Байесовский вывод** в ML: MAP, MLE, регуляризация
5. Применение: **наивный Байес** и **фильтры Калмана**

### NLP и RAG-системы

1. **RAG** — как усиливать LLM внешней памятью
2. Разные архитектуры RAG и когда какую выбирать
3. Техники улучшения retrieval и generation
4. Метрики оценки и практики
5. **Code Agents, AutoResearch и Loopy Era** — автономные multi-agent циклы с измеримым результатом
6. От разовых промптов к инженерии процесса: явные цели, evaluators и safety gates

### Fine-tuning больших языковых моделей

1. **Transformers, Attention и Vision Transformers** — архитектура Transformer
2. **LoRA** — эффективный fine-tuning
3. Когда LoRA, а когда полная настройка
4. Варианты вроде QLoRA при нехватке памяти
5. Практика с Hugging Face PEFT

### Metric learning и идентификация

1. **Эмбеддинги и матрица эмбеддингов** — база представлений
2. **Contrastive и metric learning** — пары/триплеты/InfoNCE
3. **ArcFace и angular-margin losses** — идентификационные пайплайны
4. **Лоссы metric learning и майнеры** — какой лосс, какой майнер, когда
5. **ROC AUC** + **калибровка** — пороги в open-set постановках

### Компьютерное зрение и детекция

1. **NMS** — классический пайплайн детекции
2. End-to-end подходы (YOLO26, DETR), которые убирают NMS
3. Эволюция от NMS-based к query-based детекции
4. Transformer-based детекторы и их преимущества
5. **Видеокодеки H.264/H.265** — как камера сжимает поток и как декодировать на GPU

### Фильтрация и трекинг

1. **Гауссово распределение**
2. **Unscented Kalman Filter** — нелинейная фильтрация и трекинг
3. Эволюция KF → EKF → UKF → Particle Filter
4. **Метрики action recognition и tracking** — как выбирать протокол оценки
5. Современный deep tracking
6. Связка с **NMS** в пайплайнах детекции

### RL и управление

1. **Deep Reinforcement Learning** — основы RL
2. Value-based (DQN), policy-based (PPO) и actor-critic (SAC)
3. Применения в робототехнике и автономном вождении
4. Sim-to-real и вопросы безопасности
5. Современные подходы: foundation models, diffusion policies, hierarchical RL

### Робототехника и vision-based обучение роботов

1. **VLA-модели** — как объединяют зрение, язык и действие
2. Современные архитектуры: OpenVLA, RT-1, RT-2, Octo
3. **Vision-based методы обучения роботов**
4. Подходы: Imitation Learning, RL, VLA
5. Open-source методы и датасеты (Open X-Embodiment)
6. Sim-to-real
7. Разные типы роботов: гуманоиды, четвероногие, колёсные, манипуляторы

### Системы и serving

1. **System Design для CV и NLP** — сначала $\rho$ и SLO, потом фреймворк
2. **Triton и GPU serving** — runtime и SOTA-стек (vLLM / TensorRT-LLM) на 1–N GPU
2. 100 клиентов vs 1000: горизонтальные реплики, dynamic batching
3. LLM: KV-кэш и continuous batching, не «просто RPS»
4. Камеры: latest-frame + NVDEC, не FIFO всех кадров
5. **Видеокодеки** — почему ingest часто упирается в декод
6. **Настройка гиперпараметров** — соседний слой: обучение, не serving

### Настройка гиперпараметров

1. **Деревья решений** и **ансамбли** — модели с большим числом гиперпараметров
2. **Настройка гиперпараметров** — обзор методов
3. Grid Search → Random Search → Bayesian Optimization (Optuna)
4. Продвинутые методы: Hyperband, BOHB, PBT
5. LR Finder и расписания для нейросетей
6. NAS для поиска архитектур

### Ансамбли и комбинирование моделей

1. **Деревья решений** как строительный блок большинства ансамблей
2. **Методы комбинирования моделей**
3. Bagging (Random Forest) → Boosting (XGBoost/LightGBM/CatBoost)
4. Stacking и Voting для разнородных моделей
5. **Mixture of Experts** и **Model Merging** на масштабе LLM
6. Knowledge Distillation и DL-специфичные ансамбли (TTA, SWA)

## Как дополнять книгу

При добавлении новых материалов:

- Создать или обновить директорию `topics/<topic-slug>/`
- Добавить Obsidian frontmatter (или расширить `scripts/kb_topic_metadata.py` и пересобрать через `kb_apply_obsidian_frontmatter.py`)
- Держать структуру: Оглавление, разделы, Источники
- Формулы писать в LaTeX
- Если есть код — нумерованные скрипты в `scripts/` и тесты в `tests/`
- Один скрипт — один подпункт; в комментариях (по-русски) писать, что именно он показывает
- Обновить этот README и проверить перекрёстные ссылки
- Обновить индексы `docs/` и запустить `uv run python scripts/kb_validate_links.py`

## Источники

- https://github.com/Mathews-Tom/no-magic
