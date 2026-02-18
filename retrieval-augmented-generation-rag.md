# Retrieval-Augmented Generation (RAG)

## Содержание

1. [Введение](#введение)
2. [Как работает RAG](#как-работает-rag)
3. [Математические основы](#математические-основы)
4. [Типы RAG систем](#типы-rag-систем)
   - [Naive RAG](#naive-rag)
   - [Advanced RAG](#advanced-rag)
   - [Modular RAG](#modular-rag)
   - [Self-RAG](#self-rag)
   - [Corrective RAG](#corrective-rag)
   - [LightRAG](#lightrag)
   - [Agentic RAG](#agentic-rag)
5. [Компоненты RAG системы](#компоненты-rag-системы)
6. [Техники улучшения RAG](#техники-улучшения-rag)
   - [Re-ranking и Cross-Encoder](#2-re-ranking) — детальное объяснение
7. [Реализация](#реализация)
8. [Оценка качества](#оценка-качества)
9. [Применения](#применения)
10. [Текущее состояние (2023-2026)](#текущее-состояние-2023-2026)
11. [Ссылки](#ссылки)

---

## Введение

**Retrieval-Augmented Generation (RAG)** — это архитектурный паттерн, который объединяет информационный поиск (retrieval) с генерацией текста (generation) для создания более точных и актуальных ответов языковых моделей.

### Простое объяснение (для 5-летнего)

Представь, что у тебя есть очень умный друг, который может отвечать на вопросы, но иногда забывает факты. RAG — это как дать ему библиотеку с книгами. Когда ты задаёшь вопрос, он сначала ищет нужную информацию в книгах, а потом использует её, чтобы дать правильный ответ. Это помогает ему не выдумывать и всегда говорить правду!

### Зачем нужен RAG?

**Проблемы LLM без RAG:**
- Ограниченные знания (training cutoff date)
- Галлюцинации (выдумывание фактов)
- Нет доступа к актуальным данным
- Нет доступа к приватным/корпоративным данным
- Ограниченный контекст (context window)

**Преимущества RAG:**
- ✅ Доступ к актуальной информации
- ✅ Работа с приватными данными
- ✅ Снижение галлюцинаций
- ✅ Объяснимость (можно показать источники)
- ✅ Экономичность (не нужно переобучать модель)

---

## Как работает RAG

### Базовая архитектура

RAG состоит из трёх основных этапов:

```
┌─────────────┐
│   Query     │  Пользователь задаёт вопрос
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  1. RETRIEVAL (Поиск)               │
│  ┌───────────────────────────────┐  │
│  │ Query Embedding              │  │
│  │ (векторизация запроса)       │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ Vector Search                 │  │
│  │ (семантический поиск)         │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ Top-K Documents               │  │
│  │ (релевантные документы)       │  │
│  └───────────┬───────────────────┘  │
└──────────────┼──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. AUGMENTATION (Обогащение)       │
│  ┌───────────────────────────────┐  │
│  │ Context Construction          │  │
│  │ (формирование контекста)      │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ Prompt Engineering            │  │
│  │ (создание промпта)           │  │
│  └───────────┬───────────────────┘  │
└──────────────┼──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. GENERATION (Генерация)          │
│  ┌───────────────────────────────┐  │
│  │ LLM Inference                 │  │
│  │ (генерация ответа)            │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │ Final Answer                  │  │
│  │ (финальный ответ)             │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Детальный процесс

#### Этап 1: Индексация (Offline)

Перед использованием RAG нужно подготовить данные:

```
Документы → Chunking → Embedding → Vector Store
```

1. **Chunking (Разбиение на части)**
   - Разделение документов на небольшие фрагменты (chunks)
   - Стратегии: фиксированный размер, семантическое разбиение, иерархическое

2. **Embedding (Векторизация)**
   - Преобразование текста в векторные представления
   - Модели: text-embedding-ada-002, text-embedding-3-large, E5, BGE-M3

3. **Vector Store (Векторная база данных)**
   - Сохранение эмбеддингов для быстрого поиска
   - Примеры: Pinecone, Weaviate, Qdrant, Chroma, FAISS

#### Этап 2: Retrieval (Online)

Когда пользователь задаёт вопрос:

1. **Query Embedding**
   - Векторизация запроса пользователя
   - Используется та же модель эмбеддингов

2. **Similarity Search**
   - Поиск наиболее похожих документов
   - Метрики: cosine similarity, dot product, euclidean distance
   - Возвращается Top-K наиболее релевантных chunks

#### Этап 3: Augmentation

1. **Context Construction**
   - Объединение найденных документов в контекст
   - Может включать метаданные (источник, дата, релевантность)

2. **Prompt Engineering**
   - Формирование промпта для LLM
   - Типичный формат:
     ```
     Context: [найденные документы]
     
     Question: [вопрос пользователя]
     
     Answer: [LLM генерирует ответ на основе контекста]
     ```

#### Этап 4: Generation

1. **LLM Inference**
   - Передача промпта в языковую модель
   - Генерация ответа на основе контекста

2. **Post-processing**
   - Форматирование ответа
   - Добавление ссылок на источники
   - Проверка качества

---

## Математические основы

### Векторное представление

Документ $d$ и запрос $q$ преобразуются в векторы:

$$d \rightarrow \mathbf{d} \in \mathbb{R}^n$$
$$q \rightarrow \mathbf{q} \in \mathbb{R}^n$$

где $n$ — размерность эмбеддинга (обычно 384, 768, 1536).

### Функция поиска

Для запроса $q$ находим Top-K документов:

$$\text{Retrieve}(q, D, k) = \arg\max_{d \in D, |S|=k} \text{sim}(\mathbf{q}, \mathbf{d})$$

где:
- $D$ — множество документов
- $k$ — количество возвращаемых документов
- $\text{sim}(\cdot, \cdot)$ — функция схожести

### Косинусная схожесть

Наиболее популярная метрика:

$$\text{sim}_{\cos}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{||\mathbf{q}|| \cdot ||\mathbf{d}||} = \cos(\theta)$$

где $\theta$ — угол между векторами.

### Вероятность генерации

LLM генерирует ответ $a$ с вероятностью:

$$P(a | q, D) = \prod_{i=1}^{|a|} P(a_i | q, \text{Retrieve}(q, D, k), a_{<i})$$

где:
- $a_i$ — $i$-й токен ответа
- $a_{<i}$ — предыдущие токены
- $\text{Retrieve}(q, D, k)$ — найденный контекст

### Score для релевантности

Комбинированный score для ранжирования:

$$\text{score}(d, q) = \alpha \cdot \text{sim}_{\text{dense}}(\mathbf{q}, \mathbf{d}) + \beta \cdot \text{sim}_{\text{sparse}}(q, d)$$

где:
- $\text{sim}_{\text{dense}}$ — схожесть плотных векторов (семантическая)
- $\text{sim}_{\text{sparse}}$ — схожесть разреженных векторов (BM25, ключевые слова)
- $\alpha, \beta$ — веса для гибридного поиска

---

## Типы RAG систем

### Naive RAG

**Базовая RAG архитектура** — простейшая реализация без дополнительных оптимизаций.

#### Архитектура

```
Query → Embedding → Vector Search → Top-K Docs → LLM → Answer
```

#### Характеристики

- ✅ Простота реализации
- ✅ Быстрая разработка
- ❌ Низкое качество на сложных запросах
- ❌ Нет обработки нерелевантных результатов
- ❌ Фиксированный размер контекста

#### Когда использовать

- Простые задачи (QA, поиск фактов)
- Прототипирование
- Небольшие корпуса документов

#### Визуализация

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────────┐      ┌──────────────┐
│  Embedding   │─────▶│ Vector Store │
│   Model      │      │   (FAISS)    │
└──────────────┘      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Top-K Docs  │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │     LLM      │
                      │  (GPT-4, etc)│
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │    Answer    │
                      └──────────────┘
```

---

### Advanced RAG

**Продвинутая RAG** с техниками улучшения качества поиска и генерации.

#### Ключевые улучшения

1. **Query Rewriting**
   - Переформулирование запроса для лучшего поиска
   - Multi-query generation
   - Query expansion

2. **Re-ranking**
   - Повторное ранжирование результатов
   - Cross-encoder модели
   - Learned ranking

3. **Context Compression**
   - Сжатие найденного контекста
   - Извлечение ключевой информации
   - Иерархическое сжатие

#### Архитектура

```
Query → Query Rewriting → Embedding → Vector Search 
  → Re-ranking → Context Compression → LLM → Answer
```

#### Визуализация

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│ Query Rewriting │  (Multi-query, Expansion)
└────┬────────────┘
     │
     ▼
┌──────────────┐      ┌──────────────┐
│  Embedding   │─────▶│ Vector Store │
└──────────────┘      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Top-K Docs  │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Re-ranking  │  (Cross-encoder)
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────────┐
                      │ Context          │
                      │ Compression      │
                      └──────┬───────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │     LLM      │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │    Answer    │
                      └──────────────┘
```

#### Когда использовать

- Сложные запросы
- Большие корпуса документов
- Требования к высокой точности

---

### Modular RAG

**Модульная RAG** с адаптивным выбором стратегий поиска и генерации.

#### Ключевые особенности

1. **Query Routing**
   - Определение типа запроса
   - Выбор оптимальной стратегии поиска

2. **Adaptive Retrieval**
   - Разные стратегии для разных типов запросов
   - Semantic search, keyword search, hybrid

3. **Response Synthesis**
   - Разные методы генерации ответа
   - Map-reduce, refine, map-rerank

#### Архитектура

```
Query → Query Router → [Strategy 1 | Strategy 2 | Strategy 3] 
  → Retrieval → Synthesis → Answer
```

#### Визуализация

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────────┐
│ Query Router │  (Определяет тип запроса)
└───┬──────┬───┘
    │      │      │
    ▼      ▼      ▼
┌─────┐ ┌─────┐ ┌─────┐
│Sem. │ │Key. │ │Hybr.│  (Разные стратегии)
│Srch │ │Srch │ │Srch │
└──┬──┘ └──┬──┘ └──┬──┘
   │       │       │
   └───┬───┴───┬───┘
       │       │
       ▼       ▼
┌──────────────┐
│  Synthesis   │  (Map-reduce, Refine, etc.)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Answer    │
└──────────────┘
```

#### Когда использовать

- Разнородные типы запросов
- Необходимость гибкости
- Оптимизация производительности

---

### Self-RAG

**Self-Retrieval-Augmented Generation** — система с саморефлексией и адаптивным поиском.

#### Ключевые особенности

1. **Self-Reflection**
   - LLM оценивает необходимость поиска
   - Проверка качества найденной информации

2. **Adaptive Retrieval**
   - Поиск только когда нужно
   - Итеративный поиск при необходимости

3. **Self-Critique**
   - Критическая оценка сгенерированного ответа
   - Повторная генерация при необходимости

#### Архитектура

```
Query → [Need Retrieval?] → [Yes: Retrieve] → [Is Info Good?] 
  → Generate → [Is Answer Good?] → [Yes: Output | No: Regenerate]
```

#### Визуализация

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│ Need Retrieval? │  (Self-reflection)
└───┬─────────┬───┘
   No│       │Yes
     │       │
     ▼       ▼
┌────────┐ ┌──────────────┐
│Generate│ │   Retrieve   │
└───┬────┘ └──────┬───────┘
    │             │
    │             ▼
    │      ┌──────────────┐
    │      │ Is Info Good?│
    │      └───┬──────┬───┘
    │         No│    │Yes
    │           │    │
    │           ▼    ▼
    │      ┌──────────────┐
    │      │   Generate   │
    │      └──────┬───────┘
    │             │
    └─────────────┘
           │
           ▼
    ┌──────────────┐
    │ Is Answer    │
    │ Good?        │
    └───┬──────┬───┘
       No│    │Yes
         │    │
         ▼    ▼
    ┌──────────────┐
    │   Regenerate │ │ Output
    └──────────────┘ └────────┘
```

#### Когда использовать

- Критически важные приложения
- Требования к высокой точности
- Сложные многошаговые запросы

---

### Corrective RAG

**Corrective RAG** с циклами обнаружения и исправления ошибок.

#### Ключевые особенности

1. **Error Detection**
   - Обнаружение ошибок в ответе
   - Проверка противоречий с источниками

2. **Correction Loop**
   - Исправление ошибок через повторный поиск
   - Итеративное улучшение ответа

#### Архитектура

```
Query → Retrieve → Generate → [Error?] → [Yes: Corrective Retrieve] 
  → Regenerate → [Error?] → [No: Output]
```

#### Визуализация

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────────┐      ┌──────────────┐
│   Retrieve   │─────▶│ Vector Store │
└──────┬───────┘      └──────────────┘
       │
       ▼
┌──────────────┐
│   Generate   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Error        │
│ Detection?   │
└───┬──────┬───┘
   No│    │Yes
     │    │
     ▼    ▼
┌────────┐ ┌──────────────────┐
│ Output │ │ Corrective       │
│        │ │ Retrieve         │
└────────┘ └──────┬────────────┘
                  │
                  ▼
           ┌──────────────┐
           │  Regenerate  │
           └──────┬───────┘
                  │
                  └───▶ (Loop back to Error Detection)
```

#### Когда использовать

- Высокие требования к точности
- Критически важные приложения
- Длинные и сложные ответы

---

### LightRAG

**LightRAG** — граф-ориентированный подход к организации знаний.

#### Ключевые особенности

1. **Graph Construction**
   - Построение графа знаний из документов
   - Узлы: сущности, концепции
   - Рёбра: отношения

2. **Graph Traversal**
   - Навигация по графу для поиска информации
   - Многошаговый reasoning

#### Архитектура

```
Documents → Graph Construction → Query → Graph Traversal 
  → Relevant Subgraph → LLM → Answer
```

#### Визуализация

```
┌──────────────┐
│  Documents   │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Graph           │
│ Construction    │
│                 │
│  Entity1 ──┐    │
│           │    │
│  Entity2──┼───▶│ Entity3
│           │    │
│  Entity4 ─┘    │
└──────┬─────────┘
       │
       ▼
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────────┐
│ Graph        │
│ Traversal    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Subgraph     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     LLM      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Answer    │
└──────────────┘
```

#### Когда использовать

- Структурированные знания
- Многошаговый reasoning
- Связи между концепциями важны

---

### Agentic RAG

**Agentic RAG** — агентный подход с планированием и выполнением.

#### Ключевые особенности

1. **Planning**
   - Планирование шагов для ответа на запрос
   - Разбиение сложных запросов на подзадачи

2. **Tool Use**
   - Использование различных инструментов поиска
   - Адаптивный выбор стратегий

3. **Execution Loop**
   - Итеративное выполнение плана
   - Адаптация на основе промежуточных результатов

#### Архитектура

```
Query → Plan → [Tool 1 | Tool 2 | Tool 3] → Execute 
  → Evaluate → [Done?] → [No: Continue] → [Yes: Synthesize] → Answer
```

#### Визуализация

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────────┐
│    Plan      │  (Разбиение на шаги)
└───┬──────────┘
    │
    ▼
┌─────────────────────────┐
│   Tool Selection        │
│                         │
│  ┌────┐  ┌────┐  ┌────┐│
│  │Sem.│  │Key.│  │Web ││
│  │Srch│  │Srch│  │Srch││
│  └─┬──┘  └─┬──┘  └─┬──┘│
└────┼───────┼───────┼────┘
     │       │       │
     └───┬───┴───┬───┘
         │       │
         ▼       ▼
    ┌──────────────┐
    │   Execute    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Evaluate    │
    └───┬──────┬───┘
       Done│  │Not Done
           │  │
           ▼  ▼
    ┌──────────────┐
    │  Synthesize  │ │ Continue
    └──────┬───────┘ └─────────┘
           │
           ▼
    ┌──────────────┐
    │    Answer    │
    └──────────────┘
```

#### Когда использовать

- Сложные многошаговые запросы
- Необходимость использования разных источников
- Динамическое планирование

---

## Компоненты RAG системы

### 1. Embedding Models (Модели эмбеддингов)

#### OpenAI Embeddings

- **text-embedding-ada-002**
  - Размерность: 1536
  - Хорошее качество для общего использования

- **text-embedding-3-small**
  - Размерность: 1536
  - Более эффективная версия

- **text-embedding-3-large**
  - Размерность: 3072
  - Лучшее качество для сложных задач

#### Open-Source Embeddings

- **E5 (Microsoft)**
  - E5-base, E5-large, E5-mistral-7b-instruct
  - Хорошее качество, бесплатно

- **BGE (BAAI)**
  - BGE-base-en-v1.5, BGE-large-en-v1.5
  - BGE-M3 (мультиязычный)
  - State-of-the-art для многих задач

- **Multilingual-E5**
  - Поддержка множества языков
  - Хорошее качество для неанглийских текстов

#### Сравнение моделей

| Модель | Размерность | Качество | Скорость | Стоимость |
|--------|-------------|----------|----------|-----------|
| text-embedding-ada-002 | 1536 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 💰💰💰 |
| text-embedding-3-large | 3072 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 💰💰💰💰 |
| E5-large | 1024 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Бесплатно |
| BGE-large | 1024 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Бесплатно |

### 2. Vector Stores (Векторные базы данных)

#### Pinecone

- ✅ Managed сервис
- ✅ Высокая производительность
- ✅ Автоматическое масштабирование
- ❌ Платный

#### Weaviate

- ✅ Open-source
- ✅ GraphQL API
- ✅ Гибридный поиск
- ✅ Хорошая документация

#### Qdrant

- ✅ Open-source
- ✅ Высокая производительность
- ✅ Гибридный поиск
- ✅ Хорошая поддержка фильтров

#### Chroma

- ✅ Простота использования
- ✅ Легковесный
- ✅ Хорош для прототипирования
- ❌ Меньше функций для production

#### FAISS (Facebook AI Similarity Search)

- ✅ Очень быстрый
- ✅ Open-source
- ✅ Хорош для больших корпусов
- ❌ Требует больше кода для интеграции

### 3. Chunking Strategies (Стратегии разбиения)

#### Fixed-Size Chunking

Простое разбиение на фиксированные размеры:

```python
def fixed_size_chunking(text, chunk_size=512, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

**Плюсы:**
- Простота реализации
- Предсказуемый размер

**Минусы:**
- Может разрывать семантические единицы
- Не учитывает структуру документа

#### Semantic Chunking

Разбиение по семантической близости:

```python
def semantic_chunking(text, embedding_model, threshold=0.7):
    sentences = split_into_sentences(text)
    embeddings = embedding_model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            embeddings[i-1], embeddings[i]
        )
        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    return chunks
```

**Плюсы:**
- Сохраняет семантическую целостность
- Лучшее качество поиска

**Минусы:**
- Сложнее реализовать
- Требует модель эмбеддингов

#### Hierarchical Chunking

Иерархическое разбиение (документ → разделы → параграфы):

```
Document
├── Section 1
│   ├── Paragraph 1.1
│   └── Paragraph 1.2
└── Section 2
    ├── Paragraph 2.1
    └── Paragraph 2.2
```

**Плюсы:**
- Сохраняет структуру
- Позволяет поиск на разных уровнях

**Минусы:**
- Сложная реализация
- Требует структурированные документы

### 4. Retrieval Strategies (Стратегии поиска)

#### Dense Retrieval (Плотный поиск)

Семантический поиск по векторам:

$$\text{Retrieve}(q) = \arg\max_{d \in D} \cos(\mathbf{q}, \mathbf{d})$$

**Плюсы:**
- Понимает семантику
- Хорошо для концептуальных запросов

**Минусы:**
- Может пропускать точные совпадения
- Требует хорошие эмбеддинги

#### Sparse Retrieval (Разреженный поиск)

Поиск по ключевым словам (BM25):

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

**Плюсы:**
- Точные совпадения
- Быстрый поиск
- Не требует эмбеддинги

**Минусы:**
- Не понимает синонимы
- Плохо для концептуальных запросов

#### Hybrid Search (Гибридный поиск)

Комбинация dense и sparse:

$$\text{score}(q, d) = \alpha \cdot \text{BM25}(q, d) + \beta \cdot \cos(\mathbf{q}, \mathbf{d})$$

**Плюсы:**
- Лучшее из обоих миров
- Высокое качество

**Минусы:**
- Сложнее настроить
- Требует больше ресурсов

---

## Техники улучшения RAG

### 1. Query Rewriting

#### Multi-Query Generation

Генерация нескольких вариантов запроса:

```python
def multi_query_generation(query, llm, n=3):
    prompt = f"""
    Generate {n} different versions of this query:
    {query}
    
    Each version should:
    - Use different wording
    - Focus on different aspects
    - Be semantically similar
    """
    queries = llm.generate(prompt)
    return queries
```

#### Query Expansion

Расширение запроса синонимами и связанными терминами:

```python
def query_expansion(query, thesaurus):
    expanded = [query]
    for word in query.split():
        synonyms = thesaurus.get_synonyms(word)
        expanded.extend([query.replace(word, s) for s in synonyms])
    return expanded
```

### 2. Re-ranking

#### Что такое Re-ranking и зачем он нужен?

**Re-ranking (повторное ранжирование)** — это процесс улучшения порядка документов, найденных на первом этапе поиска (retrieval). 

**Проблема первичного поиска:**
- Векторный поиск быстрый, но не всегда точный
- Может пропустить релевантные документы
- Может поставить нерелевантные документы выше релевантных
- Использует только косинусную схожесть векторов, не учитывая контекст

**Решение — Re-ranking:**
1. Первичный поиск находит Top-K документов (например, Top-100)
2. Re-ranker более точно оценивает релевантность каждого документа
3. Возвращаются Top-N наиболее релевантных (например, Top-5)

**Архитектура двухэтапного поиска:**

```
Query
  │
  ▼
┌─────────────────┐
│ Primary Search  │  (Быстрый, но менее точный)
│ Vector Search   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Top-K Docs    │  (например, Top-100)
│   (100 docs)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Re-ranker     │  (Медленнее, но точнее)
│  Cross-Encoder  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Top-N Docs    │  (например, Top-5)
│   (5 docs)      │
└─────────────────┘
```

#### Bi-Encoder vs Cross-Encoder
![[Pasted image 20260120124152.png]]

Чтобы понять Cross-Encoder, нужно сначала понять Bi-Encoder (который используется в обычном векторном поиске).

##### Bi-Encoder (Двухэтапное кодирование)

**Принцип работы:**
- Query и Document кодируются **независимо** друг от друга
- Каждый получает свой вектор-эмбеддинг
- Схожесть вычисляется как расстояние между векторами

**Архитектура:**

```
Query: "Что такое RAG?"
  │
  ▼
┌──────────────┐
│   Encoder    │
└──────┬───────┘
       │
       ▼
  [0.2, 0.5, ...]  ← Query Embedding (1536 dim)
       │
       │  Cosine Similarity
       │
       ▼
  [0.3, 0.4, ...]  ← Document Embedding (1536 dim)
       │
       ▼
┌──────────────┐
│   Encoder    │
└──────┬───────┘
       │
       ▼
Document: "RAG combines retrieval..."
```

**Математически:**

$$\mathbf{q} = \text{Encoder}_Q(\text{query})$$
$$\mathbf{d} = \text{Encoder}_D(\text{document})$$
$$\text{score} = \cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{||\mathbf{q}|| \cdot ||\mathbf{d}||}$$

**Преимущества Bi-Encoder:**
- ✅ Очень быстрый поиск (можно предвычислить эмбеддинги документов)
- ✅ Масштабируется на миллионы документов
- ✅ Эффективное использование индексов (FAISS, Pinecone)

**Недостатки Bi-Encoder:**
- ❌ Не видит взаимодействие между query и document
- ❌ Может пропустить релевантные документы с другой формулировкой
- ❌ Ограничен семантикой отдельных эмбеддингов

**Пример проблемы Bi-Encoder:**

```
Query: "Как работает машинное обучение?"
Document: "ML algorithms learn from data"

Bi-Encoder может дать низкий score, потому что:
- "машинное обучение" ≠ "ML" (разные слова)
- Эмбеддинги могут быть не очень похожи
- Но документ РЕЛЕВАНТЕН!
```

##### Cross-Encoder (Совместное кодирование)

**Принцип работы:**
- Query и Document подаются **вместе** в одну модель
- Модель видит их **взаимодействие** и контекст
- Прямо предсказывает релевантность (score)

**Архитектура:**

```
┌─────────────────────────────────────┐
│         Cross-Encoder Model        │
│                                     │
│  Input: [CLS] Query [SEP] Document │
│         ↓                           │
│     Transformer                    │
│     (BERT, RoBERTa, etc.)          │
│         ↓                           │
│     Classification Head            │
│         ↓                           │
│      Score: 0.95                   │
└─────────────────────────────────────┘
```

**Математически:**

$$\text{score} = \text{CrossEncoder}([\text{query}; \text{document}])$$

где `[query; document]` — конкатенация токенов query и document.

**Внутренняя работа Cross-Encoder:**

1. **Токенизация:**
   ```
   Input: "[CLS] Что такое RAG? [SEP] RAG combines retrieval..."
   Tokens: [CLS, Что, такое, RAG, ?, [SEP], RAG, combines, ...]
   ```

2. **Transformer обработка:**
   - Self-attention видит все токены одновременно
   - Может связать "RAG" в query с "RAG" в document
   - Понимает контекст и семантику

3. **Классификация:**
   - [CLS] токен содержит информацию о всей паре
   - Linear layer предсказывает релевантность
   - Output: score от 0 до 1 (или logit)

**Визуализация внимания в Cross-Encoder:**

```
Query:    "Что такое RAG?"
Document: "RAG combines retrieval and generation"

Attention weights показывают связи:
- "RAG" (query) ←→ "RAG" (doc)  [сильная связь]
- "что" (query) ←→ "combines" (doc)  [слабая связь]
- "такое" (query) ←→ "retrieval" (doc)  [средняя связь]
```

**Преимущества Cross-Encoder:**
- ✅ **Точность**: Видит взаимодействие query-document
- ✅ **Контекст**: Понимает семантику в контексте пары
- ✅ **Гибкость**: Может найти релевантность даже при разных формулировках

**Недостатки Cross-Encoder:**
- ❌ **Медленно**: Нужно вычислять для каждой пары query-document
- ❌ **Не масштабируется**: Нельзя предвычислить эмбеддинги документов
- ❌ **Дорого**: Требует больше вычислений

#### Сравнение Bi-Encoder vs Cross-Encoder

| Характеристика | Bi-Encoder | Cross-Encoder |
|----------------|------------|---------------|
| **Скорость** | ⚡⚡⚡⚡⚡ Очень быстрый | 🐌 Медленный |
| **Точность** | ⭐⭐⭐ Хорошая | ⭐⭐⭐⭐⭐ Отличная |
| **Масштабируемость** | ✅ Миллионы документов | ❌ Тысячи документов |
| **Предвычисление** | ✅ Да (эмбеддинги документов) | ❌ Нет |
| **Взаимодействие** | ❌ Не видит | ✅ Видит |
| **Использование** | Первичный поиск | Re-ranking |

#### Почему используется двухэтапный подход?

**Оптимальная стратегия:**

```
Stage 1: Bi-Encoder (быстрый поиск)
  └─> Находит Top-100 из миллионов документов
      Время: ~10ms

Stage 2: Cross-Encoder (точное ранжирование)
  └─> Ранжирует Top-100 → Top-5
      Время: ~100ms (10ms × 10 пар)

Total: ~110ms (быстро и точно!)
```

Если бы использовали только Cross-Encoder:
- Нужно проверить миллионы пар
- Время: ~100ms × 1,000,000 = 100,000 секунд (27 часов!) ❌

#### Cross-Encoder Re-ranking: Реализация

**Базовая реализация:**

```python
from sentence_transformers import CrossEncoder
import numpy as np

def rerank(query, documents, top_k=10):
    """
    Re-ranking документов с помощью Cross-Encoder
    
    Args:
        query: Запрос пользователя
        documents: Список документов для ранжирования
        top_k: Количество документов для возврата
    
    Returns:
        Отсортированный список документов
    """
    # Загрузка модели Cross-Encoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Создание пар [query, document]
    pairs = [[query, doc] for doc in documents]
    
    # Предсказание scores для всех пар
    # scores[i] = релевантность documents[i] к query
    scores = model.predict(pairs)
    
    # Сортировка по убыванию score
    ranked_indices = np.argsort(scores)[::-1]
    
    # Возврат Top-K документов
    return [documents[i] for i in ranked_indices[:top_k]]
```

**Продвинутая реализация с batch processing:**

```python
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Инициализация Re-ranker
        
        Args:
            model_name: Название модели Cross-Encoder
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        Re-ranking документов
        
        Args:
            query: Запрос пользователя
            documents: Список документов
            top_k: Количество документов для возврата
            batch_size: Размер батча для обработки
        
        Returns:
            Список кортежей (document, score)
        """
        # Создание пар
        pairs = [[query, doc] for doc in documents]
        
        # Batch processing для эффективности
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch)
            scores.extend(batch_scores)
        
        scores = np.array(scores)
        
        # Сортировка
        ranked_indices = np.argsort(scores)[::-1]
        
        # Возврат Top-K с scores
        results = [
            (documents[i], float(scores[i])) 
            for i in ranked_indices[:top_k]
        ]
        
        return results
```

**Пример использования:**

```python
# Инициализация
reranker = Reranker()

# Первичный поиск (Bi-Encoder)
query = "Как работает RAG?"
primary_results = vector_store.search(query, top_k=100)
# Результат: 100 документов, отсортированных по косинусной схожести

# Re-ranking (Cross-Encoder)
reranked_results = reranker.rerank(
    query=query,
    documents=[doc.text for doc in primary_results],
    top_k=5
)

# Вывод результатов
for doc, score in reranked_results:
    print(f"Score: {score:.4f}")
    print(f"Document: {doc[:100]}...")
    print()
```

#### Популярные модели Cross-Encoder

**Для релевантности поиска:**

1. **cross-encoder/ms-marco-MiniLM-L-6-v2**
   - Обучена на MS MARCO dataset
   - Баланс скорости и качества
   - Размер: ~80MB

2. **cross-encoder/ms-marco-MiniLM-L-12-v2**
   - Больше параметров
   - Лучшее качество, но медленнее

3. **cross-encoder/ms-marco-electra-base**
   - ELECTRA архитектура
   - Хорошее качество

**Для других задач:**

- **cross-encoder/stsb-roberta-base**: Для semantic similarity
- **cross-encoder/nli-deberta-v3-base**: Для natural language inference

#### Когда использовать Re-ranking?

**Используйте Re-ranking когда:**
- ✅ Нужна высокая точность (production системы)
- ✅ Количество документов после первичного поиска < 1000
- ✅ Можно пожертвовать скоростью ради качества
- ✅ Бюджет позволяет дополнительные вычисления

**Не используйте Re-ranking когда:**
- ❌ Нужна максимальная скорость
- ❌ Очень большие корпуса (> миллионы документов)
- ❌ Ограниченные вычислительные ресурсы
- ❌ Прототипирование (можно обойтись без)

#### Метрики для оценки Re-ranking

**Улучшение метрик после Re-ranking:**

```python
def evaluate_reranking(original_results, reranked_results, ground_truth):
    """
    Оценка улучшения после Re-ranking
    
    Args:
        original_results: Результаты первичного поиска
        reranked_results: Результаты после Re-ranking
        ground_truth: Правильные релевантные документы
    """
    # Precision@K до Re-ranking
    precision_before = precision_at_k(original_results, ground_truth, k=5)
    
    # Precision@K после Re-ranking
    precision_after = precision_at_k(reranked_results, ground_truth, k=5)
    
    # Улучшение
    improvement = (precision_after - precision_before) / precision_before * 100
    
    print(f"Precision@5 до Re-ranking: {precision_before:.3f}")
    print(f"Precision@5 после Re-ranking: {precision_after:.3f}")
    print(f"Улучшение: {improvement:.1f}%")
```

**Типичное улучшение:**
- Precision@5: +10-30%
- MRR: +15-40%
- NDCG@10: +20-50%

#### Визуализация процесса Re-ranking

**До Re-ranking (Bi-Encoder):**

```
Query: "Что такое RAG?"

Rank | Document                          | Score
------+-----------------------------------+-------
  1   | RAG это технология...            | 0.85
  2   | Retrieval augmented...            | 0.82
  3   | Как работает машинное обучение... | 0.80  ← Не релевантен!
  4   | RAG combines retrieval...         | 0.78
  5   | Векторные базы данных...          | 0.75
```

**После Re-ranking (Cross-Encoder):**

```
Query: "Что такое RAG?"

Rank | Document                          | Score
------+-----------------------------------+-------
  1   | RAG combines retrieval...         | 0.95  ↑ Было 4-м
  2   | RAG это технология...            | 0.92
  3   | Retrieval augmented...            | 0.89
  4   | Векторные базы данных...          | 0.65  ↓ Было 5-м
  5   | Как работает машинное обучение... | 0.45  ↓ Было 3-м
```

**Вывод:** Cross-Encoder правильно определил, что документ #4 более релевантен, несмотря на то, что Bi-Encoder дал ему более низкий score.

### 3. Context Compression

#### Summarization

Сжатие контекста через суммаризацию:

```python
def compress_context(documents, llm, max_length=1000):
    summary_prompt = f"""
    Summarize the following documents, keeping only 
    the most relevant information:
    
    {documents}
    
    Maximum length: {max_length} tokens
    """
    compressed = llm.generate(summary_prompt)
    return compressed
```

#### Extraction

Извлечение только релевантных частей:

```python
def extract_relevant(documents, query, llm):
    extraction_prompt = f"""
    Extract only the parts of these documents that 
    are relevant to: {query}
    
    Documents:
    {documents}
    """
    extracted = llm.generate(extraction_prompt)
    return extracted
```

### 4. Multi-Hop Reasoning

Итеративный поиск для сложных запросов:

```python
def multi_hop_retrieval(query, vector_store, llm, max_hops=3):
    context = []
    
    for hop in range(max_hops):
        # Поиск релевантных документов
        docs = vector_store.search(query, top_k=5)
        context.extend(docs)
        
        # Проверка, достаточно ли информации
        check_prompt = f"""
        Query: {query}
        Context: {context}
        
        Can you answer the query based on this context?
        If not, what information is missing?
        """
        
        response = llm.generate(check_prompt)
        
        if "yes" in response.lower():
            break
        
        # Обновление запроса для следующего шага
        query = llm.generate(f"""
        Based on: {response}
        Generate a new query to find missing information.
        """)
    
    return context
```

---

## Реализация

### Простой пример на Python

```python
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. Подготовка данных
documents = [
    "RAG combines retrieval and generation...",
    "Vector databases store embeddings...",
    # ... больше документов
]

# 2. Разбиение на chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Создание эмбеддингов и векторной БД
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Создание RAG цепи
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. Использование
query = "Как работает RAG?"
answer = qa_chain.run(query)
print(answer)
```

### Продвинутый пример с Re-ranking

```python
from sentence_transformers import CrossEncoder
import numpy as np

class AdvancedRAG:
    def __init__(self, vectorstore, llm, reranker_model):
        self.vectorstore = vectorstore
        self.llm = llm
        self.reranker = CrossEncoder(reranker_model)
    
    def retrieve(self, query, top_k=20):
        # Первичный поиск
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return docs
    
    def rerank(self, query, documents, top_k=5):
        # Re-ranking с cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Сортировка по score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices]
    
    def generate(self, query, documents):
        # Формирование контекста
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Генерация ответа
        prompt = f"""
        Context:
        {context}
        
        Question: {query}
        
        Answer based on the context above:
        """
        
        answer = self.llm.generate(prompt)
        return answer
    
    def query(self, user_query):
        # Полный pipeline
        docs = self.retrieve(user_query, top_k=20)
        ranked_docs = self.rerank(user_query, docs, top_k=5)
        answer = self.generate(user_query, ranked_docs)
        return answer, ranked_docs
```

### Пример с LangChain

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Загрузка документов
loader = TextLoader("documents.txt")
documents = loader.load()

# Разбиение на chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Создание векторной БД
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Создание RAG цепи
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Использование
result = qa_chain({"query": "Ваш вопрос"})
print(result["result"])
print(result["source_documents"])
```

---

## Оценка качества

### Метрики Retrieval

#### Precision@K

Доля релевантных документов среди Top-K:

$$\text{Precision@K} = \frac{|\{\text{relevant docs}\} \cap \{\text{retrieved docs}\}|}{K}$$

#### Recall@K

Доля найденных релевантных документов:

$$\text{Recall@K} = \frac{|\{\text{relevant docs}\} \cap \{\text{retrieved docs}\}|}{|\{\text{relevant docs}\}|}$$

#### MRR (Mean Reciprocal Rank)

Среднее обратное ранга первого релевантного документа:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

где $\text{rank}_q$ — ранг первого релевантного документа для запроса $q$.

#### NDCG (Normalized Discounted Cumulative Gain)

Учитывает порядок релевантных документов:

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

где:
- $\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}$
- $\text{IDCG@K}$ — идеальный DCG

### Метрики Generation

#### BLEU

Оценка качества перевода/генерации:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

где:
- $p_n$ — precision для n-грамм
- $\text{BP}$ — brevity penalty

#### ROUGE

Оценка для суммаризации:

- **ROUGE-L**: Longest Common Subsequence
- **ROUGE-N**: N-gram overlap
- **ROUGE-W**: Weighted LCS

#### Semantic Similarity

Семантическая схожесть ответа с эталоном:

```python
from sentence_transformers import SentenceTransformer

def semantic_similarity(answer, reference):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([answer, reference])
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return similarity
```

### Метрики End-to-End

#### Faithfulness

Насколько ответ соответствует источникам:

```python
def faithfulness(answer, sources, llm):
    prompt = f"""
    Answer: {answer}
    Sources: {sources}
    
    Does the answer contain any information not present 
    in the sources? Answer yes or no.
    """
    response = llm.generate(prompt)
    return "no" in response.lower()
```

#### Answer Relevance

Релевантность ответа запросу:

```python
def answer_relevance(query, answer, llm):
    prompt = f"""
    Query: {query}
    Answer: {answer}
    
    How relevant is the answer to the query? 
    Score from 1-5.
    """
    score = llm.generate(prompt)
    return int(score)
```

---

## Применения

### 1. Вопросно-ответные системы (QA)

- Корпоративные базы знаний
- Техническая поддержка
- Документация продуктов

### 2. Поисковые системы

- Семантический поиск
- Поиск по документам
- Поиск в коде

### 3. Чатботы

- Customer support
- Виртуальные ассистенты
- Образовательные боты

### 4. Анализ документов

- Юридические документы
- Медицинские записи
- Научные статьи

### 5. Генерация контента

- Статьи на основе источников
- Техническая документация
- Отчёты

---

## Текущее состояние (2023-2026)

### Тренды 2024-2025

1. **Мультимодальный RAG**
   - Работа с изображениями, видео, аудио
   - CLIP для визуального поиска

2. **Граф-ориентированный RAG**
   - LightRAG, Knowledge Graphs
   - Структурированное представление знаний

3. **Агентный RAG**
   - Планирование и выполнение
   - Использование инструментов

4. **Эффективные эмбеддинги**
   - Колоссальные модели (ColBERT, BGE-M3)
   - Квантизация и оптимизация

5. **Оценка качества**
   - Автоматические метрики
   - Human evaluation frameworks

### Актуальные исследования

- **Self-RAG** (2023): Саморефлексия и адаптивный поиск
- **Corrective RAG** (2024): Исправление ошибок
- **LightRAG** (2024): Граф-ориентированный подход
- **RAG-Fusion** (2024): Множественные запросы и слияние результатов

### Будущие направления

1. **Улучшение качества retrieval**
   - Лучшие модели эмбеддингов
   - Гибридный поиск становится стандартом

2. **Оптимизация генерации**
   - Более эффективные LLM
   - Специализированные модели для RAG

3. **Мультимодальность**
   - Интеграция различных типов данных
   - Единое представление

4. **Автоматизация**
   - Автоматический выбор стратегий
   - Self-optimizing RAG systems

---

## Визуализации и схемы

### Интерактивные диаграммы и GIF

Для лучшего понимания архитектур RAG рекомендуется посмотреть следующие визуализации:

#### 1. Базовая архитектура RAG

![RAG Basic Architecture](https://www.pinecone.io/learn/retrieval-augmented-generation/)

**Источники с визуализациями:**
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) — интерактивная диаграмма базового RAG
- [Databricks RAG Overview](https://www.databricks.com/glossary/retrieval-augmented-generation-rag) — схема компонентов RAG

#### 2. Типы RAG систем

**Сравнительная диаграмма:**
- [LangChain RAG Types](https://blog.langchain.dev/deconstructing-rag/) — сравнение Naive, Advanced, Modular RAG
- [LlamaIndex RAG Patterns](https://www.llamaindex.ai/blog) — различные паттерны RAG

#### 3. Self-RAG архитектура

![Self-RAG Flow](https://arxiv.org/abs/2310.11511)

**Визуализация процесса Self-RAG:**
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511) — детальная схема процесса саморефлексии
- [Self-RAG GitHub](https://github.com/AkariAsai/self-rag) — примеры использования с визуализациями

#### 4. Advanced RAG техники

**Визуализации компонентов:**
- **Query Rewriting**: [Multi-query RAG](https://blog.langchain.dev/query-construction/) — схема генерации множественных запросов
- **Re-ranking и Cross-Encoder**: 
  - [Cross-encoder Reranking Guide](https://www.sbert.net/examples/applications/cross-encoder/README.html) — детальное объяснение Cross-Encoder с примерами
  - [Bi-Encoder vs Cross-Encoder](https://www.sbert.net/docs/sentence_transformer/crossencoders.html) — сравнение архитектур
  - [Re-ranking Pipeline](https://www.pinecone.io/learn/reranking/) — визуализация двухэтапного поиска
- **Context Compression**: [LangChain Compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/) — схема сжатия контекста

#### 5. Agentic RAG

**Агентные архитектуры:**
- [LangGraph RAG Agents](https://blog.langchain.dev/langgraph/) — визуализация агентных циклов
- [LlamaIndex Agentic RAG](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/) — схемы планирования и выполнения

#### 6. LightRAG (Graph-based)

**Граф-ориентированные визуализации:**
- [LightRAG Paper](https://arxiv.org/abs/2405.19208) — схемы построения графа знаний
- [Knowledge Graph RAG](https://neo4j.com/developer-blog/knowledge-graphs-rag/) — визуализация графов в RAG

### Рекомендуемые видео и анимации

1. **RAG Pipeline Animation**
   - Поиск анимаций на YouTube: "RAG retrieval augmented generation animation"
   - Демонстрация потока данных от запроса к ответу

2. **Vector Search Visualization**
   - Интерактивные демо векторного поиска на сайтах Pinecone, Weaviate
   - Визуализация семантического поиска в многомерном пространстве

3. **RAG vs Fine-tuning Comparison**
   - Сравнительные диаграммы подходов улучшения LLM
   - Когда использовать RAG vs Fine-tuning

### Интерактивные инструменты

1. **LangChain Playground**
   - [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates) — готовые RAG шаблоны с визуализациями
   - Интерактивные примеры различных RAG паттернов

2. **LlamaIndex Demos**
   - [LlamaIndex Examples](https://docs.llamaindex.ai/en/stable/examples/) — интерактивные примеры с визуализацией
   - Демонстрация различных retrieval стратегий

3. **Streamlit RAG Apps**
   - Готовые Streamlit приложения с визуализацией RAG pipeline
   - Интерактивное исследование различных компонентов

---

## Ссылки

### Связанные темы

- **[Large Language Models (LLMs)](./large-language-models.md)** — основы работы LLM
- **[Attention Mechanisms](./attention-mechanisms.md)** — механизмы внимания в трансформерах
- **[Vector Databases](./vector-databases.md)** — детали векторных БД

### Полезные ресурсы

1. **Статьи и исследования:**
   - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
   - "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023) — [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
   - "Corrective Retrieval Augmented Generation" (Jiang et al., 2024) — [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
   - "LightRAG: A Lightweight Retrieval-Augmented Generation" (2024) — [arXiv:2405.19208](https://arxiv.org/abs/2405.19208)

2. **Библиотеки и фреймворки:**
   - [LangChain](https://www.langchain.com/) — фреймворк для RAG с множеством компонентов
   - [LlamaIndex](https://www.llamaindex.ai/) — специализированная библиотека для RAG
   - [Haystack](https://haystack.deepset.ai/) — end-to-end NLP framework с RAG
   - [Chroma](https://www.trychroma.com/) — векторная БД с простым API

3. **Векторные БД:**
   - [Pinecone](https://www.pinecone.io/) — managed векторная БД с хорошей документацией
   - [Weaviate](https://weaviate.io/) — open-source векторная БД с GraphQL
   - [Qdrant](https://qdrant.tech/) — высокопроизводительная векторная БД
   - [FAISS](https://github.com/facebookresearch/faiss) — библиотека для эффективного поиска похожести

4. **Визуализации и схемы:**
   - [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) — интерактивная диаграмма и объяснение
   - [Databricks RAG Overview](https://www.databricks.com/glossary/retrieval-augmented-generation-rag) — схема компонентов RAG
   - [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/) — примеры с визуализациями
   - [LlamaIndex RAG Patterns](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/) — различные паттерны RAG

5. **Обучающие материалы:**
   - [RAG Tutorials](https://www.pinecone.io/learn/) — пошаговые руководства
   - [LangChain RAG Tutorials](https://python.langchain.com/docs/tutorials/) — практические примеры
   - [LlamaIndex Getting Started](https://docs.llamaindex.ai/en/stable/getting_started/) — быстрый старт

---

## Заключение

RAG — это мощный паттерн, который решает ключевые проблемы LLM, предоставляя доступ к актуальной и релевантной информации. Развитие от Naive RAG к Advanced и Agentic RAG показывает постоянное улучшение качества и возможностей систем.

Выбор типа RAG зависит от конкретной задачи:
- **Naive RAG** — для простых случаев и прототипирования
- **Advanced RAG** — для production систем с высокими требованиями
- **Self-RAG / Corrective RAG** — для критически важных приложений
- **Agentic RAG** — для сложных многошаговых задач

Будущее RAG лежит в направлении мультимодальности, автоматизации и улучшения качества retrieval и generation.
