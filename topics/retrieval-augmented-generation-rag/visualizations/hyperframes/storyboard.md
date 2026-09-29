---
title: "Сториборд клипа: RAG — индексация, retrieval, augmentation, generation"
description: "Сториборд и команды сборки HyperFrames-клипа rag_pipeline: почему не просто LLM, offline-индексация документов в vector store, online-поиск Top-K чанков, промпт с контекстом и ответ с источниками."
tags:
  - kb/note
  - kb/visualization
  - domain/nlp
  - concept/rag
  - concept/embeddings
aliases:
  - RAG storyboard
  - rag_pipeline
related:
  - retrieval-augmented-generation-rag
  - embeddings-and-embedding-matrix
status: notes
lang: ru
type: note
slug: retrieval-augmented-generation-rag/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# RAG: индексация offline, retrieval → augmentation → generation online — сториборд

Клип `assets/visualizations/rag_pipeline.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Зачем нужен RAG?», «Как работает RAG → Базовая архитектура», «Детальный процесс», этапы 1–4).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–11 с | Почему не «просто LLM»? | Запрос «Что в нашем регламенте v3.2?» → бокс LLM → ответ «?» → «выдумывает ответ» (красный); вокруг LLM по одному появляются красные теги: знания заморожены (training cutoff), нет приватных данных, галлюцинации, ограниченный контекст; снизу въезжает «библиотека» (стопка документов), стрелка «+ retrieval», теги становятся зелёными: актуальные данные, приватные данные, меньше галлюцинаций, ответ с источниками; ответ → «ответ по документам, источник: регламент v3.2» | LLM без RAG: знания заморожены на дате обучения, нет доступа к приватным данным, может выдумывать факты · RAG = дать модели «библиотеку»: сначала найти нужное, потом отвечать по найденному — без переобучения · плюс объяснимость: можно показать, из каких документов взят ответ |
| 2 | 11–24 с | Что происходит offline: индексация | Конвейер Документы → Chunking → Embedding → Vector Store: документ разлетается на 4 chunk-карточки; каждая превращается в вектор (полоска из 6 столбиков); векторы улетают в панель Vector Store — 2D-карту эмбеддингов (16 синих точек индекса + 4 оранжевых chunk 1–4); подписи: фиксированный размер / семантически / иерархически · модель эмбеддингов: E5, BGE-M3, text-embedding-3 · vector store: FAISS, Qdrant, Chroma, Weaviate, Pinecone; тег «offline · заранее» | документы режем на chunks: фиксированный размер, семантическое или иерархическое разбиение · каждый chunk превращаем в вектор моделью эмбеддингов (E5, BGE-M3, text-embedding-3) · векторы кладём в vector store (FAISS, Qdrant, Chroma, Weaviate, Pinecone) — это делается заранее, offline |
| 3 | 24–42 с | Online: retrieval → augmentation → generation | Та же карта слева; запрос (query) → «query embedding: та же модель, что при индексации» → белое кольцо приземляется на карту; 3 ближайших chunk-точки (1, 2, 3) подсвечиваются линиями, подпись «Top-K (cosine), K = 3»; пилюли chunk 1/2/3 летят в карточку «2 · augmentation · промпт»: Context: [chunk 1, chunk 2, chunk 3] / Question: [вопрос] / Answer: …; стрелка → LLM (3 · generation) → карточка «ответ» с зелёными бейджами источников chunk 1, chunk 3 и строкой «post-processing: источники, проверка качества» | запрос векторизуем той же моделью и ищем ближайшие chunks: cosine / dot product / L2; берём Top-K · augmentation: собираем промпт = Context: [найденные chunks] + Question: [вопрос] · generation: LLM отвечает по контексту; post-processing добавляет ссылки на источники и проверку качества · **ключевая идея**: RAG = индексация offline + retrieval → augmentation → generation online: актуальные и приватные данные без переобучения, ответ с источниками |

Числа в клипе иллюстративные: K = 3 (Top-K), 4 chunk'а из одного документа, координаты точек на 2D-карте эмбеддингов заданы константами в сценах 2 и 3 (одни и те же в обеих сценах); «регламент v3.2» — вымышленный пример приватного документа. Списки стратегий chunking, моделей эмбеддингов и vector store — из README темы.

## Сборка

```bash
cd topics/retrieval-augmented-generation-rag/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 2,6,10,13,18,23,27,32,37,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/rag_pipeline.mp4
ffmpeg -i renders/rag_pipeline.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/rag_pipeline.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/rag_pipeline.mp4 -o ../../assets/visualizations/rag_pipeline.gif
```
