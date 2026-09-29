---
title: "Тег: domain/nlp"
description: Заметки с тегом domain/nlp в книге знаний.
tags:
  - domain/nlp
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `domain/nlp`

## Заметки

- [Эмбеддинги и матрица эмбеддингов](../../topics/embeddings-and-embedding-matrix/README.md) — Векторные представления токенов, матрица эмбеддингов V×d, lookup по ID и роль эмбеддингов в Transformer и RAG.
- [Сториборд клипа: эмбеддинги — матрица E и lookup по ID](../../topics/embeddings-and-embedding-matrix/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа embedding-matrix-lookup: текст → токены → ID → строки матрицы E ∈ ℝ^{V×d} → тензор n×d в Transformer; косинусное сходство и карта кластеров; pooling (mean / [CLS] / last) в один вектор для RAG.
- [System Design для Computer Vision и NLP](../../topics/ml-system-design-for-cv-and-nlp/README.md) — Ёмкость, балансировка, serving на 100 vs 1000 клиентов, dynamic batching, KV cache и обработка 10–50 видеопотоков.
- [Retrieval-Augmented Generation (RAG)](../../topics/retrieval-augmented-generation-rag/README.md) — Архитектуры RAG (Naive/Advanced/Modular/Self-RAG/Corrective/LightRAG), retriever/reranker, chunking, оценка и production-практики.
- [Сториборд клипа: RAG — индексация, retrieval, augmentation, generation](../../topics/retrieval-augmented-generation-rag/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа rag_pipeline: почему не просто LLM, offline-индексация документов в vector store, online-поиск Top-K чанков, промпт с контекстом и ответ с источниками.
- [Токенизация и сжатие текста в LLM](../../topics/tokenization-and-text-compression-in-llms/README.md) — Word/char/BPE/WordPiece/Unigram токенизация как сжатие текста перед LLM и влияние на стоимость attention.
- [Сториборд клипа: BPE — как токенизатор учит словарь и сжимает текст](../../topics/tokenization-and-text-compression-in-llms/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа bpe-tokenization-merges: текст → токены → ID → строка матрицы эмбеддингов и почему это сжатие; обучение BPE — частоты соседних пар и четыре слияния; кодирование нового слова выученными merges, компромисс размера словаря и byte-level BPE.
- [Transformers, Attention и Vision Transformers (ViT)](../../topics/transformers-attention-and-vision-transformers-vit/README.md) — Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.

