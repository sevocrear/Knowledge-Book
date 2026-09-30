---
title: "MOC: NLP, LLM и RAG"
description: "Токенизация, эмбеддинги, transformers, LoRA, RAG, code agents, протоколы агентов и serving."
tags:
  - kb/moc
  - domain/nlp
  - domain/llm
type: moc
status: canonical
updated: 2026-09-29
---

# MOC: NLP, LLM и RAG

Токенизация, эмбеддинги, transformers, LoRA, RAG, code agents, протоколы агентов и serving.

## Темы

- [Токенизация и сжатие текста в LLM](../../topics/tokenization-and-text-compression-in-llms/README.md) — Word/char/BPE/WordPiece/Unigram токенизация как сжатие текста перед LLM и влияние на стоимость attention.
- [Эмбеддинги и матрица эмбеддингов](../../topics/embeddings-and-embedding-matrix/README.md) — Векторные представления токенов, матрица эмбеддингов V×d, lookup по ID и роль эмбеддингов в Transformer и RAG.
- [Transformers, Attention и Vision Transformers (ViT)](../../topics/transformers-attention-and-vision-transformers-vit/README.md) — Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.
- [Low-Rank Adaptation (LoRA)](../../topics/low-rank-adaptation-lora/README.md) — PEFT через низкоранговые адаптеры ΔW≈BA: математика, QLoRA/AdaLoRA/DoRA, эффективность памяти и практика в Hugging Face PEFT.
- [Retrieval-Augmented Generation (RAG)](../../topics/retrieval-augmented-generation-rag/README.md) — Архитектуры RAG (Naive/Advanced/Modular/Self-RAG/Corrective/LightRAG), retriever/reranker, chunking, оценка и production-практики.
- [Code Agents, AutoResearch и Loopy Era](../../topics/code-agents-autoresearch-and-loopy-era/README.md) — Оркестрация code agents, AutoResearch loops, verification gates, harness engineering и переход от ручного кода к управлению агентными циклами.
- [MCP, ACP, UCP и Agent Harness](../../topics/agent-protocols-mcp-acp-ucp-and-harness/README.md) — Слои агентных протоколов (MCP, ACP, UCP, A2A) и agent harness: что к чему подключается, чем не путать аббревиатуры и как собрать эффективный harness в Cursor.
- [System Design для Computer Vision и NLP](../../topics/ml-system-design-for-cv-and-nlp/README.md) — Ёмкость, балансировка, serving на 100 vs 1000 клиентов, dynamic batching, KV cache и обработка 10–50 видеопотоков.

## См. также

- [Все темы](../index.md)
- [Теги](../tags/README.md)
