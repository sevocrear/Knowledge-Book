---
title: "MOC: NLP, LLM & RAG"
description: "Токенизация, эмбеддинги, transformers, LoRA, RAG и code agents."
tags:
  - kb/moc
  - domain/nlp
  - domain/llm
type: moc
status: canonical
updated: 2026-08-10
---

# MOC: NLP, LLM & RAG

Токенизация, эмбеддинги, transformers, LoRA, RAG и code agents.

## Topics

- [Tokenization and Text Compression in LLMs](../../topics/tokenization-and-text-compression-in-llms/README.md) — Word/char/BPE/WordPiece/Unigram токенизация как сжатие текста перед LLM и влияние на стоимость attention.
- [Embeddings and Embedding Matrix](../../topics/embeddings-and-embedding-matrix/README.md) — Векторные представления токенов, матрица эмбеддингов V×d, lookup по ID и роль эмбеддингов в Transformer и RAG.
- [Transformers, Attention и Vision Transformers (ViT)](../../topics/transformers-attention-and-vision-transformers-vit/README.md) — Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.
- [Low-Rank Adaptation (LoRA)](../../topics/low-rank-adaptation-lora/README.md) — PEFT через низкоранговые адаптеры ΔW≈BA: математика, QLoRA/AdaLoRA/DoRA, эффективность памяти и практика в Hugging Face PEFT.
- [Retrieval-Augmented Generation (RAG)](../../topics/retrieval-augmented-generation-rag/README.md) — Архитектуры RAG (Naive/Advanced/Modular/Self-RAG/Corrective/LightRAG), retriever/reranker, chunking, оценка и production-практики.
- [Code Agents, AutoResearch и Loopy Era](../../topics/code-agents-autoresearch-and-loopy-era/README.md) — Оркестрация code agents, AutoResearch loops, verification gates, harness engineering и переход от ручного кода к управлению агентными циклами.

## See also

- [All topics](../index.md)
- [Tags](../tags/README.md)
