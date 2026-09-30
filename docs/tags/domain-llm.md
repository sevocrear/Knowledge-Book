---
title: "Тег: domain/llm"
description: Заметки с тегом domain/llm в книге знаний.
tags:
  - domain/llm
  - kb/tag-page
type: index
status: canonical
updated: 2026-09-29
---

# Тег `domain/llm`

## Заметки

- [MCP, ACP, UCP и Agent Harness](../../topics/agent-protocols-mcp-acp-ucp-and-harness/README.md) — Слои агентных протоколов (MCP, ACP, UCP, A2A) и agent harness: что к чему подключается, чем не путать аббревиатуры и как собрать эффективный harness в Cursor.
- [Code Agents, AutoResearch и Loopy Era](../../topics/code-agents-autoresearch-and-loopy-era/README.md) — Оркестрация code agents, AutoResearch loops, verification gates, harness engineering и переход от ручного кода к управлению агентными циклами.
- [Эмбеддинги и матрица эмбеддингов](../../topics/embeddings-and-embedding-matrix/README.md) — Векторные представления токенов, матрица эмбеддингов V×d, lookup по ID и роль эмбеддингов в Transformer и RAG.
- [Сториборд клипа: эмбеддинги — матрица E и lookup по ID](../../topics/embeddings-and-embedding-matrix/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа embedding-matrix-lookup: текст → токены → ID → строки матрицы E ∈ ℝ^{V×d} → тензор n×d в Transformer; косинусное сходство и карта кластеров; pooling (mean / [CLS] / last) в один вектор для RAG.
- [Методы комбинирования моделей (Ensemble Methods)](../../topics/ensemble-methods-model-combination/README.md) — Bagging/boosting/stacking, XGBoost/LightGBM/CatBoost, MoE, distillation и model merging (TIES/DARE/SLERP) для LLM.
- [Low-Rank Adaptation (LoRA)](../../topics/low-rank-adaptation-lora/README.md) — PEFT через низкоранговые адаптеры ΔW≈BA: математика, QLoRA/AdaLoRA/DoRA, эффективность памяти и практика в Hugging Face PEFT.
- [Сториборд клипа: LoRA — низкоранговая добавка ΔW = B·A](../../topics/low-rank-adaptation-lora/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа lora-low-rank-delta-w: заморозка W и разложение ΔW = B·A, экономия параметров и памяти (4096×4096, LLaMA-7B), прямой проход с α/r, слияние на инференсе и сменные адаптеры.
- [Retrieval-Augmented Generation (RAG)](../../topics/retrieval-augmented-generation-rag/README.md) — Архитектуры RAG (Naive/Advanced/Modular/Self-RAG/Corrective/LightRAG), retriever/reranker, chunking, оценка и production-практики.
- [Токенизация и сжатие текста в LLM](../../topics/tokenization-and-text-compression-in-llms/README.md) — Word/char/BPE/WordPiece/Unigram токенизация как сжатие текста перед LLM и влияние на стоимость attention.
- [Сториборд клипа: BPE — как токенизатор учит словарь и сжимает текст](../../topics/tokenization-and-text-compression-in-llms/visualizations/hyperframes/storyboard.md) — Сториборд и команды сборки HyperFrames-клипа bpe-tokenization-merges: текст → токены → ID → строка матрицы эмбеддингов и почему это сжатие; обучение BPE — частоты соседних пар и четыре слияния; кодирование нового слова выученными merges, компромисс размера словаря и byte-level BPE.
- [Transformers, Attention и Vision Transformers (ViT)](../../topics/transformers-attention-and-vision-transformers-vit/README.md) — Scaled dot-product attention, QKV, KV cache, positional encodings (в т.ч. RoPE), ViT и DETR-подобные детекция/сегментация.
- [Triton Inference Server и развёртывание моделей на 1–N GPU](../../topics/triton-inference-server-and-gpu-model-serving/README.md) — NVIDIA Triton: dynamic batching, concurrent execution, ensembles; полезен ли на 1 GPU; SOTA serving (vLLM, TensorRT-LLM, SGLang) для 100× пользователей.

