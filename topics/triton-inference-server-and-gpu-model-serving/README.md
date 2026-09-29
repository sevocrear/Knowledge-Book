---
title: Triton Inference Server и развёртывание моделей на 1–N GPU
description: "NVIDIA Triton: dynamic batching, concurrent execution, ensembles; полезен ли на 1 GPU; SOTA serving (vLLM, TensorRT-LLM, SGLang) для 100× пользователей."
tags:
  - kb/topic
  - domain/mlops
  - domain/llm
  - domain/cv
  - concept/model-serving
  - concept/triton
  - concept/inference
aliases:
  - Triton Inference Server
  - NVIDIA Triton
  - GPU model serving
  - inference serving stack
  - vLLM vs Triton
related:
  - ml-system-design-for-cv-and-nlp
  - transformers-attention-and-vision-transformers-vit
  - tokenization-and-text-compression-in-llms
  - retrieval-augmented-generation-rag
  - video-codecs-h264-h265-and-gpu-decode
  - how-models-predict-confidence-and-calibration
status: canonical
lang: ru
type: topic
slug: triton-inference-server-and-gpu-model-serving
updated: 2026-09-18
---
# Triton Inference Server и развёртывание моделей на 1–N GPU

## Как объяснить 5-летнему ребёнку

Представь кухню ресторана с одной плитой. Если каждый официант сам готовит один заказ и ждёт, пока плита остынет, гости голодные. Умный повар делает так: собирает похожие заказы в одну сковороду (батч), параллельно греет суп и жарит котлеты на разных конфорках одной плиты и говорит официантам «не стойте у плиты — отдайте заказ на стойку». Triton — такой повар для нейросетей: принимает запросы по HTTP/gRPC, сам собирает пачки и крутит несколько моделей на одной (или нескольких) видеокартах, чтобы гости не ждали зря.

**Визуализация (HyperFrames, 43 с):** сцена 1 — одиночные запросы копятся в очереди и уходят на GPU одной пачкой (dynamic batching); сцена 2 — несколько моделей и instance делят одну GPU (concurrent execution, разные backends); сцена 3 — model repository, путь запроса и когда Triton окупается на 1 GPU.

![Triton: dynamic batching и concurrent models](./assets/visualizations/triton-dynamic-batching-concurrent-models.gif)

*Полная версия: [MP4 1080p](./assets/visualizations/triton-dynamic-batching-concurrent-models.mp4) · сториборд и исходники сцен: [`visualizations/hyperframes/`](./visualizations/hyperframes/storyboard.md).*

## Оглавление

1. [Что такое Triton Inference Server](#что-такое-triton-inference-server)
2. [Какие задачи решает](#какие-задачи-решает)
3. [Реальные примеры](#реальные-примеры)
4. [Полезен ли Triton, когда всего 1 GPU?](#полезен-ли-triton-когда-всего-1-gpu)
5. [SOTA-методы serving на 1–N GPU для «100× пользователей»](#sota-методы-serving-на-1n-gpu-для-100-пользователей)
6. [Как выбрать стек](#как-выбрать-стек)
7. [Примеры кода](#примеры-кода)
8. [Источники](#источники)

---

## Что такое Triton Inference Server

**NVIDIA Triton Inference Server** — открытый сервер инференса: ставишь модели в *model repository*, поднимаешь процесс, клиенты шлют запросы по **HTTP/REST**, **gRPC** или C API, а Triton сам планирует исполнение на GPU/CPU.

Это не «ещё один FastAPI с `model.predict`». Triton — слой **runtime + scheduler**:

| Слой | Что делает Triton |
|---|---|
| Model repository | Версии моделей на диске (`model_name/1/model.onnx`, `config.pbtxt`) |
| Backends | TensorRT, ONNX Runtime, PyTorch, OpenVINO, Python, FIL (деревья), vLLM backend и др. |
| Schedulers | Dynamic batching, sequence batching, приоритеты очередей |
| Concurrent execution | Несколько моделей / instance groups на одной GPU |
| Ensembles / BLS | Пайплайн: preprocess → model → postprocess без отдельного микросервиса на каждый шаг |
| Observability | Метрики Prometheus (очередь, GPU util, latency), health endpoints |

Архитектурно: запрос → per-model scheduler (опционально батчит) → backend → ответ. Подробности ёмкости и $\rho$ — в соседнем топике [System Design для CV и NLP](../ml-system-design-for-cv-and-nlp/README.md); здесь — *какой runtime и зачем*.

Triton **не заменяет** оптимизацию самой модели (TensorRT / квантизация / FlashAttention). Он *управляет* уже оптимизированными артефактами и смешивает нагрузку так, чтобы GPU не простаивала.

---

## Какие задачи решает

### 1. Dynamic batching (серверный батчинг)

Клиенты шлют `batch=1`. Triton ждёт до `max_queue_delay_microseconds` и собирает пачку до `preferred_batch_size`. GPU любит пачки из‑за фиксированного overhead кернела:

$$T(B) = T_0 + t_{\text{item}} \cdot B, \qquad \text{throughput} \approx \frac{B}{T(B)}.$$

Без серверного батчера каждый сервис сам пишет очередь — и почти всегда делает это хуже готового scheduler'а.

### 2. Concurrent model execution (мультиплексирование GPU)

На одной карте одновременно могут крутиться *разные* модели (или несколько instance одной). Железо GPU умеет переключать потоки; Triton отдаёт ему несколько CUDA-потоков/инстансов. Итог: пока детектор «думает», эмбеддер не обязан стоять в очереди за ним в одном процессе «по кругу».

Симуляция в `01_concurrent_model_multiplexing.py`: при двух моделях и смешанном трафике concurrent-режим поднимает суммарный throughput и режет p99 ожидания относительно строгого FIFO «одна модель за раз».

### 3. Один endpoint — много фреймворков

В репозитории рядом живут TensorRT-детектор, ONNX-классификатор и Python-постпроцессинг. Клиенту всё равно, на чём обучили; ops не плодит пять разных Docker-образов с разными портами.

### 4. Ensembles и BLS (Business Logic Scripting)

Типичный CV-пайплайн: decode/resize → YOLO → NMS → crop → embedding. В Triton это **ensemble** (граф моделей) или **BLS** (Python-логика внутри сервера). Меньше сетевых hop'ов между микросервисами, меньше копий тензоров через JSON.

### 5. Sequence batching / stateful

Для RNN, streaming ASR, трекеров с состоянием на инстансе — sequence batcher маршрутизирует шаги одной сессии на один instance и умеет держать implicit state.

### 6. Горизонталь и multi-GPU на ноде

`instance_group` с `gpus: [0]`, `gpus: [1]` или несколько инстансов на одной GPU. За нодой — Kubernetes + балансировщик; Triton даёт единый контракт модели и метрик.

### 7. Что Triton *не* решает сам

- Не обучит модель и не выберет архитектуру.
- Для LLM «как у ChatGPT» сам по себе (без TensorRT-LLM / vLLM backend) слабее специализированных LLM-серверов: continuous batching, PagedAttention, prefix cache — их родина в vLLM / TensorRT-LLM / SGLang.
- Не заменит расчёт ёмкости: если $\rho \ge 1$, нужен ещё GPU или меньше работы на запрос ([System Design](../ml-system-design-for-cv-and-nlp/README.md)).

---

## Реальные примеры

### A. Ритейл / видеоаналитика на одной ноде

**Задача:** 20–40 камер, детекция людей/SKU, re-ID эмбеддинги, лёгкий классификатор полки.

**Как без Triton:** три FastAPI-процесса, каждый держит свой CUDA context → фрагментация VRAM, ручной батчинг, три health-check'а.

**С Triton:**

1. `detector` (TensorRT) + dynamic batching, `max_queue_delay` ~2–5 мс.
2. `embedder` (ONNX) — отдельный instance на той же GPU.
3. Ensemble: crop из детектора → эмбеддер.
4. Клиент аналитики шлёт кадры gRPC; метрики очереди смотрит Prometheus → автоскейл подов.

Выигрыш: одна GPU обслуживает **смешанный** трафик (не все камеры одновременно бьют в детектор), SM util 60–80% вместо 20% при «однопоточном» FastAPI.

### B. Backend продукта: несколько моделей на одном GPU для стартапа

**Задача:** OCR + классификация документа + лёгкий LLM-summary (7B), 50–200 RPS пиков днём, ночью почти ноль.

На старте — **1× A10/L4**. Triton (или Triton + отдельный vLLM для LLM) держит OCR и классификатор с concurrent execution; LLM — отдельный процесс/backend с лимитом concurrent slots. Днём dynamic batching спасает от таймаутов; ночью задержка батча почти нулевая (очередь пустая).

### C. Банк / страхование: canary и версии

Model repository с версиями `1/`, `2/`. Canary: 5% трафика на `2` через policy на ingress или два model name. Откат = переключить версию, не пересобирать весь монолит. Калибровку и пороги смотрят отдельно ([калибровка](../how-models-predict-confidence-and-calibration/README.md)).

### D. LLM-чат на 1–4 GPU (когда Triton — обвязка)

Крупные команды часто ставят **TensorRT-LLM или vLLM** как backend, а Triton — как единый control plane (HTTP, auth на edge, ensembles «retriever → generator» для RAG). Чистый «только Triton + PyTorch» для авторегрессии обычно уступает vLLM по tokens/s.

---

## Полезен ли Triton, когда всего 1 GPU?

**Да, часто полезен** — но не всегда обязателен. Зависит от *типа* нагрузки.

| Сценарий на 1 GPU | Triton помогает? | Почему |
|---|---|---|
| Много коротких независимых запросов (классификация, детекция, эмбеддинги) | **Да** | Dynamic batching + instance groups поднимают throughput без кода клиента |
| Несколько разных моделей на одной карте | **Да** | Concurrent execution; иначе пишете свой мультиплексор |
| Пайплайн preprocess→model→postprocess | **Да** | Ensemble / BLS, меньше hop'ов |
| Один LLM, только chat, одна модель | **Скорее vLLM / TRT-LLM / SGLang** | Continuous batching и PagedAttention важнее «универсального» сервера |
| Прототип, 1 RPS, один ноутбук | **Нет** | `uvicorn` + torch достаточно; Triton — ops-стоимость |
| Модель не влезает в VRAM даже в INT4 | **Нет магии** | Нужен меньший квант, tensor parallel на N GPU или другая модель |

Интуиция: на 1 GPU Triton зарабатывает на **утилизации простоя**. Если GPU и так 95% времени в одном тяжёлом forward'е без очереди — батчер почти нечему помогать. Если приходят редкие одиночные запросы или микс моделей — выигрыш большой (см. скрипт `01_...`).

Правило большого пальца:

1. **1 модель, CV/эмбеддинги, десятки–сотни RPS** → Triton (или ORT + свой тонкий батчер) на 1 GPU — разумно.
2. **1 LLM** → специализированный LLM-сервер; Triton опционален как фасад.
3. **N моделей / ансамбль** → Triton почти всегда окупается даже на 1 GPU.

---

## SOTA-методы serving на 1–N GPU для «100× пользователей»

«100× пользователей» почти никогда не значит «×100 GPU». Обычно это: больше одновременных сессий, выше RPS, тот же или чуть больший флот карт — за счёт **батчинга, памяти, квантизации и планировщика**.

Ниже — актуальный стек (2024–2026) по классам задач.

### A. Оптимизация самой модели (обязательный первый слой)

Без этого любой сервер упрётся в потолок:

- **TensorRT / ONNX Runtime / OpenVINO** — граф, fusion, INT8/FP8.
- **Квантизация весов** LLM: GPTQ, AWQ, bitsandbytes NF4, FP8 на Hopper/Blackwell.
- **Спекулятивное декодирование** (draft + verify) — выше tokens/s при том же GPU.
- **Меньше токенов** на ответ: хороший системный промпт, constrained decoding; см. [токенизацию](../tokenization-and-text-compression-in-llms/README.md).

### B. LLM: continuous batching и память KV

Классический static batch ждёт, пока *все* в пачке закончат generate. **Continuous / in-flight batching** освобождает слот, как только один пользователь закончил, и тут же берёт следующего.

Ключевые системы:

| Система | Сильная сторона | Когда брать |
|---|---|---|
| **vLLM** | PagedAttention, continuous batching, широкий зоопарк моделей, OpenAI-compatible API | Default для open-weight LLM на 1–N GPU |
| **TensorRT-LLM** | Максимальный tokens/s на NVIDIA после компиляции | Продакшен NVIDIA, готовы к engine build |
| **SGLang** | RadixAttention / prefix cache, сложные агентные деревья вызовов | Много общих префиксов, tool-calling |
| **Hugging Face TGI** | Удобный HF-экосистемный деплой | Быстрый старт в HF-стеке |
| **LMDeploy / LightLLM** | Сильный throughput на части моделей | Альтернатива vLLM в Азии/облаках |

Память KV растёт с длиной контекста и числом одновременных слотов ([Transformers / KV cache](../transformers-attention-and-vision-transformers-vit/README.md)). «100× пользователей» часто = **не держать 100× KV сразу**, а:

- лимит `max_num_seqs` / concurrent slots;
- очередь + 429 при переполнении;
- prefix / prompt cache для одинаковых system prompt;
- chunked prefill, иногда **disaggregated prefill/decode** на разных GPU при N≥2.

Симуляция слотов continuous batching — `02_continuous_batching_slots.py`: при том же «бюджете» слотов continuous обслуживает больше завершённых диалогов, чем static batch.

### C. CV / классика: Triton + TensorRT

SOTA-практика для детекции/сегментации/эмбеддингов:

1. Экспорт в ONNX → TensorRT engine (или `torch.compile` / Torch-TensorRT).
2. Triton: dynamic batching + 1–2 instance на GPU.
3. Пре/пост на CPU или Python backend; тяжёлый decode видео — NVDEC ([кодеки](../video-codecs-h264-h265-and-gpu-decode/README.md)).
4. Latest-frame очередь для камер, не бесконечный FIFO ([System Design](../ml-system-design-for-cv-and-nlp/README.md)).

Альтернативы и надстройки: **NVIDIA Dynamo / NIM** (NIM — контейнеры поверх Triton/TensorRT-LLM), **DeepStream** (видеопайплайны), **BentoML**, **Ray Serve** + ORT.

### D. Оркестрация 1–N GPU и «100×» трафика

| Приём | 1 GPU | N GPU |
|---|---|---|
| Dynamic / continuous batching | must | must |
| Несколько instance / tensor parallel | instance groups | TP внутри модели (LLM) или data-parallel реплики |
| Pipeline parallel | редко | большие модели, аккуратно с bubble |
| Expert parallelism (MoE) | если влезает | EP across GPU |
| Балансировщик + автоскейл по queue depth | один процесс, лимит очереди | реплики за L7 / K8s HPA |
| Кэш (эмбеддинги, RAG chunks, prompt prefix) | must | + consistent hashing к реплике |
| Rate limit / fair queueing | must | must, иначе один клиент съест слоты |

**Data-parallel реплики** (N копий одной модели) масштабируют RPS почти линейно, пока хватает PCIe/сети. **Tensor parallel** нужен, когда *одна* модель не влезает или нужен больший batch на decode.

### E. RAG и агенты

Латентность = retrieval + (rerank) + generate. Кэшировать эмбеддинги запроса и чанки; не гонять генератор, если retrieval пустой. См. [RAG](../retrieval-augmented-generation-rag/README.md). Для tool-calling агентов — SGLang / vLLM с хорошим prefix cache часто выгоднее «голого» Triton.

### F. Чего избегать как «SOTA»

- Один синхронный `for request in queue: model(request)` на GPU без батча.
- Держать бесконечную очередь «чтобы никто не получил 429» — получите минуты устаревших ответов.
- Пихать LLM через обычный dynamic batcher как CNN — без continuous batching потеряете ×2–×10 throughput.
- Масштабировать по CPU util, игнорируя GPU SM util и depth очереди.

---

## Как выбрать стек

Краткая карта решений:

```text
Нужен LLM (авторегрессия)?
  да → vLLM / TensorRT-LLM / SGLang
       └─ нужен единый фасад с CV/эмбеддингами? → Triton + LLM backend
  нет → несколько моделей / ансамбль / TensorRT?
         да → Triton (часто даже на 1 GPU)
         нет, одна маленькая модель, мало RPS → FastAPI + ORT/TensorRT
```

Для «100× пользователей» на 1–N GPU чеклист:

1. Посчитать $\lambda$, $\mu$, $\rho$ и KV-память — не число DAU.
2. Оптимизировать модель (TensorRT / квант).
3. Включить правильный батчер (dynamic для CNN, continuous для LLM).
4. Поставить лимиты concurrent + очередь с политикой отказа.
5. Кэшировать повторяющееся.
6. Только потом докупать GPU / включать TP.

---

## Примеры кода

Симуляции без CUDA — показывают *смысл* планировщиков Triton/vLLM:

```bash
uv run python topics/triton-inference-server-and-gpu-model-serving/scripts/01_concurrent_model_multiplexing.py
uv run python topics/triton-inference-server-and-gpu-model-serving/scripts/02_continuous_batching_slots.py
uv run pytest topics/triton-inference-server-and-gpu-model-serving/tests -q
```

1. `01_concurrent_model_multiplexing.py` — две модели на одной GPU: FIFO vs concurrent; рост суммарного throughput.
2. `02_continuous_batching_slots.py` — static batch vs continuous slots; больше завершённых сессий при том же лимите слотов.

---

## Источники

### В этой книге

- [System Design для CV и NLP](../ml-system-design-for-cv-and-nlp/README.md) — $\rho$, SLO, dynamic batching, 100 vs 1000 клиентов, камеры.
- [Transformers, Attention и ViT](../transformers-attention-and-vision-transformers-vit/README.md) — KV-кэш как потолок concurrent LLM.
- [Токенизация и сжатие текста в LLM](../tokenization-and-text-compression-in-llms/README.md) — стоимость decode.
- [RAG](../retrieval-augmented-generation-rag/README.md) — бюджет латентности retrieval vs generate.
- [Видеокодеки и GPU-decode](../video-codecs-h264-h265-and-gpu-decode/README.md) — decode часто дороже детектора.
- [Калибровка уверенности](../how-models-predict-confidence-and-calibration/README.md) — canary и пороги в проде.

### Внешние

- [NVIDIA Triton Inference Server docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Triton: model execution / concurrent models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_execution.md)
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), 2023.
- NVIDIA TensorRT-LLM; SGLang (RadixAttention); Hugging Face Text Generation Inference.
