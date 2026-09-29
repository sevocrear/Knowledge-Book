---
title: "Сториборд клипа: Triton — dynamic batching и concurrent models"
description: "Сториборд и команды сборки HyperFrames-клипа triton-dynamic-batching-concurrent-models: сбор запросов в батч, несколько моделей на одной GPU и устройство model repository с endpoints."
tags:
  - kb/note
  - kb/visualization
  - domain/mlops
  - concept/model-serving
  - concept/triton
aliases:
  - triton-dynamic-batching-concurrent-models
  - Triton storyboard
related:
  - triton-inference-server-and-gpu-model-serving
  - ml-system-design-for-cv-and-nlp
status: notes
lang: ru
type: note
slug: triton-inference-server-and-gpu-model-serving/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Triton: dynamic batching и concurrent models — сториборд

Клип `assets/visualizations/triton-dynamic-batching-concurrent-models.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Что такое Triton Inference Server», «Какие задачи решает», «Реальные примеры», «Полезен ли Triton, когда всего 1 GPU?»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как Triton собирает запросы в батч? | Клиенты шлют по одному запросу → очередь модели с таймером окна ожидания → пачка B = 4 (иллюстративно) уезжает на GPU одним вызовом; полосы «по одному: 4·T₀ + 4·t_item» против «батчем: T₀ + 4·t_item»; карточки `dynamic_batching` (preferred_batch_size, max_queue_delay_microseconds ~2–5 мс) и T(B) = T₀ + t_item·B | batch = 1, GPU любит пачки · Triton ждёт до max_queue_delay_microseconds, собирает до preferred_batch_size · один вызов GPU вместо четырёх, цена — небольшая задержка в очереди |
| 2 | 14–29 с | Как несколько моделей делят одну GPU? | Дорожки: строгий FIFO (детектор, эмбеддер, детектор, эмбеддер подряд) против concurrent (детектор ‖ эмбеддер на двух дорожках, «раньше», длины иллюстративны); endpoint HTTP/REST · gRPC · C API и три модели: детектор TensorRT, классификатор ONNX Runtime, постпроцессинг Python backend; карточки instance_group, симуляция `01_…`, список backends | строгий FIFO: эмбеддер ждёт за детектором · concurrent: несколько CUDA-потоков и instance · backends — один endpoint, клиенту всё равно, на чём обучили |
| 3 | 29–43 с | Как Triton хранит модели и принимает запросы? | Дерево `model_repository/model_name/{config.pbtxt, 1/model.onnx, 2/}` + «canary», «откат = переключить версию»; путь клиент → scheduler модели → backend → ответ; ensemble decode/resize → YOLO → NMS → crop → embedding; карточки config.pbtxt, наблюдаемость, «Полезен ли на 1 GPU?» | версии 1/, 2/ и config.pbtxt · HTTP/REST или gRPC → scheduler → backend → ответ · ensemble в одном сервере · **ключевая идея**: Triton батчит запросы и делит GPU между моделями — GPU меньше простаивает |

Числа и названия из README: T(B) = T₀ + t_item·B, throughput ≈ B/T(B); `max_queue_delay` ~2–5 мс (пример A, детектор); `instance_group` с `gpus: [0]` / `gpus: [1]`; backends TensorRT, ONNX Runtime, PyTorch, OpenVINO, Python, FIL, vLLM.
Размер пачки B = 4, длины блоков FIFO/concurrent и ширины полос T₀ / t_item — иллюстративные (подписаны на экране); в симуляции `01_concurrent_model_multiplexing.py` README приводит только направление эффекта (throughput ↑, p99 ожидания ↓), без чисел.

## Сборка

```bash
cd topics/triton-inference-server-and-gpu-model-serving/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 2.5,6.3,12,18,20,27,30,33,41.5
npx -y hyperframes@0.8.81 render --quality looks --output renders/triton-dynamic-batching-concurrent-models.mp4
ffmpeg -i renders/triton-dynamic-batching-concurrent-models.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/triton-dynamic-batching-concurrent-models.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/triton-dynamic-batching-concurrent-models.mp4 -o ../../assets/visualizations/triton-dynamic-batching-concurrent-models.gif
```
