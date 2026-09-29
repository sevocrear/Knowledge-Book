---
title: "Сториборд клипа: эмбеддинги — матрица E и lookup по ID"
description: "Сториборд и команды сборки HyperFrames-клипа embedding-matrix-lookup: текст → токены → ID → строки матрицы E ∈ ℝ^{V×d} → тензор n×d в Transformer; косинусное сходство и карта кластеров; pooling (mean / [CLS] / last) в один вектор для RAG."
tags:
  - kb/note
  - kb/visualization
  - domain/nlp
  - domain/llm
  - concept/embeddings
aliases:
  - embedding-matrix-lookup
related:
  - embeddings-and-embedding-matrix
  - tokenization-and-text-compression-in-llms
  - retrieval-augmented-generation-rag
status: notes
lang: ru
type: note
slug: embeddings-and-embedding-matrix/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Эмбеддинги: матрица E и lookup по ID — сториборд

Клип `assets/visualizations/embedding-matrix-lookup.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Матрица эмбеддингов», «Связь с токенизатором: полный пайплайн»,
«Обучение эмбеддингов в LLM», «Размерность и размер словаря», «Эмбеддинги в других контекстах (RAG, поиск)»,
«От токенов к одному вектору на текст: pooling»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как текст попадает в Transformer? | Текст «кот сидит на коврике» → 4 пилюли-токена → ID (1042, 3517, 23, 7761 — иллюстративно) → матрица E ∈ ℝ^{V×d} (V строк, d столбцов); стрелка от каждого ID к своей строке, подсветка строки, копия строки уезжает в тензор n × d (n = 4) с подписями E[1042] … E[7761]; «+ pos» → блок Transformer (Attention, MLP, …); внизу V/d из README | токенизатор режет текст и даёт ID из {0, …, V−1}, V — размер словаря · E ∈ ℝ^{V×d} — lookup table: по ID берём строку E_id; one-hot(id) · E даёт ту же строку · n токенов → тензор n × d, + позиционное кодирование → первый слой Transformer |
| 2 | 14–28 с | Что значит «близкие» эмбеддинги? | 2D-карта (иллюстративно): 9 слов разбросаны, «обучение» стягивает их в кластеры (кот/кошка/собака, Париж/Москва/Лондон, бежать/идти/плыть); векторы из начала координат к «кот» и «кошка», дуга θ ≈ 11°, cos ≈ 0.98; вектор к «Париж», дуга θ ≈ 77°, cos ≈ 0.23; карточки «Косинусное сходство» cos(a, b) = a·b / (‖a‖·‖b‖) и «Близко или далеко» | E не задаётся вручную, а обучается с моделью — близкие по смыслу токены сближаются · похожесть меряют косинусом угла (или L2): «кот» и «кошка» почти одно направление · между кластерами угол большой: cos ≈ 0.23 (иллюстративно) |
| 3 | 28–42 с | Как получить один вектор на весь текст? | Выход энкодера: 5 строк ([CLS], кот, сидит, на, коврике) × d ячеек — тензор (n, d); три варианта pooling справа: mean (все строки съезжаются в одну усреднённую), [CLS] (первая строка), last (последняя строка); h_sent ∈ ℝ^d; карточки «Mean pooling» с формулой h_sent = (1/|I|) Σ_{i∈I} h_i и «Какой pooling где»; «→ RAG: запрос и чанки — по одному вектору, поиск по cos» | после энкодера — по вектору на токен: тензор (n, d), для поиска и RAG нужен один вектор · pooling: mean — среднее без [PAD]; [CLS] — первая позиция; last — последний токен в декодерах · **ключевая идея**: E превращает ID в векторы, pooling — n векторов в один; по нему RAG находит близкие чанки косинусом |

Числа и обозначения из README: E ∈ ℝ^{V×d}, ID ∈ {0, …, V−1}, `embeddings = E[id_seq]` → тензор n × d, «+ pos → вход в Transformer»;
V: Llama 3 — 128k, GPT‑4o — ~200k, Gemma — 256k; d (hidden size): 768, 4096, 8192; сравнение по косинусному сходству или L2;
mean pooling h_sent = (1/|I|) Σ_{i∈I} h_i (I — токены без [PAD]); [CLS] — BERT; last token — E5‑Mistral, NV‑Embed, Qwen3‑Embedding.
Иллюстративно (не из README): пример текста, токены и их ID, координаты точек на 2D-карте, θ ≈ 11° / 77° и cos ≈ 0.98 / 0.23, значения ячеек.

## Сборка

```bash
cd topics/embeddings-and-embedding-matrix/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,6.5,11,15.2,20,27,29.5,34,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/embedding-matrix-lookup.mp4
ffmpeg -i renders/embedding-matrix-lookup.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/embedding-matrix-lookup.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/embedding-matrix-lookup.mp4 -o ../../assets/visualizations/embedding-matrix-lookup.gif
```
