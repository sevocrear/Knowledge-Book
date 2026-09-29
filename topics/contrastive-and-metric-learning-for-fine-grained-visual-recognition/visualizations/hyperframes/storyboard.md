---
title: "Сториборд клипа: contrastive / metric learning и triplet loss"
description: "Сториборд и команды сборки HyperFrames-клипа contrastive_embedding_space: зачем эмбеддинги вместо классификатора, triplet loss с margin и semi-hard mining, кластеры, поиск ближайших соседей и порог τ в проде."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/contrastive-learning
  - concept/metric-learning
aliases:
  - metric learning storyboard
  - contrastive_embedding_space
related:
  - contrastive-and-metric-learning-for-fine-grained-visual-recognition
  - arcface-and-angular-margin-losses-for-identification
status: notes
lang: ru
type: note
slug: contrastive-and-metric-learning-for-fine-grained-visual-recognition/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Metric learning: пространство эмбеддингов — сториборд

Клип `assets/visualizations/contrastive_embedding_space.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Эмбеддинги и метрика сходства», «Основные семейства лоссов»,
«Сэмплинг батчей (P×K) и mining», «Калибровка порогов под бизнес-ошибки», «Как деплоить решение», «Continual learning»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Зачем эмбеддинги, а не классификатор? | Панель «пространство эмбеддингов ℝᵈ (2D-проекция, до обучения)»: 27 точек трёх сортов яблок (A зелёный, B фиолетовый, C оранжевый, по 9), перемешаны; легенда; карточки «Эмбеддинг» z = f_θ(x) ∈ ℝᵈ, ẑ = z/‖z‖₂ и «Сходство и расстояние» s = ẑᵢᵀẑⱼ (cosine), d = 1 − s или ‖ẑᵢ − ẑⱼ‖₂; в конце три соседние точки разных сортов пульсируют | Fine-grained: 15 сортов яблок отличаются мелкими деталями — жёсткая классификация ломается на новых сортах и дисбалансе · Metric learning учит карту: похожие — близко, разные — далеко по cosine / L2 · До обучения точки разных сортов перемешаны — соседи по карте ничего не значат |
| 2 | 13–29 с | Что такое triplet loss? | Панель с тройкой: anchor a (синий), positive p (зелёный), negative n (красный), отрезки d(a,p), d(a,n); столбики d(a,p) с оранжевой полосой «+ m» сверху и d(a,n); «L > 0», пока красный столбик ниже пунктира d(a,p)+m. Анимация: p притягивается к a, n отталкивается → красный столбик выше пунктира → «L = 0». Кольца r = d(a,p) и r = d(a,p)+m вокруг a — «semi-hard зона». Мини-батч P×K = 3×3. Карточки «Triplet loss» L = max(0, d(z_a,z_p) − d(z_a,z_n) + m) и «Родственники»: contrastive y·d² + (1−y)·max(0, m−d)², InfoNCE / NT-Xent (in-batch negatives, температура τ), SupCon | Тройка: anchor a, positive p (тот же класс), negative n (другой класс) · Пока d(a,n) < d(a,p) + m, лосс > 0: позитив притягиваем, негатив отталкиваем · Mining: случайные негативы слишком лёгкие (градиент ≈ 0); semi-hard — ближе, чем позитив + m, но не самый близкий — обычно лучший компромисс · Батч P классов × K примеров гарантирует и позитивы, и конкурирующие негативы |
| 3 | 29–43 с | Как это работает в проде? | Те же 27 точек стягиваются в три компактных кластера «сорт A / B / C»; запрос (белое кольцо) → три ближайших соседа (белые отрезки) → «top-3 → сорт B»; окружность «порог τ: ближе τ — принимаем, дальше — unknown». Справа пайплайн из 4 блоков, загораются по очереди: эмбеддинг-модель → ANN-индекс (FAISS / HNSW) → top-K + порог τ → мониторинг и дообучение | После обучения кластеры компактны; новый сорт = новые точки в индексе, без полного переобучения · Запрос: эмбеддинг → k ближайших соседей → решение по порогу τ (контролируем FAR и unknown) · **ключевая идея**: похожие — близко, разные — дальше минимум на margin; в проде: эмбеддинг-модель → индекс соседей → порог → мониторинг и дообучение |

Числа: все координаты точек, расстояния и margin в сцене 2 (m = 120 px, d(a,p): 291 → 112 px, d(a,n): 261 → 359 px)
и радиус порога τ = 140 px в сцене 3 — иллюстративные, заданы константами в скриптах сцен (без `Math.random`).
Три ближайших соседа запроса в сцене 3 — точки сорта B на расстояниях 93 / 117 / 117 px; четвёртый сосед (153 px) — за порогом.
Формулы — из README темы: нормировка ẑ = z/‖z‖₂, cosine similarity, d = 1 − s или L2, triplet loss с margin m,
contrastive (pairwise) loss, InfoNCE / NT-Xent с температурой τ, SupCon; P×K-сэмплинг; порог τ по validation (FAR, unknown);
пайплайн «эмбеддинг-модель → индекс ближайших соседей → порог/ранжирование → мониторинг и дообучение».

## Сборка

```bash
cd topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 4,9,17,20,22.5,26,33,37.5,42
npx -y hyperframes@0.8.81 render --quality looks --output renders/contrastive_embedding_space.mp4
ffmpeg -i renders/contrastive_embedding_space.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/contrastive_embedding_space.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/contrastive_embedding_space.mp4 -o ../../assets/visualizations/contrastive_embedding_space.gif
```
