---
title: "Сториборд клипа: triplet loss и майнеры"
description: "Сториборд и команды сборки HyperFrames-клипа metric-learning-losses-and-miners: якорь/positive/negative и triplet loss с margin, зоны easy / semi-hard / hard негативов, конвейер сэмплер P×K → майнер → лосс."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/metric-learning
  - concept/loss
aliases:
  - metric-learning-losses-and-miners
  - triplet loss storyboard
related:
  - metric-learning-losses-and-miners
  - arcface-and-angular-margin-losses-for-identification
status: notes
lang: ru
type: note
slug: metric-learning-losses-and-miners/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Triplet loss и майнеры — сториборд

Клип `assets/visualizations/metric-learning-losses-and-miners.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Краткий абстракт», «Общая рамка: эмбеддинг, метрика, „лосс + сэмплер + майнер“»,
«Triplet loss (hinge и soft-margin)», «Майнеры: зачем они нужны и какие бывают», «Каталог майнеров», «Сэмплер батча P×K»,
«Таблица „лосс ↔ майнер“», «Типовые гиперпараметры»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Что делает triplet loss? | Плоскость эмбеддингов: якорь a (синий), positive p (зелёный), negative n (красный); отрезки D<sub>ap</sub>, D<sub>an</sub>; пунктирная граница радиуса D<sub>ap</sub> + m; карточки «L = max(0, D<sub>ap</sub> − D<sub>an</sub> + m)» и «D<sub>an</sub> ≥ D<sub>ap</sub> + m ⇒ L = 0»; шаг градиента: p стягивается к a, n уходит за границу, статус «L > 0» → «L = 0» | якорь, positive, negative — лосс сравнивает D<sub>ap</sub> и D<sub>an</sub> · negative должен быть дальше positive хотя бы на m · градиент тянет p и толкает n за D<sub>ap</sub> + m, потом лосс = 0 |
| 2 | 13–29 с | Какие тройки полезны для обучения? | Якорь a и positive p, кольца D<sub>ap</sub> (зелёное) и D<sub>ap</sub> + m (оранжевое); негативы по зонам: easy снаружи (серые), semi-hard в кольце (оранжевые), hard внутри (красные); таблица «тройки по трудности при margin m» с условием и градиентом; карточка «без майнера: > 95 % троек — easy» | easy: D<sub>an</sub> > D<sub>ap</sub> + m — градиента нет · semi-hard: D<sub>ap</sub> < D<sub>an</sub> < D<sub>ap</sub> + m — «золотая середина» FaceNet · hard: D<sub>an</sub> < D<sub>ap</sub> — часто шум разметки → коллапс · случайные тройки: > 95 % easy, обучение «замерзает» — нужен майнер |
| 3 | 29–42 с | Что делает майнер перед лоссом? | Конвейер «1 · Сэмплер P×K → 2 · Майнер → 3 · Лосс»; батч 4×4 (строка = класс); ось расстояний от якоря a: 3 positives и 12 negatives, batch-hard выбирает самый далёкий positive и самый близкий negative (D<sub>an</sub> − D<sub>ap</sub> = 0.07 < m); тройка (a, p, n) уезжает в блок «Лосс»; карточки «Майнеры (pytorch-metric-learning)» и «Кому майнер не нужен» | сэмплер P×K — иначе в батче почти нет positives · майнер batch-hard: самый далёкий positive + самый близкий negative · **ключевая идея**: лосс не выбирают отдельно от сэмплера и майнера; triplet без semi-hard mining почти не учится, ArcFace и Proxy-Anchor майнер не нужен |

Числа и формулы из README: triplet loss L = max(0, D<sub>ap</sub> − D<sub>an</sub> + m) (FaceNet, 2015); типичный margin m = 0.2 (FaceNet), 0.3 (re-ID);
классы троек easy / semi-hard / hard и их градиент; «после первой эпохи > 95 % троек — easy», троек O(N³); batch-hard (Hermans 2017) — самый далёкий positive
и самый близкий negative, P×K с K ≥ 4; TripletMarginMiner(semihard): 0 < D<sub>an</sub> − D<sub>ap</sub> < m; DistanceWeightedMiner: негативы с весом ∝ 1/q(D);
майнер не нужен ArcFace/CosFace, Proxy-Anchor, SupCon, InfoNCE. Координаты точек, m в пикселях, батч 4×4 и значения расстояний на оси — иллюстративные (помечены на экране).

## Сборка

```bash
cd topics/metric-learning-losses-and-miners/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.5,11.8,16,20,24,28,32,36.5,40
npx -y hyperframes@0.8.81 render --quality looks --output renders/metric-learning-losses-and-miners.mp4
ffmpeg -i renders/metric-learning-losses-and-miners.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/metric-learning-losses-and-miners.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/metric-learning-losses-and-miners.mp4 -o ../../assets/visualizations/metric-learning-losses-and-miners.gif
```
