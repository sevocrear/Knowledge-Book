---
title: "Сториборд клипа: метрики MOT — MOTA, IDF1, HOTA"
description: "Сториборд и команды сборки HyperFrames-клипа tracking-metrics-mota-idf1-hota: покадровое сопоставление детекций с GT-треками по IoU (TP / FN / FP / IDSW), чем MOTA отличается от IDF1 и как HOTA объединяет DetA и AssA с усреднением по порогам α."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/metrics
  - concept/tracking
aliases:
  - tracking-metrics-mota-idf1-hota
  - MOT metrics storyboard
related:
  - action-recognition-and-object-tracking-metrics
  - unscented-kalman-filter-and-tracking
status: notes
lang: ru
type: note
slug: action-recognition-and-object-tracking-metrics/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Метрики MOT: MOTA, IDF1, HOTA — сториборд

Клип `assets/visualizations/tracking-metrics-mota-idf1-hota.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (раздел «Метрики для Object Tracking» → «2) Multiple Object Tracking (MOT)»: CLEAR MOT, Identity-aware метрики, HOTA; «3) Вспомогательные MOT-метрики»; «Как выбирать метрики под задачу», п. 4).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как сопоставляют детекции с GT-треками? | Сетка «кадры t = 1…6 × GT-треки A, B»: пунктирные GT-боксы, поверх — залитые предсказания (цвет = ID трека). Кадры 1–3: ID 1 на A, ID 2 на B → TP. Кадры 4–6: B в кадре 4 без пары → «FN»; лишний бокс ID 4 в кадре 5 в строке «без GT» → «FP»; A с кадра 4 идёт под ID 3 → скобка «IDSW: тот же GT A, но ID 1 → ID 3». Карточки «Сопоставление в каждом кадре» (IoU(pred, GT) ≥ α → TP) и «Итог по 6 кадрам (иллюстративно)»: TP = 11, FN = 1, FP = 1, IDSW = 1, GT = 12 | в каждом кадре боксы сопоставляют с GT по IoU ≥ α: пара найдена — TP · GT без пары — FN, предсказание без пары — FP · тот же GT-трек, но новый ID с кадра 4 — IDSW |
| 2 | 14–29 с | Что считает MOTA, а что — IDF1? | Та же последовательность в компактном виде; красные кольца на трёх ошибках (IDSW + FN + FP) → «MOTA = 1 − 3/12 = 0.75», столбик MOTA 0.75. Затем оранжевая рамка на A, кадры 4–6: «3 кадра из 6 под чужим ID → IDP = IDR = 8/12 ≈ 0.67», столбик IDF1 0.67 рядом; пометка «MOTA скрывает потерю identity, IDF1 её показывает». Карточки «CLEAR MOT · MOTA» (формула) и «Identity-aware · IDF1» (гармоническое среднее IDP и IDR) | MOTA складывает все ошибки в одну сумму FN + FP + IDSW / GT · IDSW стоит MOTA одну ошибку, хотя половина трека A идёт под чужим ID · IDF1 сопоставляет identity на всей длине: 0.67 < 0.75 |
| 3 | 29–43 с | Почему HOTA — современный стандарт? | Столбики DetA = 11/(11+1+1) ≈ 0.85 и AssA ≈ 0.65 (по TP: A 3/6, B 5/6) → стрелка «√» → HOTA_α ≈ 0.74; ось α = 0.05 … 0.95 с 19 точками «HOTA = среднее HOTA_α по 19 порогам»; примечание про LocA. Карточки «HOTA при пороге overlap» (HOTA_α = √(DetA_α · AssA_α), DetA_α, AssA_α через TPA / FNA / FPA) и «Итог по порогам» (𝒜 = {0.05, 0.10, …, 0.95}) | HOTA раскладывает качество на детекцию DetA и ассоциацию AssA · HOTA_α = √(DetA_α · AssA_α) — геометрическое среднее, затем среднее по α · **ключевая идея**: MOT оценивают набором — минимум HOTA + IDF1 + MOTA, отдельно IDs / Frag / FN / FP |

Числа: игрушечная последовательность **иллюстративная** (в README нет числового примера): 6 кадров, 2 GT-объекта → GT = 12 боксов; TP = 11, FN = 1, FP = 1, IDSW = 1. Отсюда по формулам README: MOTA = 1 − (1 + 1 + 1)/12 = 0.75; IDP = IDR = 8/12 ≈ 0.67 → IDF1 ≈ 0.67 (трек A засчитан под одним ID только в 3 кадрах из 6); DetA = 11/13 ≈ 0.85, AssA = (6·3/6 + 5·5/6)/11 ≈ 0.65, HOTA_α = √(0.85·0.65) ≈ 0.74 при одном пороге α. Из README без изменений: формулы MOTA, IDF1 (гармоническое среднее IDP/IDR), HOTA_α, DetA_α, AssA_α, набор порогов 𝒜 = {0.05, 0.10, …, 0.95}, LocA репортится отдельно, «на MOTChallenge смотрят вместе HOTA, IDF1, MOTA», «MOT: минимум HOTA + IDF1 + MOTA, и отдельно IDs/Frag/FN/FP».

## Сборка

```bash
cd topics/action-recognition-and-object-tracking-metrics/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 2,8,12.5,16,23,27.5,31,35.5,42
npx -y hyperframes@0.8.81 render --quality looks --output renders/tracking-metrics-mota-idf1-hota.mp4
ffmpeg -i renders/tracking-metrics-mota-idf1-hota.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/tracking-metrics-mota-idf1-hota.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/tracking-metrics-mota-idf1-hota.mp4 -o ../../assets/visualizations/tracking-metrics-mota-idf1-hota.gif
```
