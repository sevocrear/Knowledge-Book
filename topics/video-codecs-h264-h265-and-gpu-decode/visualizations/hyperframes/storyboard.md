---
title: "Сториборд клипа: видеокодеки — intra, inter и GOP из I/P/B"
description: "Сториборд и команды сборки HyperFrames-клипа gop-i-p-b-prediction: блок → DCT → квантование, вектор движения и остаток, GOP с I/P/B-кадрами, ссылками между ними и бюджетом бит."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/compression
aliases:
  - codec storyboard
  - gop-i-p-b-prediction
related:
  - video-codecs-h264-h265-and-gpu-decode
  - ml-system-design-for-cv-and-nlp
status: notes
lang: ru
type: note
slug: video-codecs-h264-h265-and-gpu-decode/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Видеокодеки: intra, inter и GOP (I/P/B) — сториборд

Клип `assets/visualizations/gop-i-p-b-prediction.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (раздел «В чём суть сжатия»: 1. Intra, 2. Inter, 3. GOP; раздел «Как устроен H.264 (AVC)» — размеры блоков 4×4 / 8×8, CAVLC/CABAC).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как сжимается один кадр (intra)? | Три сетки 8×8: блок X (градиент серого) → Y = C·X·Cᵀ (яркость ∝ \|коэффициент\|, светло только слева сверху) → ŷ = round(y/Q) с числами; ползунок Q: 8 → 32, ненулевых 13 → 6; полоска «биты» сжимается под заголовком «CAVLC / CABAC: нули дёшевы». Карточки «DCT», «Квантование», «Качество» (PSNR) | блоки 4×4 / 8×8 и DCT: энергия в низких частотах · round(y/Q) обнуляет высокие частоты, больше Q → больше нулей и ниже PSNR · нули дёшевы энтропийно (CAVLC/CABAC) — так устроен I-кадр и JPEG |
| 2 | 14–28 с | Как сжимается движение (inter)? | Кадры t−1 и t (20×14 блоков, оранжевый объект сдвинут на (dy, dx) = (−2, +3)); блок X подсвечен, рамка перебирает кандидатов в t−1, стрелка находит совпадение, рисуется вектор движения; ряд панелей 4×4: X − X̃ = R (R почти из нулей); столбики энергии: сырая разница кадров vs остаток R. Карточки «Motion compensation», «Residual» | кодек ищет вектор движения (dy, dx) для каждого блока · из декодированного прошлого строим X̃ и кодируем остаток R = X − X̃ · энергия остатка на порядок меньше — после DCT и квантования блок почти пустой |
| 3 | 28–42 с | Что такое GOP: I, P, B? | Ряд I B B P B B P B B I; P ссылается на предыдущий I/P (стрелки снизу), B — на предыдущий и следующий якорь (стрелки сверху); столбики бюджета бит (I ≫ P > B, иллюстративно); ряд «камера: I P P P P … — IPPP без B, GOP 1–2 с, или intra-refresh». Карточки «I (IDR)», «P», «B» | GOP — пачка кадров между якорями, I (IDR) хранит картинку и служит точкой входа · P из прошлого, B из прошлого и будущего: сжатие лучше, задержка выше · **ключевая идея**: I хранит картинку, P и B — только отличие; на камерах IPPP без B и короткий GOP либо intra-refresh |

Числа: блок X, его DCT Y и ŷ при Q = 8 / Q = 32 посчитаны заранее (ортонормированное DCT-II, `round`) и вшиты в сцену константами: ненулевых коэффициентов 13 → 6 из 64. Кадры сцены 2 и остаток R = X − X̃ тоже посчитаны заранее (объект 6×6 с градиентом, сдвиг на (−2, +3) блока): у R шесть значений от −2 до 5 на краях, остальные нули; энергия сырой разницы кадров на порядки больше. Высоты столбиков бюджета бит в сцене 3 — иллюстративные (I ≫ P > B), без конкретных чисел.

## Сборка

```bash
cd topics/video-codecs-h264-h265-and-gpu-decode/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,9,13.3,15,19,27.3,29,33,41.3
npx -y hyperframes@0.8.81 render --quality looks --output renders/gop-i-p-b-prediction.mp4
ffmpeg -i renders/gop-i-p-b-prediction.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/gop-i-p-b-prediction.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/gop-i-p-b-prediction.mp4 -o ../../assets/visualizations/gop-i-p-b-prediction.gif
```
