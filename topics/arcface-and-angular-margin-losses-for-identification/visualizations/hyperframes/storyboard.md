---
title: "Сториборд клипа: ArcFace — угловой margin на гиперсфере"
description: "Сториборд и команды сборки HyperFrames-клипа arcface_angular_margin: единичная сфера и угол θ, что делает additive angular margin m, компактные классы с зазором и сравнение SphereFace / CosFace / ArcFace."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/metric-learning
  - concept/face-recognition
aliases:
  - ArcFace storyboard
  - arcface_angular_margin
related:
  - arcface-and-angular-margin-losses-for-identification
  - contrastive-and-metric-learning-for-fine-grained-visual-recognition
status: notes
lang: ru
type: note
slug: arcface-and-angular-margin-losses-for-identification/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# ArcFace: угловой margin на гиперсфере — сториборд

Клип `assets/visualizations/arcface_angular_margin.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Математика ArcFace», «Геометрическая интуиция», «ArcFace vs CosFace vs SphereFace»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Где живут эмбеддинги в ArcFace? | Единичная сфера, эмбеддинг z и прототипы W₁, W₂; дуга θ₁; карточки «Нормировка» и «Похожесть = угол» | L2-нормировка → все векторы на сфере · z·Wⱼ = cos θⱼ — похожесть это угол · обычный softmax сравнивает cos θ₁ и cos θ₂ |
| 2 | 13–29 с | Что делает угловой margin m? | Дуга θ_y (синяя) от W_y к z, дуга +m (оранжевая) и пунктирный «худший» вектор θ_y+m; карточка логитов s·cos(θ_y+m) vs s·cos θⱼ; столбики cos θ_y = 0.906 → cos(θ_y+m) = 0.593; z доворачивается к W_y (θ_y 25° → 7°): 0.993 / 0.813; cos θ₂ = 0 → < 0 | к углу своего класса прибавляем m · cos(θ_y+m) < cos θ_y — класс намеренно занижен · z должен довернуться к W_y сильнее · классы получают запас ≥ m |
| 3 | 29–42 с | Как выглядят эмбеддинги после ArcFace? | 3 класса по 9 точек на сфере: разброс ±49° → ±9°, между кластерами красные дуги «зазор ≥ m»; таблица SphereFace cos(m·θ) / CosFace cos θ − m / ArcFace cos(θ + m) | до: классы перемешаны · после: внутри компактно, между — запас ≥ m · **ключевая идея**: softmax на сфере + угловой зазор m; де-факто стандарт face recognition, база для re-ID и SKU |

Числа: m = 0.5 рад ≈ 28.6°, s = 64 — значения из статьи ArcFace (Deng et al., 2019); cos 25° = 0.906, cos 53.65° = 0.593, cos 7° = 0.993, cos 35.65° = 0.813.

## Сборка

```bash
cd topics/arcface-and-angular-margin-losses-for-identification/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7,11,17,20,26,32,37,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/arcface_angular_margin.mp4
ffmpeg -i renders/arcface_angular_margin.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/arcface_angular_margin.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/arcface_angular_margin.mp4 -o ../../assets/visualizations/arcface_angular_margin.gif
```
