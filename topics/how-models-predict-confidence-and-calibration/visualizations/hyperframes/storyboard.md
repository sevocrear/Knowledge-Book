---
title: "Сториборд клипа: калибровка — softmax, reliability diagram, temperature scaling"
description: "Сториборд и команды сборки HyperFrames-клипа softmax-calibration-temperature-scaling: overconfident softmax из логитов, reliability diagram с ECE и temperature scaling softmax(z/T), которое меняет уверенность, но не argmax."
tags:
  - kb/note
  - kb/visualization
  - domain/ml-foundations
  - concept/calibration
  - concept/confidence
aliases:
  - softmax-calibration-temperature-scaling
related:
  - how-models-predict-confidence-and-calibration
  - roc-curve-and-roc-auc
status: notes
lang: ru
type: note
slug: how-models-predict-confidence-and-calibration/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Калибровка: softmax → reliability diagram → temperature scaling — сториборд

Клип `assets/visualizations/softmax-calibration-temperature-scaling.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Откуда берётся “уверенность”: logits -> вероятности», «Когда вероятности “врут”: проблема калибровки»,
«Как измеряют калибровку», «Temperature Scaling: базовая калибровка»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Откуда берётся уверенность модели? | Четыре класса (кот/собака/лиса/волк), чипы логитов z = [4.2, 1.0, 0, −1.0] (иллюстративно), стрелки → горизонтальные столбики softmax 94 / 4 / 1 / 1 %; оранжевая рамка на max p; карточка «Softmax» с формулой p_k = e^{z_k} / Σ_j e^{z_j}; карточка «Overconfident»: предсказано 0.95 vs верно на деле 0.70 | классификатор выдаёт logits z_k, softmax превращает их в «вероятности» p_k · уверенность = максимальная p_k, здесь 0.94 · уверенность ≠ правильность: при p = 0.95 верны лишь 70% — overconfident |
| 2 | 13–29 с | Как измерить калибровку: reliability diagram? | Оси conf (x) / acc (y), 10 бинов; синие столбики acc(B_m) (иллюстративно: 0.05…0.70), зелёная диагональ acc = conf, красные разрывы \|acc − conf\|; карточки «Reliability diagram» и «ECE = Σ_m \|B_m\|/N · \|acc(B_m) − conf(B_m)\|»; значение ECE ≈ 0.24 (иллюстративно) | разбиваем предсказания на 10 бинов, в каждом считаем conf и acc · идеал — диагональ acc = conf; столбики ниже — overconfident · разрыв \|acc − conf\| в бине — ошибка калибровки, ECE взвешивает её долей \|B_m\|/N · ECE ≈ 0.24: чем меньше, тем лучше; как уменьшить, не переобучая модель? |
| 3 | 29–42 с | Что делает temperature scaling? | Вертикальные столбики softmax(z/T) для тех же логитов; чип T = 1 → T = 2 → T = 0.5: 94/4/1/1 % → 72/14/9/5 % → 100/0/0/0 %; метка «▲ argmax» под классом «кот» не двигается; карточки «Temperature scaling» (p_k(T) = softmax(z/T)_k; T > 1 мягче, T < 1 жёстче) и «Как подобрать T» (валидация, min NLL = −log p_y(T); argmax и порядок сохраняются) | логиты делим на T и только потом softmax; при T = 1 — обычный softmax · T > 1: распределение мягче, 94% → 72%, но argmax тот же · **ключевая идея**: один параметр T, подобранный на валидации по NLL; ранжирование не меняется, калибровка заметно улучшается (Guo et al., 2017) |

Числа из README: пример overconfident «при уверенности 0.95 правильны только 70% ответов»; 10 бинов по 0.0..1.0; формулы softmax, ECE, softmax(z/T); T > 1 → мягче, T < 1 → жёстче; T подбирают на валидации по NLL; ссылка на Guo et al., 2017.
Иллюстративные величины (помечены на экране): логиты z = [4.2, 1.0, 0, −1.0] → softmax 0.942 / 0.038 / 0.014 / 0.005 (T = 1), 0.715 / 0.144 / 0.088 / 0.053 (T = 2), 0.998 / … (T = 0.5); точности по бинам 0.05…0.70 и ECE ≈ 0.24.

## Сборка

```bash
cd topics/how-models-predict-confidence-and-calibration/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3.5,7,11.5,16,21,26,31.5,36,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/softmax-calibration-temperature-scaling.mp4
ffmpeg -i renders/softmax-calibration-temperature-scaling.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/softmax-calibration-temperature-scaling.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/softmax-calibration-temperature-scaling.mp4 -o ../../assets/visualizations/softmax-calibration-temperature-scaling.gif
```
