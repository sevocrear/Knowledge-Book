---
title: "Сториборд клипа: свёртка 3×3 — размер карты, receptive field и параметры"
description: "Сториборд и команды сборки HyperFrames-клипа convolution-3x3-receptive-field-params: окно 3×3 скользит по входу и формула размера feature map, две 3×3 вместо одной 5×5 (receptive field и 18 против 25 весов), подсчёт параметров Conv2d / depthwise / Linear."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/cnn
  - concept/convolution
aliases:
  - convolution-3x3-receptive-field-params
related:
  - convolutions-and-parameters-in-cnn
  - normalization-layers-batchnorm-layernorm
status: notes
lang: ru
type: note
slug: convolutions-and-parameters-in-cnn/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Свёртка 3×3: размер карты, receptive field и параметры — сториборд

Клип `assets/visualizations/convolution-3x3-receptive-field-params.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы 1 «Введение», 2.1 «Баланс качество / цена у 3×3», 3 «Почему большие свёртки
неэффективны», 4 «Формула размера feature map», 6 «Как считать число обучаемых параметров», 7 «Как бы я объяснил…»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как окно 3×3 строит карту признаков? | Вход 6×6 (иллюстративно), оранжевое окно 3×3 проходит 16 позиций (S = 1) и зажигает клетки выхода 4×4; карточка формулы H_out = ⌊(H_in + 2P − K)/S + 1⌋; затем кайма padding P = 1 — выход становится 6×6; окно прыгает через клетку (S = 2); карточка с примерами 4 / 6 / 3 | окно 3×3 скользит по входу: одна позиция → одна клетка выхода · формула размера; 6×6, K = 3, P = 0, S = 1 → 4×4 · padding = 1 сохраняет размер, stride = 2 уменьшает примерно вдвое |
| 2 | 14–28 с | Зачем две 3×3 вместо одной 5×5? | Ряд A: одна 5×5 покрывает вход 5×5 → один выход, «25 весов»; столбики весов ядра 3×3 / 5×5 / 7×7 = 9 / 25 / 49; ряд B: окно 3×3 проходит 9 позиций → промежуточная 3×3 → вторая 3×3 → один выход; зелёная рамка receptive field 5×5, «9 + 9 = 18 весов»; чип «ReLU · BN» между слоями; карточка 2 × (3×3) = 18 < 25, 3 × (3×3) = 27 < 49 | стоимость ядра растёт как K²: 9, 25, 49; одна 5×5 смотрит на область 5×5 · две 3×3 подряд: receptive field тот же 5×5, но весов 9 + 9 = 18 вместо 25 · плюс ReLU и BatchNorm между слоями — так строят VGG, ResNet, Inception v3 |
| 3 | 28–42 с | Сколько параметров у Conv2d? | Вход 4 плоскости → один фильтр 3×3×4 («4·3·3 = 36 весов», «+ 1 bias = 37») → пять фильтров («5·37 = 185») → выход 5 плоскостей; счётчик 0 → 37 → 185 параметров; карточка params = out·(in·K_h·K_w + 1); карточка «Другие слои»: depthwise in·K_h·K_w, pointwise in·out, Linear in·out + out (128·64 + 64), BatchNorm 2·C | Conv2d(in = 4, out = 5, K = 3): фильтр 4·3·3 = 36 весов, + bias → 37 · фильтров 5 → 5·37 = 185; H×W входа в формуле нет · **ключевая идея**: параметры = чисел в одном фильтре × число фильтров; depthwise + pointwise (MobileNet) режут это произведение |

Числа из README: 3×3 → 9, 5×5 → 25, 7×7 → 49 весов (§2.1); две 3×3 ≈ receptive field 5×5, три 3×3 ≈ 7×7 (§2.1);
одна 5×5 = 25 против двух 3×3 = 9 + 9 = 18 (§3); формула H_out = ⌊(H_in + 2P − D·(K − 1) − 1)/S + 1⌋, при D = 1
⌊(H_in + 2P − K)/S + 1⌋; 3×3, P = 1, S = 1 сохраняет размер, S = 2 — примерно вдвое (§4); Conv2d(4, 5, 3) с bias:
4·3·3 = 36, +1 = 37, 5·37 = 185 (§6.2); depthwise in·K_h·K_w (+bias), pointwise in·out (+bias) (§6.3);
Linear 128·64 + 64 (§6.1); BatchNorm 2·C (§6.4). Вход 6×6 в сцене 1 и вход 5×5 в сцене 2 — иллюстративные размеры;
27 = 9 + 9 + 9 для трёх 3×3 — арифметика по числам README.

## Сборка

```bash
cd topics/convolutions-and-parameters-in-cnn/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 1,3,7.5,12.8,17,22.5,26.5,31.5,36,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/convolution-3x3-receptive-field-params.mp4
ffmpeg -i renders/convolution-3x3-receptive-field-params.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/convolution-3x3-receptive-field-params.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/convolution-3x3-receptive-field-params.mp4 -o ../../assets/visualizations/convolution-3x3-receptive-field-params.gif
```
