---
title: "Сториборд клипа: VLA — от кадра и инструкции к действию робота"
description: "Сториборд и команды сборки HyperFrames-клипа vla-vision-language-action-pipeline: входные токены кадра и инструкции, два типа action head (дискретные токены и chunk непрерывных действий), замкнутый цикл и данные для обучения."
tags:
  - kb/note
  - kb/visualization
  - domain/robotics
  - domain/multimodal
  - concept/vla
aliases:
  - vla-vision-language-action-pipeline
  - VLA storyboard
related:
  - vision-language-action-models-vla
  - vision-based-robot-training-methods
status: notes
lang: ru
type: note
slug: vision-language-action-models-vla/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# VLA: от кадра и инструкции к действию робота — сториборд

Клип `assets/visualizations/vla-vision-language-action-pipeline.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы 2 «Архитектура VLA», 4 «Обучение», 5 «Современные VLA модели», 6.1 «Манипуляция», 8.2–8.4 «Реализация»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Что получает VLA на вход? | Кадр камеры с сеткой патчей → Vision Encoder (ViT; у OpenVLA SigLIP + DINOv2) → 6 визуальных токенов; инструкция «подними красный кубик» → Tokenizer → 4 токена текста; обе цепочки сходятся в блок VLM (OpenVLA: MLP-проектор → LLaMA 2 7B) | кадр → Vision Encoder → визуальные токены · инструкция → токены текста, как в LLM · оба набора идут одним потоком в предобученную VLM |
| 2 | 13–29 с | Как токены становятся действиями? | Вариант A: 7 ячеек dx, dy, dz, droll, dpitch, dyaw, gripper с бинами (иллюстративно) появляются по одной → карточка a = Detokenize(VLM(I, T)). Вариант B: 12 столбиков шум → плавный chunk (иллюстративно) → карточка «chunk a₁ … aₙ», у Octo [batch, action_horizon, action_dim] | вариант A: дискретные токены, 256 бинов на измерение, авторегрессивно · вариант B: diffusion / flow matching превращает шум в chunk непрерывных действий · итог — команда роботу: смещение, поворот, захват |
| 3 | 29–42 с | Как замыкается цикл и откуда данные? | Цикл из 4 узлов: наблюдение → VLA-модель → действие a → робот выполняет → новый кадр (точка обходит стрелки); справа карточка «Чему учится VLA»: демонстрации, Open X-Embodiment 1M+ / 22 робота / ~970k у OpenVLA, cross-entropy по токенам действия, fine-tuning / LoRA | цикл: кадр + инструкция → VLA → действие → робот → новый кадр · учится на демонстрациях, новые задачи — fine-tuning, сотни вместо миллионов · **ключевая идея**: VLA = кадр + инструкция → действие робота в одной модели, по кругу; учится на демонстрациях |

Числа и названия (из README): 256 бинов на измерение (RT-1, RT-2, OpenVLA); 7 действий для Bridge: dx, dy, dz, droll, dpitch, dyaw, gripper (8.2); SigLIP + DINOv2, MLP-проектор, LLaMA 2 7B (5.3); a = Detokenize(VLM(I, T)) (5.2); Octo — diffusion head, chunk действий, `[batch, action_horizon, action_dim]` (5.5, 8.4); π0, SmolVLA — VLM-бэкбон + flow-matching action expert (2.3, 5.6); Open X-Embodiment — 1M+ траекторий, 22 робота, OpenVLA — ~970k (4.1, 5.3); loss OpenVLA — next-token cross-entropy по токенам действия (8.3); fine-tuning на сотнях демонстраций, LoRA (4.3); цикл «получить кадр → предсказать → выполнить, пока задача не выполнена» (6.1).
Иллюстративно: значения бинов 131, 127, 118, 128, 129, 141, 255; высоты столбиков chunk; число визуальных (6) и текстовых (4) токенов; форма кадра.

## Сборка

```bash
cd topics/vision-language-action-models-vla/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7,11.5,17,21,24.5,31,34,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/vla-vision-language-action-pipeline.mp4
ffmpeg -i renders/vla-vision-language-action-pipeline.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/vla-vision-language-action-pipeline.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/vla-vision-language-action-pipeline.mp4 -o ../../assets/visualizations/vla-vision-language-action-pipeline.gif
```
