---
title: "Сториборд клипа: обучение робота по камере — IL, RL и VLA"
description: "Сториборд и команды сборки HyperFrames-клипа robot-learning-il-rl-vla: петля камера-политика-действие и imitation learning, RL в симуляторе с sim-to-real и domain randomization, VLA на данных многих роботов."
tags:
  - kb/note
  - kb/visualization
  - domain/robotics
  - concept/imitation-learning
  - concept/sim-to-real
aliases:
  - robot-learning-il-rl-vla
  - robot training storyboard
related:
  - vision-based-robot-training-methods
  - vision-language-action-models-vla
status: notes
lang: ru
type: note
slug: vision-based-robot-training-methods/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Обучение робота по камере: IL, RL, VLA — сториборд

Клип `assets/visualizations/robot-learning-il-rl-vla.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы 1, 2.1, 2.2, 2.4, 3.1, 5.1, 5.4, 6.3, 7).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Как робот учится по демонстрациям? | Петля «Камера → Политика π (ResNet50 → 7-DoF) → Действие → Робот и сцена → камера», точка обходит петлю; карточки «демонстрации: пары (кадр, действие оператора)» и «loss = MSE(π(image), a_expert)»; график distribution shift: путь демонстраций (зелёный) и уходящий путь политики (красный), зазор «ошибки копятся» (иллюстративно) | end-to-end обучение от изображений к действиям · imitation learning: копируем действия эксперта (behavioral cloning) · distribution shift: ошибки накапливаются |
| 2 | 14–29 с | RL в симуляторе: что с реальностью? | Политика ⇄ Симулятор (Isaac Sim · MuJoCo · PyBullet), действие и «кадр + награда r», столбики награды (иллюстративно); карточки RL (плюсы/минусы), Sim-to-Real gap (физика, картинка, сенсоры, железо), Domain randomization (диапазоны параметров); «Реальный робот», красный зазор gap; четыре варианта симулятора (яркость 0.6…1.4, иллюстративно); зелёная стрелка «перенос + fine-tune» | RL: пробует действия, получает награду · Sim-to-Real gap: физика, картинка, сенсоры не совпадают · Domain randomization + дообучение на реальных данных |
| 3 | 29–43 с | Одна модель для многих роботов? | Четыре типа роботов → Open X-Embodiment (1M+ траекторий, 60 датасетов, 22 типа роботов, 21 институт) → VLA-модель (OpenVLA 7B: SigLIP + DinoV2, LLaMA 2 7B) с входами «изображение» и «pick up the red block» → action-токены (256 бинов); строка гибридов «pre-train на демонстрациях → fine-tune через RL»; 970k демонстраций, +16.5% к RT-2-X (55B) | данные многих роботов вместе · VLA: картинка + инструкция → action-токены, OpenVLA обучен на 970k демонстраций · **ключевая идея**: IL, RL и VLA на данных многих роботов |

Числа из README: 7-DoF, ResNet50 (пример behavioral cloning, п. 2.1); MSE-лосс (п. 2.1); PPO, Isaac Sim / MuJoCo / PyBullet (пп. 2.2, 7.3); диапазоны domain randomization — свет 0.5–1.5, трение 0.3–1.5, масса 0.5–2.0, шум камеры 0.0–0.1, текстуры smooth / rough / textured, 100 эпизодов реальных данных (пп. 6.3, 7.2); Open X-Embodiment: 1M+ траекторий, 60 датасетов, 22 типа роботов, 21 институт (п. 5.1); OpenVLA: 7B, 970k демонстраций, SigLIP + DinoV2, LLaMA 2 7B, 256 бинов на измерение действия, +16.5% к RT-2-X (55B) (п. 3.1). Столбики награды, траектории distribution shift и значения яркости 0.6–1.4 — иллюстративные.

## Сборка

```bash
cd topics/vision-based-robot-training-methods/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.5,11.5,13.3,17,22,24,27.5,32,36,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/robot-learning-il-rl-vla.mp4
ffmpeg -i renders/robot-learning-il-rl-vla.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/robot-learning-il-rl-vla.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/robot-learning-il-rl-vla.mp4 -o ../../assets/visualizations/robot-learning-il-rl-vla.gif
```
