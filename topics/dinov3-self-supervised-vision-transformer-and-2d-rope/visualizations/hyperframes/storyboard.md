---
title: "Сториборд клипа: DINOv3 — student–teacher без разметки и 2D RoPE"
description: "Сториборд и команды сборки HyperFrames-клипа dino-student-teacher-2d-rope: self-supervised обучение student–teacher на глобальных и локальных кропах с EMA-teacher’ом и DINO-loss, 2D RoPE — вращение половин каналов Q и K на углы, пропорциональные координатам патча, и что дают глобальные и dense-фичи для классификации, детекции и сегментации."
tags:
  - kb/note
  - kb/visualization
  - domain/cv
  - concept/self-supervised
  - concept/rope
aliases:
  - dino-student-teacher-2d-rope
related:
  - dinov3-self-supervised-vision-transformer-and-2d-rope
  - transformers-attention-and-vision-transformers-vit
status: notes
lang: ru
type: note
slug: dinov3-self-supervised-vision-transformer-and-2d-rope/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# DINOv3: student–teacher без разметки и 2D RoPE — сториборд

Клип `assets/visualizations/dino-student-teacher-2d-rope.{mp4,gif}`, 44 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 2–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы 3 «Архитектура DINOv3 как ViT-бэкбон», 4 «Self-Supervised обучение», 5 «2D RoPE», 6 «Как формируются фичи», 7 «Классификация», 8 «Детекция и сегментация»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–16 с | Как DINOv3 учится без разметки? | Собственная «картинка» (небо, солнце, дом, холм) без меток; на ней рисуются 2 глобальных кропа (оранжевые) и 3 локальных (синие); боксы teacher (видит глобальные, без градиента) и student (все кропы, градиент); стрелка EMA student → teacher; карточки «Teacher = EMA student’а» θ_teacher ← τ·θ_teacher + (1 − τ)·θ_student и «Глобальный DINO-loss» L_DINO = CE(p_t, p_s), p_t = softmax((t − c)/T_t), p_s = softmax(s/T_s); столбики распределений p_t и p_s (иллюстративно): student подстраивается под teacher; сноска «в DINOv2/v3 вместо центра c — Sinkhorn-Knopp; плюс patch-loss в духе iBOT» | из одного изображения — 2 глобальных кропа и несколько локальных · student видит все кропы, teacher — глобальные; teacher = EMA весов student’а · выходы teacher’а центрируем (c — EMA его выходов), softmax с температурой, student учится через cross-entropy · итог: согласованные представления всех видов; patch-loss в духе iBOT для dense-фич |
| 2 | 16–31 с | Как 2D RoPE кодирует координаты патча? | Сетка патчей h × w (здесь 6 × 6), патч A = (3, 2); вектор q ∈ ℝ^D, разделённый на «каналы x» и «каналы y»; два круга: стрелка в круге x поворачивается на θˣᵢ = αᵢ·x, в круге y — на θʸⱼ = βⱼ·y; патч B = (1, 4) даёт другие фазы, красная дуга — разность фаз ∝ (Δx, Δy); карточки «1D RoPE: поворот пары компонент» (q′₂ᵢ, q′₂ᵢ₊₁) = R(θᵢ(p))·(q₂ᵢ, q₂ᵢ₊₁), «2D RoPE: две оси — два поворота» Q′ₓ,ᵧ = RoPEₓ(Q⁽ˣ⁾, x) ⊕ RoPEᵧ(Q⁽ʸ⁾, y) и «Зачем это DINOv3» | позицию не прибавляем к X, а зашиваем во вращение пар компонент Q и K · каналы делим пополам: первая половина вращается на θˣᵢ = αᵢ·x, вторая — на θʸⱼ = βⱼ·y · патч B получает другие фазы; в Q·K остаётся разность фаз — attention видит (Δx, Δy) · нет таблицы под фиксированное h × w: координаты масштабируют |
| 3 | 31–44 с | Что дают фичи DINOv3 на выходе? | Пайплайн: изображение H × W × 3 → DINOv3-ViT (патчи P × P, L блоков MHA + MLP, 2D RoPE) → патч-токены Z⁽ᴸ⁾ ∈ ℝ^{N×D}; ветка 1: CLS или mean → z_global → linear probe → класс; ветка 2: reshape → карта h × w × D (4 × 4, клетки раскрашены по «классам» региона, иллюстративно) → детектор / декодер (DETR, Mask2Former, U-Net) → боксы · маски; карточки с формулами z_global = (1/N)·Σᵢ zᵢ⁽ᴸ⁾, ŷ = softmax(W_cls·z_global + b), mask = σ(f(query)·feature_map) | глобальный вектор: CLS или среднее по патчам → линейная голова · патч-токены reshape’ятся в карту h × w × D — бэкбон для детекции и сегментации · **ключевая идея**: DINOv3 = ViT, обученный без меток (student–teacher + patch-loss) с 2D RoPE; один бэкбон даёт и глобальные, и dense-фичи |

Числа и обозначения: 2 глобальных кропа + несколько локальных, EMA teacher’а с τ ∈ [0, 1), центр c — EMA выходов teacher’а, температуры T_t, T_s — из README, раздел 4; θˣᵢ(x) = αᵢ·x, θʸⱼ(y) = βⱼ·y, конкатенация ⊕ — раздел 5; z_global = (1/N)·Σ zᵢ, reshape в h × w × D — разделы 3 и 6; ŷ = softmax(W_cls·z_global + b), linear probe — раздел 7; DETR / Mask2Former / U-Net-декодер, mask = σ(f(query)·feature_map) — раздел 8.
Иллюстративные величины (помечены на экране): сетка 6 × 6, углы αᵢ = 24° и βⱼ = 30° на патч, высоты столбиков p_t / p_s, раскраска карты 4 × 4.

## Сборка

```bash
cd topics/dinov3-self-supervised-vision-transformer-and-2d-rope/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7,11,14,19,23,27,30,33,38,42
npx -y hyperframes@0.8.81 render --quality looks --output renders/dino-student-teacher-2d-rope.mp4
ffmpeg -i renders/dino-student-teacher-2d-rope.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/dino-student-teacher-2d-rope.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/dino-student-teacher-2d-rope.mp4 -o ../../assets/visualizations/dino-student-teacher-2d-rope.gif
```
