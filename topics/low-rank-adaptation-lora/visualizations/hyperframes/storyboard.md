---
title: "Сториборд клипа: LoRA — низкоранговая добавка ΔW = B·A"
description: "Сториборд и команды сборки HyperFrames-клипа lora-low-rank-delta-w: заморозка W и разложение ΔW = B·A, экономия параметров и памяти (4096×4096, LLaMA-7B), прямой проход с α/r, слияние на инференсе и сменные адаптеры."
tags:
  - kb/note
  - kb/visualization
  - domain/llm
  - concept/peft
  - concept/lora
aliases:
  - lora-low-rank-delta-w
  - LoRA storyboard
related:
  - low-rank-adaptation-lora
  - transformers-attention-and-vision-transformers-vit
status: notes
lang: ru
type: note
slug: low-rank-adaptation-lora/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# LoRA: низкоранговая добавка ΔW = B·A — сториборд

Клип `assets/visualizations/lora-low-rank-delta-w.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы 2 «Математическая основа LoRA», 3 «Архитектура LoRA», 4 «Параметры и эффективность», 7.4 «Мульти-таск обучение»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Почему не обучать все веса W? | Сетка W (d × k, 14×14 клеток — иллюстративно): при full FT все клетки «загораются» оранжевым (ΔW везде); затем W синеет и «замораживается» (grad = 0); справа появляется разложение: + B (d × r, узкая высокая) · A (r × k, низкая широкая), ΔW = B·A, rank r ≪ min(d, k). Карточки «Full fine-tuning: W_new = W + ΔW» и «Гипотеза LoRA: ΔW = B·A, r ∈ {1…64}, старт r = 8» | full FT обновляет всю матрицу W (d × k) — на каждую задачу своя копия модели · LoRA замораживает W (градиент = 0) и учит только добавку ΔW · гипотеза: у ΔW низкий ранг r ≪ min(d, k) ⇒ ΔW = B·A, B — d × r, A — r × k |
| 2 | 14–29 с | Сколько параметров экономит LoRA? | Столбики «один слой d = k = 4096»: full FT 16 777 216 vs LoRA r = 8 — 65 536 (×256 меньше; столбик LoRA ≈ 1/256 высоты); карточка LLaMA-7B (r = 8, α = 16, Q/K/V/O): ≈ 8.4M vs 7B → ≈ 800×; столбики «LLaMA-7B: память при обучении»: ≈ 112 GB (FP32 + Adam) vs ≈ 28.1 GB (≈ ×4 меньше) | слой 4096 × 4096: full FT учит 16 777 216 весов, LoRA с r = 8 — 65 536: в 256 раз меньше · LLaMA-7B, r = 8 на Q/K/V/O: ≈ 8.4M обучаемых против 7B — примерно в 800 раз меньше · память обучения: ≈ 112 GB → ≈ 28.1 GB — градиенты и моменты Adam нужны только для B и A |
| 3 | 29–43 с | Как LoRA работает в слое и на инференсе? | Граф прямого прохода: x → [W, заморожена] → ⊕ и x → [A r×k] → [B d×r] → [α/r] → ⊕ → h; пакеты бегут по обеим веткам; B стартует как «B = 0» и «включается»; затем ветки сливаются в один блок W′ = W + B·A («слито заранее»); сверху к базе по очереди пристыковываются адаптеры «задача 1: B₁A₁» и «задача 2: B₂A₂». Карточки «Прямой проход: h = Wx + (α/r)·BAx» и «Инференс и адаптеры: W′ = W + B·A считаем заранее, set_adapter» | forward: h = Wx + (α/r)·BAx; B стартует нулями ⇒ ΔW = 0, модель начинает с базовых весов · на инференсе W + B·A вычисляем заранее — обычный линейный слой, лишних вычислений нет · **ключевая идея**: LoRA = замороженная база + маленькие адаптеры B·A: ≪ 1% параметров, свой адаптер на задачу, база общая |

Числа (все из README темы): W ∈ ℝ^{d×k}, A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, r ∈ {1, 2, 4, 8, 16, 32, 64}, рекомендуемый старт r = 8 (§2.1, §6.3);
параметры d·k = 4096·4096 = 16 777 216 vs r(d + k) = 8·(4096 + 4096) = 65 536 — в 256 раз меньше (§2.1);
LLaMA-7B, r = 8, α = 16, Q/K/V/O: ≈ 8.4M vs 7B — ≈ 800× (§4.1); память обучения ≈ 112 GB → ≈ 28.1 GB, ≈ 4× (§4.2);
h = Wx + (α/r)·BAx, α = r или 2r (§3.3); A — случайная, B = 0 ⇒ ΔW = 0 на старте (§3.2);
на инференсе (W + BA)x можно вычислить заранее (§3.1); несколько адаптеров на одной базе, `set_adapter` (§7.4).
Размер сеток (14×14, 14×2, 2×14 клеток) — иллюстративный.

## Сборка

```bash
cd topics/low-rank-adaptation-lora/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.5,12,17,22,26.5,32,34.5,39.5
npx -y hyperframes@0.8.81 render --quality looks --output renders/lora-low-rank-delta-w.mp4
ffmpeg -i renders/lora-low-rank-delta-w.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/lora-low-rank-delta-w.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/lora-low-rank-delta-w.mp4 -o ../../assets/visualizations/lora-low-rank-delta-w.gif
```
