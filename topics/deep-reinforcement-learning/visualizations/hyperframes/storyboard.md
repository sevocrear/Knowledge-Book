---
title: "Сториборд клипа: Deep RL — цикл агент–среда, DQN vs policy gradient, PPO clip"
description: "Сториборд и команды сборки HyperFrames-клипа rl-agent-environment-loop-ppo: цикл агент–среда и MDP с дисконтированным возвратом, value-based (DQN: target-сеть, replay buffer) против policy gradient (∇ log π · Â, actor-critic) и обрезанная целевая функция PPO с ε-трубкой [1−ε, 1+ε]."
tags:
  - kb/note
  - kb/visualization
  - domain/rl
  - concept/rl
  - concept/ppo
aliases:
  - Deep RL storyboard
  - rl-agent-environment-loop-ppo
related:
  - deep-reinforcement-learning
  - vision-language-action-models-vla
status: notes
lang: ru
type: note
slug: deep-reinforcement-learning/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Deep RL: цикл агент–среда → DQN vs policy gradient → PPO clip — сториборд

Клип `assets/visualizations/rl-agent-environment-loop-ppo.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Основы Reinforcement Learning → MDP, Основные компоненты», «Deep Reinforcement Learning → Основные подходы», «Value-Based Методы → DQN», «Policy Gradient Методы», «Actor-Critic Методы → A2C», «Современные Методы → PPO», «Сравнение Методов»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Как агент учится у среды? | Блоки «Агент (политика π(a\|s))» и «Среда (P(s′\|s,a), R(s,a,s′))», замкнутый цикл: оранжевый пакет a_t едет к среде, зелёный (r_{t+1}, s_{t+1}) возвращается к агенту, счётчик «шаг t» 0 → 3; карточки «Формализация: MDP (S, A, P, R, γ)» и «Цель агента: G_t = Σ γ^{k−t} r_{k+1}, π* = argmax E[Σ γ^t R_{t+1}]»; столбики r₁, γr₂, γ²r₃, γ³r₄, γ⁴r₅ с γ = 0.9 (иллюстративно) | агент видит s_t, выбирает a_t по π(a\|s), среда отвечает r_{t+1} и s_{t+1} · формально это MDP (S, A, P, R, γ): P и R — свойство среды, π — то, что учим · цель — максимум ожидаемого возврата G_t, дисконт γ ∈ [0,1] делает далёкие награды дешевле |
| 2 | 13–28 с | Учить ценность или сразу политику? | Две колонки. Слева DQN: s → Q-сеть Q(s,a;θ) → столбики Q(s,a₁..a₃) (иллюстративно), argmax → a₂; replay buffer D с чипами (s,a,r,s′), из него «случайный батч» → Q-сеть; пунктирная target-сеть Q(s′,a′;θ⁻); карточка «y = r + γ·max_{a′} Q(s′,a′;θ⁻), L(θ) = E_D[(y − Q(s,a;θ))²]». Справа policy gradient: s → policy-сеть π_θ(a\|s) → столбики вероятностей (Σ = 1, иллюстративно), выбрано a₂ с Â_t > 0 → π(a₂\|s) растёт; карточка «∇_θJ = E_τ[Σ ∇ log π_θ(a_t\|s_t)·Â_t], θ ← θ + α ∇ log π · A^π»; плашка Actor-critic: A^π(s,a) = Q^π(s,a) − V^π(s) | value-based (DQN): сеть оценивает Q(s,a;θ), агент берёт argmax, переходы копятся в replay buffer · цель r + γ·max Q(s′,a′;θ⁻) считает целевая сеть θ⁻, loss — квадрат ошибки, батч случайный · policy-based: сеть выдаёт π_θ(a\|s), шаг по ∇ log π·Â_t повышает вероятность действий с Â_t > 0 · actor-critic: actor — политика, critic оценивает V(s), advantage A = Q − V снижает дисперсию; PPO построен так |
| 3 | 28–43 с | Почему PPO обрезает отношение политик? | График L^CLIP от r_t(θ) при Â_t = ±1 (иллюстративно): синяя ε-трубка [0.8, 1.2] при ε = 0.2, метки 1−ε, 1, 1+ε; зелёная кривая (Â > 0) растёт до 1+ε и переходит в плато, серый пунктир — как рос бы r·Â без clip; оранжевая точка едет по кривой и останавливается на плато с плашкой «градиент = 0»; красная кривая (Â < 0) симметрично: плато левее 1−ε; карточки «Отношение вероятностей r_t(θ) = π_θ/π_θold» и «Целевая функция PPO: E_t[min(r_t·Â_t, clip(r_t, 1−ε, 1+ε)·Â_t)]», ε обычно 0.1–0.2, в реализации из README eps_clip = 0.2, k_epochs = 10 | r_t(θ) — во сколько раз новая политика изменила вероятность выбранного действия · Â_t > 0: выгодно растить r_t, но выше 1+ε выигрыш обрезан — градиент 0 · Â_t < 0: симметрично, ниже 1−ε штраф не растёт; min берёт пессимистичную оценку · **ключевая идея**: PPO = policy gradient «на поводке», шаг ограничен ε-трубкой вокруг старой политики; стабильно и просто — стартовый выбор для роботов и игр |

Числа и формулы из README: MDP (S, A, P, R, γ), γ ∈ [0,1]; V^π, Q^π, π* = argmax_π E_π[Σ γ^t R_{t+1}]; G_t = Σ_{k=t}^{T} γ^{k−t} r_{k+1} (REINFORCE);
DQN: L(θ) = E_{(s,a,r,s′)∼D}[(r + γ max_{a′} Q(s′,a′;θ⁻) − Q(s,a;θ))²], replay buffer, target network (`target_update`);
policy gradient: ∇_θJ(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) Â_t]; A2C: A^π = Q^π − V^π, θ ← θ + α ∇ log π_θ(a|s) A^π;
PPO: L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1−ε, 1+ε)Â_t)], r_t(θ) = π_θ(a_t|s_t)/π_θold(a_t|s_t), ε обычно 0.1–0.2, в коде README `eps_clip=0.2`, `k_epochs=10`.
Иллюстративные величины (помечены на экране): γ = 0.9 для столбиков возврата, значения Q(s,a) и π(a|s) в сцене 2, Â_t = ±1 на графике PPO.

## Сборка

```bash
cd topics/deep-reinforcement-learning/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.5,11.5,15.5,20,25,27,31,36.5,40,42
npx -y hyperframes@0.8.81 render --quality looks --output renders/rl-agent-environment-loop-ppo.mp4
ffmpeg -i renders/rl-agent-environment-loop-ppo.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/rl-agent-environment-loop-ppo.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/rl-agent-environment-loop-ppo.mp4 -o ../../assets/visualizations/rl-agent-environment-loop-ppo.gif
```
