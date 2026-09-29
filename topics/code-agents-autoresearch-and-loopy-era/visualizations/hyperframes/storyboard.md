---
title: "Сториборд клипа: агентный цикл с verification gate"
description: "Сториборд и команды сборки HyperFrames-клипа code-agent-verification-loop: переход 80/20 → 20/80 и Ralph Loop, verification gate с тремя итерациями до done, agent harness и AutoResearch с evaluator."
tags:
  - kb/note
  - kb/visualization
  - domain/agents
  - concept/orchestration
  - concept/verification
aliases:
  - code-agent-verification-loop
related:
  - code-agents-autoresearch-and-loopy-era
  - agent-protocols-mcp-acp-ucp-and-harness
status: notes
lang: ru
type: note
slug: code-agents-autoresearch-and-loopy-era/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Агентный цикл с verification gate — сториборд

Клип `assets/visualizations/code-agent-verification-loop.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Введение», «Главные идеи», «Ключевые инсайты», «Правила большого пальца»,
«Ralph Loop: что это и какие разновидности бывают», «Must-have техники на 2026 год», «Транскрипт-выжимка: два видео про Ralph»,
«Stop Babysitting Your Agents», «AI Harness Engineering»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Что такое агентный цикл? | Полоса «роль инженера»: доля ручного кода 80 % → 20 %, доля intent · constraints · review-стандартов 20 % → 80 %; инженер → кольцо plan → implement → test → fix (repeat, пока не done); карточка «Ralph Loop» с `while ! done; do ai_agent "$PROMPT"; done`; блок «внешнее состояние» (код, git, PRD, progress-файлы, тестовые отчёты); карточка «один агент = одна роль» (Planner / Implementer / Reviewer, Evaluator); точка пробегает кольцо и уходит в «done-сигнал ✓» | ручной код 80/20 → 20/80 и ниже; инженер формулирует intent, constraints и review-стандарты · Ralph Loop: тот же high-level prompt, прогресс во внешнем состоянии · implement → test → fix → repeat до явного done-сигнала |
| 2 | 14–29 с | Что добавляет verification gate? | Конвейер implement / fix → verification gate (лампы tests · lint · types · perf) → done; красная стрелка «fail → fix»; запрос бегает три итерации: 1 — tests ✗ → fix, 2 — tests ✓ · lint ✗ → fix, 3 — все gates ✓ → done; журнал итераций; плашка guardrails; карточки «Test-gated loop», «Objective — верифицируемо» (✗ «Сделай лучше» / ✓ «Уменьши p95 latency на 20 % без регрессии test-suite»), «Guardrails» (max iterations, completion signal, human approve) | gate = tests, lint, type checks, perf budget — тот же цикл проверки, что у человека · красный gate возвращает на fix; человек задал objective и guardrails · objective верифицируемо: «сделай лучше» плохо, «p95 latency −20 % без регрессии» хорошо · все gates зелёные → done; guardrails: max iterations и completion signal |
| 3 | 29–42 с | Как снять человека с критического пути? | Блок «человек: objective, guardrails» → пунктирная рамка «agent harness» вокруг модели с tools, guardrails, verify, outer loop и подписью «не “prompt harder”, а код обвязки»; справа AutoResearch: три параллельных кандидата с метрикой pass@tests 0.71 / 0.84 / 0.78 (иллюстративно), evaluator выбирает лучшего, плашка «оценка раньше генерации · eval before scale»; карточки «AutoResearch: где работает» (объективная метрика loss / latency / pass@tests / recall@k, дешёвая верификация, параллелизуемое пространство) и «Leverage ≈ полезные агентные токены / человеческие токены и время» | harness — всё вокруг модели: tools, guardrails, verify, outer loop; чинит код обвязки, а не «prompt harder» · AutoResearch: цель и метрика формализованы → кандидаты параллельно, evaluator выбирает по метрике · **ключевая идея**: человек задаёт objective и guardrails; цикл крутится сам, а verification gate решает, что «готово» |

Числа и формулы: 80/20 → 20/80 (и ниже) по ручному коду — «Главные идеи», п. 1; `while ! done; do ai_agent "$PROMPT"; done`
и implement → test → fix → repeat — раздел «Ralph Loop»; quality gates tests, lint, type checks, perf budget — «Test-gated loop»
и «Must-have для Python»; «Уменьши p95 latency на 20 % без регрессии test-suite» — «Правила большого пальца», п. 1;
harness = tools, guardrails, verify, outer loop и «не prompt harder» — «AI Harness Engineering»; критерии AutoResearch и
Leverage ≈ полезные агентные токены / человеческие токены и время — «Ключевые инсайты», пп. 2–3; max iterations,
completion signal, human approve — «Транскрипт-выжимка» и «Ralph Loop». Значения pass@tests 0.71 / 0.84 / 0.78 — иллюстративные.

## Сборка

```bash
cd topics/code-agents-autoresearch-and-loopy-era/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 4,7.5,13,17,20.5,25,28,31.5,36,40
npx -y hyperframes@0.8.81 render --quality looks --output renders/code-agent-verification-loop.mp4
ffmpeg -i renders/code-agent-verification-loop.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/code-agent-verification-loop.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/code-agent-verification-loop.mp4 -o ../../assets/visualizations/code-agent-verification-loop.gif
```
