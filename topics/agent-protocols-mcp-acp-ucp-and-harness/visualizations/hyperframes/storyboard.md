---
title: "Сториборд клипа: слои агентных протоколов и harness"
description: "Сториборд и команды сборки HyperFrames-клипа agent-protocol-layers-and-harness: карта слоёв ACP / MCP / A2A / UCP вокруг агента, три примитива MCP и направление разговора, цикл agent harness с шагом verify."
tags:
  - kb/note
  - kb/visualization
  - domain/agents
  - concept/mcp
  - concept/harness
aliases:
  - agent-protocol-layers-and-harness
related:
  - agent-protocols-mcp-acp-ucp-and-harness
  - code-agents-autoresearch-and-loopy-era
status: notes
lang: ru
type: note
slug: agent-protocols-mcp-acp-ucp-and-harness/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# Слои агентных протоколов (MCP / ACP / A2A / UCP) и agent harness — сториборд

Клип `assets/visualizations/agent-protocol-layers-and-harness.{mp4,gif}`, 42 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Карта слоёв», «MCP — Model Context Protocol», «ACP — три разных
протокола с одним именем» (п. 1), «UCP — Universal Commerce Protocol», «Agent Harness», «Как протоколы входят в harness»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–14 с | Какой стык закрывает каждый протокол? | Цепочка блоков Человек → Редактор / IDE → Агент (LLM + loop); стрелки рисуются по очереди, на них пилюли «промпт, review» и «ACP · сессия, diffs, permissions»; пакет ACP едет редактор → агент и обратно; затем от агента три стрелки во внешний мир: «MCP · tools, resources, prompts» → Инструменты и данные (почта, БД, тикеты, браузер), «A2A · делегирование» → Другие агенты, «UCP · shopping lifecycle» → Коммерция (каталог, checkout, оплата) | Протоколы не конкурируют: каждый закрывает свой стык · ACP (Agent Client Protocol) сажает агента в IDE, как LSP даёт IDE язык: сессия, diffs, запрос permission · MCP — агент ↔ инструменты, A2A — агент ↔ агент, UCP — агент ↔ магазин: разные стыки |
| 2 | 14–28 с | Что внутри MCP и куда идёт разговор? | Карточка «MCP-сервер · 3 примитива»: Tools (действия: create_pr, get_build_logs), Resources (данные: схема БД, файл, страница тикета), Prompts (шаблоны: «разбери failing CI»); подвал «транспорт JSON-RPC: stdio или HTTP»; чипы «tool schema» летят с сервера в бар «контекст модели» внутри агента, бар заполняется; тег «схемы tools едят контекст: 10 MCP × 100 tools = размытый каталог»; слева появляется Редактор / IDE (Zed / JetBrains / Neovim, хост ACP): стрелки «ACP · редактор спрашивает» / «агент отвечает, просит permission» и «MCP · агент спрашивает» / «tool-сервер отвечает» с пакетами туда-обратно; тег «MCP ≠ безопасность — это канал; guardrails живут в harness» | Сервер MCP описывает три примитива: Tools — действия, Resources — данные, Prompts — шаблоны · Транспорт — JSON-RPC по stdio или HTTP; клиент кладёт схемы tools в контекст модели — они едят контекст · ACP: редактор спрашивает, агент отвечает. MCP: агент спрашивает, tool-сервер отвечает |
| 3 | 28–42 с | Что такое agent harness? | Пунктирная рамка «Agent harness — всё вокруг модели»; по схеме README: Задача → Rules / AGENTS.md / skills → Agent loop; сверху MCP tool registry, снизу Guardrails (max steps, permissions); Agent loop → Verify (tests, hooks, browser); круг 1: пакет → Verify краснеет «✗ тесты упали», стрелка «fail» возвращает пакет в loop; круг 2: Verify зеленеет «✓ side effect есть», стрелка «pass» → Done; карточка формулы «надёжность ≈ f(verify, guardrails, tool quality) ≫ f(длина system prompt)» | Harness — всё вокруг модели: инструменты, контекст, guardrails, цикл агента и шаг verify · Verify смотрит на side effect — тесты, hooks, браузер, — а не на слова агента: fail → ещё круг · **ключевая идея**: протоколы — провода, harness выбирает и ограничивает их. Не промптить сильнее, а чинить harness |

Числа и названия: «10 MCP × 100 tools» — из README («десять MCP с сотней tools — размытый каталог»); примеры tools
(`create_pr`, `get_build_logs`), resources и prompts — из таблицы примитивов README; определение harness и схема
компонентов — из раздела «Agent Harness»; формула надёжности — из «Как протоколы входят в harness». Fail/pass в сцене 3 —
иллюстративный прогон двух кругов verify, конкретных тестов нет. Логотипов продуктов нет — только блоки и стрелки.

## Сборка

```bash
cd topics/agent-protocols-mcp-acp-ucp-and-harness/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7,12,17,22,27,31,36,41
npx -y hyperframes@0.8.81 render --quality looks --output renders/agent-protocol-layers-and-harness.mp4
ffmpeg -i renders/agent-protocol-layers-and-harness.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/agent-protocol-layers-and-harness.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/agent-protocol-layers-and-harness.mp4 -o ../../assets/visualizations/agent-protocol-layers-and-harness.gif
```
