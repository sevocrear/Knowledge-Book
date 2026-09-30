---
name: kb-video
description: >
  Explainer-клипы для топиков книги знаний: HyperFrames (HTML + GSAP → MP4 + GIF) по пайплайну nd-video-studio.
  Использовать, когда просят визуализировать / анимировать / «объяснить наглядно» любую тему книги (архитектуры,
  лоссы, пайплайны, алгоритмы, очереди и serving), добавить или перерендерить клип в `topics/<slug>/`, поправить
  сцену, подписи или сториборд. Триггеры: «визуализируй X», «сделай клип/гифку про X», «анимируй как работает X»,
  «перерендери визуализацию», «добавь озвучку». Результат: `topics/<slug>/visualizations/hyperframes/` (исходники)
  и `topics/<slug>/assets/visualizations/<name>.{mp4,gif}` (артефакты), встроенные в README темы.
---

# kb-video: от идеи к MP4/GIF в топике

Пайплайн взят из [nd-video-studio](https://github.com/vakovalskii/nd-video-studio) (скилл `nd-video`):
сториборд «один вопрос — одна сцена» → композиция HyperFrames → `check` + контактный лист → рендер → сжатие.
Голос и музыка в nd-video опциональные; в книге клипы **без звука** (подписи на экране обязательны),
озвучка — см. раздел «Опционально: голос и музыка».

Контракт композиции HyperFrames: скилл `hyperframes-core` (ставится в проект командой `npx hyperframes init`,
либо читайте `vendor/hyperframes/skills/hyperframes-core/SKILL.md` в репозитории nd-video-studio).
Эталонный проект в книге: `topics/arcface-and-angular-margin-losses-for-identification/visualizations/hyperframes/`
— копируйте его структуру, `theme.css` и приёмы, не изобретайте новую вёрстку.

## Что где лежит (обязательная раскладка)

```
topics/<slug>/
  visualizations/hyperframes/         # исходники (коммитятся)
    storyboard.md                     # сториборд: сцены, вопросы, подписи, числа, команды сборки
    index.html                        # тонкий оркестратор: root + N сцен-подкомпозиций
    compositions/s1-<name>.html …     # сцены (<template> + theme.css + свой <style> + один paused GSAP timeline)
    theme.css                         # общая тема книги (копия из эталона, не менять)
    gsap.min.js                       # локальная копия GSAP 3.14 (CDN в рендере недоступен)
    hyperframes.json, package.json    # конфиг проекта
    renders/, snapshots/              # НЕ коммитятся (.gitignore)
  assets/visualizations/<name>.mp4    # 1080p, без звука, сжатый (libx264 crf 28)
  assets/visualizations/<name>.gif    # 960px, 12 fps, < 8 MB — то, что показывает Markdown
```

Конвертация только общими скриптами `scripts/viz/mp4_to_gif.py` и `scripts/viz/gif_to_mp4.py` (ffmpeg).

## Workflow

1. **Факты.** Прочитать разделы README темы, которые покрывает клип. Все числа и формулы — из README
   (или из первоисточника с пометкой на экране: «в статье …», «иллюстративно»). Ничего не выдумывать.
2. **Сториборд** (`storyboard.md`): 3 сцены (максимум 4), 12–16 с каждая, всего 40–45 с. У сцены —
   заголовок-вопрос, 2–4 подписи, у последней сцены — «ключевая идея». Без подписей и маршрута зритель
   не понимает, на что смотрит (главный фидбек nd-video по первому монтажу).
3. **Композиция.** Скопировать из эталона `theme.css`, `gsap.min.js`, `hyperframes.json`, `package.json`,
   структуру `index.html` и сцены. Правила — ниже, приёмы — `references/scene-patterns.md`, скелет сцены —
   `references/scene-template.html`.
4. **Проверка.** `npx -y hyperframes@0.8.81 check` → во всех секциях `0 errors, 0 warnings`.
   Затем `npx -y hyperframes@0.8.81 snapshot --at t1,t2,…` (2–3 момента на сцену) и **смотреть каждый PNG**:
   наложения, обрезки, неверная геометрия, ошибки в формулах. Минимум два круга правок.
5. **Рендер.** `npx -y hyperframes@0.8.81 render --quality looks --output renders/<name>.mp4` (42 с ≈ 1–2 мин).
6. **Сжатие + GIF** (из корня репозитория):
   ```bash
   ffmpeg -y -i topics/<slug>/visualizations/hyperframes/renders/<name>.mp4 \
     -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart \
     topics/<slug>/assets/visualizations/<name>.mp4
   uv run python scripts/viz/mp4_to_gif.py topics/<slug>/assets/visualizations/<name>.mp4 \
     -o topics/<slug>/assets/visualizations/<name>.gif        # --fps 10 --width 896, если > 8 MB
   ```
7. **Встраивание в README** (см. ниже), затем `uv run python scripts/kb_validate_links.py` и `uv run pytest`.

## Встраивание в README темы

GIF через обычную картинку рендерится везде (GitHub, Obsidian, IDE-превью); `<video>` с относительным
путём на GitHub и в Obsidian не играет. Поэтому:

```markdown
**Визуализация (HyperFrames, 42 с):** одна фраза, что показывает клип, сцена за сценой.

![Подпись клипа](./assets/visualizations/<name>.gif)

*Полная версия: [MP4 1080p](./assets/visualizations/<name>.mp4) · сториборд и исходники сцен: [`visualizations/hyperframes/`](./visualizations/hyperframes/storyboard.md).*
```

## Правила композиции (нарушение = красный `check` или битый рендер)

- Оркестратор `index.html`: `root` с `data-duration` = сумме сцен; сцены — подкомпозиции подряд на одном
  `data-track-index`; `window.__timelines["root"]` — пустой paused timeline. Сцены не вкладывать в root
  напрямую (lint `nested_structure_needs_subcomposition`).
- Сцена: всё внутри `<template>` (стили, разметка, скрипт); `<link rel="stylesheet" href="theme.css">`
  внутри шаблона; корень `#root` с `data-composition-id="sN"`; ключ `window.__timelines["sN"]` совпадает с id
  хоста в `index.html`. Регистрировать timeline **после** построения.
- Один `gsap.timeline({ paused: true })` на сцену. Входы — `fromTo`; повторный `fromTo` на тот же элемент —
  с `immediateRender: false`. Fade сцены — opacity на `#sN-wrap`, не на хосте/клипе.
- Детерминизм: без `Math.random`, `Date.now`, `performance.now`, сетевых загрузок, `repeat: -1`,
  `onUpdate` для визуального состояния (колбэки глушатся при seek). Координаты — константы/массивы в скрипте.
- Движение только трансформами (`x`, `y`, `scale`, `rotation`, `opacity`) или SVG-`attr` (`cx`, `cy`, `x`, `y`,
  `width`, `height`, `stroke-dasharray`); `top/left/bottom` и др. layout-свойства на тексте — lint
  `gsap_non_transform_motion`. Не сочетать CSS `transform` и GSAP `x/y` на одном элементе.
- Дуги/линии «рисуются» через `attr: {"stroke-dasharray": "<длина> 9999"}`; точки едут через `attr: {cx, cy}`;
  вращение SVG-групп — `rotation` + `svgOrigin: "cx cy"`. Счётчики — CSS `@property --n` + `counter-reset`
  и tween `"--n"` со `snap: "--n"` (пример: сцена 2 эталона).
- id уникальны во всей сборке: префикс сцены (`s2-bar1`). Намеренный кроссфейд двух текстов —
  `data-layout-allow-overlap` на обоих.
- Шрифты: только стек из `theme.css` (system-ui…), без `@font-face` и именованных семейств — иначе компилятор
  тянет замену с Google Fonts. Формулы — Unicode/HTML (`θ`, `<sub>`), без LaTeX.
- Контраст ≥ 4.5:1: серый текст `#aab3c2` и светлее. Подпись ≤ 2 строк при 38px (~80 символов), одна подпись
  на экране; последняя подпись последней сцены — `class="cap key"`.
- Только собственная графика (клетки, градиенты, фигуры); без фотографий и чужих персонажей.

## Опционально: голос и музыка (пайплайн nd-video-studio)

Если нужны озвучка/музыка — клонировать nd-video-studio и использовать его скрипты рядом с проектом сцены:
`narration.json` (`start` + `text` по сценам) → `tts_hub.py` или `tts_gpt_audio.py` (голос, нужны ключи
`ND_API_KEY` / `OPENROUTER_API_KEY`) → `nd_align.py` (whisper-проверка, тайминги слов) → `build_audio.sh`
(fit → voice_fx → mix с ducking) → в композиции один `<audio id="soundtrack" src="audio/mix.wav">` на всю длину.
Ключи только в `.env`, не печатать и не коммитить. Для книги озвучка не требуется; MP4/GIF в README — немые.

## Тесты и CI

- `tests/test_hyperframes_topic_compositions.py` — статические проверки каждого проекта (структура, ключи
  timeline, уникальные id, запрещённые вызовы, наличие MP4/GIF и ссылки из README). Гоняются в CI без Node.
- Тесты с маркером `hyperframes` запускают настоящий `npx hyperframes check`; включаются локально
  переменной `KB_RUN_HYPERFRAMES=1` (нужны Node 22+, ffmpeg, кэш Chrome: `npx hyperframes browser ensure`).

## Чек-лист перед коммитом

- [ ] `storyboard.md` с вопросами сцен, подписями, числами и командами сборки
- [ ] `check`: 0 ошибок и 0 предупреждений; кадры просмотрены глазами (2+ момента на сцену)
- [ ] MP4 сжат (обычно 1–3 MB), GIF < 8 MB, оба в `assets/visualizations/`, старые Manim-исходники удалены
- [ ] README: GIF-картинка + ссылка на MP4 + ссылка на сториборд; `kb_validate_links.py` и `pytest` зелёные
- [ ] `renders/`, `snapshots/`, `node_modules/` не в коммите
