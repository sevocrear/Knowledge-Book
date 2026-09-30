# Приёмы сцен kb-video (HyperFrames + GSAP)

Все приёмы проверены `npx hyperframes check` (0 ошибок / 0 предупреждений) и рендером в headless Chrome.
Примеры — в эталонном проекте `topics/arcface-and-angular-margin-losses-for-identification/visualizations/hyperframes/`.

## Оглавление

1. [Раскладка кадра](#1-раскладка-кадра)
2. [Подписи и ключевая идея](#2-подписи-и-ключевая-идея)
3. [Дуги, линии и стрелки, которые «рисуются»](#3-дуги-линии-и-стрелки-которые-рисуются)
4. [Точки, которые едут (эмбеддинги, кластеры)](#4-точки-которые-едут-эмбеддинги-кластеры)
5. [Поворот вектора вокруг центра](#5-поворот-вектора-вокруг-центра)
6. [Столбики с живыми числами](#6-столбики-с-живыми-числами)
7. [Пайплайн из блоков](#7-пайплайн-из-блоков)
8. [Пакеты/запросы, бегущие по пути](#8-пакетызапросы-бегущие-по-пути)
9. [Кроссфейд состояний](#9-кроссфейд-состояний)
10. [Что ловит lint (и как чинить)](#10-что-ловит-lint-и-как-чинить)

## 1. Раскладка кадра

1920×1080, тёмный фон `#0f1117`. `theme.css` задаёт зоны:

- `.head` (y 44–140): pill «N / 3» + заголовок-вопрос 48px; `.badge` справа «Knowledge Book · Тема».
- `.stage` (x 80–1840, y 160–820): слева SVG-диаграмма `980×660` (viewBox `0 0 980 660`), справа карточки
  `.card` шириной 700px (`left: 1060px`).
- `.caps` (y 846–1026): подписи `.cap` друг над другом, видима одна.

Палитра: blue `#4a9eff`, orange `#ff7a2f`, green `#3dba7a`, red `#ff4d6d`, purple `#b48ce6`, teal `#2dd4bf`,
gray `#aab3c2` (только для текста ≥ 4.5:1), панель `#161a24`, линия `#2a3140`.

## 2. Подписи и ключевая идея

```js
const cap = (id, t0, t1) => {
  tl.fromTo(id, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.45, ease: "power2.out" }, t0);
  if (t1) tl.to(id, { opacity: 0, duration: 0.35 }, t1);
};
cap("#s1-cap1", 0.6, 5.0); cap("#s1-cap2", 5.3, 9.3); cap("#s1-cap3", 9.5, 0); // последняя без t1 — держится
```

Окна подписей не пересекаются (t1 предыдущей < t0 следующей). `class="cap key"` — оранжевая карточка
«Ключевая идея» только в последней сцене. Цветные акценты: `<b class="o|g|b|p|t|r">`.

## 3. Дуги, линии и стрелки, которые «рисуются»

`d` у `<path>` не интерполируется; рисуйте через `stroke-dasharray`:

```js
const pt = (r, deg) => [cx + r * Math.cos(deg * Math.PI / 180), cy - r * Math.sin(deg * Math.PI / 180)];
const arc = (r, a0, a1) => { const [x0, y0] = pt(r, a0), [x1, y1] = pt(r, a1); const sweep = a1 > a0 ? 0 : 1;
  return `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 0 ${sweep} ${x1.toFixed(1)} ${y1.toFixed(1)}`; };
const len = (r, a0, a1) => (r * Math.abs(a1 - a0) * Math.PI / 180).toFixed(1);
el.setAttribute("d", arc(150, 15, 40));                  // один раз при сборке
tl.fromTo(el, { attr: { "stroke-dasharray": "0 9999" } },
             { attr: { "stroke-dasharray": len(150, 15, 40) + " 9999" }, duration: 0.8, ease: "power2.inOut" }, t);
```

Углы — математические (против часовой от оси x), `pt` переворачивает y для экрана. Для прямой линии длина —
`Math.hypot(dx, dy)`. Укорачивание дуги (угол уменьшается) — просто tween к меньшей длине.

## 4. Точки, которые едут (эмбеддинги, кластеры)

Позиции «до» и «после» — явные массивы, никакого `Math.random`:

```js
const offsBefore = [-46, -31, -18, -7, 4, 13, 24, 37, 49], offsAfter = [-9, -6.5, -4, -1.5, 0.5, 2.5, 4.5, 6.5, 9];
points.forEach(([el, cls, i]) => { const [x, y] = pt(R, proto[cls] + offsAfter[i]);
  tl.to(el, { attr: { cx: x.toFixed(1), cy: y.toFixed(1) }, duration: 2.2, ease: "power2.inOut" }, 4.6); });
```

Появление — `fromTo(el, {opacity: 0}, {opacity: 1, duration: 0.3}, t + 0.04 * i)` (ручной stagger).

## 5. Поворот вектора вокруг центра

```js
tl.fromTo("#s2-z", { rotation: 0, svgOrigin: "430 340" }, { rotation: 18, svgOrigin: "430 340", duration: 2, ease: "power2.inOut" }, 9.4);
```

`rotation > 0` — по часовой на экране. Группа `<g>` с линией, кружком и подписью крутится целиком; подпись
внутри группы наклонится — если это мешает, держите подпись отдельно и двигайте `x/y`.

## 6. Столбики с живыми числами

Столбик — `div.bar` с `height` (можно tween `height`), число — блок над ним, который едет `y`, а не `bottom`:

```css
@property --n { syntax: "<integer>"; inherits: true; initial-value: 0; }
.bar-val::before { counter-reset: n var(--n); content: "0." counter(n); }   /* 906 → «0.906» */
```
```js
tl.fromTo("#s2-bar1", { height: 0 }, { height: 0.906 * H, duration: 0.8 }, 5.0);
tl.fromTo("#s2-val1", { y: 0 }, { y: -0.906 * H, duration: 0.8 }, 5.0);
tl.to("#s2-val1", { y: -0.993 * H, "--n": 993, snap: "--n", duration: 2 }, 9.4);  // число меняется вместе со столбиком
```

`inherits: true` обязателен, иначе `::before` видит 0. Отрицательные/особые значения — отдельный текстовый
блок и кроссфейд (п. 9).

## 7. Пайплайн из блоков

Ряд/колонка `.card`-подобных блоков с SVG-стрелками между ними; блоки «загораются» по очереди:

```js
["#s3-b1", "#s3-b2", "#s3-b3", "#s3-b4"].forEach((id, i) => {
  tl.fromTo(id, { opacity: 0.35, scale: 0.97 }, { opacity: 1, scale: 1, duration: 0.4, ease: "power2.out" }, 6 + 1.2 * i);
  tl.to(id + "-arrow", { attr: { "stroke-dasharray": "120 9999" }, duration: 0.5 }, 6.4 + 1.2 * i);
});
```

Блоки — `display:block` с явными размерами (иначе `scale` невидим). Не тянуть `width/height` карточек с текстом.

## 8. Пакеты/запросы, бегущие по пути

Несколько кружков-`div` (block, sized), позиции — ключевые точки пути, повторы конечные:

```js
const cycle = 1.6, window_ = 8, reps = Math.max(0, Math.floor(window_ / cycle) - 1);
tl.fromTo("#s2-pkt1", { x: 0, y: 0, opacity: 1 }, { x: 420, y: 0, duration: 0.8, ease: "none", repeat: reps, repeatDelay: 0.8 }, 3.0);
```

`repeat: -1` запрещён; `floor`, не `ceil` (иначе выход за `data-duration`).

## 9. Кроссфейд состояний

Два блока в одной позиции; первый гаснет, второй появляется; на оба — `data-layout-allow-overlap`:

```html
<div id="s2-val3" class="bar-val" data-layout-allow-overlap>0.0</div>
<div id="s2-val3b" class="bar-val" style="opacity:0" data-layout-allow-overlap>&lt; 0</div>
```
```js
tl.to("#s2-val3", { opacity: 0, duration: 0.4 }, 10.4); tl.to("#s2-val3b", { opacity: 1, duration: 0.4 }, 10.4);
```

## 10. Что ловит lint (и как чинить)

| Находка | Причина | Решение |
|---|---|---|
| `nested_structure_needs_subcomposition` | сцены-`section` внутри root | каждая сцена — файл в `compositions/`, host в `index.html` |
| `gsap_non_transform_motion` | tween `top/left/bottom/…` на тексте | `x`/`y` |
| `gsap_repeated_fromto_without_baseline` | два `fromTo` на один элемент | `immediateRender: false` во втором |
| `gsap_css_transform_conflict` | CSS `transform` + GSAP `x/y` | убрать CSS transform, задать старт в `fromTo` |
| `content_overlap` (Layout) | два текста в одной зоне | развести или `data-layout-allow-overlap` при кроссфейде |
| `request_failed` / `gsap is not defined` (Runtime) | GSAP с CDN | локальный `./gsap.min.js` |
| `sweep_static` | timeline не двигается при seek | timeline paused и зарегистрирован после сборки |
| `font_family_without_font_face` / «Fetched … from Google Fonts» | именованный шрифт | только стек `theme.css` |
| контраст < 4.5:1 | тёмно-серый текст | `#aab3c2` и светлее |
