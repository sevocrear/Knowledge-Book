---
title: "Сториборд клипа: BPE — как токенизатор учит словарь и сжимает текст"
description: "Сториборд и команды сборки HyperFrames-клипа bpe-tokenization-merges: текст → токены → ID → строка матрицы эмбеддингов и почему это сжатие; обучение BPE — частоты соседних пар и четыре слияния; кодирование нового слова выученными merges, компромисс размера словаря и byte-level BPE."
tags:
  - kb/note
  - kb/visualization
  - domain/nlp
  - domain/llm
  - concept/bpe
aliases:
  - BPE storyboard
  - bpe-tokenization-merges
related:
  - tokenization-and-text-compression-in-llms
  - embeddings-and-embedding-matrix
status: notes
lang: ru
type: note
slug: tokenization-and-text-compression-in-llms/visualizations/hyperframes/storyboard
updated: 2026-09-29
---
# BPE: как токенизатор учит словарь и сжимает текст — сториборд

Клип `assets/visualizations/bpe-tokenization-merges.{mp4,gif}`, 43 с, 1920×1080, без звука.
Одна идея на сцену; каждая сцена — вопрос в заголовке, 3–4 подписи внизу, в конце ключевая идея.
Факты сверены с `../../README.md` (разделы «Что такое токенизатор», «Зачем нужен токенизатор и почему это сжатие»,
«Сабворд-токенизация → BPE / Byte-level BPE», «Компрессия и выбор словаря токенов», «Практические рекомендации и грабли»).

| # | Время | Вопрос | Что на экране | Подписи (по порядку) |
|---|---|---|---|---|
| 1 | 0–13 с | Что такое токенизатор и почему это сжатие? | Строка «Привет, как дела?» и три разрезания: по символам (17 чипов), по словам (5 чипов + «новое слово → ⟨UNK⟩»), subword (7 чипов из README: Пр · ивет · , · Ġкак · Ġдел · а · ?) → стрелки к ID 10342, 9121, 11, 845, 12987, 42, 30 → строка E₁₀₃₄₂ ∈ ℝᵈ, d = 4096 (иллюстративный градиент ячеек); карточки «Словарь V (например, 50k)», «Эмбеддинги E ∈ ℝ^{V×d}», «Сжатие» | по символам — 17 ID, по словам — словарь-гигант и OOV · subword — компромисс: частое — один токен, редкое — по кусочкам; 7 токенов вместо 17 · каждый ID — строка E, число токенов n задаёт стоимость attention O(n²) |
| 2 | 13–29 с | Как BPE учит словарь токенов? | Корпус из статьи Sennrich et al. (2016), упрощён (без маркера конца слова): low ×5, lower ×2, newest ×6, widest ×3 как чипы символов; таблица частот пар шага 1 (e s 9, s t 9, w e 8, l o 7, o w 7, n e 6, e w 6); четыре слияния с подсветкой пары и кроссфейдом чипов: e+s → es (9), es+t → est (9), l+o → lo (7), lo+w → low (7); полоса словаря 10 базовых символов + новые токены; счётчик \|V\| = 10 → 14; карточки «Алгоритм BPE» (4 шага из README) и «Merges по порядку» | старт: словарь = алфавит базовых токенов (символы или байты) · самая частая пара e s (9) → новый токен es · повторяем: es t → est, l o → lo, lo w → low — по одному токену за слияние · останавливаемся, когда словарь достиг нужного размера (тысячи–десятки тысяч) |
| 3 | 29–43 с | Как BPE кодирует новое слово? | Слово «lowest» (в корпусе не было) → l o w e s t → трасса из 4 строк, merges применяются в порядке обучения → [low, est]: 2 токена вместо 6; ползунок «размер словаря \|V\|» (иллюстративно): n падает, V·d растёт; карточки «Кодирование», «Компромисс» (\|V\| мал → n ↑, attention O(n²); \|V\| велик → E ∈ ℝ^{V×d} ↑), «Byte-level BPE» (байты 0–255, GPT-2, RoBERTa; токенизатор модели менять нельзя) | применяем выученные merges в том же порядке → [low, est] · компромисс: маленький словарь → длинные последовательности и O(n²); большой — растёт матрица E (V×d) · **ключевая идея**: BPE — обученное сжатие текста: частое — одним токеном, редкое — по кусочкам; byte-level BPE (байты 0–255) обходится без OOV |

Числа и факты: пример «Привет, как дела?» → 7 токенов и ID из раздела README «Что такое токенизатор»; V ≈ 50k, d = 4096 — оттуда же;
четыре шага BPE, «символы или байты», «тысячи–десятки тысяч токенов», byte-level BPE (0–255, GPT-2, RoBERTa), стоимость attention O(n²),
компромисс размера словаря и «не смешивать токенизаторы» — из README. Обучающий корпус (low ×5, lower ×2, newest ×6, widest ×3) и частоты пар
(e s = 9, s t = 9, w e = 8, l o = 7, o w = 7, n e = 6, e w = 6) — упрощённый пример из статьи Sennrich, Haddow, Birch (2016), помечен на экране;
17 символов в «Привет, как дела?» — прямой подсчёт; ползунок \|V\| и градиент вектора эмбеддинга — иллюстративные.

## Сборка

```bash
cd topics/tokenization-and-text-compression-in-llms/visualizations/hyperframes
npx -y hyperframes@0.8.81 check                       # 0 ошибок, 0 предупреждений
npx -y hyperframes@0.8.81 snapshot --at 3,7.5,11,17,20,25,28,32,37.5,41.5
npx -y hyperframes@0.8.81 render --quality looks --output renders/bpe-tokenization-merges.mp4
ffmpeg -i renders/bpe-tokenization-merges.mp4 -c:v libx264 -preset slow -crf 28 -tune animation -pix_fmt yuv420p -movflags +faststart ../../assets/visualizations/bpe-tokenization-merges.mp4
uv run python ../../../../scripts/viz/mp4_to_gif.py ../../assets/visualizations/bpe-tokenization-merges.mp4 -o ../../assets/visualizations/bpe-tokenization-merges.gif
```
