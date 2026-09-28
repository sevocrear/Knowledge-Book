---
title: Лоссы metric learning и подбор майнеров
description: "Каталог лоссов metric learning (contrastive, triplet, N-pair, Multi-Similarity, Circle, InfoNCE/SupCon, Proxy-NCA/Anchor, SoftTriple, ArcFace/CosFace/AdaFace): формулы, интуиция, какие майнеры к какому лоссу и когда что выбирать."
tags:
  - kb/topic
  - domain/cv
  - concept/metric-learning
  - concept/loss
  - concept/embeddings
  - concept/contrastive-learning
aliases:
  - triplet loss
  - Multi-Similarity loss
  - Circle loss
  - Proxy-Anchor
  - hard negative mining
  - semi-hard mining
  - miners
related:
  - arcface-and-angular-margin-losses-for-identification
  - contrastive-and-metric-learning-for-fine-grained-visual-recognition
  - classification-losses-cross-entropy-focal-loss
  - embeddings-and-embedding-matrix
  - roc-curve-and-roc-auc
status: canonical
lang: ru
type: topic
slug: metric-learning-losses-and-miners
updated: 2026-09-18
---
# Лоссы metric learning и подбор майнеров

## Оглавление

- Краткий абстракт и объяснение для 5-летнего ребёнка
- Общая рамка: эмбеддинг, метрика, «лосс + сэмплер + майнер»
- Таксономия лоссов
- Pair-based лоссы
  - Contrastive loss
  - Multi-Similarity loss
  - Circle loss
- Triplet и tuple-based лоссы
  - Triplet loss (hinge и soft-margin)
  - N-pair loss
  - Lifted Structured loss
- Batch-contrastive (InfoNCE-семейство)
  - InfoNCE / NT-Xent
  - SupCon
- Proxy- и classification-based лоссы
  - Normalized Softmax и Center loss
  - Proxy-NCA / Proxy-NCA++
  - Proxy-Anchor
  - SoftTriple
  - SphereFace / CosFace / ArcFace
  - Sub-center ArcFace, CurricularFace, AdaFace, MagFace, Partial FC
- Сводная таблица лоссов
- Майнеры: зачем они нужны и какие бывают
  - Почему случайные пары/триплеты не работают
  - Каталог майнеров
  - Сэмплер батча P×K и cross-batch memory
- Таблица «лосс ↔ майнер»
- Когда что использовать: decision guide
- Обновляемый набор классов: ArcFace vs pair-based
- Типовые гиперпараметры
- Минимальный пример на pytorch-metric-learning
- Частые ошибки и анти-паттерны
- Источники

---

## Краткий абстракт и объяснение для 5-летнего ребёнка

**Кратко по-взрослому.**
Metric learning учит сеть $f_\theta$ отображать объекты в векторы $z$ так, чтобы расстояние (или косинус) между ними отражало «похожесть». Все лоссы этой области делают одно и то же — **стягивают positives и раздвигают negatives** — но отличаются тем, *что* они сравнивают (пары, тройки, целый батч, обучаемые прототипы классов) и *как* взвешивают трудные примеры. Из этого следует главное правило: лосс нельзя выбирать отдельно от **сэмплера батча** и **майнера** (правила отбора информативных пар/троек). Triplet без semi-hard mining почти не учится; ArcFace и Proxy-Anchor майнер не нужен вообще, потому что softmax по всем классам уже «видит» все негативы.

**Как объяснить 5-летнему ребёнку.**
Мы расставляем игрушки на столе: одинаковые — рядом, разные — далеко. Один способ — брать по две игрушки и сравнивать (пары), другой — по три (тройки), третий — нарисовать для каждого вида игрушек «домик» и подвигать игрушки к своему домику (прототипы). А «майнер» — это помощник, который приносит только те игрушки, которые ты пока путаешь: с очевидными и так всё понятно.

---

## Общая рамка: эмбеддинг, метрика, «лосс + сэмплер + майнер»

Обозначения на весь топик:

- $z_i = f_\theta(x_i)\in\mathbb{R}^d$, чаще всего $\|z_i\|_2=1$ (L2-нормировка);
- сходство $S_{ij} = z_i^\top z_j$ (cosine), расстояние $D_{ij}=\|z_i-z_j\|_2$; для единичных векторов $D_{ij}^2 = 2-2S_{ij}$;
- $P_i$ — множество positives для якоря $i$ (тот же класс/экземпляр), $N_i$ — negatives;
- $W_c$ или $p_c$ — обучаемый прототип (proxy) класса $c$, $\theta_c$ — угол между $z$ и $W_c$.

Обучение metric learning — это система из трёх частей:

1. **Сэмплер батча**: гарантирует, что в батче есть positives (обычно $P$ классов × $K$ примеров).
2. **Майнер**: из всех пар/троек батча выбирает информативные (ненулевой градиент, но не мусор).
3. **Лосс**: превращает выбранные пары/тройки (или сравнения с прототипами) в число.

Часть лоссов «встраивает» майнинг в себя (log-sum-exp, softmax по батчу или классам), часть — нет, и тогда майнер обязателен.

---

## Таксономия лоссов

| Семейство | Что сравнивается | Нужны ли метки классов | Встроенный майнинг | Примеры |
|---|---|---|---|---|
| Pair-based | пары $(i,j)$ | пары «same/different» | нет / частично | Contrastive, Multi-Similarity, Circle (pair-форма) |
| Triplet / tuple-based | тройки $(a,p,n)$ или $(a,p,n_1..n_K)$ | метки или пары | нет (triplet) / да (N-pair, Lifted) | Triplet, N-pair, Lifted Structured |
| Batch-contrastive | якорь против всего батча | нет (SSL) или да (SupCon) | да (softmax по батчу) | InfoNCE, NT-Xent, SupCon |
| Proxy / classification-based | эмбеддинг против прототипов классов | да, фиксированный набор классов | да (softmax по классам) | NormSoftmax, Proxy-NCA, Proxy-Anchor, SoftTriple, ArcFace/CosFace, AdaFace |

Практически: **pair/triplet** = гибко, работает с любыми метками, но дорого и чувствительно к майнингу; **proxy** = быстро сходится, стабильно, но требует известного набора классов и памяти $O(C\cdot d)$ на прототипы.

---

## Pair-based лоссы

### Contrastive loss

Классическая форма (Hadsell, Chopra, LeCun, 2006) для пары с меткой $y\in\{0,1\}$:

$$
L = y\,D_{ij}^2 + (1-y)\,\max(0,\; m - D_{ij})^2 .
$$

Positives стягиваются к нулю, negatives выталкиваются за margin $m$. Современный вариант с двумя порогами (как в `pytorch-metric-learning`):

$$
L = y\,\max(0,\; D_{ij} - m_{pos}) + (1-y)\,\max(0,\; m_{neg} - D_{ij}),
$$

то есть positives не обязаны схлопываться в точку — достаточно быть ближе $m_{pos}$. Это уменьшает переобучение на внутриклассовой вариативности.

- Плюсы: простой, объяснимый, работает без меток классов (только пары).
- Минусы: сравнивает абсолютные расстояния, а не относительные; большинство пар после первых эпох лежат «за margin» и не дают градиента → нужен майнер (`PairMarginMiner`) или P×K-сэмплер.

### Multi-Similarity loss

Wang et al., CVPR 2019. Идея: взвешивать каждую пару по трём видам сходства — self-similarity (само $S_{ij}$), negative relative similarity (относительно других негативов якоря) и positive relative similarity (относительно других позитивов). Практически это реализовано как **шаг майнинга + лосс**:

$$
L = \frac{1}{N}\sum_{i=1}^N \left[
\frac{1}{\alpha}\log\Big(1+\sum_{k\in P_i} e^{-\alpha (S_{ik}-\lambda)}\Big)
+ \frac{1}{\beta}\log\Big(1+\sum_{k\in N_i} e^{\beta (S_{ik}-\lambda)}\Big)
\right].
$$

Здесь $\alpha$ управляет силой для positives, $\beta$ — для negatives ($\beta\gg\alpha$: трудные негативы весят экспоненциально больше), $\lambda$ — центр (порог) сходства. Log-sum-exp — это «мягкий max», то есть лосс сам фокусируется на самых трудных парах внутри якоря. Типично $\alpha=2,\ \beta=50,\ \lambda=0.5\text{–}1$.

Собственный майнер (`MultiSimilarityMiner`, $\epsilon\approx0.1$): для якоря $i$ берём негативы с $S_{in} > \min_{p\in P_i} S_{ip} - \epsilon$ и позитивы с $S_{ip} < \max_{n\in N_i} S_{in} + \epsilon$ — то есть только те пары, которые «нарушают порядок» с запасом $\epsilon$.

На retrieval-бенчмарках (SOP, CUB, Cars196) MS + MS-miner долго был одним из самых сильных pair-based рецептов и остаётся хорошим дефолтом.

### Circle loss

Sun et al., CVPR 2020. Объединяет pair- и proxy-формы в один лосс и исправляет «негибкость» triplet: в triplet/softmax градиент по $s_p$ и $s_n$ одинаков по модулю, хотя далёкий positive надо тянуть сильнее, чем уже далёкий negative толкать. Circle взвешивает каждое сходство отклонением от оптимума:

$$
L = \log\Big[1 + \sum_{j} e^{\gamma\,\alpha_n^j (s_n^j - \Delta_n)} \cdot \sum_{i} e^{-\gamma\,\alpha_p^i (s_p^i - \Delta_p)}\Big],
$$

$$
\alpha_p^i = [O_p - s_p^i]_+,\quad \alpha_n^j = [s_n^j - O_n]_+,\quad
O_p = 1+m,\ O_n=-m,\ \Delta_p = 1-m,\ \Delta_n = m .
$$

Граница решения в плоскости $(s_n, s_p)$ — окружность (отсюда название), а не прямая $s_p - s_n = m$. Работает и с парами из батча (тогда $s_p, s_n$ — сходства с примерами), и с прототипами (тогда $s_p=\cos\theta_y$, $s_n=\cos\theta_{j\ne y}$, и лосс становится обобщением ArcFace/CosFace). Типично $\gamma = 80\text{–}256$, $m = 0.25\text{–}0.4$. Майнер не обязателен: самовзвешивание уже играет его роль.

---

## Triplet и tuple-based лоссы

### Triplet loss (hinge и soft-margin)

FaceNet (Schroff et al., 2015). Для тройки якорь $a$, positive $p$, negative $n$:

$$
L = \max\big(0,\; D_{ap} - D_{an} + m\big)
\quad\text{или в квадратах}\quad
\max\big(0,\; D_{ap}^2 - D_{an}^2 + m\big).
$$

Требование **относительное**: negative должен быть дальше positive хотя бы на $m$. Абсолютные расстояния не фиксируются → эмбеддинг свободнее, чем в contrastive.

Soft-margin вариант (Hermans et al., 2017, «In Defense of the Triplet Loss»):

$$
L = \log\big(1 + e^{\,D_{ap} - D_{an}}\big) = \text{softplus}(D_{ap}-D_{an}),
$$

без гиперпараметра $m$ и без «мёртвой зоны» — лосс никогда не обнуляется полностью, продолжая слегка растягивать пары.

Классификация троек по трудности при margin $m$:

| Тип | Условие | Градиент | Что происходит |
|---|---|---|---|
| easy | $D_{an} > D_{ap} + m$ | 0 | бесполезна |
| semi-hard | $D_{ap} < D_{an} < D_{ap} + m$ | ненулевой, умеренный | «золотая середина» (FaceNet) |
| hard | $D_{an} < D_{ap}$ | максимальный | информативна, но часто это шум разметки/outlier → коллапс |

Отсюда вся культура майнинга (см. ниже). Batch-hard (Hermans): в батче $P\times K$ для каждого якоря берём **самый далёкий positive** и **самый близкий negative** — получаем $PK$ троек «умеренно трудных», потому что жёсткость ограничена батчем, а не всем датасетом.

### N-pair loss

Sohn, NeurIPS 2016. Вместо одного negative — по одному positive на каждый из $N$ классов батча; каждый anchor сравнивается сразу со всеми чужими positives:

$$
L = \frac{1}{N}\sum_{i=1}^N \log\Big(1 + \sum_{j\ne i} e^{\,z_i^\top z_j^+ - z_i^\top z_i^+}\Big)
= -\frac{1}{N}\sum_i \log\frac{e^{z_i^\top z_i^+}}{\sum_j e^{z_i^\top z_j^+}} .
$$

По сути softmax-cross-entropy по батчу — это ранняя форма InfoNCE (без температуры) с батчем «по 2 примера на класс». Майнинг встроен: log-sum-exp автоматически подчёркивает самые близкие negatives.

### Lifted Structured loss

Oh Song et al., CVPR 2016. Для каждой positive пары $(i,j)$ используется **все** negatives обоих концов пары через log-sum-exp (мягкий max по трудному негативу):

$$
L = \frac{1}{2|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}
\max\Big(0,\; D_{ij} + \log\Big(\sum_{(i,k)\in\mathcal{N}} e^{\,m - D_{ik}} + \sum_{(j,l)\in\mathcal{N}} e^{\,m - D_{jl}}\Big)\Big)^2 .
$$

Исторически важен как первый лосс, «поднимающий» (lift) батч в полную матрицу расстояний; сейчас его нишу заняли Multi-Similarity и SupCon.

---

## Batch-contrastive (InfoNCE-семейство)

### InfoNCE / NT-Xent

Oord et al. 2018 (CPC); NT-Xent — форма из SimCLR (Chen et al., 2020). Для якоря $i$ ровно один positive $i^+$ (вторая аугментация, парный текст и т. п.), все остальные элементы батча — negatives:

$$
L_i = -\log\frac{\exp(S_{i,i^+}/\tau)}{\sum_{j\ne i}\exp(S_{ij}/\tau)} .
$$

Температура $\tau$ — ключевой гиперпараметр: маленькая $\tau$ (0.05–0.1) резко увеличивает вес трудных негативов (это и есть «встроенный hard mining»), слишком маленькая — нестабильность и штраф за семантически близкие «ложные негативы». Качество сильно зависит от числа негативов → большие батчи (SimCLR: 4096) или очередь-память (MoCo).

Подробнее о self-supervised применении: [DINOv3 и SSL в CV](../dinov3-self-supervised-vision-transformer-and-2d-rope/README.md).

### SupCon

Khosla et al., NeurIPS 2020. Supervised обобщение NT-Xent: positives — все примеры того же класса в батче:

$$
L_i = -\frac{1}{|P_i|}\sum_{p\in P_i}\log\frac{\exp(S_{ip}/\tau)}{\sum_{a\ne i}\exp(S_{ia}/\tau)} .
$$

Это «triplet со всеми positives и всеми negatives сразу» без майнера. При $|P_i|=1$ вырождается в N-pair/InfoNCE. Хорошо работает как pre-training или замена CE при классификации, при наличии меток обычно стабильнее SimCLR-стиля.

---

## Proxy- и classification-based лоссы

Общая идея: заменить сравнение «пример–пример» на «пример–прототип класса». Прототипов $C$ штук, каждый шаг сравнивает батч со **всеми** классами — майнинг по датасету становится не нужен, сложность падает с $O(N^2)$/$O(N^3)$ до $O(N\cdot C)$, сходимость ускоряется на порядок.

### Normalized Softmax и Center loss

**NormSoftmax / NormFace** (Wang et al., 2017; Zhai & Wu, 2018): обычный softmax, но $\|z\|=\|W_c\|=1$ и масштаб $s$:

$$
L = -\log\frac{e^{\,s\cos\theta_y}}{\sum_{j=1}^{C} e^{\,s\cos\theta_j}} .
$$

Это baseline для всего angular-семейства: без margin, но уже даёт cosine-геометрию.

**Center loss** (Wen et al., ECCV 2016) — регуляризатор к обычному softmax: стягивает эмбеддинги к обучаемому центру своего класса:

$$
L = L_{softmax} + \frac{\lambda}{2}\sum_i \|z_i - c_{y_i}\|_2^2 .
$$

Улучшает внутриклассовую компактность, но не межклассовую разделимость; сегодня почти полностью вытеснен margin-лоссами.

### Proxy-NCA / Proxy-NCA++

Movshovitz-Attias et al., ICCV 2017. NCA (Neighbourhood Component Analysis), где соседи заменены на прокси:

$$
L = -\log\frac{\exp(-D(z, p_y))}{\sum_{j\ne y}\exp(-D(z, p_j))} .
$$

Proxy-NCA++ (Teh et al., 2020) добавляет нормировку, температуру и сумму по всем классам в знаменателе (включая $y$) — получается NormSoftmax с прокси и низкой температурой; на SOP/CUB даёт заметный прирост над оригиналом.

### Proxy-Anchor

Kim et al., CVPR 2020. Прокси выступает **якорем** и сравнивается со всеми примерами батча, а log-sum-exp внутри даёт взвешивание по трудности (как в Multi-Similarity):

$$
L = \frac{1}{|P^+|}\sum_{p\in P^+}\log\Big(1+\sum_{x\in X_p^+} e^{-\alpha (S(x,p) - \delta)}\Big)
+ \frac{1}{|P|}\sum_{p\in P}\log\Big(1+\sum_{x\in X_p^-} e^{\,\alpha (S(x,p) + \delta)}\Big),
$$

где $P^+$ — прокси, у которых есть positives в батче, $\alpha$ — масштаб (32), $\delta$ — margin (0.1). Ключевой трюк: **learning rate для прокси в 100 раз больше**, чем для backbone. Сходится за единицы эпох и на retrieval-бенчмарках был SOTA среди proxy-методов; де-факто дефолт, когда классов много, а batch маленький.

### SoftTriple

Qian et al., ICCV 2019. Один прототип на класс плохо описывает мультимодальные классы (сорт яблок в трёх ракурсах). SoftTriple заводит $K$ центров $w_c^k$ на класс и «мягко» выбирает ближайший:

$$
S'_{i,c} = \sum_{k=1}^{K}\frac{\exp(z_i^\top w_c^k/\gamma)}{\sum_{k'}\exp(z_i^\top w_c^{k'}/\gamma)}\; z_i^\top w_c^k,
\qquad
L = -\log\frac{e^{\lambda (S'_{i,y} - \delta)}}{e^{\lambda (S'_{i,y} - \delta)} + \sum_{j\ne y} e^{\lambda S'_{i,j}}} .
$$

Плюс регуляризатор, склеивающий лишние центры одного класса. Показано, что SoftTriple эквивалентен triplet loss с бесконечным числом троек, но без сэмплирования.

### SphereFace / CosFace / ArcFace

Классификационные лоссы с угловым margin для истинного класса. Подробный разбор геометрии и практики — в топике [ArcFace и angular-margin losses](../arcface-and-angular-margin-losses-for-identification/README.md); здесь только формулы для сравнения.

Общая форма:

$$
L = -\log\frac{e^{\,s\,\psi(\theta_y)}}{e^{\,s\,\psi(\theta_y)} + \sum_{j\ne y} e^{\,s\cos\theta_j}},
$$

| Лосс | $\psi(\theta_y)$ | Тип margin | Типичные $s, m$ |
|---|---|---|---|
| SphereFace (2017) | $\cos(m\theta_y)$ | multiplicative angular | $m=4$ (с annealing) |
| CosFace / AM-Softmax (2018) | $\cos\theta_y - m$ | additive cosine | $s=64,\ m=0.35$ |
| ArcFace (2019) | $\cos(\theta_y + m)$ | additive angular | $s=64,\ m=0.5$ |

Обобщённая форма (Deng et al.): $\psi(\theta) = \cos(m_1\theta + m_2) - m_3$. ArcFace даёт постоянный угловой зазор по всей окружности (геодезическое расстояние на сфере), поэтому обычно чуть лучше CosFace при той же $s$; CosFace проще в оптимизации.

### Sub-center ArcFace, CurricularFace, AdaFace, MagFace, Partial FC

Развитие ArcFace под реальные данные (шум разметки, низкое качество, миллионы классов):

- **Sub-center ArcFace** (Deng et al., ECCV 2020): $K$ суб-центров на класс, в логите используется $\max_k \cos\theta_{y,k}$. Шумные/чужие фото «уходят» во второстепенный суб-центр и не тянут главный. Рецепт для web-собранных датасетов: обучить с $K=3$, отбросить примеры далеко от доминирующего суб-центра, дообучить с $K=1$.
- **CurricularFace** (Huang et al., CVPR 2020): вес трудных негативов растёт по ходу обучения (curriculum) — на старте лосс похож на CosFace-mining-lite, к концу усиливает hard negatives: $N(t,\cos\theta_j) = \cos\theta_j\,(t + \cos\theta_j)$, если $\cos\theta_j > \cos(\theta_y+m)$.
- **AdaFace** (Kim et al., CVPR 2022): margin зависит от нормы признака $\|z\|$, которая коррелирует с качеством картинки. Для низкокачественных примеров margin уменьшается (чтобы не переобучаться на неразличимых лицах), для качественных — увеличивается: $m_{angle} = -m\cdot\hat{\|z\|},\ m_{add} = m\cdot\hat{\|z\|} + m$, где $\hat{\|z\|}$ — нормированная по батч-статистике норма.
- **MagFace** (Meng et al., CVPR 2021): margin $m(a)$ — возрастающая функция нормы $a=\|z\|$, плюс регуляризатор $g(a)$, который толкает норму вверх для «лёгких» примеров. В итоге норма эмбеддинга становится оценкой качества без отдельной модели.
- **Partial FC** (An et al., 2022): при $C\sim10^6$ классов softmax по всем прототипам не влезает в память; на каждом шаге берётся positive-классы батча + случайная выборка ($\sim10\%$) негативных классов. Качество почти не падает, память и время падают на порядок.

---

## Сводная таблица лоссов

| Лосс | Вход | Ключевые гиперпараметры | Майнер | Когда брать |
|---|---|---|---|---|
| Contrastive | пары | $m_{pos}, m_{neg}$ | желателен (`PairMarginMiner`) | простые бинарные «same/different», siamese |
| Triplet (hinge) | тройки | $m\approx0.2$–$0.3$ | **обязателен** (semi-hard / batch-hard / distance-weighted) | re-ID, лица (исторически), любые метки |
| Triplet (soft-margin) | тройки | — | batch-hard | re-ID (BoT-рецепт с ID-loss) |
| N-pair | батч 2/класс | — | встроен | замена triplet при малом $K$ |
| Lifted Structured | батч | $m$ | встроен | исторический baseline |
| Multi-Similarity | батч P×K | $\alpha=2,\beta=50,\lambda$ | `MultiSimilarityMiner` ($\epsilon=0.1$) | retrieval-бенчмарки, fine-grained |
| Circle | батч или прототипы | $\gamma, m$ | не нужен | универсальная замена triplet/ArcFace |
| InfoNCE / NT-Xent | пары аугментаций | $\tau=0.1$–$0.5$ | не нужен (большой батч / очередь) | SSL, image–text, нет меток |
| SupCon | батч с метками | $\tau\approx0.07$–$0.1$ | не нужен | supervised pre-training, мало классов |
| NormSoftmax | батч + прокси | $s$ | не нужен | baseline proxy |
| Proxy-NCA(++) | батч + прокси | $\tau$ | не нужен | много классов, мало данных на класс |
| Proxy-Anchor | батч + прокси | $\alpha=32,\delta=0.1$, lr прокси ×100 | не нужен | дефолт для retrieval с известными классами |
| SoftTriple | батч + $K$ прокси/класс | $K, \gamma, \lambda, \delta$ | не нужен | мультимодальные классы |
| CosFace / ArcFace | батч + прокси | $s=64,\ m=0.35/0.5$ | не нужен | face / SKU / re-ID identification, open-set пороги |
| Sub-center ArcFace | батч + $K$ прокси/класс | $K=3$ | не нужен | шумная разметка |
| AdaFace / MagFace | батч + прокси | $m, h$ / $l_a,u_a$ | не нужен | низкое качество изображений |

---

## Майнеры: зачем они нужны и какие бывают

### Почему случайные пары/триплеты не работают

При $C$ классах и случайных тройках уже после первой эпохи $>95\%$ троек — easy (лосс $=0$). Градиент в среднем по батчу почти нулевой, обучение «замерзает». Дополнительно триплетов $O(N^3)$, перебрать нельзя. Майнер решает обе проблемы: отбирает из батча (или из внешней памяти) только те пары/тройки, где лосс ненулевой, но при этом отбрасывает «слишком трудные» — часто это ошибки разметки, дубликаты, outliers, на которых hard mining схлопывает эмбеддинги в точку.

Ментальная модель: майнер — это **curriculum**. Semi-hard в начале, hard ближе к концу; либо лосс с log-sum-exp/температурой, который делает это плавно.

### Каталог майнеров

Имена — как в `pytorch-metric-learning` (PML), где майнер — отдельный модуль, возвращающий индексы пар/троек.

| Майнер | Что возвращает | Правило отбора | Комментарий |
|---|---|---|---|
| `TripletMarginMiner(margin, type)` | тройки | `easy`: $D_{an} - D_{ap} > m$; `semihard`: $0 < D_{an}-D_{ap} < m$; `hard`: $D_{an} < D_{ap}$; `all`: все с ненулевым лоссом | Semi-hard — рецепт FaceNet. `hard` использовать только на чистых данных |
| `BatchHardMiner` | по одной тройке на якорь | hardest positive + hardest negative внутри батча | Hermans 2017; требует P×K; де-факто стандарт в re-ID |
| `BatchEasyHardMiner(pos_strategy, neg_strategy)` | пары/тройки | любая комбинация `easy / semihard / hard / all` отдельно для positives и negatives | Обобщение двух предыдущих; например `pos=easy, neg=semihard` — щадящий режим для шумных меток |
| `MultiSimilarityMiner(epsilon)` | пары | negatives с $S_{an} > \min_p S_{ap} - \epsilon$; positives с $S_{ap} < \max_n S_{an} + \epsilon$ | Родной майнер MS-loss; работает и с Contrastive/Triplet |
| `PairMarginMiner(pos_margin, neg_margin)` | пары | positives с $D > m_{pos}$, negatives с $D < m_{neg}$ | Зеркало contrastive loss: убирает пары с нулевым лоссом |
| `DistanceWeightedMiner(cutoff, nonzero_loss_cutoff)` | тройки | negatives сэмплируются с весом $\propto 1/q(D)$, где $q(D)\propto D^{d-2}(1-D^2/4)^{(d-3)/2}$ — плотность расстояний на сфере | Wu et al. 2017, «Sampling Matters». Равномерно покрывает диапазон расстояний вместо пика около $\sqrt2$; устойчивее hard mining |
| `AngularMiner(angle)` | тройки | тройки с углом между $(z_p - z_a)$ и $(z_n - z_a)$ меньше порога | Для Angular loss (Wang et al. 2017) |
| `HDCMiner(filter_percentage)` | пары | каскад: верхние $x\%$ самых трудных пар | Hard-aware Deeply Cascaded (Yuan et al. 2017) — для ансамбля голов разной глубины |
| `UniformHistogramMiner` | пары | равномерно по гистограмме расстояний | Диагностический/регуляризирующий; снимает перекос в лёгкие пары |
| `EmbeddingsAlreadyPackagedAsTriplets` | тройки | батч уже сформирован как $(a,p,n)$ | Когда тройки собирает внешний даталоадер (например, с offline-майнингом по всему датасету) |

**Offline vs online mining.** Всё выше — online (внутри батча, дёшево, «свежие» эмбеддинги). Offline mining — периодический прогон датасета, построение kNN по эмбеддингам и сборка троек из глобально трудных пар; сильнее, но эмбеддинги устаревают между прогонами. Обычно: online majority + offline раз в N эпох для узкого набора «конфликтных пар».

### Сэмплер батча P×K и cross-batch memory

- **`MPerClassSampler` / P×K**: $P$ классов по $K$ примеров ($32\times4$, $16\times8$…). Без него в случайном батче почти нет positives, и любой майнер бесполезен. $K\ge4$ нужен batch-hard (иначе «hardest positive» — единственный).
- **Cross-Batch Memory, XBM** (Wang et al., CVPR 2020): очередь эмбеддингов прошлых батчей ($10^3$–$10^5$ штук) как источник дополнительных негативов для pair-based лоссов (MS, Contrastive). Опирается на «медленный дрейф» признаков: старые эмбеддинги ещё валидны несколько сотен итераций. Даёт эффект большого батча на одной GPU.
- **MoCo-очередь** — тот же приём для InfoNCE, но с momentum-энкодером, чтобы старые ключи не расходились с текущими.

---

## Таблица «лосс ↔ майнер»

| Лосс | Рекомендованный майнер | Альтернативы | Почему |
|---|---|---|---|
| Triplet (hinge) | `TripletMarginMiner(semihard)` при чистых метках, `BatchHardMiner` при P×K | `DistanceWeightedMiner` (стабильнее), `BatchEasyHardMiner(pos=hard, neg=semihard)` | Без майнера ~все тройки easy; чистый hard → коллапс на шуме |
| Triplet (soft-margin) | `BatchHardMiner` | — | Рецепт Hermans/BoT для re-ID |
| Contrastive | `PairMarginMiner` | `MultiSimilarityMiner`, XBM | Отбрасывает пары за margin; XBM даёт больше негативов |
| Multi-Similarity | `MultiSimilarityMiner(ε=0.1)` | XBM | Так задуман в статье: майнер = шаг 1, лосс = шаг 2 |
| Circle | без майнера | `MultiSimilarityMiner` | Самовзвешивание $\alpha_p,\alpha_n$ уже акцентирует трудные пары |
| N-pair / Lifted | без майнера | — | Log-sum-exp по всему батчу = встроенный мягкий mining |
| InfoNCE / NT-Xent | без майнера; большой батч или MoCo-очередь | температура $\tau$ как «регулятор жёсткости» | Softmax по батчу взвешивает негативы экспоненциально; явный hard mining ухудшает SSL из-за ложных негативов |
| SupCon | без майнера | — | То же, positives уже все в батче |
| NormSoftmax / Proxy-NCA / Proxy-Anchor / SoftTriple | без майнера | — | Каждый шаг сравнивает со всеми прокси; трудные классы взвешены softmax/LSE |
| CosFace / ArcFace | без майнера | CurricularFace (встроенный curriculum), Partial FC (сэмплинг классов, не примеров) | Softmax по всем $C$ прототипам; «майнинг» происходит на уровне выбора margin и качества данных |
| Sub-center ArcFace / AdaFace / MagFace | без майнера | — | Устойчивость к шуму/качеству встроена в margin |
| Center loss | без майнера | — | Регуляризатор к CE |

Правило: **майнер нужен лоссам, которые считают hinge по отдельным парам/тройкам** (Contrastive, Triplet). Лоссам с log-sum-exp/softmax по батчу или по прототипам майнер либо не нужен, либо вреден (двойная фокусировка на шуме).

---

## Когда что использовать: decision guide

1. **Есть фиксированный набор классов с метками, классов много ($10^3$–$10^6$), примеров на класс мало–средне** (лица, SKU, person/vehicle re-ID, товарные каталоги):
   → **ArcFace / CosFace** (или Circle в proxy-форме). Шумная разметка → **Sub-center ArcFace**; низкое качество снимков → **AdaFace/MagFace**; миллионы ID → **Partial FC**. Затем опциональный fine-tune triplet/MS с hard negatives на целевом домене. Open-set пороги — см. [ArcFace-топик](../arcface-and-angular-margin-losses-for-identification/README.md).
2. **Retrieval с известными классами, но нужны компактные батчи и быстрая сходимость** (SOP, fashion, fine-grained):
   → **Proxy-Anchor** или **Multi-Similarity + MS-miner** (+ XBM, если батч маленький).
3. **Person re-ID** (стандартный BoT-рецепт):
   → **Triplet soft-margin + BatchHardMiner + ID cross-entropy** (с BNNeck), P×K = 16×4, center loss как опциональный регуляризатор.
4. **Классов мало (десятки), а важно различать тонкие детали** (15 сортов яблок):
   → **SupCon** или **Multi-Similarity + MS-miner** с P×K. ArcFace на десятках классов тоже работает, но при малом $C$ прототипы плохо покрывают вариативность → чаще берут pair-based. Практический разбор — [fine-grained топик](../contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md).
5. **Нет меток классов, есть только пары «same/different»** (siamese-верификация, дубликаты):
   → **Contrastive + PairMarginMiner** или **Triplet + semi-hard** (тройки собираются из пар).
6. **Нет меток вообще (SSL) или multi-modal пары (image–text)**:
   → **InfoNCE / NT-Xent** с большим батчем или MoCo-очередью; $\tau$ вместо майнера.
7. **Классы постоянно добавляются (обновляемый ассортимент, новые ID)**:
   → это **не** аргумент против ArcFace. Голова с прототипами $W$ нужна только на обучении; на инференсе любой metric-learning-эмбеддинг используется одинаково — галерея эталонов + kNN, и новый класс добавляется в индекс без retrain (так работает face recognition на невиданных людях). Разница проявляется только при **дообучении**: pair-based лоссы потребляют новые данные без головы и словаря классов, proxy-лоссам нужно расширить $W$ (новые строки инициализируются средним эмбеддингом класса) и class-balanced sampler для малочисленных новых классов. Выбирать pair-based стоит, если классов на train мало (сотни) или нужен continual-поток пар из продакшена; при тысячах классов — ArcFace/CosFace + периодический retrain. Подробнее — раздел «Обновляемый набор классов» ниже.
8. **Шумная разметка / много дубликатов**:
   → избегать `hard`-майнеров (они находят именно ошибки); брать semi-hard, distance-weighted или proxy-лоссы (Sub-center ArcFace, Proxy-Anchor).
9. **Нужны калиброванные пороги для verification (1:1)**:
   → margin-лоссы (ArcFace/CosFace/Circle) дают более компактные кластеры и стабильнее пороги TAR@FAR, чем triplet.

Обобщённо: proxy-лоссы — дефолт, когда классов на train много; pair/triplet — когда классов мало или меток классов нет; InfoNCE — когда меток нет вовсе. Появление новых классов после обучения само по себе выбор лосса не меняет.

---

## Обновляемый набор классов: ArcFace vs pair-based

Типовая задача: распознавание товаров в сети магазинов, ассортимент обновляется еженедельно. Что здесь важно:

- **На инференсе оба семейства эквивалентны.** Модель — экстрактор эмбеддингов, классификационная голова ArcFace отбрасывается. Новый SKU = эталонные фото → эмбеддинги → ANN-индекс. Retrain не нужен.
- **Обобщение на невиданные классы** зависит от числа классов на train, а не от лосса как такового: ArcFace при $C\sim10^3$–$10^5$ переносится на новые классы отлично (Product10K, Shopee Product Matching, Google Landmark — angular-margin эмбеддинги + kNN), при $C$ в десятки–сотни эмбеддинг «заучивает» разделение именно этих классов, и pair-based/SupCon обычно переносятся лучше.
- **Retrain**: pair-based — просто докинуть данные; ArcFace — расширить $W$ новыми строками (init средним эмбеддингом класса), warm-start от старых весов, class-balanced sampler (у новых SKU мало фото, иначе их прототипы недообучены).
- **Дрейф эмбеддингов** после любого retrain требует пересчёта индекса; для товаров это дёшево (эталонов мало). Если нельзя — Backward-Compatible Training (Shen et al., CVPR 2020).
- **Рецепт для ритейла**: ArcFace/CosFace (Sub-center при шумном каталоге) на всём каталоге, опционально + triplet/MS на тех же эмбеддингах; деплой через индекс с open-set порогом и «unknown» на ревью; мониторинг Recall@1 отдельно на новых SKU; retrain по расписанию или по триггеру «новый SKU коллидирует со старым».

---

## Типовые гиперпараметры

| Настройка | Значение |
|---|---|
| Triplet margin (L2-нормированные эмбеддинги) | $0.2$ (FaceNet), $0.3$ (re-ID); soft-margin — без $m$ |
| ArcFace | $s=64,\ m=0.5$; CosFace $m=0.35$; для мелких датасетов $s=30$ |
| SupCon / NT-Xent $\tau$ | SupCon $0.07$–$0.1$; SimCLR $0.1$–$0.5$; CLIP — обучаемая, старт $0.07$ |
| Multi-Similarity | $\alpha=2,\ \beta=50,\ \lambda=1$ (PML) или $0.5$ (статья); miner $\epsilon=0.1$ |
| Proxy-Anchor | $\alpha=32,\ \delta=0.1$; lr прокси $=100\times$ lr backbone |
| Circle | $\gamma=80$ (pair) / $256$ (class), $m=0.25$–$0.4$ |
| P×K сэмплер | $P=32,\ K=4$ (батч 128); re-ID $16\times4$ |
| XBM | размер очереди $10^3$–$10^5$, включать после «прогрева» ~1000 итераций |
| Размерность эмбеддинга | 128–512 (лица 512; retrieval 128–256 для ANN) |

---

## Минимальный пример на pytorch-metric-learning

```python
from pytorch_metric_learning import losses, miners, samplers, distances, reducers

# 1) Triplet + semi-hard mining (классика FaceNet)
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
loss_fn = losses.TripletMarginLoss(margin=0.2, distance=distances.LpDistance(normalize_embeddings=True))

# 2) Multi-Similarity + собственный майнер
ms_miner = miners.MultiSimilarityMiner(epsilon=0.1)
ms_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=1)

# 3) Proxy-Anchor — майнер не нужен, прокси учатся с lr ×100
pa_loss = losses.ProxyAnchorLoss(num_classes=C, embedding_size=D, margin=0.1, alpha=32)
optimizer = torch.optim.AdamW([
    {"params": model.parameters(), "lr": 1e-4},
    {"params": pa_loss.parameters(), "lr": 1e-2},
])

# 4) ArcFace — тоже без майнера
arc_loss = losses.ArcFaceLoss(num_classes=C, embedding_size=D, margin=28.6, scale=64)  # margin в градусах ≈ 0.5 rad

# Сэмплер P×K, без него майнеры бесполезны
sampler = samplers.MPerClassSampler(labels, m=4, batch_size=128)

for x, y in loader:  # loader построен на sampler
    z = model(x)
    hard = miner(z, y)              # индексы троек
    loss = loss_fn(z, y, hard)      # или ms_loss(z, y, ms_miner(z, y)); pa_loss(z, y); arc_loss(z, y)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

Реализация triplet + semi-hard mining без библиотек — в [скрипте fine-grained топика](../contrastive-and-metric-learning-for-fine-grained-visual-recognition/scripts/01_metric_learning_triplet_synthetic.py).

---

## Частые ошибки и анти-паттерны

- Triplet loss с случайными тройками и без P×K-сэмплера → лосс быстро уходит в ноль, embedding не учится.
- `hard`-майнер на шумных данных → коллапс эмбеддингов (все точки в одну) или «обучение на ошибках разметки».
- Майнер поверх ArcFace/Proxy-Anchor/SupCon «для надёжности» → двойная фокусировка на шуме и потеря стабильности.
- Слишком маленькая $\tau$ в InfoNCE при малом батче → нестабильность; слишком большая → нет градиента от трудных негативов.
- Proxy-лоссы с одинаковым lr для прокси и backbone → прокси не успевают «разбежаться», медленная сходимость.
- Не нормировать эмбеддинги при cosine-лоссах и одновременно использовать L2-margin → margin теряет смысл масштаба.
- Сравнивать лоссы по classification accuracy вместо Recall@K / mAP / TAR@FAR (см. [ROC AUC](../roc-curve-and-roc-auc/README.md)).
- Забывать, что «известный набор классов» на train ≠ open-set на inference: пороги калибруются отдельно.

---

## Источники

- Внутри knowledge-book:
  - `./topics/arcface-and-angular-margin-losses-for-identification/README.md`
  - `./topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md`
  - `./topics/classification-losses-cross-entropy-focal-loss/README.md`
  - `./topics/embeddings-and-embedding-matrix/README.md`
  - `./topics/roc-curve-and-roc-auc/README.md`
- Pair / triplet / tuple:
  - Hadsell, Chopra, LeCun. Dimensionality Reduction by Learning an Invariant Mapping (CVPR 2006) — contrastive loss
  - Schroff et al. FaceNet: A Unified Embedding for Face Recognition and Clustering (CVPR 2015) — triplet + semi-hard
  - Hermans, Beyer, Leibe. In Defense of the Triplet Loss for Person Re-Identification (2017) — batch-hard, soft-margin
  - Sohn. Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NeurIPS 2016)
  - Oh Song et al. Deep Metric Learning via Lifted Structured Feature Embedding (CVPR 2016)
  - Wu et al. Sampling Matters in Deep Embedding Learning (ICCV 2017) — distance-weighted sampling
  - Wang et al. Multi-Similarity Loss with General Pair Weighting (CVPR 2019)
  - Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)
  - Wang et al. Cross-Batch Memory for Embedding Learning (CVPR 2020) — XBM
- Batch-contrastive:
  - Oord, Li, Vinyals. Representation Learning with Contrastive Predictive Coding (2018) — InfoNCE
  - Chen et al. A Simple Framework for Contrastive Learning of Visual Representations (ICML 2020) — NT-Xent
  - Khosla et al. Supervised Contrastive Learning (NeurIPS 2020)
- Proxy / classification-based:
  - Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition (ECCV 2016) — center loss
  - Movshovitz-Attias et al. No Fuss Distance Metric Learning using Proxies (ICCV 2017) — Proxy-NCA
  - Teh, DeVries, Taylor. ProxyNCA++ (ECCV 2020)
  - Kim et al. Proxy Anchor Loss for Deep Metric Learning (CVPR 2020)
  - Qian et al. SoftTriple Loss: Deep Metric Learning Without Triplet Sampling (ICCV 2019)
  - Liu et al. SphereFace (CVPR 2017); Wang et al. CosFace (CVPR 2018); Deng et al. ArcFace (CVPR 2019)
  - Deng et al. Sub-center ArcFace (ECCV 2020); Huang et al. CurricularFace (CVPR 2020)
  - Meng et al. MagFace (CVPR 2021); Kim et al. AdaFace (CVPR 2022); An et al. Partial FC (CVPR 2022)
- Обзоры и инструменты:
  - Musgrave, Belongie, Lim. A Metric Learning Reality Check (ECCV 2020) — честное сравнение лоссов
  - Shen et al. Towards Backward-Compatible Representation Learning (CVPR 2020) — retrain без пересчёта галереи
  - Roth et al. Revisiting Training Strategies and Generalization Performance in Deep Metric Learning (ICML 2020)
  - pytorch-metric-learning: <https://kevinmusgrave.github.io/pytorch-metric-learning/> (losses, miners, samplers)
