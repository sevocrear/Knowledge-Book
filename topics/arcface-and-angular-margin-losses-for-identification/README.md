## Table of Contents

- Краткий абстракт и объяснение для 5-летнего ребёнка
- Что такое ArcFace и зачем он нужен
- Математика ArcFace: additive angular margin
- Геометрическая интуиция на гиперсфере
- ArcFace vs CosFace vs SphereFace
- Как использовать ArcFace для идентификации лиц
- Как использовать ArcFace для идентификации SKU и товаров
- Open-set режим и выбор порогов
- Современные датасеты для metric learning и re-ID
  - Face recognition
  - Person re-identification
  - Product/SKU retrieval
  - Vehicle re-identification
- Практический протокол обучения и оценки
- Частые ошибки и анти-паттерны
- References

---

## Краткий абстракт и объяснение для 5-летнего ребёнка

**Кратко по-взрослому.**  
ArcFace - это функция потерь для обучения эмбеддингов, где классы разделяются не только по косинусному сходству, но и с дополнительным угловым зазором (margin). За счет этого внутри-классовая компактность и межклассовая разделимость становятся лучше, особенно в задачах идентификации (1:1 verification и 1:N identification). На практике ArcFace стал де-факто стандартом для face recognition и часто используется как сильная база в re-ID и product retrieval.

**Как объяснить 5-летнему ребёнку.**  
Представь, что у каждого человека или товара есть точка на круглой карте. ArcFace учит ставить точки так, чтобы "свои" стояли очень кучно, а "чужие" - с заметным промежутком. Тогда легче понять, кто есть кто.

**Визуализация (Manim):** эмбеддинг и веса классов на единичной сфере, угол $\theta$, угловой margin $m$, идея $s\cos(\theta_y+m)$.

<video src="./assets/visualizations/arcface_angular_margin.mp4" controls muted loop playsinline width="100%"></video>

*Fallback GIF:* `![](./assets/visualizations/arcface_angular_margin.gif)`

---

## Что такое ArcFace и зачем он нужен

В обычной классификации с softmax модель может давать высокую точность на train/val, но эмбеддинги оказываются неустойчивыми для real-world поиска и verification. ArcFace решает это, потому что:

- работает в нормированном пространстве (эмбеддинги и class weights на единичной сфере);
- оптимизирует именно угловую геометрию, близкую к cosine similarity;
- добавляет явный запас разделимости для правильного класса.

Итог: более стабильные эмбеддинги для open-set сценариев, где классы могут появляться после обучения.

---

## Математика ArcFace: additive angular margin

Пусть $z_i = f_\theta(x_i)$ - эмбеддинг, $W_j$ - вес (прототип) класса $j$. В ArcFace обычно используют:

- нормировку: $z_i_2 = 1$, $W_j_2 = 1$;
- масштаб логитов $s$;
- угловой margin $m$ для истинного класса $y_i$.

Тогда логиты имеют вид:

$$
\text{logit}*{y_i} = s \cdot \cos(\theta*{y_i} + m), \quad
\text{logit}_{j \neq y_i} = s \cdot \cos(\theta_j),
$$

где $\theta_j$ - угол между $z_i$ и $W_j$. Лосс:

$$
L = -\frac{1}{N}\sum_{i=1}^N
\log \frac{e^{s\cos(\theta_{y_i}+m)}}
{e^{s\cos(\theta_{y_i}+m)} + \sum_{j\neq y_i}e^{s\cos(\theta_j)}}.
$$

Интуитивно: чтобы классифицировать sample правильно, модель должна "довернуть" его ближе к центру своего класса сильнее, чем в обычном softmax.

---

## Геометрическая интуиция на гиперсфере

После L2-нормировки все эмбеддинги лежат на сфере, и важен угол между векторами:

- меньше угол -> больше cosine similarity;
- ArcFace увеличивает требуемый угол-отступ между классами;
- границы решений становятся более "строгими", что помогает в hard-парах.

Для идентификации это критично: ошибки часто возникают именно между визуально похожими идентичностями (люди с похожим лицом, похожие SKU-упаковки).

---

## ArcFace vs CosFace vs SphereFace

- **SphereFace**: multiplicative angular margin ($\cos(m\theta)$), исторически важен, но обучение менее стабильное.
- **CosFace**: additive cosine margin ($\cos(\theta)-m$), проще в оптимизации.
- **ArcFace**: additive angular margin ($\cos(\theta + m)$), обычно лучший баланс геометрической интерпретируемости и качества.

На практике ArcFace часто берут как baseline; CosFace - как более "легкую" альтернативу при жестких вычислительных ограничениях.

---

## Как использовать ArcFace для идентификации лиц

Типичный production pipeline:

1. **Backbone** (ResNet/IR-ResNet/MobileFaceNet/ViT-variant) выдает эмбеддинг.
2. **Обучение** с ArcFace на большом ID-супервизированном датасете.
3. **Inference**:
  - face detect + align;
  - embedding extraction;
  - cosine similarity с gallery/prototypes.
4. **Режимы**:
  - verification (1:1): сравнить пару, принять/отклонить по порогу;
  - identification (1:N): найти top-K в gallery.
5. **Калибровка порога** по TAR@FAR на валидации, близкой к продовым условиям.

Важно для лиц:

- alignment (5-point или dense landmarks) влияет на качество не меньше, чем выбор лосса;
- следить за demographic/domain shift и quality bias (освещение, blur, pose);
- держать отдельные пороги для разных operational policy (дверной доступ, KYC, watchlist).

---

## Как использовать ArcFace для идентификации SKU и товаров

ArcFace хорошо работает для SKU, когда задача ближе к instance/fine-grained retrieval, чем к "грубому" multi-class.

Рекомендованный процесс:

1. **Определить единицу идентичности**:
  - SKU-level (один ID на товар);
  - variant-level (объем, вкус, упаковка как отдельные ID).
2. **Собрать hard negatives**:
  - visually similar SKU той же категории;
  - private label vs брендовые аналоги.
3. **Обучать эмбеддинг** с ArcFace или гибридом (ArcFace + retrieval fine-tuning).
4. **Использовать ANN-индекс** (HNSW/IVF/PQ) для 1:N поиска в каталоге.
5. **Open-set fallback**:
  - если max similarity ниже порога -> "unknown SKU";
  - отправлять на human review / OCR / barcode fallback.

Практический паттерн для ритейла: двухступенчатая схема

- stage 1: coarse retrieval по эмбеддингам (top-50);
- stage 2: rerank (cross-encoder/siamese head/правила по метаданным).

---

## Open-set режим и выбор порогов

ArcFace обучается как классификационный лосс, но обычно применяется в open-set. Поэтому пороги - это отдельная инженерная задача.

Что делать:

- валидировать порог не только на "чистом" in-domain, но и на unknown/novel классах;
- мониторить две кривые:
  - false accept (чужой принят как свой),
  - false reject (свой отклонен);
- выбирать порог по бизнес-стоимости ошибок, а не только по EER.

Для SKU часто применяют class-conditional или category-conditional thresholds (например, напитки и косметика имеют разную "плотность" признаков).

---

## Современные датасеты для metric learning и re-ID

Ниже - практичный список: "классика + актуальные бенчмарки", которые реально используют для обучения или честной оценки.

### Face recognition

- **MS1MV3 / MS1M-RetinaFace clean** - популярный large-scale train set для ArcFace-подобных пайплайнов.
- **Glint360K** - большой набор ID, часто используется для pretraining.
- **WebFace42M** - более современный крупный датасет для обучения face embeddings.
- **VGGFace2** - сильный датасет по вариативности поз/возраста.
- **IJB-B / IJB-C** - де-факто для сложной evaluation (verification/identification в "грязных" условиях).
- **AgeDB-30, CFP-FP, CALFW/CPLFW** - узкоспециализированные evaluation-наборы.

### Person re-identification

- **Market-1501** - классический baseline benchmark.
- **DukeMTMC-reID** - долго использовался как второй ключевой benchmark (часто сохраняют для сравнимости со старыми работами).
- **MSMT17** - более сложный и ближе к real-world по камерам/условиям.
- **CUHK03** - исторически важный re-ID benchmark.
- **Occluded-DukeMTMC, Partial-REID** - для сценариев окклюзий/частичных наблюдений.
- **LaST** и новые long-tail/large-scale re-ID бенчмарки - полезны для современных исследовательских setup.

### Product/SKU retrieval

- **Stanford Online Products (SOP)** - базовый metric learning benchmark для товаров.
- **In-Shop Clothes Retrieval (DeepFashion)** - retrieval для fashion SKU.
- **DeepFashion2** - detection + retrieval/landmarks для fashion pipeline.
- **Product10K** - реальный e-commerce сценарий поиска похожих товаров.
- **RPC (Retail Product Checkout)** - retail shelf/checkout сценарии, особенно полезен для domain shift.
- **SKU110K** - в первую очередь detection на полках, но полезен как источник hard visual conditions.
- **ABO (Amazon Berkeley Objects)** - multi-view product data, удобен для instance-level matching.

### Vehicle re-identification

- **VeRi-776** - базовый vehicle re-ID benchmark.
- **VehicleID** - классический датасет для идентификации машин.
- **VERI-Wild** - крупнее и сложнее, ближе к real-world вариативности.

---

## Практический протокол обучения и оценки

Универсальный recipe для ArcFace в metric/re-ID задачах:

1. Backbone pretrain (ImageNet/SSL) -> ArcFace fine-tuning.
2. Batch sampler с классовым балансом (P x K).
3. Сильный, но реалистичный augment policy (без уничтожения fine-grained признаков).
4. Валидация:
  - retrieval: Recall@K, mAP;
  - verification: ROC-AUC, TAR@FAR, EER.
5. Domain-specific threshold calibration.
6. Deployment через embedding + ANN + rerank.

Если данных мало, хорошо работает transfer:

- pretrained ArcFace backbone -> адаптация head/projection;
- плюс hard-negative mining на целевом домене.

---

## Частые ошибки и анти-паттерны

- Оптимизировать только classification accuracy и не смотреть retrieval/verification метрики.
- Случайный split с leakage по камере/магазину/поставке.
- Слишком агрессивные аугментации, стирающие идентичность.
- Отсутствие unknown-классов в валидации open-set режима.
- Один глобальный порог для всех категорий SKU без калибровки.

---

## References

- Внутри knowledge-book:
  - `./topics/contrastive-and-metric-learning-for-fine-grained-visual-recognition/README.md`
  - `./topics/embeddings-and-embedding-matrix/README.md`
  - `./topics/roc-curve-and-roc-auc/README.md`
  - `./topics/how-models-predict-confidence-and-calibration/README.md`
  - `./topics/dinov3-self-supervised-vision-transformer-and-2d-rope/README.md`
- Базовые статьи:
  - ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR 2019)
  - CosFace: Large Margin Cosine Loss for Deep Face Recognition (CVPR 2018)
  - SphereFace: Deep Hypersphere Embedding for Face Recognition (CVPR 2017)

