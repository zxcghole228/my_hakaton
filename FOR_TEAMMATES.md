# Ветка `minicooper1`: полный handoff для команды

Актуально на **20 августа 2026 года**.

Этот файл — основная карта ветки `minicooper1`. Он объясняет, что уже сделано,
как устроен репозиторий, где находятся baseline, обученная E5-модель, метрики и
готовый submission, а также что нужно восстановить локально после клонирования.

## 1. Коротко: текущее состояние

В ветке сохранены baseline и два независимых готовых E5 solution:

1. **Исходный CrossEncoder baseline организаторов** — оставлен в корне
   репозитория без изменения его интерфейса запуска.
2. **E5-small V2 submission** — находится в `solutions/e5_small_v2/`, полностью
   проверен и уже дал Public LB `0.4838757641`.
3. **E5-small V3 Fashion Specialist** — находится в
   `solutions/e5_small_v3_hybrid/`. Это готовый offline hybrid pipeline с
   отдельной fashion-моделью и routing; он прошёл локальный smoke-test.

Проверенный V2 fallback:

- backbone: `intfloat/multilingual-e5-small`;
- fine-tuned checkpoint: step `30000`;
- обучение: LLM stage-A only;
- preprocessing: V2;
- `MAX_LEN=192`;
- Full LLM group holdout Macro PR-AUC: **0.786482**;
- готовый локальный архив: `e5_small_macro_v2_30k_submission.zip`;
- формат запуска совместим с аргументами соревнования
  `--items_path`, `--matches_path`, `--output_path`;
- модель и tokenizer загружаются полностью offline.

Новый готовый V3 solution:

- specialist инициализирован из V2 checkpoint;
- исправлены exact fashion category names;
- добавлены variant-сигналы и hard-example mining;
- Best fast hybrid Macro PR-AUC: `0.790609`;
- Hybrid Full LLM Macro PR-AUC: **`0.790182`**;
- прирост относительно V2: **`+0.003700`**;
- routing использует specialist для обуви, одежды и ювелирных изделий, а base
  для галантереи.
- готовый локальный архив: `e5_small_macro_v3_hybrid_submission.zip`;
- формат запуска совпадает с V2 и интерфейсом соревнования;
- модель и tokenizer загружаются полностью offline;
- leaderboard score V3 пока не получен.

Код решения, ноутбуки, отчёты и метрики находятся на GitHub. Данные, веса,
checkpoint и ZIP-архивы намеренно не коммитятся из-за размера.

## 2. Что уже сделано

К текущему моменту выполнено следующее:

1. Сохранён и проверен исходный baseline соревнования.
2. Репозиторий разделён на baseline, notebooks, experiments, models, docs и
   финальный submission pipeline.
3. Подготовлен и выполнен эксперимент E5-small Macro LLM Stage-A V2.
4. Выбран лучший checkpoint шага 30 000.
5. Checkpoint восстановлен и проверен повторным расчётом fast validation:
   значение воспроизвелось без расхождения (`delta=0.0`).
6. Посчитаны полный LLM group holdout, manual diagnostic holdout и метрики по
   категориям.
7. Модель экспортирована в Hugging Face-совместимый offline-формат.
8. В точности перенесён preprocessing V2 из тренировочного ноутбука.
9. Собран автономный V2 solution под интерфейс организаторов.
10. Проведён smoke-test на локальных parquet.
11. Проверены схема `id1,id2,predict`, полнота строк, порядок ID, отсутствие
    NaN/Inf и диапазон вероятностей `[0, 1]`.
12. Собран чистый submission ZIP без кэшей, тестовых данных, `.DS_Store` и
    лишнего корневого каталога.
13. Код E5 submission запушен в ветку `minicooper1`.
14. Получен Public LB V2 submission: `0.4838757641`.
15. Обучен V3 fashion specialist с исправленными категориями, variant-сигналами
    и hard mining.
16. Выполнена полная hybrid validation V3: `0.790182347`, delta к V2
    `+0.003700013`.
17. Сохранены best specialist, полный resume checkpoint и hybrid export с base,
    specialist, tokenizer и routing.
18. Реализован отдельный V3 inference pipeline с точными variant-сигналами и
    маршрутизацией по категории `id1`.
19. V2 и V3 разнесены по `solutions/e5_small_v2/` и
    `solutions/e5_small_v3_hybrid/`, поэтому они не перезаписывают друг друга.
20. V3 прошёл smoke-test на 100 парах: CSV валиден, все строки полны, NaN/Inf
    нет, base-строки совпадают с отдельным V2 pipeline.
21. Собран и проверен чистый V3 submission ZIP.

Для V3 пока отсутствует только leaderboard score и full-size benchmark в
контейнере соревнования.

## 3. Полное дерево ветки

Обозначения:

- **Git** — файл хранится на GitHub;
- **local** — файл есть в подготовленной локальной копии, но игнорируется Git;
- **pending** — материал ещё нужно добавить.

```text
matching-baseline-submit/
├── FOR_TEAMMATES.md                         # Git: этот handoff
├── README.md                                # Git: краткое описание проекта
├── .gitignore                               # Git: защита данных и весов
│
├── run.py                                   # Git: entry point исходного baseline
├── metadata.json                            # Git: metadata исходного baseline
├── baseline_logreg_l12.joblib               # Git: небольшой baseline classifier
├── src/
│   └── utils.py                             # Git: baseline preprocessing/inference
│
├── solutions/                               # независимые готовые E5 submissions
│   ├── README.md                            # Git: индекс V2/V3
│   ├── e5_small_v2/                         # проверенный V2 fallback
│   │   ├── README.md                        # Git: запуск и метрики V2
│   │   ├── metadata.json                    # Git: metadata V2
│   │   ├── run.py                           # Git: entry point V2
│   │   ├── src/                             # Git: preprocessing/inference V2
│   │   └── models/                          # local, ignored
│   │       └── e5_small_macro_v2_30k/
│   │           ├── config.json
│   │           ├── model.safetensors
│   │           ├── tokenizer.json
│   │           └── tokenizer_config.json
│   └── e5_small_v3_hybrid/                  # готовый V3 hybrid
│       ├── README.md                        # Git: запуск, routing, проверки
│       ├── metadata.json                    # Git: metadata V3
│       ├── run.py                           # Git: entry point V3
│       ├── src/                             # Git: V2 text + V3 signals/routing
│       └── models/                          # local, ignored
│           └── e5_small_v3_hybrid/
│               ├── base_model/
│               ├── fashion_specialist/
│               ├── tokenizer/
│               └── routing.json
│
├── notebooks/                               # Git: воспроизводимые ноутбуки
│   ├── e5_small_macro_llm_stageA_v2.ipynb
│   ├── e5_small_restore30k_fullval_export.ipynb
│   └── e5_small_macro_v3_fashion_specialist_hybrid.ipynb
│
├── experiments/
│   ├── README.md                            # Git: правила хранения экспериментов
│   ├── e5_small_macro_v2/
│   │   ├── RUN_SUMMARY.md                   # Git: основной отчёт эксперимента
│   │   ├── metrics.json                     # Git: агрегированные метрики
│   │   ├── huggingface_source.json          # Git: ревизия backbone
│   │   ├── llm_by_category.csv              # Git: LLM holdout по категориям
│   │   ├── fashion_by_category.csv          # Git: четыре fashion-категории
│   │   └── manual_by_category.csv           # Git: manual diagnostic
│   └── e5_small_macro_v3/
│       ├── RUN_SUMMARY.md
│       ├── KAGGLE_RUN_SUMMARY.md
│       ├── metrics.json
│       ├── routing.json
│       ├── artifacts.json
│       ├── fashion_comparison.csv
│       ├── hybrid_full_by_category.csv
│       └── huggingface_source.json
│
├── models/
│   ├── README.md                            # Git: инвентаризация и SHA-256
│   ├── cross-encoder-ms-marco-MiniLM-L12-v2/ # local: веса baseline
│   ├── e5_small_macro_v2/                   # local: V2-артефакты
│   │   ├── checkpoints/
│   │   │   └── e5_macro_v2_best.zip
│   │   ├── exports/
│   │   │   └── e5_macro_v2_30k_export.zip
│   │   └── hf_export_30k/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       └── tokenizer_config.json
│   └── e5_small_macro_v3/                   # local: V3-артефакты
│       ├── checkpoints/
│       │   ├── e5_v3_best_specialist.pt
│       │   └── e5_v3_resume.pt
│       └── exports/
│           └── e5_v3_hybrid_export.zip
│
├── data/                                    # local, целиком игнорируется Git
│   ├── items_human.parquet
│   ├── matches.parquet
│   ├── items_test.parquet
│   ├── matches_test.parquet
│   └── submit_test.csv
│
├── docs/
│   └── PENDING_FILES.md                     # Git: список ранних материалов
│
├── e5_small_macro_v2_30k_submission.zip     # local: готовый V2 ZIP, ignored
└── e5_small_macro_v3_hybrid_submission.zip  # local: готовый V3 ZIP, ignored
```

## 4. Почему в корне и в `solutions/` есть отдельные `run.py`

Это сделано намеренно.

### Корневой `run.py`: исходный baseline

Файлы baseline:

- `run.py`;
- `src/utils.py`;
- `metadata.json`;
- `baseline_logreg_l12.joblib`;
- локальная модель `models/cross-encoder-ms-marco-MiniLM-L12-v2/`.

Baseline строит общий текст товара, получает CLS embeddings от
`cross-encoder/ms-marco-MiniLM-L12-v2`, после чего применяет сохранённый
Logistic Regression classifier.

Пример запуска baseline из корня:

```bash
python -u run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_test.csv
```

Корневой pipeline оставлен как контрольная точка и пример формата организаторов.
Для новой отправки E5 использовать его не нужно.

### `solutions/`: наши готовые submissions

У каждой версии собственные entry point, preprocessing/inference, metadata и
локальная модель:

- `solutions/e5_small_v2/` — уже проверенный на leaderboard V2 fallback;
- `solutions/e5_small_v3_hybrid/` — новый V3 hybrid с двумя моделями и routing.

Для новой V3 отправки использовать `solutions/e5_small_v3_hybrid/run.py`. Если
понадобится точно воспроизвести уже оценённую V2 посылку, использовать
`solutions/e5_small_v2/run.py`. Ни одна версия не зависит от кода другой.

## 5. Основной эксперимент E5-small Macro V2

### Конфигурация

| Параметр | Значение |
|---|---|
| Backbone | `intfloat/multilingual-e5-small` |
| Backbone revision | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Архитектура export | `BertForSequenceClassification`, один logit |
| Fine-tuned checkpoint | step `30000` |
| Обучение | LLM stage-A only |
| Manual stage-B | не использовался |
| `MAX_LEN` | `192` |
| `MAX_ATTR_CHARS` | `460` |
| Loss | category-balanced soft BCE |
| Labels | confidence-weighted LLM soft labels |
| Random pair swap | `0.5` |
| LLM validation pairs | `191555` |
| Manual validation pairs | `72948` |

### Почему этот эксперимент устроен так

- E5-small выбран как более сильный multilingual backbone.
- Обучение выполнялось только на LLM stage-A, потому что manual fine-tune по
  предыдущим командным A/B-тестам мог ухудшать leaderboard.
- Category weights приближают train objective к macro-метрике соревнования.
- Soft labels не бинаризуются во время обучения; менее уверенные примеры
  получают меньший вес.
- Случайная перестановка товаров снижает зависимость от порядка `id1/id2`.
- Validation split сделан по компонентам связности товаров, поэтому один товар
  не попадает одновременно в train и validation.

### Результаты

| Проверка | Macro PR-AUC |
|---|---:|
| Fast LLM group holdout во время обучения | `0.786378890` |
| Fast LLM group holdout после восстановления | `0.786378890` |
| Full LLM group holdout | **`0.786482334`** |
| Manual group holdout, diagnostic only | `0.677023885` |
| Fashion subset | `0.484432198` |
| Non-fashion subset | `0.861994868` |

Fast validation после восстановления checkpoint совпала с записанной метрикой
точно (`delta=0.0`). Это подтверждает соответствие checkpoint, tokenizer,
preprocessing, split и inference.

### Сильные и слабые категории на Full LLM holdout

Слабые категории:

| Категория | PR-AUC |
|---|---:|
| Обувь | `0.323267` |
| Одежда | `0.431439` |
| Галантерея и аксессуары | `0.541161` |
| Ювелирные изделия | `0.641862` |

Сильные категории:

| Категория | PR-AUC |
|---|---:|
| Автотовары | `0.940917` |
| Музыкальные инструменты | `0.922032` |
| Товары для животных | `0.918800` |
| Аптека | `0.917181` |
| Бытовая химия | `0.916806` |

Основной резерв качества находится в fashion, особенно в обуви и одежде.

### Известный баг тренировочного V2

Дополнительный fashion boost `x1.15` не применился к двум реальным категориям:

- `Галантерея и аксессуары`;
- `Ювелирные изделия`.

В training использовались несовпадающие exact strings. Общая category-balanced
часть loss при этом работала, поэтому эксперимент валиден. При следующем
fashion-oriented обучении exact category names нужно исправить.

## 6. E5-small V3 Fashion Specialist + Hybrid Routing

V3 реализует следующий шаг, намеченный после анализа V2. Он не заменяет base
глобально: отдельная модель дообучена на fashion и подключается по категориям,
где fast validation подтверждает преимущество.

### Конфигурация V3

| Параметр | Значение |
|---|---:|
| Initialization | точные V2 weights, step 30 000 |
| Fashion train source rows | 2 227 727 |
| Balanced candidate pool | 800 000 |
| Mined train | 600 000 |
| Hard / random / stable | 50% / 30% / 20% |
| Epochs | 2 |
| Optimizer steps | 4 688 |
| Best step | 4 000 |
| Backbone LR | `8e-6` |
| Head LR | `3e-5` |
| Effective batch | 256 |
| GPU / training time | Tesla T4 / 1.78 h |

V3 сохраняет V2 group split, `MAX_LEN=192`, `MAX_ATTR_CHARS=460` и основной
product text, но дополнительно извлекает размер, цвет, артикул, модель, пол и
материал. Если размер или цвет отсутствуют в attributes, используется fallback
из названия товара.

### Метрики V3

| Проверка | Macro PR-AUC |
|---|---:|
| V2 base fast | `0.786378890` |
| Best V3 fast hybrid | `0.790608644` |
| V2 base full | `0.786482334` |
| V3 specialist fashion full | `0.504527989` |
| **V3 hybrid full** | **`0.790182347`** |
| **Delta к V2** | **`+0.003700013`** |

Routing:

```json
{
  "Галантерея и аксессуары": "base",
  "Обувь": "specialist",
  "Одежда": "specialist",
  "Ювелирные изделия": "specialist"
}
```

Подробный отчёт находится в
`experiments/e5_small_macro_v3/RUN_SUMMARY.md`. Готовый offline pipeline
находится в `solutions/e5_small_v3_hybrid/`; он использует этот routing и
последовательно загружает base и specialist, чтобы не держать обе модели в
памяти одновременно.

## 7. Preprocessing V2

Точный V2-код находится в
`solutions/e5_small_v2/src/preprocessing.py`. V3 сохраняет тот же основной
product text в `solutions/e5_small_v3_hybrid/src/preprocessing.py` и добавляет
только обученные pair-level variant signals для specialist.

Для каждого товара используется `name`, `attributes`, `category`:

1. Текст переводится в нижний регистр.
2. `ё` заменяется на `е`.
3. Знаки `×`, `х`, `Х` нормализуются в `x`.
4. Запятые заменяются точками.
5. Повторяющиеся пробелы схлопываются.
6. `attributes` разбираются как JSON object.
7. Сначала выводятся identity/variant-атрибуты: бренд, артикул, part number,
   OEM, код, SKU, модель, размер, цвет, материал, объём, вес, габариты,
   комплектация, количество и тип.
8. Затем добавляются остальные атрибуты в исходном порядке.
9. Строка атрибутов обрезается до 460 символов.

Итоговый формат:

```text
категория: ... | название: ... | атрибуты: ...
```

Пара товаров токенизируется совместно с dynamic padding, truncation и
`max_length=192`. Добавлять E5-префиксы `query:`/`passage:` нельзя: модель была
обучена как sequence classifier без них.

## 8. Как работает готовый E5 inference

Оба solution выполняют общий набор проверок. V3-код находится в
`solutions/e5_small_v3_hybrid/src/pipeline.py` и дополнительно применяет
routing:

1. Загружает из `matches_path` только `id1` и `id2`.
2. Проверяет отсутствие null ID.
3. Собирает множество реально нужных item ID.
4. Потоково сканирует items parquet через PyArrow батчами по 400 000 строк.
5. Строит тексты только для товаров, участвующих в matches.
6. Падает с понятной ошибкой, если какой-либо нужный товар отсутствует.
7. Выбирает base/specialist по категории `id1` из `routing.json`.
8. Загружает model/tokenizer только из локальной директории с
   `local_files_only=True`.
9. Последовательно выполняет base и specialist inference, освобождая первую
   модель до загрузки второй.
10. Сортирует пары по примерной длине текста для уменьшения padding.
11. Возвращает predictions в исходный порядок и применяет sigmoid.
12. Проверяет колонки, количество строк, NaN/Inf и диапазон `[0, 1]`.
13. Записывает CSV строго в формате `id1,id2,predict`.

Offline mode включается до импорта Transformers:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Выбор batch size:

| Устройство | Batch size |
|---|---:|
| CUDA, VRAM >= 60 GB | 512 |
| CUDA, VRAM >= 35 GB | 256 |
| CUDA, VRAM >= 14 GB | 64 |
| CUDA с меньшей памятью | 32 |
| Apple MPS | 32 |
| CPU | 32 |

Для ручного override можно задать `E5_BATCH_SIZE`, например:

```bash
E5_BATCH_SIZE=128 python -u solutions/e5_small_v3_hybrid/run.py ...
```

На CUDA используются FP16 для старых GPU и BF16 для Ampere/Hopper и новее.

## 9. Как запустить E5 solution локально

### 9.1. Необходимые зависимости

Официальный `metadata.json` использует image:

```text
odsai/ecup26-matching-baseline:1.0
```

Для локального запуска нужны как минимум:

- Python;
- NumPy;
- pandas;
- PyArrow;
- PyTorch;
- Transformers;
- safetensors.

Локальная `.venv/` в Git не хранится.

### 9.2. Восстановить модели

После обычного `git clone` папок с весами не будет. Для V2 нужно получить export
`e5_macro_v2_30k_export.zip` и разместить четыре файла так:

```text
solutions/e5_small_v2/models/e5_small_macro_v2_30k/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Контрольная сумма `model.safetensors`:

```text
b7263e2c9f39cf73bfd1217c91ef613b9cbc9529fa95a062784215d4c568d92c
```

Эти файлы можно также скопировать из локального
`models/e5_small_macro_v2/hf_export_30k/`.

Для V3 нужно распаковать `e5_v3_hybrid_export.zip` в структуру:

```text
solutions/e5_small_v3_hybrid/models/e5_small_v3_hybrid/
├── base_model/
├── fashion_specialist/
├── tokenizer/
└── routing.json
```

### 9.3. Подготовить данные

Минимальные схемы:

```text
items parquet:   id, name, attributes, category
matches parquet: id1, id2
```

Для smoke-test в подготовленной локальной копии используются:

- `data/items_test.parquet`: 199 товаров;
- `data/matches_test.parquet`: 100 пар.

### 9.4. Запустить

V3 из корня репозитория:

```bash
python -u solutions/e5_small_v3_hybrid/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path /tmp/e5_v3_predictions.csv
```

V2 fallback:

```bash
python -u solutions/e5_small_v2/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path /tmp/e5_v2_predictions.csv
```

## 10. Что проверено в готовых solution

V2 ранее проверялся из build-директории, распакованного ZIP и финальной папки в
репозитории. V3 отдельно проверен на тех же тестовых parquet.

Проверки:

- 100 входных пар -> 100 выходных строк;
- найдены все 199 нужных товаров;
- колонки строго `id1,id2,predict`;
- порядок `id1/id2` совпадает с `matches_test.parquet`;
- NaN и Inf отсутствуют;
- predictions находятся в `[0, 1]`;
- модель и tokenizer загрузились без обращения к сети.

Локальные benchmark на доступном CPU:

| Решение / этап | Результат |
|---|---:|
| V2 cold start | около `4.05 s` |
| V2 inference | около `44 pair/s` |
| V3 cold start, две модели | около `4.38 s` |
| V3 base inference | около `43.8 pair/s` |
| V3 specialist inference | около `30.3 pair/s` |

В V3 smoke-test 84 пары ушли в base и 16 в specialist. На base-строках
максимальное расхождение с отдельным V2 solution было меньше `4e-8`; все 16
specialist-строк получили новые предсказания.

Это CPU smoke benchmark, а не оценка H100. Реальный полный test и скорость в
контейнере соревнования ещё нужно измерить по логам первой отправки.

## 11. Готовые submission ZIP

### V2

```text
e5_small_macro_v2_30k_submission.zip
```

- размер на диске: около 344 MB;
- SHA-256:
  `e958c0733af328e15a52e02f3d56c865a7b279474dbd6e111616e7cc739a002b`;
- ZIP прошёл полную CRC-проверку;
- внутри нет wrapper-директории: `metadata.json` и `run.py` лежат в корне.

В архиве ровно девять файлов:

```text
metadata.json
run.py
src/__init__.py
src/preprocessing.py
src/pipeline.py
models/e5_small_macro_v2_30k/config.json
models/e5_small_macro_v2_30k/model.safetensors
models/e5_small_macro_v2_30k/tokenizer.json
models/e5_small_macro_v2_30k/tokenizer_config.json
```

Архив не хранится на GitHub. Git LFS и GitHub CLI на машине на момент сборки не
были настроены. Для командного резервного хранения лучше загрузить ZIP как
GitHub Release asset, Kaggle Dataset или в другое хранилище больших файлов.

### V3 Hybrid

```text
e5_small_macro_v3_hybrid_submission.zip
```

- размер: `717517727` bytes (около 684 MiB);
- SHA-256:
  `d81b1a03fddf2fa7b75dc6554fd2b12c5a0cf9a5ee915bd7e0c85050aec652f8`;
- ZIP прошёл полную CRC-проверку;
- внутри 12 обязательных файлов без wrapper-директории: entry point, V3 code,
  routing, tokenizer, base model и fashion specialist.

Оба ZIP игнорируются Git.

## 12. Локальные модельные артефакты

### Fine-tuned checkpoint

```text
models/e5_small_macro_v2/checkpoints/e5_macro_v2_best.zip
```

- размер исходного файла: `470701556` bytes;
- SHA-256:
  `e3ceb948ddb41961dc5db08b55b317019074365836c337d9f3923d25ed1ce29c`.

### Hugging Face export ZIP

```text
models/e5_small_macro_v2/exports/e5_macro_v2_30k_export.zip
```

- размер исходного файла: `361191807` bytes;
- SHA-256:
  `926a31dc7c300e33735c1aa9e0172a2b7a8dd5a3b274250eea4021ea97152837`.

### Распакованный export

```text
models/e5_small_macro_v2/hf_export_30k/
```

Это канонический источник четырёх файлов, скопированных в
`solutions/e5_small_v2/models/e5_small_macro_v2_30k/`.

Архив `e5_macro_v2_best_repacked.zip` был проверен как дубликат содержимого
checkpoint: 207 внутренних файлов совпали по именам, размерам и CRC. Повторно в
репозиторий он не переносился. Bundle `results.zip` также состоял в основном из
уже сохранённых копий export/checkpoint/metrics.

### V3 best specialist checkpoint

```text
models/e5_small_macro_v3/checkpoints/e5_v3_best_specialist.pt
```

- best step: 4 000;
- размер: `470702271` bytes;
- SHA-256:
  `0546814b8950782e7116248ac557ed67ebd04098e5c0f9362aa5d9a8bbca7ada`.

### V3 resume checkpoint

```text
models/e5_small_macro_v3/checkpoints/e5_v3_resume.pt
```

- содержит model, optimizer, scheduler, scaler и состояние полного run;
- размер: `1412119762` bytes;
- SHA-256:
  `0ae0d6a752893faa8bbd4131e03b1d730ffe2fc1a5618887d433aab62fcededf`.

### V3 hybrid export

```text
models/e5_small_macro_v3/exports/e5_v3_hybrid_export.zip
```

- содержит V2 base model, V3 fashion specialist, tokenizer, routing и метрики;
- размер: `718451718` bytes;
- SHA-256:
  `ea84f56b9b0f6bfbabed5d3dfbe75292992234c8beffbd5f9141614c392d99a8`.

Новый `results (1).zip` прошёл CRC-проверку. Его V2 repacked checkpoint совпал
с уже сохранённым V2 по всем 207 внутренним файлам. Распакованный hybrid export
совпал с вложенным export ZIP по всем 10 файлам, поэтому дубликаты не копируются.

## 13. Ноутбуки

### `notebooks/e5_small_macro_llm_stageA_v2.ipynb`

Главный training notebook. В нём находятся:

- preprocessing V2;
- потоковая загрузка items;
- group splits;
- macro-aware category weights;
- Dataset, random swap и weighted soft BCE;
- fast validation;
- обучение stage-A;
- сохранение best checkpoint;
- full holdout evaluation;
- опциональная идея fashion continuation;
- экспорт финальной модели.

### `notebooks/e5_small_restore30k_fullval_export.ipynb`

Notebook восстановления и проверки. В нём находятся:

- автоматический поиск/восстановление checkpoint;
- проверка checkpoint metadata;
- точное воспроизведение group splits;
- загрузка только нужных validation items;
- восстановление модели step 30000;
- sanity check fast validation;
- full LLM holdout;
- fashion analysis;
- manual diagnostic holdout;
- Hugging Face export и краткий run summary.

Именно второй notebook подтвердил, что checkpoint восстановлен корректно и дал
Full LLM Macro PR-AUC `0.786482334`.

### `notebooks/e5_small_macro_v3_fashion_specialist_hybrid.ipynb`

Полный V3 notebook. В нём находятся:

- восстановление и контроль V2 checkpoint;
- тот же group split, что в V2;
- balanced fashion candidate pool;
- извлечение variant-сигналов;
- base predictions и hard-example mining;
- mined train на 600 000 пар;
- specialist, инициализированный из V2 weights;
- частые hybrid validation и resume checkpoints;
- загрузка best step 4000;
- full hybrid validation;
- rough LB signal;
- export base + specialist + tokenizer + routing.

## 14. Папка `experiments/`

`experiments/e5_small_macro_v2/` хранит небольшие результаты, которые можно
безопасно держать в Git:

- `RUN_SUMMARY.md` — человекочитаемый отчёт;
- `metrics.json` — точные агрегированные числа и параметры;
- `huggingface_source.json` — repo ID и revision исходного backbone;
- `llm_by_category.csv` — 20 категорий Full LLM holdout;
- `fashion_by_category.csv` — четыре слабые fashion-категории;
- `manual_by_category.csv` — 20 категорий manual diagnostic holdout.

Manual metric используется только как диагностика. Решение о submission
принималось по LLM group holdout и проверке воспроизводимости checkpoint.

`experiments/e5_small_macro_v3/` хранит историю нового fashion specialist:

- `RUN_SUMMARY.md` — полный проверенный отчёт;
- `KAGGLE_RUN_SUMMARY.md` — исходный summary из Kaggle output;
- `metrics.json` — точные V3 параметры и метрики;
- `routing.json` — категории base/specialist и variant config;
- `fashion_comparison.csv` — V2/V3 comparison для четырёх категорий;
- `hybrid_full_by_category.csv` — итоговый hybrid по 20 категориям;
- `huggingface_source.json` — ревизия исходного backbone.
- `artifacts.json` — provenance, размеры, SHA-256 и проверки дублей.

## 15. Что хранится на GitHub, а что нет

### Хранится на GitHub

- Python-код baseline и обоих E5 solution;
- `metadata.json` для baseline, V2 и V3;
- небольшой baseline Logistic Regression classifier;
- training/restore notebooks;
- отчёты, JSON с метриками и category CSV;
- SHA-256 и описание локальных моделей;
- проектная документация.

### Не хранится на GitHub

- `data/`, `*.parquet`;
- `.venv/`;
- `models/*`, кроме `models/README.md`;
- `solutions/*/models/`;
- `*.safetensors`, `*.pt`, `*.pth`, `*.bin`, `*.onnx`;
- `*.zip`;
- кэши и generated outputs.

Не использовать `git add -f` для этих файлов без отдельного командного решения
по Git LFS или внешнему artifact storage.

Полезная проверка перед коммитом:

```bash
git status --short
git status --ignored --short
git check-ignore -v solutions/e5_small_v2/models/e5_small_macro_v2_30k/model.safetensors
git check-ignore -v solutions/e5_small_v3_hybrid/models/e5_small_v3_hybrid/base_model/model.safetensors
git check-ignore -v e5_small_macro_v2_30k_submission.zip
git check-ignore -v e5_small_macro_v3_hybrid_submission.zip
```

## 16. История ветки

Ветка `minicooper1` основана на `main` с исходным baseline.

Ключевые коммиты до этого handoff:

```text
6c2d8ee  Initial commit: E-CUP 2026 matching baseline
8458193  do first experement
2a292e9  feat: add E5-small submission pipeline
```

- `8458193` добавил notebooks, experiment reports, metrics, model inventory и
  проектную структуру.
- `2a292e9` добавил проверенный E5 offline submission pipeline.

## 17. Что ещё не сделано

1. V3 ещё не прошёл full-size inference в контейнере/H100.
2. V3 submission ZIP ещё не отправлен на leaderboard; Public LB неизвестен.
3. Крупные V2/V3-артефакты пока не имеют командного удалённого backup.
4. Не добавлены три ранних notebook, если они сохранились:
   - `ozon_ecup_crossencoder_v1.ipynb`;
   - `ce_tiny_2ep.ipynb`;
   - `ozon_ecup_catboost_v1.ipynb`.
5. Нет отдельного `docs/FOR_MISHA.md`, если такая персональная памятка всё ещё
   нужна.

Проверенный V2 в `solutions/e5_small_v2/` уже дал Public LB `0.4838757641` и
остаётся production fallback независимо от V3.

## 18. Рекомендуемые следующие действия команды

1. Выполнить V3 submit, сохранить execution logs и leaderboard score.
2. Сравнить V3 LB с V2 Public LB `0.4838757641` и rough signal `0.486152`.
3. Выполнить full-size benchmark V3 в реальном контейнере.
4. Загрузить V3 export/checkpoints в командное хранилище больших файлов.
5. После LB решить, продолжать ли specialist или делать blending/ensemble.

## 19. Самый короткий маршрут для нового участника

Если времени мало:

1. Прочитать этот файл.
2. Посмотреть V2 и V3 отчёты в `experiments/`.
3. Открыть индекс `solutions/README.md` и выбрать V2 fallback или V3 hybrid.
4. Для нового решения изучить `solutions/e5_small_v3_hybrid/run.py` и `src/`.
5. Восстановить локальную модель по README выбранного solution.
6. Запустить smoke-test на `data/items_test.parquet` и
   `data/matches_test.parquet`.
7. Для истории обучения V3 открыть
   `notebooks/e5_small_macro_v3_fashion_specialist_hybrid.ipynb` и
   `experiments/e5_small_macro_v3/RUN_SUMMARY.md`.

Главный вывод: **V2 и V3 — два независимых готовых solution в `solutions/`.
V2 уже оценён на leaderboard и остаётся fallback; V3 лучше на локальном
holdout, прошёл smoke-test и готов к первой отправке; корневой pipeline —
сохранённый baseline.**
