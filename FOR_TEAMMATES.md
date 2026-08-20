# Ветка `minicooper1`: полный handoff для команды

Актуально на **20 августа 2026 года**.

Этот файл — основная карта ветки `minicooper1`. Он объясняет, что уже сделано,
как устроен репозиторий, где находятся baseline, обученная E5-модель, метрики и
готовый submission, а также что нужно восстановить локально после клонирования.

## 1. Коротко: текущее состояние

В ветке сохранены две независимые линии решения:

1. **Исходный CrossEncoder baseline организаторов** — оставлен в корне
   репозитория без изменения его интерфейса запуска.
2. **Наш основной E5-small submission** — находится в `solution/` и является
   готовым решением для первой отправки.

Главный результат на текущий момент:

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
9. Собран автономный `solution/` под интерфейс организаторов.
10. Проведён smoke-test на локальных parquet.
11. Проверены схема `id1,id2,predict`, полнота строк, порядок ID, отсутствие
    NaN/Inf и диапазон вероятностей `[0, 1]`.
12. Собран чистый submission ZIP без кэшей, тестовых данных, `.DS_Store` и
    лишнего корневого каталога.
13. Код E5 submission запушен в ветку `minicooper1`.

Публичного leaderboard score для этого E5 submission в ветке пока нет. Первую
реальную отправку ещё нужно выполнить и затем записать её результат.

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
├── solution/                                # основной готовый E5 submission
│   ├── README.md                            # Git: краткий запуск solution
│   ├── metadata.json                        # Git: metadata submission
│   ├── run.py                               # Git: entry point E5 submission
│   ├── src/
│   │   ├── __init__.py                      # Git
│   │   ├── preprocessing.py                 # Git: точный preprocessing V2
│   │   └── pipeline.py                      # Git: offline inference + CSV checks
│   └── models/                              # local, целиком игнорируется Git
│       └── e5_small_macro_v2_30k/
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           └── tokenizer_config.json
│
├── notebooks/                               # Git: воспроизводимые ноутбуки
│   ├── e5_small_macro_llm_stageA_v2.ipynb
│   └── e5_small_restore30k_fullval_export.ipynb
│
├── experiments/
│   ├── README.md                            # Git: правила хранения экспериментов
│   └── e5_small_macro_v2/
│       ├── RUN_SUMMARY.md                   # Git: основной отчёт эксперимента
│       ├── metrics.json                     # Git: агрегированные метрики
│       ├── huggingface_source.json          # Git: ревизия backbone
│       ├── llm_by_category.csv              # Git: LLM holdout по категориям
│       ├── fashion_by_category.csv          # Git: четыре fashion-категории
│       └── manual_by_category.csv           # Git: manual diagnostic
│
├── models/
│   ├── README.md                            # Git: инвентаризация и SHA-256
│   ├── cross-encoder-ms-marco-MiniLM-L12-v2/ # local: веса baseline
│   └── e5_small_macro_v2/                   # local: E5-артефакты
│       ├── checkpoints/
│       │   └── e5_macro_v2_best.zip
│       ├── exports/
│       │   └── e5_macro_v2_30k_export.zip
│       └── hf_export_30k/
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           └── tokenizer_config.json
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
└── e5_small_macro_v2_30k_submission.zip     # local: готовый submission, ignored
```

## 4. Почему в корне и в `solution/` есть отдельные `run.py`

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

### `solution/run.py`: наш текущий submission

Файлы E5 submission:

- `solution/run.py`;
- `solution/src/preprocessing.py`;
- `solution/src/pipeline.py`;
- `solution/metadata.json`;
- локальная модель `solution/models/e5_small_macro_v2_30k/`.

Именно содержимое `solution/` без `README.md` упаковано в готовый ZIP. Если
задача — проверить или отправить наш лучший текущий вариант, использовать нужно
эту линию.

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

## 6. Preprocessing V2

Точный код находится в `solution/src/preprocessing.py` и совпадает с training.

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

## 7. Как работает готовый E5 inference

`solution/src/pipeline.py` выполняет следующие шаги:

1. Загружает из `matches_path` только `id1` и `id2`.
2. Проверяет отсутствие null ID.
3. Собирает множество реально нужных item ID.
4. Потоково сканирует items parquet через PyArrow батчами по 400 000 строк.
5. Строит тексты только для товаров, участвующих в matches.
6. Падает с понятной ошибкой, если какой-либо нужный товар отсутствует.
7. Загружает model/tokenizer только из локальной директории с
   `local_files_only=True`.
8. Сортирует пары по примерной длине текста для уменьшения padding.
9. Выполняет batched inference и возвращает predictions в исходный порядок.
10. Применяет sigmoid к единственному logit.
11. Проверяет колонки, количество строк, NaN/Inf и диапазон `[0, 1]`.
12. Записывает CSV строго в формате `id1,id2,predict`.

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
E5_BATCH_SIZE=128 python -u solution/run.py ...
```

На CUDA используются FP16 для старых GPU и BF16 для Ampere/Hopper и новее.

## 8. Как запустить E5 solution локально

### 8.1. Необходимые зависимости

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

### 8.2. Восстановить модель

После обычного `git clone` папки `solution/models/` не будет. Нужно получить
экспорт `e5_macro_v2_30k_export.zip` из сохранённых Kaggle/local artifacts и
разместить четыре файла так:

```text
solution/models/e5_small_macro_v2_30k/
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

### 8.3. Подготовить данные

Минимальные схемы:

```text
items parquet:   id, name, attributes, category
matches parquet: id1, id2
```

Для smoke-test в подготовленной локальной копии используются:

- `data/items_test.parquet`: 199 товаров;
- `data/matches_test.parquet`: 100 пар.

### 8.4. Запустить

Из корня репозитория:

```bash
python -u solution/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path /tmp/e5_predictions.csv
```

Из `solution/`:

```bash
cd solution
python -u run.py \
  --items_path ../data/items_test.parquet \
  --matches_path ../data/matches_test.parquet \
  --output_path /tmp/e5_predictions.csv
```

## 9. Что проверено перед первой отправкой

Smoke-test запускался трижды: из build-директории, из распакованного ZIP и из
финальной папки `solution/` в репозитории.

Проверки:

- 100 входных пар -> 100 выходных строк;
- найдены все 199 нужных товаров;
- колонки строго `id1,id2,predict`;
- порядок `id1/id2` совпадает с `matches_test.parquet`;
- NaN и Inf отсутствуют;
- predictions находятся в `[0, 1]`;
- повторный запуск из распакованного ZIP дал идентичный CSV;
- модель и tokenizer загрузились без обращения к сети.

Локальный benchmark на доступном CPU:

| Этап | Результат |
|---|---:|
| Полный cold start | около `4.05 s` |
| Чистый inference 100 пар | около `2.27 s` |
| Средняя скорость | около `44 pair/s` |

Это CPU smoke benchmark, а не оценка H100. Реальный полный test и скорость в
контейнере соревнования ещё нужно измерить по логам первой отправки.

## 10. Готовый submission ZIP

Локальный файл:

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

## 11. Локальные модельные артефакты

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
`solution/models/e5_small_macro_v2_30k/`.

Архив `e5_macro_v2_best_repacked.zip` был проверен как дубликат содержимого
checkpoint: 207 внутренних файлов совпали по именам, размерам и CRC. Повторно в
репозиторий он не переносился. Bundle `results.zip` также состоял в основном из
уже сохранённых копий export/checkpoint/metrics.

## 12. Ноутбуки

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

## 13. Папка `experiments/`

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

## 14. Что хранится на GitHub, а что нет

### Хранится на GitHub

- Python-код baseline и E5 solution;
- `metadata.json` для обоих pipeline;
- небольшой baseline Logistic Regression classifier;
- training/restore notebooks;
- отчёты, JSON с метриками и category CSV;
- SHA-256 и описание локальных моделей;
- проектная документация.

### Не хранится на GitHub

- `data/`, `*.parquet`;
- `.venv/`;
- `models/*`, кроме `models/README.md`;
- `solution/models/`;
- `*.safetensors`, `*.pt`, `*.pth`, `*.bin`, `*.onnx`;
- `*.zip`;
- кэши и generated outputs.

Не использовать `git add -f` для этих файлов без отдельного командного решения
по Git LFS или внешнему artifact storage.

Полезная проверка перед коммитом:

```bash
git status --short
git status --ignored --short
git check-ignore -v solution/models/e5_small_macro_v2_30k/model.safetensors
git check-ignore -v e5_small_macro_v2_30k_submission.zip
```

## 15. История ветки

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

## 16. Что ещё не сделано

1. Не выполнена первая leaderboard-отправка E5 ZIP либо её score ещё не записан.
2. Full-size inference не измерен в реальном контейнере соревнования/H100.
3. Большие E5-артефакты пока не имеют командного удалённого backup.
4. Не добавлены три ранних notebook, если они сохранились:
   - `ozon_ecup_crossencoder_v1.ipynb`;
   - `ce_tiny_2ep.ipynb`;
   - `ozon_ecup_catboost_v1.ipynb`.
5. Нет отдельного `docs/FOR_MISHA.md`, если такая персональная памятка всё ещё
   нужна.
6. Не выполнен следующий fashion-specific эксперимент с исправленными exact
   category names.

Важно: раздел «Нужно подготовить позже» в `docs/PENDING_FILES.md`, где ещё
предлагалось собрать E5 `solution/`, теперь устарел. E5 solution уже собран,
протестирован и находится в `solution/`. Актуальный список незавершённых задач —
в этом разделе.

## 17. Рекомендуемые следующие действия команды

1. Отправить `e5_small_macro_v2_30k_submission.zip` как первый E5 submission.
2. Сохранить execution time, сообщения контейнера и leaderboard score.
3. Записать score в `README.md`, `RUN_SUMMARY.md` и этот handoff.
4. Загрузить ZIP/export/checkpoint в командное хранилище больших файлов.
5. После первого LB решить, делать ли:
   - fashion continuation;
   - отдельную fashion-модель/ensemble;
   - исправленное category weighting;
   - калибровку или blending с другим независимым решением.
6. Не менять preprocessing отдельно от модели без повторной валидации: текущие
   веса обучены именно с V2 и `MAX_LEN=192`.

## 18. Самый короткий маршрут для нового участника

Если времени мало:

1. Прочитать этот файл.
2. Посмотреть `experiments/e5_small_macro_v2/RUN_SUMMARY.md`.
3. Для кода submission открыть `solution/run.py`,
   `solution/src/preprocessing.py`, `solution/src/pipeline.py`.
4. Получить четыре model/tokenizer файла и положить их в
   `solution/models/e5_small_macro_v2_30k/`.
5. Запустить smoke-test на `data/items_test.parquet` и
   `data/matches_test.parquet`.
6. Для обучения/анализа открыть два notebook в `notebooks/`.

Главный вывод: **готовое текущее решение — E5-small в `solution/`; корневой
pipeline — сохранённый baseline.**
