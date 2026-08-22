# Карта репозитория для команды

Актуально на **21 августа 2026 года** для ветки `minicooper1`.

Этот файл — основной handoff по репозиторию E-CUP 2026 product matching. Здесь
описаны назначение всех каталогов, четыре независимых решения, локальные
артефакты и правила добавления следующих экспериментов.

## 1. Главный принцип структуры

Репозиторий разделён по жизненному циклу ML-решения:

```text
notebooks/ -> experiments/ -> models/ -> solutions/ -> artifacts/submissions/
```

- `notebooks/` отвечает за исследование и обучение;
- `experiments/` фиксирует результаты, конфигурации и принятое решение;
- `models/` хранит локальные training checkpoints и exports;
- `solutions/` содержит только код автономных inference pipelines;
- `artifacts/submissions/` содержит готовые ZIP для отправки.

В корне нет ни одного `run.py`, веса или submission-архива. Поэтому нельзя
случайно запустить не ту версию или принять ZIP за исходный код.

## 2. Полное дерево

Обозначения:

- **Git** — файл должен храниться в репозитории;
- **local** — файл есть в рабочей копии, но исключён из Git;
- **generated** — временный локальный файл, который можно пересоздать;
- **pending** — материал ещё нужно получить или проверить.

```text
matching-baseline-submit/
├── .gitignore                                      # Git: правила хранения
├── README.md                                       # Git: короткая навигация
├── FOR_TEAMMATES.md                                # Git: этот handoff
│
├── solutions/                                      # все runnable решения
│   ├── README.md                                   # Git: реестр и контракт
│   ├── cross_encoder_baseline/                     # baseline организаторов
│   │   ├── README.md                               # Git
│   │   ├── metadata.json                           # Git
│   │   ├── run.py                                  # Git
│   │   ├── baseline_logreg_l12.joblib              # Git: 12 KiB classifier
│   │   ├── src/
│   │   │   └── utils.py                            # Git
│   │   └── models/                                 # local, ignored
│   │       └── cross-encoder-ms-marco-MiniLM-L12-v2/
│   │
│   ├── e5_small_v2/                                # проверенный fallback
│   │   ├── README.md                               # Git
│   │   ├── metadata.json                           # Git
│   │   ├── run.py                                  # Git
│   │   ├── src/                                    # Git
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py
│   │   │   └── pipeline.py
│   │   └── models/                                 # local, ignored
│   │       └── e5_small_macro_v2_30k/
│   │           ├── config.json
│   │           ├── model.safetensors
│   │           ├── tokenizer.json
│   │           └── tokenizer_config.json
│   │
│   ├── e5_small_v3_hybrid/                         # V3 leaderboard solution
│   │   ├── README.md                               # Git
│   │   ├── metadata.json                           # Git
│   │   ├── run.py                                  # Git
│   │   ├── src/                                    # Git
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py
│   │   │   └── pipeline.py
│   │   └── models/                                 # local, ignored
│   │       └── e5_small_v3_hybrid/
│   │           ├── base_model/
│   │           ├── fashion_specialist/
│   │           ├── tokenizer/
│   │           └── routing.json
│   │
│   └── e5_small_v4_structured_ensemble/            # последний candidate
│       ├── README.md                               # Git
│       ├── metadata.json                           # Git
│       ├── run.py                                  # Git
│       ├── src/                                    # Git
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   ├── pipeline.py
│       │   └── structured_v4.py
│       └── models/                                 # local, ignored
│           └── e5_small_v3_hybrid/
│               ├── base_model/
│               ├── fashion_specialist/
│               ├── tokenizer/
│               ├── routing.json
│               ├── structured_model.so
│               ├── blend_config_v4.json
│               ├── feature_config_v4.json
│               └── metrics_v4.json
│
├── notebooks/                                      # обучение и анализ
│   ├── README.md                                   # Git
│   ├── e5_small_macro_llm_stageA_v2.ipynb          # Git
│   ├── e5_small_restore30k_fullval_export.ipynb    # Git
│   ├── e5_small_macro_v3_fashion_specialist_hybrid.ipynb # Git
│   └── v3-ozonecup-fix-old-bugs.ipynb              # новый рабочий notebook
│
├── experiments/                                    # компактные результаты
│   ├── README.md                                   # Git
│   ├── e5_small_macro_v2/
│   │   ├── RUN_SUMMARY.md
│   │   ├── metrics.json
│   │   ├── huggingface_source.json
│   │   ├── llm_by_category.csv
│   │   ├── fashion_by_category.csv
│   │   └── manual_by_category.csv
│   ├── e5_small_macro_v3/
│   │   ├── RUN_SUMMARY.md
│   │   ├── KAGGLE_RUN_SUMMARY.md
│   │   ├── metrics.json
│   │   ├── routing.json
│   │   ├── artifacts.json
│   │   ├── fashion_comparison.csv
│   │   ├── hybrid_full_by_category.csv
│   │   └── huggingface_source.json
│   └── e5_small_v4_structured_ensemble/
│       ├── RUN_SUMMARY.md
│       ├── metrics.json
│       ├── blend_config.json
│       ├── feature_config.json
│       ├── feature_importance.csv
│       ├── eval_by_category.csv
│       └── tune_category_weights.csv
│
├── models/                                         # training artifacts
│   ├── README.md                                   # Git: inventory + hashes
│   ├── e5_small_macro_v2/                          # local, ignored
│   │   ├── checkpoints/
│   │   ├── exports/
│   │   └── hf_export_30k/
│   ├── e5_small_macro_v3/                          # local, ignored
│   │   ├── checkpoints/
│   │   └── exports/
│   └── e5_small_macro_v4/                          # local, ignored
│       └── structured_catboost.cbm
│
├── data/                                           # datasets и predictions
│   ├── README.md                                   # Git
│   ├── items_human.parquet                         # local
│   ├── matches.parquet                             # local
│   ├── items_test.parquet                          # local
│   ├── matches_test.parquet                        # local
│   └── submit_test.csv                             # generated
│
├── artifacts/                                      # готовые локальные bundles
│   ├── README.md                                   # Git
│   ├── submissions/                                # local, ignored
│   │   ├── e5_small_macro_v2_30k_submission.zip
│   │   ├── e5_small_macro_v3_hybrid_submission.zip
│   │   └── e5_small_v4_structured_ensemble.zip
│   ├── submission_builds/                          # local, ignored
│   │   └── e5_small_macro_v3_hybrid_submission/
│   └── experiment_exports/                         # local, ignored
│       └── e5_small_v4_structured_ensemble.zip
│
├── docs/
│   └── PENDING_FILES.md                            # Git
│
├── .venv/                                          # local Python environment
├── .tmp/                                           # generated temporary files
└── .torch_cache/                                   # generated Torch cache
```

## 3. Реестр решений

| Версия | Метод | Offline Macro PR-AUC | Public LB | Роль |
|---|---|---:|---:|---|
| CrossEncoder baseline | CLS + Logistic Regression | — | — | контрольный baseline |
| E5-small V2 | один fine-tuned E5 checkpoint | `0.786482` | `0.4838757641` | проверенный fallback |
| E5-small V3 Hybrid | V2 base + fashion specialist | `0.790182` | `0.4848439268` | лучшая оценённая версия |
| E5-small V4 | V3 + structured CatBoost rank blend | `0.789531` на V4 eval | pending | submission candidate |

Метрики V2/V3 и V4 считались не на полностью одинаковых eval split, поэтому
значения в таблице нельзя интерпретировать как прямой честный leaderboard.

### 3.1 CrossEncoder baseline

Папка: `solutions/cross_encoder_baseline/`.

Это исходная контрольная реализация организаторов. Она строит объединённый
текст товара, получает CLS embeddings CrossEncoder и применяет небольшой
Logistic Regression classifier.

```bash
python -u solutions/cross_encoder_baseline/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_baseline.csv
```

Classifier хранится в Git. Веса Transformer находятся локально в
`solutions/cross_encoder_baseline/models/`.

### 3.2 E5-small V2

Папка: `solutions/e5_small_v2/`.

- backbone: `intfloat/multilingual-e5-small`;
- checkpoint: step `30000`;
- обучение: LLM stage-A only;
- preprocessing V2;
- `MAX_LEN=192`, `MAX_ATTR_CHARS=460`;
- Full LLM group holdout Macro PR-AUC: `0.786482334`;
- Public LB: `0.4838757641`.

```bash
python -u solutions/e5_small_v2/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_v2.csv
```

V2 оставлен неизменяемым fallback. Новые эксперименты не должны переписывать
его код, веса или archive.

### 3.3 E5-small V3 Hybrid

Папка: `solutions/e5_small_v3_hybrid/`.

V3 сохраняет V2 как base и использует fashion specialist только там, где он
улучшил validation:

```text
Галантерея и аксессуары -> base
Обувь                    -> specialist
Одежда                   -> specialist
Ювелирные изделия        -> specialist
все остальные категории  -> base
```

К specialist-текстам добавляются pair-level variant signals: сравнение
размера, цвета, артикула, модели, пола и материала.

- best specialist checkpoint: step `4000`;
- V3 Hybrid Full LLM Macro PR-AUC: `0.790182347`;
- delta к V2 на том же split: `+0.003700013`;
- Public LB: `0.4848439268`.

```bash
python -u solutions/e5_small_v3_hybrid/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_v3.csv
```

### 3.4 E5-small V4 Structured Ensemble

Папка: `solutions/e5_small_v4_structured_ensemble/`.

V4 использует V3 predictions и отдельную structured lexical CatBoost-модель.
Structured features включают fuzzy-сходство названий, числа, коды, единицы,
brand/model/article/size/color/material/gender, attribute overlaps и word/char
hash cosine. Score обеих моделей переводится в percentile rank внутри
категории.

Зафиксированная формула submission:

```text
predict = 0.90 * E5_V3_category_rank + 0.10 * structured_category_rank
```

Основные результаты V4 experiment:

| Проверка | Macro PR-AUC |
|---|---:|
| V3 на V4 eval split | `0.786570708` |
| Structured CatBoost | `0.586165657` |
| Global blend | `0.789530729` |
| Category-tuned diagnostic | `0.789536118` |
| Delta global blend к V3 eval | `+0.002960021` |

В исходном `metrics.json` поле `delta_vs_v3_eval` содержит `0.002965410` — это
разница category-tuned diagnostic и V3. Сам submission использует global
weight, поэтому выше отдельно показана его фактическая разница.

```bash
python -u solutions/e5_small_v4_structured_ensemble/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_v4.csv
```

Leaderboard score V4 пока не записан.

Важно: `structured_model.so` в submission — Linux x86-64 ELF. Полный V4
inference нужно запускать в Linux-контейнере из `metadata.json`; macOS не может
загрузить эту библиотеку как Mach-O. На macOS отдельно проверяется V3 E5-часть,
а исходная CatBoost-модель лежит в
`models/e5_small_macro_v4/structured_catboost.cbm`.

## 4. Где лежат большие файлы

### Runtime-модели

Каждый pipeline ожидает модели только внутри собственной папки:

```text
solutions/<solution>/models/
```

Это важное правило: `solutions/` можно упаковать независимо, и код не зависит
от центрального каталога training artifacts.

В текущей локальной копии одинаковые immutable E5-веса V3 и V4 связаны через
hardlinks, чтобы не дублировать около 900 MiB на диске. У обеих версий есть
полные собственные пути, и при упаковке ZIP файлы становятся обычными
независимыми entries. Не перезаписывайте model-файлы in-place.

### Training checkpoints и exports

Они находятся в `models/e5_small_macro_v*/`. Это исходные артефакты для
восстановления, продолжения обучения или повторной сборки solution. Полный
реестр и SHA-256 указан в `models/README.md`.

### Submission ZIP

```text
artifacts/submissions/e5_small_macro_v2_30k_submission.zip
artifacts/submissions/e5_small_macro_v3_hybrid_submission.zip
artifacts/submissions/e5_small_v4_structured_ensemble.zip
```

Контрольные суммы текущих локальных архивов:

| Архив | SHA-256 |
|---|---|
| V2 | `e958c0733af328e15a52e02f3d56c865a7b279474dbd6e111616e7cc739a002b` |
| V3 | `d81b1a03fddf2fa7b75dc6554fd2b12c5a0cf9a5ee915bd7e0c85050aec652f8` |
| V4 | `8507e0eace4456031621d5f44e21e8b98cab5ef7c668ade798aa1dad217e6f6e` |
| V4 experiment export | `976e3483480c6cd98dd59672b03aa6f7fe381d48c6131207a6cc12a7bbfc8fce` |

После клонирования GitHub этих файлов не будет. Их нужно получить из
командного хранилища или пересобрать из documented exports.

## 5. Что хранит Git

В Git должны попадать:

- Python-код и `metadata.json` всех решений;
- ноутбуки без встроенных больших данных и секретов;
- README, run summaries, JSON-конфигурации и небольшие CSV с метриками;
- компактный classifier исходного baseline;
- документация каталогов.

В Git не должны попадать:

- `parquet`, Arrow, Feather и другие datasets;
- `safetensors`, PyTorch checkpoints, ONNX, CatBoost-модели;
- submission ZIP и исходные experiment bundles;
- generated predictions;
- `.venv`, кэши, `__pycache__`, `.DS_Store`.

## 6. Как добавить новое решение

Допустим, появляется V5.

1. Создать notebook с содержательным именем в `notebooks/`.
2. Создать `experiments/e5_small_v5_<idea>/` и сохранить туда summary,
   configuration, metrics и category tables.
3. Положить checkpoints/exports локально в `models/e5_small_macro_v5/` и
   добавить размеры и SHA-256 в `models/README.md`.
4. Создать `solutions/e5_small_v5_<idea>/` по обязательному шаблону из
   `solutions/README.md`.
5. Проверить, что solution не импортирует код предыдущей версии и работает
   offline.
6. Собрать ZIP в `artifacts/submissions/`, не в корне.
7. Обновить таблицы в `README.md`, `solutions/README.md` и этом файле.

Не используйте названия `new`, `final`, `final2` или `best`. Имя должно
фиксировать версию и идею: например `e5_small_v5_calibrated_blend`.

## 7. Проверки перед коммитом

Проверить состояние и игнорирование:

```bash
git status --short
git status --ignored --short
git check-ignore -v artifacts/submissions/e5_small_v4_structured_ensemble.zip
git check-ignore -v models/e5_small_macro_v3/checkpoints/e5_v3_resume.pt
git check-ignore -v solutions/e5_small_v4_structured_ensemble/models/e5_small_v3_hybrid/structured_model.so
```

Проверить Python entry points:

```bash
python -m py_compile \
  solutions/cross_encoder_baseline/run.py \
  solutions/e5_small_v2/run.py \
  solutions/e5_small_v3_hybrid/run.py \
  solutions/e5_small_v4_structured_ensemble/run.py
```

Перед отправкой archive дополнительно проверить:

- отсутствие wrapper-директории, `.DS_Store`, `__pycache__` и тестовых данных;
- наличие `metadata.json`, `run.py`, `src/` и всех runtime-моделей;
- CRC архива через `unzip -t`;
- точную схему результата `id1,id2,predict`;
- сохранение порядка и числа входных пар;
- отсутствие NaN/Inf и диапазон score `[0, 1]`.

## 8. Что ещё не закрыто

- Получить leaderboard score V4 и записать его в V4 README, общий README и
  experiment metrics.
- Провести full-size benchmark V4 в контейнере соревнования.
- Сделать командный удалённый backup крупных checkpoints, exports и submission
  ZIP: сейчас они хранятся только локально.
- Найти ранние CrossEncoder/CatBoost notebooks из `docs/PENDING_FILES.md`.

До появления V4 leaderboard **V3 остаётся лучшим подтверждённым решением**, а
V2 — стабильным fallback.
