# E5-small V4 Structured Ensemble

Submission candidate, объединяющий E5-small V3 hybrid и structured lexical
CatBoost.

## Идея

1. V3 выбирает base или fashion specialist по категории товара `id1`.
2. Structured-модель строит lexical/attribute признаки: названия, числа, коды,
   единицы, бренд, модель, артикул, размер, цвет, материал и hash similarities.
3. Оба score переводятся в percentile ranks внутри категории.
4. Финальный score: `0.90 * E5 rank + 0.10 * structured rank`.

## Метрики

| Проверка | Macro PR-AUC |
|---|---:|
| V3 combined holdout | `0.790182` |
| V3 на V4 eval split | `0.786571` |
| Structured CatBoost | `0.586166` |
| Global rank blend | `0.789531` |
| Category-tuned blend, diagnostic | `0.789536` |

Зафиксированный submission использует global blend. По experiment report его
решение — `SUBMISSION_CANDIDATE`. Leaderboard score V4 пока не записан.

## Состав

```text
e5_small_v4_structured_ensemble/
├── README.md
├── metadata.json
├── run.py
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── structured_v4.py
└── models/                              # local, ignored
    └── e5_small_v3_hybrid/
        ├── base_model/
        ├── fashion_specialist/
        ├── tokenizer/
        ├── routing.json
        ├── structured_model.so
        ├── blend_config_v4.json
        ├── feature_config_v4.json
        └── metrics_v4.json
```

## Запуск

```bash
python -u solutions/e5_small_v4_structured_ensemble/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_v4.csv
```

Inference полностью offline. Восстановить полный набор runtime-моделей можно
из `artifacts/submissions/e5_small_v4_structured_ensemble.zip`.

`structured_model.so` собран как Linux x86-64 ELF для контейнера соревнования.
На macOS его нельзя загрузить через `ctypes` (`Mach-O` и другая архитектура),
поэтому полный V4 smoke-test нужно выполнять в Linux-образе из
`metadata.json`. Нативно на macOS можно проверять V3 E5-часть; исходная
CatBoost-модель сохранена в `models/e5_small_macro_v4/structured_catboost.cbm`.

В текущей локальной рабочей копии неизменяемые E5-веса V3 представлены
hardlink-копиями в V4 `models/`, чтобы не занимать повторно около 900 MiB.
Путь V4 при этом полноценен: при сборке ZIP hardlinks упаковываются как обычные
файлы. Не изменяйте файлы весов in-place; новую модель сохраняйте новым файлом.

Experiment report находится в
`experiments/e5_small_v4_structured_ensemble/`.
