# CrossEncoder baseline

Исходный baseline организаторов вынесен из корня в самостоятельное решение.
Пайплайн строит тексты товаров, получает CLS embeddings модели
`cross-encoder/ms-marco-MiniLM-L12-v2` и применяет Logistic Regression.

## Состав

```text
cross_encoder_baseline/
├── README.md
├── metadata.json
├── run.py
├── baseline_logreg_l12.joblib
├── src/
│   └── utils.py
└── models/                              # local, ignored
    └── cross-encoder-ms-marco-MiniLM-L12-v2/
```

## Запуск

Из корня репозитория:

```bash
python -u solutions/cross_encoder_baseline/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_test.csv
```

`run.py` вычисляет пути к classifier и модели относительно своей директории,
поэтому команда работает из любой текущей папки.

## Локальные зависимости

Веса должны находиться в
`solutions/cross_encoder_baseline/models/cross-encoder-ms-marco-MiniLM-L12-v2/`.
Они исключены из Git. Компактный файл `baseline_logreg_l12.joblib` остаётся в
Git как часть самого baseline.
