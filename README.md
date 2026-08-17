# E-CUP 2026: матчинг товаров

Рабочий репозиторий для baseline и экспериментов по матчингу товаров на E-CUP 2026.

## Структура

- `run.py` — точка входа для запуска решения.
- `src/` — код пайплайна и вспомогательные функции.
- `metadata.json` — конфигурация запуска в среде организаторов.
- `baseline_logreg_l12.joblib` — компактный классификатор baseline.
- `models/` — локальные веса моделей; не хранятся в Git.
- `data/` — локальные данные соревнования и результаты; не хранятся в Git.

## Локальные файлы

Исходный baseline ожидает веса CrossEncoder в:

```text
models/cross-encoder-ms-marco-MiniLM-L12-v2/
```

Каталог с весами занимает более 1 ГБ и содержит файлы крупнее лимита GitHub, поэтому он исключён из Git. Для восстановления рабочей копии возьмите его из оригинального архива `matching-baseline-submit.zip`, выданного организаторами.

## Пример запуска

```bash
python run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_test.csv
```
