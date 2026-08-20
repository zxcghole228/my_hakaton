# E-CUP 2026: матчинг товаров Ozon

Рабочий репозиторий для baseline, экспериментов и итогового решения задачи product matching.

## Структура

- `run.py` — текущая точка входа исходного CrossEncoder baseline.
- `src/` — код baseline-пайплайна.
- `metadata.json` — конфигурация запуска baseline в среде организаторов.
- `baseline_logreg_l12.joblib` — компактный классификатор baseline.
- `notebooks/` — воспроизводимые ноутбуки обучения, восстановления и анализа.
- `experiments/` — метрики, таблицы по категориям и краткие отчёты без весов моделей.
- `solution/` — описание текущего состояния и место для итогового submission pipeline.
- `models/` — локальные checkpoint и экспорты; в Git попадает только `models/README.md`.
- `data/` — локальные данные соревнования и результаты; в Git не попадают.
- `docs/` — проектные заметки и список ещё не добавленных материалов.

## Текущий baseline

Исходный baseline ожидает веса CrossEncoder в:

```text
models/cross-encoder-ms-marco-MiniLM-L12-v2/
```

Пример локального запуска из корня репозитория:

```bash
python run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_test.csv
```

Baseline пока оставлен в корне, чтобы не нарушать его текущую конфигурацию запуска и относительные пути. План переноса итогового решения описан в `solution/README.md`.

## E5-small Macro LLM Stage-A V2

Основной текущий эксперимент использует `intfloat/multilingual-e5-small`, checkpoint шага 30 000 и длину входа 192.

Ключевые результаты:

| Валидация | Macro PR-AUC |
|---|---:|
| Fast LLM group holdout | 0.786379 |
| Full LLM group holdout | 0.786482 |
| Manual group holdout, diagnostic | 0.677024 |
| Fashion subset | 0.484432 |
| Non-fashion subset | 0.861995 |

Полный отчёт и таблицы по категориям находятся в `experiments/e5_small_macro_v2/`.

## Что хранится только локально

В репозиторий не коммитятся:

- веса и checkpoint моделей;
- архивы;
- `parquet`, Arrow и Feather datasets;
- виртуальные окружения;
- кэши, временные каталоги и generated outputs.

Расположение и контрольные суммы локальных модельных артефактов записаны в `models/README.md`.

## Следующие материалы

Список файлов, которые ещё предстоит скачать или подготовить, находится в `docs/PENDING_FILES.md`.
