# Готовые решения

Каждое решение живёт в собственной папке и не импортирует код другого
решения. Это позволяет отдельно упаковывать, проверять и откатывать любую
версию.

## Реестр

| Папка | Назначение | Статус |
|---|---|---|
| `cross_encoder_baseline/` | исходный CrossEncoder + Logistic Regression | baseline |
| `e5_small_v2/` | E5-small checkpoint 30k | fallback, Public LB `0.4838757641` |
| `e5_small_v3_hybrid/` | V2 base + fashion specialist | Public LB `0.4848439268` |
| `e5_small_v4_structured_ensemble/` | V3 + structured CatBoost rank blend | candidate |

## Обязательная структура решения

```text
solutions/<solution_name>/
├── README.md       # идея, метрики, запуск, состав локальных файлов
├── metadata.json   # образ и entry point платформы
├── run.py          # единая CLI-точка входа
├── src/            # preprocessing и inference
└── models/         # runtime-веса; local, ignored
```

Дополнительный небольшой classifier можно хранить рядом с `run.py`, только
если он действительно является частью исходного baseline и явно разрешён в
`.gitignore`. Большие веса всегда остаются в `models/`.

## Интерфейс

Все entry point принимают:

```text
--items_path
--matches_path
--output_path
```

Запуск выполняется из корня репозитория:

```bash
python -u solutions/<solution_name>/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/predictions.csv
```

Готовые архивы лежат локально в `artifacts/submissions/`. Они не коммитятся.
