# E5-small submission

Рабочий submission для E-CUP 2026 Ozon matching.

- Backbone: `intfloat/multilingual-e5-small`
- Checkpoint: step 30000
- Preprocessing: V2
- `MAX_LEN`: 192
- Full LLM group holdout Macro PR-AUC: 0.786482

Запуск из этой директории:

```bash
python -u run.py \
  --items_path ../data/items_test.parquet \
  --matches_path ../data/matches_test.parquet \
  --output_path predictions.csv
```

Модель и tokenizer загружаются только из локального каталога
`models/e5_small_macro_v2_30k/`; обращений к Hugging Face во время
inference нет.

Готовый архив для отправки находится в корне репозитория:

```text
e5_small_macro_v2_30k_submission.zip
```

Архив намеренно игнорируется Git. Локальная модель внутри `solution/models/`
также игнорируется, чтобы веса и tokenizer не попали в обычный коммит.
