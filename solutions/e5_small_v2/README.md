# E5-small V2 submission

Рабочий submission для E-CUP 2026 Ozon matching.

- Backbone: `intfloat/multilingual-e5-small`
- Checkpoint: step 30000
- Preprocessing: V2
- `MAX_LEN`: 192
- Full LLM group holdout Macro PR-AUC: 0.786482
- Public LB: 0.4838757641

Запуск из корня репозитория:

```bash
python -u solutions/e5_small_v2/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_v2.csv
```

Модель и tokenizer загружаются только из локального каталога
`models/e5_small_macro_v2_30k/`; обращений к Hugging Face во время
inference нет.

Готовый архив для отправки находится в каталоге артефактов:

```text
artifacts/submissions/e5_small_macro_v2_30k_submission.zip
```

Архив намеренно игнорируется Git. Локальная модель внутри
`solutions/e5_small_v2/models/` также игнорируется, чтобы веса и tokenizer не
попали в обычный коммит.

V3 сохранён независимо в `../e5_small_v3_hybrid/`; обновление V3 не должно
изменять этот проверенный fallback.
