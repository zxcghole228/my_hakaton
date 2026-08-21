# E5-small V3 Hybrid submission

Готовый offline solution для E-CUP 2026 Ozon matching. Он сохраняет V2 как
base и использует отдельно обученный fashion specialist только в категориях,
где validation показала преимущество.

## Состав и качество

- base: E5-small V2 checkpoint step 30 000;
- specialist: V3 fashion checkpoint step 4 000;
- preprocessing: V2 product text плюс точные V3 pair-level variant signals;
- `MAX_LEN=192`, `MAX_ATTR_CHARS=460`;
- V2 Full LLM Macro PR-AUC: `0.786482334`;
- V3 Hybrid Full LLM Macro PR-AUC: `0.790182347`;
- delta: `+0.003700013`.
- Public LB: `0.4848439268`.

Routing:

```text
Галантерея и аксессуары -> base
Обувь                    -> specialist
Одежда                   -> specialist
Ювелирные изделия        -> specialist
все остальные категории  -> base
```

Specialist получает одинаковый pair signal в обоих текстах: сравнение размера,
цвета, артикула, модели, пола и материала. Реализация дословно воспроизводит
логику обучающего V3 notebook.

## Локальная модель

Pipeline загружает всё только из:

```text
models/e5_small_v3_hybrid/
├── base_model/
│   ├── config.json
│   └── model.safetensors
├── fashion_specialist/
│   ├── config.json
│   └── model.safetensors
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── routing.json
```

Источником служит локальный export
`models/e5_small_macro_v3/exports/e5_v3_hybrid_export.zip`. Веса и tokenizer
игнорируются Git.

## Запуск

Из корня репозитория:

```bash
python -u solutions/e5_small_v3_hybrid/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path /tmp/e5_v3_predictions.csv
```

Интерфейс полностью совместим с аргументами соревнования `--items_path`,
`--matches_path`, `--output_path`. Hugging Face работает в offline mode.

## Проверка сборки

Локальный smoke-test на 100 парах и 199 товарах завершён успешно:

- 100 входных пар -> 100 выходных строк;
- колонки строго `id1,id2,predict`;
- порядок ID сохранён;
- NaN/Inf отсутствуют, значения находятся в `[0, 1]`;
- 84 пары направлены в base, 16 — в specialist;
- base-строки совпали с отдельным V2 solution с максимальным отклонением
  меньше `4e-8`;
- specialist изменил все 16 предназначенных ему строк.

CPU smoke benchmark:

- полный cold start: около `4.38 s`;
- base inference: `43.8 pair/s`;
- specialist inference: `30.3 pair/s`;
- end-to-end на 100 парах: около `22.8 pair/s`, включая загрузку двух моделей.

Готовый локальный архив находится в каталоге артефактов:

```text
artifacts/submissions/e5_small_macro_v3_hybrid_submission.zip
```

- размер: `717517727` bytes (около 684 MiB);
- SHA-256:
  `d81b1a03fddf2fa7b75dc6554fd2b12c5a0cf9a5ee915bd7e0c85050aec652f8`;
- 12 файлов, без wrapper-директории и служебных файлов;
- ZIP прошёл полную CRC-проверку.

V2 fallback сохранён независимо в `../e5_small_v2/`.
Следующая V4-версия сохранена независимо в
`../e5_small_v4_structured_ensemble/`.
