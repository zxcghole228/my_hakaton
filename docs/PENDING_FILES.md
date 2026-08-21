# Незакрытые материалы и проверки

## Ранние ноутбуки

Если файлы сохранились в Kaggle или истории проекта, положить их в
`notebooks/`:

- `ozon_ecup_crossencoder_v1.ipynb` — ранний CrossEncoder-эксперимент;
- `ce_tiny_2ep.ipynb` — короткий CrossEncoder training run;
- `ozon_ecup_catboost_v1.ipynb` — ранний CatBoost baseline.

Если существует отдельная командная памятка, положить её в
`docs/FOR_MISHA.md`.

## V4 Structured Ensemble

V4 оформлен как отдельный pipeline в
`solutions/e5_small_v4_structured_ensemble/`. Код и компактные результаты
эксперимента сохранены, локальный submission archive находится в
`artifacts/submissions/e5_small_v4_structured_ensemble.zip`.

Осталось:

- выполнить full-size benchmark в контейнере соревнования/H100;
- отправить V4 archive;
- записать leaderboard score и execution logs;
- сравнить Public LB V4 с V3 (`0.4848439268`) и V2 (`0.4838757641`).

## Backup

Нужен командный удалённый backup крупных checkpoints, exports и submission
archives. Сейчас они намеренно игнорируются Git и хранятся только локально.
