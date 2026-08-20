# Материалы, которые ещё нужно добавить

## Нужно скачать

Если эти файлы сохранились в Kaggle или в истории проекта, положить их в `notebooks/`:

- `ozon_ecup_crossencoder_v1.ipynb` — ранний CrossEncoder-эксперимент.
- `ce_tiny_2ep.ipynb` — короткий CrossEncoder training run.
- `ozon_ecup_catboost_v1.ipynb` — CatBoost baseline/experiment.

Если существует отдельная командная памятка, положить её сюда:

- `docs/FOR_MISHA.md` — инструкция по воспроизведению и передаче решения.

## Осталось после сборки V3 solution

V3 Fashion Specialist обучен, экспортирован и собран в отдельный проверенный
pipeline `solutions/e5_small_v3_hybrid/`. Чистый локальный ZIP также готов.
Осталось:

- выполнить full-size benchmark в контейнере соревнования/H100;
- отправить `e5_small_macro_v3_hybrid_submission.zip`;
- записать V3 leaderboard score и execution logs;
- сравнить Public LB V3 с V2 (`0.4838757641`).

Также нужно сделать командный удалённый backup крупных V3 checkpoint/export,
которые сейчас хранятся только локально и игнорируются Git.

Проверенный V2 сохранён независимо в `solutions/e5_small_v2/` и остаётся
production fallback независимо от дальнейших изменений V3.
