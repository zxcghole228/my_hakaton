# Материалы, которые ещё нужно добавить

## Нужно скачать

Если эти файлы сохранились в Kaggle или в истории проекта, положить их в `notebooks/`:

- `ozon_ecup_crossencoder_v1.ipynb` — ранний CrossEncoder-эксперимент.
- `ce_tiny_2ep.ipynb` — короткий CrossEncoder training run.
- `ozon_ecup_catboost_v1.ipynb` — CatBoost baseline/experiment.

Если существует отдельная командная памятка, положить её сюда:

- `docs/FOR_MISHA.md` — инструкция по воспроизведению и передаче решения.

## Нужно подготовить позже

После выбора финальной модели собрать в `solution/`:

- `run.py` — единая точка входа;
- `preprocessing.py` — preprocessing V2;
- `model.py` — загрузка модели и inference;
- `metadata.json` — финальная конфигурация контейнера;
- при необходимости небольшой `requirements.txt` или точное описание зависимостей.

Перед заменой текущего baseline следует проверить полный локальный запуск и формат итогового submission-файла.
