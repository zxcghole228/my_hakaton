# Эксперименты

Здесь хранятся небольшие воспроизводимые результаты: `RUN_SUMMARY.md`,
конфигурации, агрегированные метрики и таблицы по категориям.

```text
experiments/
├── e5_small_macro_v2/
├── e5_small_macro_v3/
├── e5_small_v4_structured_ensemble/
└── e5_bge_selective_v6/
```

Checkpoint, веса, datasets и generated predictions сюда не добавляются.
Training artifacts размещаются локально в `models/`, а исходные bundles — в
`artifacts/experiment_exports/`.

Для нового эксперимента создайте отдельную папку и как минимум сохраните:

- цель и гипотезу;
- конфигурацию train/validation split;
- основные метрики и решение по результату;
- ссылки на notebook, model artifact и основанное на нём solution.
