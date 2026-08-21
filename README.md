# E-CUP 2026: матчинг товаров Ozon

Рабочий репозиторий команды для baseline, NLP-экспериментов и независимых
submission pipelines задачи product matching.

Подробный handoff, полное дерево и правила добавления новых решений находятся
в [`FOR_TEAMMATES.md`](FOR_TEAMMATES.md).

## Навигация

| Каталог | Что хранится |
|---|---|
| `solutions/` | Самодостаточный код каждого baseline/solution |
| `notebooks/` | Обучение, восстановление, анализ и export |
| `experiments/` | Отчёты, конфигурации, метрики и небольшие таблицы |
| `models/` | Локальные training checkpoints и exports; в Git только README |
| `data/` | Локальные входные данные и predictions; в Git только README |
| `artifacts/` | Submission ZIP, распакованные builds и исходные bundles |
| `docs/` | Командные заметки и список незакрытых материалов |

Корень намеренно не содержит `run.py`, моделей и submission-архивов. Все точки
входа находятся рядом со своим решением в `solutions/`.

## Решения

| Решение | Статус | Основной результат |
|---|---|---:|
| `cross_encoder_baseline` | baseline организаторов | контрольная точка |
| `e5_small_v2` | проверенный fallback | Public LB `0.4838757641` |
| `e5_small_v3_hybrid` | проверено на leaderboard | Public LB `0.4848439268` |
| `e5_small_v4_structured_ensemble` | submission candidate | offline eval `0.789531` |

V4 объединяет V3 E5 scores и structured CatBoost через category-wise
percentile-rank blend с весами `0.90 / 0.10`. Его leaderboard score пока не
зафиксирован.

## Пример запуска

Все решения принимают единый интерфейс:

```text
--items_path
--matches_path
--output_path
```

Например, запуск последнего кандидата из корня:

```bash
python -u solutions/e5_small_v4_structured_ensemble/run.py \
  --items_path data/items_test.parquet \
  --matches_path data/matches_test.parquet \
  --output_path data/submit_v4.csv
```

Для запуска нужны локальные веса в `models/` выбранного решения. Инструкции и
состав модели описаны в README этого решения.

## Что не попадает в Git

Git игнорирует datasets, веса, checkpoints, ZIP-архивы, predictions, кэши и
виртуальные окружения. В репозитории остаются код, ноутбуки, документация,
конфигурации и небольшие результаты экспериментов.
