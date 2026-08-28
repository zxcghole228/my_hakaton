# Stage A retrained

Повторное обучение `deepvk/USER-bge-m3` на официальном
`matches_llm.parquet`. Запуск `research_worker_s42_02` выполнен на двух NVIDIA
H100 PCIe, завершил 42 774 optimizer steps и получил full LLM holdout macro
PR-AUC `0.860026`.

## Основные параметры

| Параметр | Значение |
|---|---:|
| max length | 320 |
| epochs | 2 |
| learning rate | `2e-5` |
| global batch | 512 |
| component split seed | 13 |
| training seed | 42 |
| swap probability | 0.5 |
| validation interval | 8 000 steps |

Preprocessing `pair_text_v1` включает приоритетные атрибуты, ограничение 520
символов, нормализацию гомоглифов, единиц измерения и количества.

## Состояние воспроизводимости

Исходный notebook содержит полный успешный training log и таблицу артефактов.
В Git сохранены notebook и точный training script. Для финального Docker ещё
нужно получить `export/`, tokenizer, exact inference runtime и контрольные суммы
весов. Полный список находится в `artifacts/manifest.json`.

Терминологически это fine-tuning из исходного pretrained checkpoint, а не
обучение BGE со случайно инициализированными весами.
