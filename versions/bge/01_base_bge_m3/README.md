# Base USER-BGE-M3

Базовая BGE-версия решения для матчинга товаров Ozon E-CUP 2026.

Модель получена полным fine-tuning предобученной `deepvk/USER-bge-m3` на официальной LLM-разметке. Эта версия является отправной точкой BGE-линейки и не использует Human Stage B, checkpoint soup, ensemble или test-time augmentation.

## Результат

Public leaderboard Macro PR-AUC:

```text
0.5136543845676982
```

Локальные метрики обучения:

| Метрика                       |             Значение |
| ----------------------------- | -------------------: |
| best fast-val Macro PR-AUC    | `0.8633551084605952` |
| full LLM holdout Macro PR-AUC |           `0.860026` |
| best fast-val step            |              `40000` |
| optimizer steps               |              `42774` |

## Обучение

| Параметр             |               Значение |
| -------------------- | ---------------------: |
| base model           |   `deepvk/USER-bge-m3` |
| training data        |  `matches_llm.parquet` |
| item rows            |           `13 397 761` |
| labeled pairs        |           `11 187 780` |
| epochs               |                    `2` |
| max length           |                  `320` |
| learning rate        |                 `2e-5` |
| target global batch  |                  `512` |
| component split seed |                   `13` |
| training seed        |                   `42` |
| swap probability     |                  `0.5` |
| validation interval  |           `8000` steps |
| hardware             | `2 × NVIDIA H100 PCIe` |

Во время обучения обновляются все параметры BGE-M3. Classification head инициализируется заново. Поэтому это fine-tuning предобученной модели, а не обучение BGE со случайно инициализированными весами.

## Preprocessing

Контракт `pair_text_v1` используется одинаково при обучении и инференсе.

Он включает:

* приоритизацию ключевых атрибутов товара;
* ограничение текста атрибутов до 520 символов;
* нормализацию смешанных латинских и кириллических гомоглифов;
* канонизацию единиц измерения;
* извлечение количества товаров в наборе;
* итоговое ограничение текста до 2000 символов;
* токенизацию пары с `max_length=320`.

## Структура

```text
training/original/
├── pipeline_public.ipynb
└── train_user_bge_m3_320.py

runs/research_worker_s42_02/
├── metrics.json
└── run_config.json

solution/
├── original/
└── runtime/
```

`solution/original/` сохраняет исходную структуру solution, предоставленную автором модели. Малые runtime- и tokenizer-файлы зафиксированы в Git. `model.safetensors` хранится локально и игнорируется Git из-за размера.

`solution/runtime/` содержит исправленную рабочую версию инференса:

* без TTA и checkpoint ensemble;
* с поддержкой официального CLI;
* с автономной загрузкой модели без интернета;
* с сохранением порядка, дубликатов и всех входных пар;
* с выходными столбцами `id1`, `id2`, `predict`.

## Веса

```text
file: model.safetensors
size_bytes: 1436159092
sha256: d1c65bd374abf91cc8373bfee05b1c706c34acf0c8aa83e88e4f0a409ea20735
```

Хеш весов и принадлежность solution базовой BGE подтверждены автором модели.

## Статус

Это первая из трёх BGE-версий-кандидатов. После оформления и проверки всех трёх версий команда выберет две финальные submission.
