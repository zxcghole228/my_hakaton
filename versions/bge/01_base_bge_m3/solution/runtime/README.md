# Baseline USER-BGE-M3 runtime

Исправленная версия runtime-кода для первого кандидата BGE.

Модель основана на `deepvk/USER-bge-m3` и дообучена на базовом корпусе `matches_llm.parquet`. Это baseline fine-tuning предобученной модели, а не обучение BGE со случайно инициализированными весами.

Public leaderboard score: `0.5136543845676982`.

Эта версия не использует:

- Human Stage B;
- объединение двух checkpoint;
- model soup;
- symmetry TTA;
- ансамблирование нескольких моделей.

## Отличие от original

Папка `../original` содержит исходный solution в том виде, в котором он был получен от автора модели.

Папка `runtime` содержит исправленный вариант для воспроизводимого запуска. В нём:

- удалены ошибочные подписи про `v2 soup`, `LB 0.5522` и symmetry TTA;
- добавлена поддержка аргумента `--output-path`;
- сохранена совместимость с вариантами аргументов через подчёркивание;
- модель загружается только из локальной директории;
- предсказания создаются для всех входных пар;
- сохраняются исходный порядок строк и повторяющиеся пары;
- результат записывается в столбец `predict`;
- исходный preprocessing `pair_text_v1` не изменён.

## Подготовка модели

Файлы модели не дублируются в Git. Перед запуском runtime необходимо создать директорию:

`models/user-bge-m3-baseline`

и скопировать в неё файлы из:

`../original/models/user-bge-human-ft`

Итоговая структура должна выглядеть так:

```text
runtime/
├── metadata.json
├── README.md
├── run.py
├── models/
│   └── user-bge-m3-baseline/
│       ├── config.json
│       ├── model.safetensors
│       ├── sentencepiece.bpe.model
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── src/
    ├── __init__.py
    ├── pair_text.py
    └── utils.py
```

SHA-256 файла `model.safetensors`:

```text
d1c65bd374abf91cc8373bfee05b1c706c34acf0c8aa83e88e4f0a409ea20735
```

Размер файла:

```text
1436159092 bytes
```

## Запуск

Команду необходимо выполнять из директории `runtime`:

```bash
python3 run.py \
  --items_path /path/to/items.parquet \
  --matches_path /path/to/matches.parquet \
  --output-path /path/to/submit.csv
```

Также поддерживаются варианты:

- `--items-path`;
- `--matches-path`;
- `--output_path`.

## Результат

Runtime создаёт CSV-файл со столбцами:

```text
id1,id2,predict
```

Для каждой строки входного файла `matches.parquet` создаётся ровно одно предсказание. Порядок строк и повторяющиеся пары сохраняются.

## Статус проверки

Для runtime выполнены следующие статические проверки:

- Python-файлы успешно компилируются;
- JSON-файлы имеют корректный синтаксис;
- CLI принимает поддерживаемые варианты аргументов;
- preprocessing совпадает с исходным solution;
- пары с отсутствующим или пустым текстом не удаляются;
- ошибочные подписи от решений `v2 soup` и TTA отсутствуют.

Полный запуск с реальными весами и проверка времени выполнения должны проводиться отдельно в целевом контейнере.
