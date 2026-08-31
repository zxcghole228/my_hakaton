# Stage-B v3 cross-soup USER-BGE-M3 runtime with symmetry TTA

Исправленная версия runtime-кода для третьего кандидата BGE.

Public leaderboard score: `0.5539900397479464`.

## Происхождение модели

Базовая архитектура — `deepvk/USER-bge-m3`.

Финальная модель получена смешиванием двух checkpoint:

```text
0.83 × Stage-B v2 step_02000.pt
+
0.17 × Stage-B v3 step_00400.pt
```

После смешивания параметры двух checkpoint сохранены как одна модель. Во время инференса не требуется загружать две модели.

Stage-B v2 соответствует второму кандидату репозитория:

```text
versions/bge/02_stage_b_v2_checkpoint_soup_8515
```

Stage-B v3 обучалась отдельно с использованием:

* ручной разметки;
* LLM replay;
* hard examples;
* distillation;
* gray-set distillation;
* ranking loss;
* symmetry loss;
* EMA checkpoint;
* checkpoint sweep.

Для самостоятельной Stage-B v3 лучшим был выбран `step_00600.pt`. Однако для финального cross-soup использовался `step_00400.pt`, поскольку он показал лучший баланс при смешивании со Stage-B v2.

## Symmetry TTA

Для каждой входной пары модель выполняет два forward pass:

```text
p_forward = sigmoid(model(item_1, item_2))
p_reverse = sigmoid(model(item_2, item_1))
predict = 0.5 × (p_forward + p_reverse)
```

Это inference-time техника. Она не создаёт вторую модель и не изменяет веса.

Preprocessing соответствует контракту `pair_text_v1` и не изменён относительно исходного solution.

## Original и runtime

Папка `../original` содержит исходный solution в том виде, в котором он был получен от автора модели.

Папка `runtime` содержит исправленную рабочую копию. В ней:

* добавлены аргументы `--items-path`, `--matches-path` и `--output-path`;
* сохранена совместимость с аргументами через подчёркивание;
* добавлен аргумент `--batch-size`;
* модель загружается только из локальной директории;
* отключена возможность скачивания модели из сети;
* сохраняются все входные пары;
* сохраняется исходный порядок строк;
* сохраняются повторяющиеся пары;
* отсутствующие товары и пустые тексты не приводят к удалению строк;
* результат записывается в столбец `predict`;
* исходный preprocessing `pair_text_v1` сохранён;
* исходная формула symmetry TTA сохранена.

## Подготовка модели

Веса не дублируются в Git.

Перед запуском необходимо создать директорию:

```text
models/user-bge-m3-v3-cross-soup-tta
```

и скопировать в неё содержимое:

```text
../original/models/user-bge-human-ft
```

Локальный файл весов:

```text
model.safetensors
```

Размер:

```text
1436159092 bytes
```

SHA-256:

```text
b9d98750751a2442a946f6d9188fa3cda251df4d818ad0b23fa13abf223469da
```

Итоговая структура runtime должна выглядеть так:

```text
runtime/
├── metadata.json
├── README.md
├── run.py
├── models/
│   └── user-bge-m3-v3-cross-soup-tta/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── src/
    ├── __init__.py
    └── utils.py
```

## Запуск

```bash
python -u run.py \
  --items-path /path/to/items.parquet \
  --matches-path /path/to/matches.parquet \
  --output-path /path/to/submit.csv
```

При необходимости можно изменить batch size:

```bash
python -u run.py \
  --items-path /path/to/items.parquet \
  --matches-path /path/to/matches.parquet \
  --output-path /path/to/submit.csv \
  --batch-size 128
```

Также поддерживаются старые варианты аргументов:

```text
--items_path
--matches_path
--output_path
--batch_size
```

## Формат результата

Runtime создаёт CSV со столбцами:

```text
id1,id2,predict
```

Количество строк соответствует количеству строк во входном `matches.parquet`.

## Статус проверки

Статически проверено:

* синтаксис Python;
* поддержка CLI-алиасов;
* локальный путь модели;
* запрет сетевой загрузки;
* сохранение всех входных пар;
* сохранение порядка и дубликатов;
* неизменность preprocessing;
* неизменность symmetry TTA;
* формат выходных столбцов;
* SHA-256 исходного solution;
* SHA-256 финальных весов.

Полный запуск с реальными весами, проверка времени выполнения и потребления памяти должны проводиться отдельно в целевом контейнере.
