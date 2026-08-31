# BGE Candidate 3 — Stage-B v3 cross-soup with symmetry TTA

Третий кандидат семейства BGE для задачи матчинга товаров E-CUP 2026.

Public leaderboard score: `0.5539900397479464`.

## Краткое описание

Решение объединяет три компонента:

1. Stage-B v2 checkpoint `step_02000.pt` из второго BGE-кандидата.
2. Stage-B v3 checkpoint `step_00400.pt` из отдельного обучения.
3. Symmetry TTA во время инференса.

Финальные параметры модели получены по формуле:

```text
theta =
    0.83 × Stage-B v2 step_02000.pt
    +
    0.17 × Stage-B v3 step_00400.pt
```

После смешивания checkpoint параметры экспортированы как одна модель. На этапе инференса загружается только один `model.safetensors`.

## Lineage

Базовая модель:

```text
deepvk/USER-bge-m3
```

Лицензия базовой модели:

```text
Apache-2.0
```

Stage-B v3 инициализировалась checkpoint первого кандидата:

```text
../01_base_bge_m3
```

Второй компонент cross-soup взят из второго кандидата:

```text
../02_stage_b_v2_checkpoint_soup_8515
```

Использованный checkpoint второго кандидата:

```text
step_02000.pt
```

SHA-256:

```text
553d3a165aca6db16c26244a4eefe9ff6d57f7687ef5a8031b398711956f37a5
```

## Stage-B v3

Stage-B v3 обучалась отдельно от Stage-B v2.

В обучении использовались:

* ручная разметка;
* существующий LLM replay;
* hard examples по категориям;
* anchor examples;
* label loss;
* distillation loss;
* gray-set distillation;
* ranking loss;
* symmetry loss;
* EMA weights;
* checkpoint sweep.

Основные параметры:

```text
seed:                         20260826
epochs:                       1
maximum optimizer steps:      1200
reported optimizer steps:     710
maximum sequence length:      320
backbone learning rate:       5e-6
classifier learning rate:     2e-5
warmup ratio:                 0.05
swap probability:             0.5
unfrozen transformer layers:  12
GPU count:                    2
batch per GPU:                128
gradient accumulation:        2
effective global batch:       512
checkpoint interval:          200
EMA decay:                    0.999
```

Loss weights:

```text
label:          0.45
distillation:   0.18
gray distill:   0.40
ranking:        0.20
symmetry:       0.10
```

Для самостоятельной Stage-B v3 лучшим checkpoint по внутреннему composite-score стал:

```text
step_00600.pt
```

Его метрики:

```text
tune macro AP:   0.761367671851467
problem AP:      0.5351681227196512
full gray AP:    0.5934142254224464
composite:       0.6066187787622722
```

Однако в финальном cross-soup использовался `step_00400.pt`, поскольку checkpoint с лучшей самостоятельной метрикой не обязательно даёт лучший результат при смешивании с другой моделью.

## Выбор коэффициента soup

Для финального кандидата использовался коэффициент:

```text
alpha = 0.17
```

Формула:

```text
theta = (1 - alpha) × v2_step_02000
        + alpha × v3_step_00400
```

То есть:

```text
weight Stage-B v2 = 0.83
weight Stage-B v3 = 0.17
```

Внутренние метрики варианта `alpha=0.17`:

```text
tune macro AP:   0.7987693804111746
problem AP:      0.634591481855638
full gray AP:    0.5554132425182203
composite:       0.64367358976552
```

В исходном snapshot также сохранены эксперименты с другими значениями alpha. Они находятся только в `recovery` и не считаются финальной конфигурацией этого кандидата.

## Symmetry TTA

Во время инференса каждая пара оценивается в двух направлениях:

```text
p_forward = sigmoid(model(item_1, item_2))
p_reverse = sigmoid(model(item_2, item_1))
```

Итоговое предсказание:

```text
predict = 0.5 × p_forward + 0.5 × p_reverse
```

Symmetry TTA не изменяет веса и не требует второй модели, но выполняет два forward pass для каждой пары.

## Preprocessing

Используется контракт:

```text
pair_text_v1
```

Основные ограничения:

```text
attribute character limit: 520
maximum text characters:   2000
maximum token length:      320
```

Каноническая реализация:

```text
training/reconstructed/pair_text_v1.py
```

Проверка соответствия:

```text
training/reconstructed/verify_pair_text.py
```

Preprocessing исправленного runtime побайтово и структурно сопоставлен с исходным solution.

## Структура версии

```text
03_stage_b_v3_cross_soup_symmetry_tta_5540/
├── README.md
├── version.json
├── artifacts/
│   └── manifest.json
├── construction/
│   ├── original/
│   │   └── build_submit.py
│   └── runtime/
│       └── build_submit.py
├── recovery/
│   └── solution_lb554_staging/
├── runs/
│   └── cross_tta_s42_01/
│       ├── stageb_v3/
│       ├── stageb_v3.log
│       ├── cross_soup_fine_alpha/
│       └── cross_soup_a017/
├── solution/
│   ├── original/
│   └── runtime/
└── training/
    ├── environment.json
    ├── source_manifest.json
    ├── original/
    │   └── pipeline_public.ipynb
    └── reconstructed/
        ├── train_bge_stageb_v3.py
        ├── blend_checkpoint_soup.py
        ├── score_ensemble.py
        ├── score_val_gray.py
        ├── pair_text_v1.py
        └── verify_pair_text.py
```

## Recovery snapshot

Папка:

```text
recovery/solution_lb554_staging
```

содержит распакованный source snapshot, полученный от автора модели.

В ней сохранены:

* все переданные Python-скрипты;
* выполненный публичный notebook;
* вспомогательные notebooks;
* training logs;
* checkpoint sweep;
* fine alpha grid;
* метрики вариантов `0.15`, `0.17` и `0.18`;
* исходный TTA solution template;
* исходные README и MANIFEST.

Recovery snapshot не редактировался. Канонические файлы были скопированы из него в основные директории версии.

Исходный архив:

```text
solution_source_lb554.zip
```

Размер:

```text
141472 bytes
```

SHA-256:

```text
d6dc53d7cd5144664a2bc4d434cf033dd6ea65ec1c6b7e11afb86ad08460e2d9
```

## Original solution

Папка:

```text
solution/original
```

содержит solution в том виде, в котором он был получен от автора модели.

Файл весов:

```text
solution/original/models/user-bge-human-ft/model.safetensors
```

Размер:

```text
1436159092 bytes
```

SHA-256:

```text
b9d98750751a2442a946f6d9188fa3cda251df4d818ad0b23fa13abf223469da
```

Вес хранится локально и исключён из Git правилом `*.safetensors`.

Облегчённый архив solution без весов:

```text
matching-bge-human-ft-v3-soup-tta.zip
```

Размер:

```text
825859 bytes
```

SHA-256:

```text
c94933a30e8a8e7381af0bac1d064a2464d9be91797f409196c942fa09a1abc4
```

Полная папка solution и облегчённый ZIP были побайтово сравнены. За исключением `model.safetensors` и служебных файлов macOS их содержимое совпадает.

## Сборка submission

В репозитории сохранены три варианта сборки.

Исходный исторический сборщик:

```text
construction/original/build_submit.py
```

Поздний технически усиленный runtime-сборщик:

```text
construction/runtime/build_submit.py
```

Сборщик фактически выбранного V3 после ручного
исправления corner case:

```text
construction/fix/build_submit.py
```

Точный fixed-runtime без весов находится в:

```text
construction/fix/runtime
```

Именно `construction/fix/build_submit.py` следует
использовать для воспроизведения выбранного V3.

Он:

- проверяет SHA-256 каждого runtime- и model-файла;

- использует точный исторический `run.py`;

- использует исправленный исторический `src/utils.py`;

- сохраняет строки с пустыми текстами, их порядок
  и дубликаты;

- кладёт модель в `models/user-bge-human-ft`;

- создаёт плоский ZIP с `metadata.json` в корне;

- исключает `.DS_Store`, `__MACOSX` и другие
  посторонние файлы;

- повторно проверяет структуру и SHA-256 каждого
  файла внутри готового ZIP;

- не выполняет сетевых запросов;

- поддерживает ZIP64 и фиксированные timestamps.

Две последовательные локальные сборки дали одинаковый
архив:

```text
size_bytes: 1324785706
sha256: 8be6ae608eaa80a64522c26522a00f58e35a50df1ac41276779b5640fbd63879
```

Полный SHA-256 ZIP относится к данной воспроизведённой
сборке. Историческому platform ZIP побайтовое совпадение
не приписывается, поскольку его внешние ZIP-метаданные
не сохранились.

Подробности находятся в:

```text
construction/fix/sad_story.md
```

Большой submission ZIP не хранится в Git.

## Исправленный runtime

Папка:

```text
solution/runtime
```

содержит исправленную рабочую копию исходного runtime.

Исправления:

* добавлены официальные CLI-аргументы с дефисами;
* сохранены старые аргументы через подчёркивание;
* модель загружается только локально;
* все входные пары сохраняются;
* сохраняется исходный порядок;
* сохраняются повторяющиеся пары;
* пары с отсутствующими товарами не удаляются;
* пары с пустым текстом не удаляются;
* выходной столбец называется `predict`;
* preprocessing не изменён;
* symmetry TTA не изменена.

Подробная инструкция находится в:

```text
solution/runtime/README.md
```

## Формат запуска

```bash
python -u run.py \
  --items-path /path/to/items.parquet \
  --matches-path /path/to/matches.parquet \
  --output-path /path/to/submit.csv
```

Формат результата:

```text
id1,id2,predict
```

## Данные

Ожидаемые файлы:

```text
items.parquet
items_human.parquet
matches.parquet
matches_llm.parquet
```

Данные не хранятся в Git.

Reported dataset sizes:

```text
items:          13397761 rows
manual matches: 365654 rows
LLM matches:    11187780 rows
```

Дополнительная пользовательская LLM-переразметка для этого кандидата не выполнялась.

## Окружение

```text
Python:        3.12
torch:         2.6.0+cu124
transformers:  4.57.6
numpy:         2.2.6
pandas:        2.3.3
pyarrow:       23.0.1
scikit-learn:  1.8.0
GPU:           2 × NVIDIA H100 PCIe
precision:     bfloat16
backend:       NCCL
```

`config.json` финального solution сообщает версию экспорта `transformers 5.6.2`. Это отдельно отражено в `training/environment.json`.

## Ограничения воспроизводимости

В репозиторий не включены крупные training checkpoint:

* Stage-B v3 `step_00400.pt`;
* Stage-B v3 `step_00600.pt`;
* финальный soup `best.pt`;
* Stage-A initialization `best.pt`.

Финальный deploy-файл `model.safetensors` доступен локально, проверен по SHA-256 и исключён из Git.

Revision базовой модели `deepvk/USER-bge-m3` в исходном запуске не была зафиксирована. Поэтому повторное обучение в другой программной или аппаратной среде может не дать побитово идентичные веса.

Статический аудит исходников и runtime выполнен. Полный повторный запуск обучения и полный container benchmark должны проводиться отдельно.
