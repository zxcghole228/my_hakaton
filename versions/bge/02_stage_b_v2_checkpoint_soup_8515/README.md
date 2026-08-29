# Stage-B v2 USER-BGE-M3 checkpoint soup

Вторая версия BGE для задачи матчинга товаров E-CUP 2026.

Модель продолжает обучение первого baseline-кандидата на ручной разметке и LLM replay. Во время Stage B сохранялись промежуточные checkpoint, после чего два выбранных состояния модели были объединены в один checkpoint soup.

Public leaderboard score: `0.5520634383301408`.

## Итоговая модель

Финальная модель получена по формуле:

```text
0.15 × step_00400.pt + 0.85 × step_02000.pt
```

Веса смешиваются на уровне `state_dict`. После этого создаётся одна итоговая модель и выполняется экспорт в формат Hugging Face `model.safetensors`.

Это не ансамбль предсказаний. Во время инференса загружается одна модель.

Symmetry TTA в отправленном solution не используется.

## Цепочка обучения

```text
Stage-A baseline checkpoint
→ расчёт teacher probabilities
→ Stage-B v2: human labels + LLM replay
→ checkpoint sweep
→ step_00400.pt + step_02000.pt
→ checkpoint soup 0.15 / 0.85
→ model.safetensors
→ submit.zip
```

Основные параметры фактического запуска:

- base model: `deepvk/USER-bge-m3`;
- maximum sequence length: `320`;
- seed: `20260825`;
- Stage-B epochs: `2`;
- обучение: `2 × NVIDIA H100`;
- checkpoint interval: `400` optimizer steps;
- preprocessing: `pair_text_v1`;
- финальный инференс: single pass, без symmetry TTA.

## Данные

Pipeline использует предоставленные организаторами данные:

```text
items.parquet
items_human.parquet
matches.parquet
matches_llm.parquet
```

`matches.parquet` содержит ручную разметку.

`matches_llm.parquet` содержит предоставленную организаторами вероятностную LLM-разметку. Pipeline не вызывает внешние LLM, API или проприетарные сервисы.

Файлы соревнования не публикуются в этой версии репозитория. Их необходимо разместить локально и передать пути через переменные окружения.

## Stage-B v2

Обучение включает:

- разбиение ручной разметки по связанным компонентам товаров;
- отдельные `train`, `tune` и `eval` части;
- дополнительный вес проблемных fashion-категорий;
- повторное добавление товаров категории «Обувь»;
- hard и anchor примеры из LLM replay;
- distillation от Stage-A модели;
- supervised loss;
- ranking loss;
- symmetry consistency loss во время обучения;
- оценку сохранённых checkpoint на полной gray-выборке.

Итоговая метрика выбора checkpoint:

```text
0.50 × problem_categories_ap
+ 0.30 × full_gray_ap
+ 0.20 × tune_macro_ap
```

Symmetry loss при обучении не означает использование symmetry TTA во время финального инференса.

## Teacher cache

`teacher_probs.npz` является промежуточным кешем, а не внешним источником разметки.

Он воспроизводимо создаётся Stage-A моделью:

```bash
CUDA_VISIBLE_DEVICES=0 \
python training/reconstructed/train_bge_stageb_v2.py \
  --precompute-teacher
```

После этого запускается распределённое обучение:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
  --nproc_per_node=2 \
  training/reconstructed/train_bge_stageb_v2.py
```

Пути к данным и артефактам задаются переменными:

```text
STAGE_B_ITEMS_PATH
STAGE_B_ITEMS_HUMAN_PATH
STAGE_B_MATCHES_HUMAN
STAGE_B_MATCHES_LLM
STAGE_B_INIT_CKPT
STAGE_B_OUT_DIR
```

## Структура версии

```text
02_stage_b_v2_checkpoint_soup_8515/
├── README.md
├── version.json
├── artifacts/
│   └── manifest.json
├── construction/
│   ├── original/
│   └── runtime/
├── recovery/
│   └── executed_source/
├── runs/
│   └── v2_soup_s42_01/
├── solution/
│   ├── original/
│   └── runtime/
└── training/
    ├── environment.json
    ├── source_manifest.json
    ├── original/
    └── reconstructed/
```

### `training/original`

Содержит публичный notebook, полученный от автора модели без изменения содержимого.

### `recovery/executed_source`

Содержит полный восстановленный snapshot:

- training-код;
- checkpoint scoring;
- blending;
- preprocessing;
- сборщик solution;
- сохранённые логи;
- метрики;
- шаблон отправленного solution.

Snapshot сохранён отдельно, чтобы не выдавать восстановленные файлы за побайтово сохранившиеся первоначальные исходники.

### `training/reconstructed`

Содержит минимальный набор восстановленных скриптов, соответствующий сохранённым выводам notebook и фактическому запуску.

### `runs/v2_soup_s42_01`

Содержит сохранённые результаты Stage-B v2 и checkpoint soup:

- training log;
- checkpoint evaluations;
- checkpoint sweep;
- best-checkpoint metadata;
- soup grid;
- итоговые локальные метрики.

### `solution/original`

Содержит исходный solution в том виде, в котором он был получен от автора и отправлен на соревнование.

### `solution/runtime`

Содержит исправленную версию runtime:

- официальный аргумент `--output-path`;
- локальная загрузка модели;
- сохранение всех входных пар;
- сохранение порядка и повторяющихся пар;
- результат `id1,id2,predict`;
- неизменённый preprocessing `pair_text_v1`.

## Артефакты

Исторические checkpoint:

```text
step_00400.pt
SHA-256: 517c8f0ac9b4d74c982d31157914c876bf541803bef2fff9b79e40ec615bbaec
size_bytes: 1436274567

step_02000.pt
SHA-256: 553d3a165aca6db16c26244a4eefe9ff6d57f7687ef5a8031b398711956f37a5
size_bytes: 1436274567

soup best.pt
SHA-256: c842d53b45dec9aa00419dfc5f7a2bd911e1163d7273eecd31bf7f76fb208c18
size_bytes: 1436212729
```

Финальный экспорт:

```text
model.safetensors
SHA-256: d51bd8b0170a4a0d307d803050fa8aa1bc525fd2780e8b4faef6a13e8c6e3d93
size_bytes: 1436159092
```

Отправленный solution:

```text
matching-bge-human-ft-submit.zip
SHA-256: 908f60451f01463d7574e488c4655bea2b52bbac7fa19be43842a8e0a3ee5260
size_bytes: 1330444872
```

Финальные веса находятся локально в:

```text
solution/original/models/user-bge-human-ft/model.safetensors
```

Они исключены из обычного Git с помощью `.gitignore`.

## Лицензия базовой модели

`deepvk/USER-bge-m3` опубликована под лицензией Apache-2.0:

<https://huggingface.co/deepvk/USER-bge-m3>

Использование базовой модели не требует проприетарных API или закрытого inference-сервиса.

## Ограничения воспроизводимости

Первоначальный исполнявшийся training-скрипт не сохранился как отдельный побайтовый файл. Представленный код восстановлен по выполненному notebook, сохранённым логам, метрикам и связанным source snapshot.

Промежуточные `step_00400.pt`, `step_02000.pt` и `soup best.pt` не хранятся в Git. Для них сохранены SHA-256 и размеры.

Финальный отправленный `model.safetensors` и полный solution сохранились и имеют подтверждённые контрольные суммы.