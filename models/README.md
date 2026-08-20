# Локальные модели

В этой папке локально хранятся веса baseline и экспериментальных моделей. Все вложенные модельные файлы исключены из Git; отслеживается только этот README.

## CrossEncoder baseline

```text
models/cross-encoder-ms-marco-MiniLM-L12-v2/
```

Каталог восстановлен из baseline-архива организаторов и нужен текущему корневому `run.py`.

## E5-small Macro V2

Checkpoint обучения:

```text
models/e5_small_macro_v2/checkpoints/e5_macro_v2_best.zip
```

- Формат: PyTorch checkpoint в ZIP-контейнере
- Размер исходного файла: 470 701 556 bytes
- SHA-256: `e3ceb948ddb41961dc5db08b55b317019074365836c337d9f3923d25ed1ce29c`

Экспорт Hugging Face после восстановления checkpoint 30k:

```text
models/e5_small_macro_v2/exports/e5_macro_v2_30k_export.zip
```

- Содержит `model.safetensors`, tokenizer, config, metrics и CSV по категориям
- Размер исходного файла: 361 191 807 bytes
- SHA-256: `926a31dc7c300e33735c1aa9e0172a2b7a8dd5a3b274250eea4021ea97152837`

Распакованный экспорт для прямой локальной загрузки:

```text
models/e5_small_macro_v2/hf_export_30k/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

- `model.safetensors` SHA-256: `b7263e2c9f39cf73bfd1217c91ef613b9cbc9529fa95a062784215d4c568d92c`
- Backbone revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`

Архив `e5_macro_v2_best_repacked.zip` в рабочую папку не переносился. Его внутренние 207 файлов совпадают с `e5_macro_v2_best.zip` по пути, размеру и CRC; различается только упаковка архива.

Исходный bundle `results.zip` также не переносился: вложенные export, checkpoint, метрики и таблицы дублируют уже разложенные файлы. Из него сохранён только распакованный Hugging Face export и сведения, необходимые для воспроизводимости.

## E5-small V3 Fashion Specialist

Лучший specialist checkpoint:

```text
models/e5_small_macro_v3/checkpoints/e5_v3_best_specialist.pt
```

- Best step: 4 000
- Размер: 470 702 271 bytes
- SHA-256: `0546814b8950782e7116248ac557ed67ebd04098e5c0f9362aa5d9a8bbca7ada`

Полный resume checkpoint после обучения:

```text
models/e5_small_macro_v3/checkpoints/e5_v3_resume.pt
```

- Содержит model, optimizer, scheduler, scaler и состояние обучения
- Размер: 1 412 119 762 bytes
- SHA-256: `0ae0d6a752893faa8bbd4131e03b1d730ffe2fc1a5618887d433aab62fcededf`

Hybrid export:

```text
models/e5_small_macro_v3/exports/e5_v3_hybrid_export.zip
```

- Содержит base model, fashion specialist, tokenizer, routing и метрики
- Размер: 718 451 718 bytes
- SHA-256: `ea84f56b9b0f6bfbabed5d3dfbe75292992234c8beffbd5f9141614c392d99a8`
- Base `model.safetensors` SHA-256 совпадает с V2:
  `b7263e2c9f39cf73bfd1217c91ef613b9cbc9529fa95a062784215d4c568d92c`
- Specialist `model.safetensors` SHA-256:
  `cb781e2a4d33cbe5825453fa9986f7f49b0a0c270b46cd4e22debf46449475ff`

Распакованные файлы, которые напрямую использует готовый solution:

```text
solutions/e5_small_v3_hybrid/models/e5_small_v3_hybrid/
├── base_model/
├── fashion_specialist/
├── tokenizer/
└── routing.json
```

Чистый submission archive:

```text
e5_small_macro_v3_hybrid_submission.zip
```

- Размер: 717 517 727 bytes
- SHA-256: `d81b1a03fddf2fa7b75dc6554fd2b12c5a0cf9a5ee915bd7e0c85050aec652f8`
- Содержит только 12 обязательных файлов, без wrapper-директории

В исходном `results (1).zip` также находились:

- `e5_macro_v2_best_repacked.pt` — подтверждён как точный дубликат уже
  сохранённого V2 checkpoint по всем 207 внутренним файлам и CRC;
- распакованный `e5_v3_hybrid_export/` — совпадает с вложенным export ZIP по
  всем 10 файлам, поэтому повторно не сохранялся.

Исходный bundle остался в `Downloads` и не перемещался.

## Правило Git

Перед `git add` полезно проверить:

```bash
git status --ignored --short
git check-ignore -v models/e5_small_macro_v2/checkpoints/e5_macro_v2_best.zip
git check-ignore -v models/e5_small_macro_v2/exports/e5_macro_v2_30k_export.zip
git check-ignore -v solutions/e5_small_v3_hybrid/models/e5_small_v3_hybrid/base_model/model.safetensors
git check-ignore -v e5_small_macro_v3_hybrid_submission.zip
```
