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

## Правило Git

Перед `git add` полезно проверить:

```bash
git status --ignored --short
git check-ignore -v models/e5_small_macro_v2/checkpoints/e5_macro_v2_best.zip
git check-ignore -v models/e5_small_macro_v2/exports/e5_macro_v2_30k_export.zip
```
