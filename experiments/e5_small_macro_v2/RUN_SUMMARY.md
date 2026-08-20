# E5-small Macro LLM Stage-A V2 — checkpoint 30k

## Конфигурация

- Backbone: `intfloat/multilingual-e5-small`
- Обучение: LLM stage-A only
- Checkpoint: step 30 000
- `MAX_LEN`: 192
- `MAX_ATTR_CHARS`: 460
- LLM validation pairs: 191 555
- Manual validation pairs: 72 948
- Random pair swap during training: 0.5
- Loss: category-balanced soft BCE
- Labels: confidence-weighted LLM soft labels
- Manual stage-B: not used

Исходная модель зафиксирована на ревизии Hugging Face `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Машинно-читаемые сведения находятся в `huggingface_source.json`.

## Результаты

| Валидация | Macro PR-AUC |
|---|---:|
| Fast LLM group holdout, записанный при обучении | 0.786379 |
| Fast LLM group holdout после восстановления | 0.786379 |
| Full LLM group holdout | 0.786482 |
| Manual group holdout, diagnostic only | 0.677024 |
| Fashion subset | 0.484432 |
| Non-fashion subset | 0.861995 |

Fast validation воспроизведена без расхождения: delta = `0.0`.

## Fashion-категории

- `Галантерея и аксессуары`
- `Обувь`
- `Одежда`
- `Ювелирные изделия`

В исходном training V2 дополнительный fashion boost не применился к двум категориям из-за несовпадения точных строк: `Галантерея и аксессуары` и `Ювелирные изделия`. При этом общая macro-aware category weighting работала.

## Файлы

- `metrics.json` — машинно-читаемая конфигурация и агрегированные метрики.
- `llm_by_category.csv` — full LLM group holdout по категориям.
- `fashion_by_category.csv` — fashion subset по категориям.
- `manual_by_category.csv` — diagnostic manual holdout по категориям.
- `huggingface_source.json` — точная ревизия исходного backbone и список полученных файлов.

Ноутбуки:

- `notebooks/e5_small_macro_llm_stageA_v2.ipynb` — обучение и выбор checkpoint.
- `notebooks/e5_small_restore30k_fullval_export.ipynb` — восстановление, полная валидация и экспорт.

Локальные веса и архивы перечислены в `models/README.md` и не должны попадать в Git.
