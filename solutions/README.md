# Готовые решения

В этой папке независимо сохранены обе рабочие версии E5 submission. Они не
перезаписывают друг друга и имеют собственные `run.py`, `metadata.json`, код
preprocessing/inference и локальную папку `models/`.

## `e5_small_v2/`

Проверенный production fallback, уже отправленный на leaderboard.

- checkpoint: E5-small step 30 000;
- preprocessing V2, `MAX_LEN=192`;
- Full LLM Macro PR-AUC: `0.786482334`;
- Public LB: `0.4838757641`;
- локальный архив: `e5_small_macro_v2_30k_submission.zip` в корне репозитория.

## `e5_small_v3_hybrid/`

Новый hybrid solution: V2 base плюс V3 fashion specialist.

- specialist checkpoint: step 4 000;
- Hybrid Full LLM Macro PR-AUC: `0.790182347`;
- specialist используется для `Обувь`, `Одежда` и `Ювелирные изделия`;
- base используется для остальных категорий, включая
  `Галантерея и аксессуары`;
- локальный архив: `e5_small_macro_v3_hybrid_submission.zip` в корне
  репозитория.

Оба entry point принимают одинаковые аргументы соревнования:

```text
--items_path
--matches_path
--output_path
```

Веса, tokenizer и ZIP-архивы игнорируются Git. После обычного клонирования их
нужно восстановить по инструкциям в README выбранного решения и
`models/README.md`.
