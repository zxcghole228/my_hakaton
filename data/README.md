# Локальные данные

В этой папке лежат входные данные соревнования и локальные результаты. Все
файлы, кроме этого README, игнорируются Git.

Текущая локальная раскладка:

```text
data/
├── items_human.parquet   # обучающие товары
├── matches.parquet       # обучающие пары
├── items_test.parquet    # тестовые товары для smoke/full inference
├── matches_test.parquet  # тестовые пары
└── submit_test.csv       # локальный generated output
```

Не добавляйте datasets и predictions в обычный коммит. Если схема данных или
название файла меняются, обновите этот README и примеры запуска решений.
