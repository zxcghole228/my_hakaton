# Solution

Здесь будет находиться итоговый submission pipeline.

Сейчас рабочий CrossEncoder baseline намеренно оставлен в корне репозитория:

- `run.py`
- `src/utils.py`
- `metadata.json`
- `baseline_logreg_l12.joblib`

Это сохраняет исходную конфигурацию `python -u run.py` и относительные пути до локальных весов.

Экспорт E5-small уже сохранён локально и задокументирован, но полноценный production pipeline для него ещё не собран. Когда он будет готов, целевая структура будет такой:

```text
solution/
├── run.py
├── preprocessing.py
├── model.py
└── metadata.json
```

Перенос baseline или замена корневой точки входа должны выполняться только вместе с проверкой запуска на тестовых данных.
