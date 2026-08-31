# Сторонние компоненты и лицензии

Документ перечисляет основные сторонние компоненты, используемые BGE-решениями команды для E-CUP 2026.

Сведения проверены по официальным страницам проектов 31 августа 2026 года. Документ не заменяет полные тексты лицензий. При распространении моделей, Python-пакетов, Docker-образов и других бинарных материалов необходимо сохранять применимые файлы `LICENSE`, `NOTICE` и уведомления об авторских правах.

## Базовая модель

| Компонент | Использование | Лицензия | Источник |
|---|---|---|---|
| `deepvk/USER-bge-m3` | Базовая модель BGE-решений | Apache-2.0 | [Hugging Face model card](https://huggingface.co/deepvk/USER-bge-m3) |

Представленные BGE-веса получены дообучением или объединением контрольных точек этой модели.

Точная ревизия базовой модели, использованная при обучении, не была сохранена. Это ограничение зафиксировано в `version.json`.

## Основные библиотеки

| Компонент | Использование | Лицензия | Источник |
|---|---|---|---|
| Python | Среда выполнения | PSF License Version 2 | [CPython LICENSE](https://github.com/python/cpython/blob/main/LICENSE) |
| PyTorch | Обучение и инференс | BSD-style и дополнительные notices | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers | Работа с моделью и tokenizer | Apache-2.0 | [Transformers LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| Tokenizers | Токенизация | Apache-2.0 | [Tokenizers LICENSE](https://github.com/huggingface/tokenizers/blob/main/LICENSE) |
| Safetensors | Формат хранения весов | Apache-2.0 | [Safetensors LICENSE](https://github.com/safetensors/safetensors/blob/main/LICENSE) |
| SentencePiece | Токенизация XLM-RoBERTa | Apache-2.0 | [SentencePiece LICENSE](https://github.com/google/sentencepiece/blob/master/LICENSE) |
| NumPy | Работа с массивами | BSD-3-Clause | [NumPy license metadata](https://github.com/numpy/numpy/blob/main/pyproject.toml) |
| pandas | Обработка таблиц | BSD-3-Clause | [pandas LICENSE](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| PyArrow | Чтение Parquet | Apache-2.0 и дополнительные notices | [Apache Arrow LICENSE](https://github.com/apache/arrow/blob/main/LICENSE.txt) |
| scikit-learn | Метрики и валидация | BSD-3-Clause | [scikit-learn COPYING](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| tqdm | Отображение прогресса | MPL-2.0 и MIT notices | [tqdm LICENCE](https://github.com/tqdm/tqdm/blob/master/LICENCE) |

Список охватывает основные непосредственно используемые компоненты, но не является полным перечнем транзитивных зависимостей Docker-образа. Отдельные пакеты могут содержать дополнительные лицензионные уведомления.

## Среда выполнения

Решения используют Docker-образ `odsai/ecup26-matching-baseline:1.0`.

Обучение выполнялось с PyTorch `2.6.0+cu124` на NVIDIA H100. Docker-образ, драйверы NVIDIA и CUDA Runtime поставляются отдельно и не входят в состав репозитория. Их использование регулируется лицензиями соответствующих поставщиков.

## Данные соревнования

В решениях используются следующие файлы:

- `items.parquet`;
- `items_human.parquet`;
- `matches.parquet`;
- `matches_llm.parquet`.

Датасеты не добавляются в Git. Права на их использование и распространение определяются правилами E-CUP 2026 и условиями предоставления данных организаторами.

## Код команды

Документ описывает только сторонние компоненты и не предоставляет лицензию на собственный код команды.

Если команда решит лицензировать собственный код, условия должны быть зафиксированы в отдельном корневом файле `LICENSE` после согласования со всеми авторами.
