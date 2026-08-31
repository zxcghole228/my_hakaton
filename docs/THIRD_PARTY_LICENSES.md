# Сторонние компоненты и лицензии

Этот документ перечисляет основные сторонние модели и библиотеки, непосредственно используемые BGE-решениями E-CUP 2026.

Проверка выполнена по официальным страницам проектов 31 августа 2026 года.

Документ не заменяет полные тексты лицензий. При распространении Docker-образа, Python wheels или других бинарных сборок необходимо сохранять входящие в них `LICENSE` и `NOTICE`.

## Базовая модель

| Компонент            | Использование                       | Лицензия   | Официальный источник                                                 |
| -------------------- | ----------------------------------- | ---------- | -------------------------------------------------------------------- |
| `deepvk/USER-bge-m3` | Базовая модель всех трёх кандидатов | Apache-2.0 | [Hugging Face model card](https://huggingface.co/deepvk/USER-bge-m3) |

Все финальные BGE-веса получены дообучением или объединением checkpoint этой модели.

Точная исходная ревизия модели не была записана. Это ограничение отдельно отмечено в `version.json` и документации воспроизводимости.

## Основные библиотеки

| Компонент     | Использование                | Основная лицензия                  | Официальный источник                                                                   |
| ------------- | ---------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------- |
| Python        | Язык выполнения              | PSF License Version 2              | [CPython LICENSE](https://github.com/python/cpython/blob/main/LICENSE)                 |
| PyTorch       | Обучение и инференс          | BSD-style, дополнительные notices  | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE)                |
| Transformers  | Модель и tokenizer           | Apache-2.0                         | [Transformers LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE)  |
| Tokenizers    | Быстрый tokenizer            | Apache-2.0                         | [Tokenizers LICENSE](https://github.com/huggingface/tokenizers/blob/main/LICENSE)      |
| Safetensors   | Формат финальных весов       | Apache-2.0                         | [Safetensors LICENSE](https://github.com/safetensors/safetensors/blob/main/LICENSE)    |
| SentencePiece | XLM-RoBERTa tokenizer        | Apache-2.0                         | [SentencePiece LICENSE](https://github.com/google/sentencepiece/blob/master/LICENSE)   |
| NumPy         | Работа с массивами           | BSD-3-Clause для основного проекта | [NumPy license metadata](https://github.com/numpy/numpy/blob/main/pyproject.toml)      |
| pandas        | Табличные данные             | BSD-3-Clause для основного проекта | [pandas LICENSE](https://github.com/pandas-dev/pandas/blob/main/LICENSE)               |
| PyArrow       | Чтение Parquet               | Apache-2.0, дополнительные notices | [Apache Arrow LICENSE](https://github.com/apache/arrow/blob/main/LICENSE.txt)          |
| scikit-learn  | Метрики обучения и валидации | BSD-3-Clause                       | [scikit-learn COPYING](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| tqdm          | Индикаторы выполнения        | MPL-2.0 и MIT notices              | [tqdm LICENCE](https://github.com/tqdm/tqdm/blob/master/LICENCE)                       |

## Что проверено

Статический анализ исходных Python-файлов и code cells notebook подтвердил прямое использование:

```text
numpy
pandas
pyarrow
sklearn
torch
tqdm
transformers
```

`tokenizers`, `sentencepiece` и `safetensors` используются модельным форматом или устанавливаются как зависимости Transformers.

В training- и runtime-коде не обнаружено импортов:

```text
tensorflow
catboost
xgboost
lightgbm
```

Также runtime не содержит вызовов платных или закрытых inference API.

## Транзитивные зависимости

Этот список не является полным software bill of materials Docker-образа.

Python wheels могут включать дополнительные компоненты и лицензии. Например:

* NumPy wheels могут содержать сторонние математические библиотеки;
* pandas распространяется с дополнительными license notices;
* Apache Arrow содержит собственный `NOTICE` и список включённых компонентов;
* PyTorch содержит дополнительные notices для встроенных и связанных проектов.

Полный аудит распространяемого Docker-образа должен выполняться по фактическому содержимому конкретного image digest, а не только по прямым импортам репозитория.

## Docker-образ

Все исходные submission ссылаются на:

```text
odsai/ecup26-matching-baseline:1.0
```

Это внешний Docker-образ, указанный в `metadata.json`. Репозиторий не содержит его слои и не может независимо подтвердить полный перечень находящихся внутри пакетов.

Перед финальной проверкой необходимо убедиться, что:

* образ доступен площадке;
* используется ожидаемый tag или digest;
* его использование разрешено организаторами;
* содержащиеся в нём license notices сохранены поставщиком образа.

## CUDA и системное окружение

Обучение выполнялось с PyTorch `2.6.0+cu124` на NVIDIA H100.

NVIDIA driver и CUDA runtime являются системной инфраструктурой вычислительной площадки и не распространяются в этом Git-репозитории.

Если требование об отсутствии проприетарного ПО распространяется также на системную GPU-инфраструктуру, допустимость CUDA необходимо отдельно подтвердить у организаторов. Репозиторий не скрывает её использование.

## Данные

Лицензии библиотек и модели не определяют права на распространение датасетов.

Файлы соревнования:

```text
items.parquet
items_human.parquet
matches.parquet
matches_llm.parquet
```

не добавляются в Git. Их использование и распространение регулируются правилами E-CUP 2026.

## Код команды

Этот документ описывает только сторонние компоненты и не назначает лицензию собственному коду команды.

Лицензия собственного кода должна быть оформлена отдельным корневым файлом `LICENSE` после согласования всеми правообладателями команды. До появления такого файла нельзя автоматически считать код опубликованным под Apache-2.0 или другой open-source лицензией.

## Итог

Основная модель и непосредственно используемые Python-библиотеки опубликованы под открытыми лицензиями.

Открытыми вопросами остаются:

* полный состав и notices Docker-образа;
* допустимость системной CUDA-инфраструктуры в трактовке правил;
* лицензия собственного кода команды;
* условия распространения датасетов соревнования.
