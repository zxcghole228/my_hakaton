# Ресерч веток команды: фичи и приёмы, которых нет в egor/llm-solution

Сжатая версия. Полные таблицы переносимости — спрашивайте у Егора.

## Берём сразу (код есть, переносится легко)

### Из zhuk (Даня, catboost V2 + CE):
- **Нормализация физических единиц** (0.5л=500мл, кг/г, см/мм, GB/TB, Вт, мАч) —
  30 фич, чистая функция, наши nums_jaccard это не ловят. [cell 1, QRE/quantities()]
- **Раздельные код-блоки**: article / manufacturer_article / oem / part_number / model
  как 5 отдельных спец-полей (у нас один общий пул кодов). [cell 1, SPECIAL]
- **Interaction-фичи**: hard_conf (max по 10 жёстким конфликтам), lex_mean,
  name_num_conf, name_attr_conf; идея semantic_conflict = CE-скор × hard_conf
  («похожи, но вариант другой» — прицельно фэшн). [cell 1, INTERACTIONS]
- **Абсолютные счётчики** и тройки both/exact/disjoint. [cell 1, BASE_F]
- rank-нормализация перед блендом; category experts + per-category rank-blend.
- CE: категория в тексте, det-сабсэмпл по хэшу, category-balanced loss, swap p=0.5,
  swap-TTA, stage-B на LR 3e-6 + rank-blend (способ вернуть пользу stage B — перепроверить).

### Из minicooper1 (Мишаня, e5-small):
- **Confidence weighting** LLM-меток: w = 0.75 + 0.25*|2y-1| — применимо и к CatBoost
  как sample_weight. [cell 13]
- **Нормализация текста в CE**: lower, ё→е, ×/х→x, ,→. (у нас сырой текст!).
- **KEY_ORDER 35 ключей** с фэшн-ключами: рост, обхват, пол, сезон. MAX_ATTR 460, MAX_LEN 192.
- Loss с нормировкой на сумму весов батча (не .mean()) — LR не плывёт.
- Balanced fast-val (3000 пар/категорию) для выбора чекпойнта.
- Инференс: потоковая загрузка ТОЛЬКО нужных товаров, bf16-autocast по compute
  capability, fallback токенайзера, валидация выхода. [solution/src/pipeline.py]

## Рецепт большого CE-прогона (собран из проверенного):
категория в тексте + нормализация + maxlen 192 / attrs 460 + фэшн-ключи +
swap p=0.5 + category-balanced + confidence weighting + differential LR
(head 1e-4 / backbone 3e-5) + best-checkpoint по balanced fast-val + 2 эпохи.

## Баги в чужом коде (не тащить как есть)
1. zhuk category-attrs: `A[i, j+1 if eq else j+2-j]` — все conflict-флаги пишутся
   в колонку 2. Переносить с фиксом.
2. zhuk jac(): пустые множества -> 1.0 (у нас 0.0) — проверить при переносе.
3. minicooper fashion boost: работал только на Обувь/Одежда (имена категорий) — известно.
4. zhuk использует difflib — при переносе менять на rapidfuzz (в разы быстрее).

## Чего в репо НЕТ (просить у авторов)
- zhuk: catboost v3 (кода нет вообще), артефакты v2 (tfidf/pca/веса), метрики,
  feature importance топ-50 (критично: какие из 400+ фич работают), инференс v2.
  ВАЖНО: V2 обучался на ручных парах и не валидировался на LLM-распределении.
- minicooper: **веса e5-small** (e5_macro_v2_30k_export.zip) — нужны для стека!
  Результаты fashion continuation (код есть, не запускался).
