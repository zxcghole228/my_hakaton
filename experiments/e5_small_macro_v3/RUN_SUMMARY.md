# E5-small V3 — Fashion Specialist + Hybrid Routing

## Статус

Обучение и полная локальная валидация завершены. Модели, tokenizer и routing
экспортированы. Production submission pipeline для одновременной загрузки base
и specialist ещё не собран: текущий `solution/` продолжает содержать V2.

## Отправная точка V2

- Backbone: `intfloat/multilingual-e5-small`
- Checkpoint: step 30 000
- Public LB: `0.4838757641`
- Full LLM group holdout Macro PR-AUC: `0.7864823339791696`
- `MAX_LEN`: 192
- `MAX_ATTR_CHARS`: 460

Specialist инициализирован точными весами V2, обучение с нуля не выполнялось.

## Что исправлено и добавлено в V3

- Использованы точные названия четырёх fashion-категорий:
  - `Галантерея и аксессуары`
  - `Обувь`
  - `Одежда`
  - `Ювелирные изделия`
- Добавлены variant-сигналы: размер, цвет, артикул, модель, пол и материал.
- Для размера и цвета предусмотрено извлечение из названия, если атрибуты
  отсутствуют.
- Собран balanced candidate pool по 200 000 пар на категорию.
- Выполнен hard-example mining на predictions базовой V2.
- Итоговый train содержит до 150 000 пар на категорию:
  - hard: 50% (`75 000`);
  - random: 30% (`45 000`);
  - stable: 20% (`30 000`).
- Hybrid routing выбирает specialist только там, где он устойчиво лучше base.

## Обучение

| Параметр | Значение |
|---|---:|
| GPU | Tesla T4, 14.6 GB |
| Fashion train source rows | 2 227 727 |
| Balanced candidate pool | 800 000 |
| Итоговый mined train | 600 000 |
| Epochs | 2 |
| Optimizer steps | 4 688 |
| Best specialist step | 4 000 |
| Backbone LR | `8e-6` |
| Head LR | `3e-5` |
| Effective batch | 256 |
| Prediction batch | 64 |
| Время training | 1.78 h |

Group split, LLM validation и V2 preprocessing сохранены совместимыми с V2.

## Результаты

| Проверка | Macro PR-AUC |
|---|---:|
| V2 base fast LLM | `0.786378890` |
| Best V3 fast hybrid | `0.790608644` |
| V2 base full LLM | `0.786482334` |
| V3 specialist fashion full | `0.504527989` |
| **V3 hybrid full LLM** | **`0.790182347`** |
| **Delta V3 vs V2** | **`+0.003700013`** |

Грубый LB-сигнал через историческое отношение V2 public LB к local metric:
`0.486152`. Это только эвристика, а не предсказание leaderboard score.

## Routing

```json
{
  "Галантерея и аксессуары": "base",
  "Обувь": "specialist",
  "Одежда": "specialist",
  "Ювелирные изделия": "specialist"
}
```

Хотя specialist улучшил full AP для всех четырёх категорий, routing зафиксирован
по fast validation с safety margin `0.002`. Поэтому для `Галантерея и
аксессуары` оставлена базовая модель.

## Fashion по категориям

| Категория | V2 base AP | V3 specialist AP | Delta | Route |
|---|---:|---:|---:|---|
| Галантерея и аксессуары | `0.541161` | `0.547544` | `+0.006383` | base |
| Обувь | `0.323267` | `0.348282` | `+0.025015` | specialist |
| Одежда | `0.431439` | `0.457674` | `+0.026235` | specialist |
| Ювелирные изделия | `0.641862` | `0.664612` | `+0.022751` | specialist |

## Файлы эксперимента

- `RUN_SUMMARY.md` — этот проверенный отчёт.
- `KAGGLE_RUN_SUMMARY.md` — исходный автоматически созданный summary.
- `metrics.json` — точные параметры и агрегированные метрики.
- `routing.json` — production routing и параметры preprocessing.
- `fashion_comparison.csv` — base/specialist по fashion-категориям.
- `hybrid_full_by_category.csv` — итоговый hybrid по всем 20 категориям.
- `huggingface_source.json` — точная ревизия исходного backbone.
- `artifacts.json` — размеры, SHA-256, provenance и проверки дублей.
- `notebooks/e5_small_macro_v3_fashion_specialist_hybrid.ipynb` — полный
  training/mining/validation/export notebook.

## Локальные артефакты

```text
models/e5_small_macro_v3/
├── checkpoints/
│   ├── e5_v3_best_specialist.pt
│   └── e5_v3_resume.pt
└── exports/
    └── e5_v3_hybrid_export.zip
```

Они документированы в `models/README.md` и игнорируются Git.

## Следующий технический шаг

Собрать отдельный V3 submission pipeline, который:

1. строит V2 text и variant-сигналы точно как в notebook;
2. загружает base и fashion specialist offline;
3. маршрутизирует пары по категории первого товара через `routing.json`;
4. сохраняет все строки в формате `id1,id2,predict`;
5. проходит smoke-test и full-size benchmark до замены текущего V2 `solution/`.
