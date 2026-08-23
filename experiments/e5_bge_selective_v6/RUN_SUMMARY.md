# V6.2 — E5 Stage-B + selective BGE reranking

## Гипотеза

E5 Stage-B лучше согласован с human labels, а BGE-M3 Stage-A даёт независимый
семантический сигнал. Полный ансамбль двух cross-encoder не укладывается в лимит
inference, поэтому BGE используется только для reranking верхней части списка E5
внутри каждой категории.

## Данные и validation

- human component-safe untouched: `72 948` пар;
- untouched разделён по компонентам на tune/eval половины;
- exact LLM component holdout: `191 555` пар;
- веса и top fraction подбирались по Macro PR-AUC;
- финальный Public LB не использовался для обучения моделей.

## Модели

- E5-base@320 Human Stage-B, swap-TTA — все пары;
- BGE-M3@320 Stage-A, swap-TTA — только top-10% пар по E5 внутри категории;
- внутри выбранной области: `0.80 * rank(E5) + 0.20 * rank(BGE)`;
- нижние 90% сохраняют исходный порядок E5.

## Validation

| Срез | E5 solo | Selective E5+BGE | Delta |
|---|---:|---:|---:|
| Manual tune half | 0.774751 | 0.775224 | +0.000473 |
| Manual eval half | 0.766715 | 0.768955 | +0.002240 |
| Manual full | 0.770112 | 0.771457 | +0.001344 |

Глобальный validation-optimum `top-30%, BGE=0.15` давал manual full `0.771864`,
но две версии такого контейнера превысили лимит времени. Поэтому в production
выбран немного более слабый, но существенно более быстрый top-10% вариант.

## Leaderboard

- предыдущий лучший сабмит: **0.5244670740**;
- V6.2 selective top-10%: **0.5290960422**;
- абсолютный прирост: **+0.0046289682**;
- относительный прирост: **+0.8826%**.

## Production-решение

Для прошедшего контейнера BGE читает и преобразует только товары, встречающиеся
в выбранных парах. Это устраняет лишнюю CPU-обвязку, описанную в
`docs/FOR_ZHUKOV_V61_TIMEOUT.md`. Модели не переобучались.

## Артефакты

- notebook: `notebooks/selective_e5_bge_reranking_v6.ipynb`;
- grid: `experiments/e5_bge_selective_v6/grid.csv`;
- metrics: `experiments/e5_bge_selective_v6/metrics.json`;
- локальный submission archive: `ecup-submit-e5-selective-bge-top10.zip`
  (в Git не добавляется из-за весов и размера).
