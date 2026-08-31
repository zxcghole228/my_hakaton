# v2 checkpoint soup — LB **0.5522**

Воспроизведение лучшего **submitted** soup: `0.85·step_2000 + 0.15·step_400` (v2 checkpoints).

## Важно: pair_text **v1**

Эти модели обучены с **`pair_text v1`** (KEYS: бренд первым, attr cap **520**).  
**Нельзя** использовать `ecup_v2/pair_text.py` (v2) — тексты будут другими → LB сломается.

Проверка: `python ../lib/verify_pair_text.py`

## Структура

```
v2_soup_lb_5522/
├── README.md
├── pipeline.ipynb      ← Run All
├── build_submit.py     # zip, pair_text v1, NO TTA
├── submit/             # inference template
└── output/             # soup run dir (создаётся notebook)
```

## Данные и чекпойнты (на сервере)

| Артефакт | Путь |
|----------|------|
| parquets | `items*.parquet`, `matches*.parquet` |
| v2 step_400 / 2000 | `final_4_models/runs/01_v2_step2000/checkpoints/` |
| готовый soup | `final_4_models/runs/02_v2_soup/` |
| LB submit sha | `2c32c73ad2ae540d378db16087893bbca52d7f15fa78f53b95017e3387c2bdf1` |

## Pipeline (кратко)

1. **verify** pair_text v1  
2. *(опционально)* Stage-B v2 train → `final_4_models/scripts/train_bge_stageb_v2.py`  
3. **blend** `blend_checkpoint_soup.py --alphas 0.15,...`  
4. **pack** `python build_submit.py --run-dir output/soup_run`

## Submit

- **pair_text v1** в `submit/src/pair_text.py`
- **Без symmetry TTA** (как LB 0.5522)
- import: `from src.pair_text import build_text`

## Ожидания offline

| Метрика | v2 solo | v2 soup α=0.15 |
|---------|---------|----------------|
| LB | 0.5516 | **0.5522** |
| problem | ~0.642 | ~0.635 |
| gray | ~0.550 | ~0.551 |

Полный retrain: ±0.002–0.005 LB.
