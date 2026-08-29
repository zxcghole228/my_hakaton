# Symmetry TTA + v3×v2 cross-soup

## BASE model (LB best **0.5539**)

| | |
|---|---|
| Run | `output/cross_tta_gray_a015/soup_run/` |
| Blend | `(1−0.15)·v2@2000 + 0.15·v3@400` + **symmetry TTA** |
| pair_text | **v1** |
| Manifest | `output/BASE_MODEL.json` |
| Notebook | `from_best_public.ipynb` — все эксперименты от этой точки |

Offline @α=0.15: gray **0.5544**, problem **0.6357**.

---

Воспроизведение **04_v3_soup_tta**: blend `0.90·v2_step2000 + 0.10·v3_step400` + **symmetry TTA** в submit.

## pair_text **v1** (обязательно)

Train v2/v3 и submit используют **v1** (`final_4_models/scripts/score_ensemble.py` или `notebooks/lib/pair_text_v1.py`).

Submit template: `final_4_models/submit/` — `build_text` inline v1 + avg(p(a,b), p(b,a)).

Проверка: `python ../lib/verify_pair_text.py`

## Структура

```
symmetry_tta_v3_soup/
├── README.md
├── pipeline.ipynb               ← internal (absolute paths)
├── pipeline_public.ipynb        ← full train + grid
├── reblend_gray_alpha_public.ipynb  ← re-blend фикс. α (без train)
├── build_submit.py
└── output/
```

Скрипты: `final_4_models/scripts/` (не копируем — pin версия в репо).

## Чекпойнты

| | Путь |
|---|------|
| v2 step_2000 | `final_4_models/runs/01_v2_step2000/checkpoints/step_02000.pt` |
| v3 step_400 | `user_bge_stageb_final_run/checkpoints/step_00400.pt` |
| готовый run | `final_4_models/runs/04_v3_soup_tta/` |
| submit sha | `9ba6879c77aaf133d68e2b85991e629c9e72edd62597b2faae1f3bc6ee32f06f` |

## Pipeline

1. verify pair_text v1  
2. *(опционально)* `train_bge_stageb_v3.py` от Stage-A `best.pt`  
3. blend v3 step_400 × v2 step_2000 — grid `0.05…0.30` + **`--symmetry-tta`**  
4. `python build_submit.py --run-dir output/<run>/soup_run`

**Commission:** `pipeline_public.ipynb` — берёт v2 step_2000 из `v2_soup_lb_5522/output/v2_soup_s42_01`, Stage-A из `initial_stage_a_user_bge`.

**Re-blend (LB 0.5527 → α по gray):** `reblend_gray_alpha_public.ipynb` — без train, default **α=0.15**, TTA, ~5 мин GPU.

## Symmetry TTA

В inference: `score = 0.5 * sigmoid(fwd) + 0.5 * sigmoid(rev)`  
См. `final_4_models/submit/matching-bge-human-ft/src/utils.py`

**Не путать** с v2 soup (LB 0.5522) — там TTA **выключен**.

## Препроцессинг

| | v1 (этот pipeline) | ecup_v2 v2 |
|---|-------------------|------------|
| attr cap | 520 | 720 |
| size/color priority | нет | да |
| submit TTA | **да** | нет |
