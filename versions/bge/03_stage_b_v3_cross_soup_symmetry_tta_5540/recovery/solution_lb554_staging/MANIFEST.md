# Symmetry TTA cross-soup — source snapshot (LB 0.5539 / 0.5540)

Архив исходников, ноутбуков и **логов** для решения:

```
(1−α)·v2@2000 + α·v3@400  +  symmetry TTA  +  pair_text v1
```

## LB

| Submit | α | LB macro AP |
|--------|---|-------------|
| `cross_tta_gray_a015` | 0.15 | **0.5539** |
| `cross_tta_a017` | 0.17 | **0.5540** |

## Структура архива

```
solution_source_lb554.zip
├── MANIFEST.md          ← этот файл
├── scripts/             ← все .py, использованные в pipeline
├── notebooks/           ← ноутбуки + build_submit.py
├── logs/                ← логи train/blend (отдельная папка)
└── metrics/             ← metrics.json, soup_grid, checkpoint sweeps (без весов)
```

## Чекпойнты (не в архиве — слишком большие)

| Роль | Путь на диске |
|------|---------------|
| v2@2000 (B, high-problem) | `notebooks/v2_soup_lb_5522/output/v2_soup_s42_01/stageb_v2/checkpoints/step_02000.pt` |
| v3@400 (A, high-gray) | `notebooks/symmetry_tta_v3_soup/output/cross_tta_s42_01/stageb_v3/checkpoints/step_00400.pt` |
| Stage-A init v3 | `notebooks/initial_stage_a_user_bge/output/research_worker_s42_02/best.pt` |
| Best soup export | `notebooks/symmetry_tta_v3_soup/output/cross_tta_gray_a015/soup_run/export_fp16/` |
| Submit zip | `.../cross_tta_gray_a015/soup_run/matching-bge-human-ft-submit.zip` |

## Pipeline (порядок запуска)

1. **v2 Stage-B** — `notebooks/v2_soup_lb_5522/pipeline_public.ipynb`  
   → `train_bge_stageb_v2.py`, soup 400×2000  
   → log: `logs/v2_stageb_v2_soup_s42_01.log`

2. **v3 Stage-B** — `notebooks/symmetry_tta_v3_soup/pipeline_public.ipynb`  
   → `train_bge_stageb_v3.py` от Stage-A, cross-soup grid + TTA  
   → log: `logs/v3_stageb_cross_tta_s42_01.log`

3. **Re-blend α по gray** — `reblend_gray_alpha_public.ipynb`  
   → `blend_checkpoint_soup.py --symmetry-tta`  
   → metrics: `metrics/cross_tta_gray_a015_soup/`

4. **Fine α + submit** — `from_best_public.ipynb`  
   → α=0.17 submit, metrics: `metrics/cross_tta_a017_soup/`

## Ключевые скрипты

| Файл | Назначение |
|------|------------|
| `scripts/final_4_models/scripts/blend_checkpoint_soup.py` | checkpoint soup + TTA eval |
| `scripts/final_4_models/scripts/score_ensemble.py` | pair_text **v1**, eval pairs |
| `scripts/final_4_models/scripts/score_val_gray.py` | gray holdout |
| `scripts/train_bge_stageb_v3.py` | Stage-B v3 train |
| `scripts/final_4_models/scripts/train_bge_stageb_v2.py` | Stage-B v2 train |
| `scripts/notebooks/lib/pair_text_v1.py` | canonical pair_text v1 |
| `scripts/notebooks/lib/verify_pair_text.py` | sanity check v1 vs v2 |
| `scripts/final_4_models/submit/.../utils.py` | **symmetry TTA** inference |
| `notebooks/symmetry_tta_v3_soup/build_submit.py` | pack submit zip |

> Для eval/blend используйте копии из `scripts/final_4_models/scripts/`, не корневые `score_ensemble_root.py` (там ecup_v2 pair_text).

## Логи (`logs/`)

| Файл | Содержание |
|------|------------|
| `v2_stageb_v2_soup_s42_01.log` | train v2 + precompute teacher |
| `v3_stageb_cross_tta_s42_01.log` | train v3 (2 GPU) |
| `blend_cross_v3_600_x_v2_tta.log` | эксперимент v3@600 × v2 |
| `metrics_cross_tta_*.json` | копии metrics после re-blend / fine α |

## Окружение

Python 3.12, torch 2.6.0+cu124, transformers 4.57.6, numpy 2.2.6, pandas 2.3.3, pyarrow 23.0.1, scikit-learn 1.8.0.

Base model: `deepvk/USER-bge-m3`.

## Проверка

```bash
cd scripts
PYTHONPATH=final_4_models/scripts:notebooks/lib python notebooks/lib/verify_pair_text.py
# OK — use pair_text_v1 for v2 soup & symmetry TTA pipelines
```

Создано: 2026-08-29
