# Executed source snapshot — v2 soup LB 0.5522 (`v2_soup_s42_01`)

Файлы подобраны так, чтобы совпасть с **сохранённым stdout** `pipeline_public.ipynb`, а не с более новыми копиями в корне репозитория.

## Какую копию брать (это и было претензией)

| Файл в архиве раньше | Почему отклонили | Что лежит здесь |
|---|---|---|
| корневой `score_ensemble.py` | `from ecup_v2.pair_text import build_text` → `verify_main()` падает | `final_4_models/scripts/score_ensemble.py`: свой `build_text` = pair_text **v1** |
| корневой `train_bge_stageb_v2.py` | печатает `model=... init=... out=...` | `final_4_models/scripts/train_bge_stageb_v2.py`: печатает `init=... out=...` |
| `final_4_models/scripts/blend_checkpoint_soup.py` (новый) | печатает `model=` / `trust_remote_code=` / `symmetry_tta=` | корневой `blend_checkpoint_soup.py`: только `checkpoint soup` / `blend: theta` / `OUT=` |

Отпечаток логов сохранённого запуска (27 Aug):

```
STAGE-B v2 | gpus=2 batch=128 accum=2 amp=torch.bfloat16
init=.../research_worker_s42_02/best.pt out=.../stageb_v2
...
checkpoint soup | A=.../step_00400.pt | B=.../step_02000.pt
blend: theta = (1-alpha)*B + alpha*A  | alphas=[0.15, 0.2, 0.25, 0.3, 0.35]
OUT=...
```

`ecup_v2/pair_text.py` нужен **только** `verify_pair_text.compare_v1_vs_v2`. Executed `score_ensemble.py` его **не импортирует**.

## Как проверять

Из корня этого snapshot:

```bash
python notebooks/lib/verify_pair_text.py
# ожидается: v1 vs v2 equal: False  и  OK — use pair_text_v1
```

Должно совпасть с ячейкой notebook.

## Чего здесь нет (и не должно быть в source ZIP)

- `items.parquet`, `items_human.parquet`, `matches.parquet`, `matches_llm.parquet`
- Stage-A `best.pt`, `step_00400.pt`, `step_02000.pt`, `export_fp16/`
- HuggingFace `deepvk/USER-bge-m3`

Без весов нельзя побайтно повторить soup. Формула того запуска: `0.85·step_02000 + 0.15·step_00400`.

## SHA-256 executed sources

```
029e49a55044d8e7af26eb32259cce121f61b9b805079bda7b88045409e8efaf  train_bge_stageb_v2.py
78f7ff9e528ddf7dc18ee8941c3aa5fdcc9f757ac480373fcecaa84b61ce3b1f  score_ensemble.py
21de0803ce25857992ddd19d7160c42bf672775fecfa9eda3eda579ce8d6110f  score_val_gray.py
eebc6ae0ca5c9bd57993d5554a361f1bb3ab1424e5ecce5d6320818cef6b5b23  final_4_models/scripts/blend_checkpoint_soup.py
cdce31cd07265078ed592eed003e3db695a028247ea89389f979c295ee822d52  notebooks/lib/pair_text_v1.py
586afe24da3ba93279f2f071cd96b8b60a94c75cbd35d126c1fd2caa28ac629e  notebooks/lib/verify_pair_text.py
a4915027dc457887758a307ff5e45272ff9409099237c49e8749cc3a3dbdf49e  ecup_v2/pair_text.py
381a45686dcd23175aac753998b3a2e0d35f23cdcdef9ea442772145cd1638a2  notebooks/v2_soup_lb_5522/build_submit.py
```

## Окружение из notebook

Python 3.12, torch 2.6.0+cu124, transformers 4.57.6, numpy 2.2.6, pandas 2.3.3, pyarrow 23.0.1, scikit-learn 1.8.0.
Revision `deepvk/USER-bge-m3` в том запуске не фиксировалась.
