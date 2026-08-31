#!/usr/bin/env python3
"""
Checkpoint soup: blend step_400 + step_2000 into one model (single submit).

  python3 blend_checkpoint_soup.py
  python3 blend_checkpoint_soup.py --alphas 0.15,0.20,0.25,0.30,0.35
  python3 build_bge_human_submit.py --run-dir ./user_bge_stageb_v2_soup_run --full-only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from score_ensemble import MAX_LEN, load_texts
from score_val_gray import load_gray_holdout

ITEMS_HUMAN_PATH = "./items_human.parquet"
MATCHES_HUMAN = "./matches.parquet"

MODEL_NAME = "deepvk/USER-bge-m3"
V2_RUN = "./user_bge_stageb_v2_320_run"
CKPT_A = f"{V2_RUN}/checkpoints/step_00400.pt"   # high gray
CKPT_B = f"{V2_RUN}/checkpoints/step_02000.pt"  # high problem (LB 0.5516)
OUT_DIR = "./user_bge_stageb_v2_soup_run"

PROBLEM_CATS = frozenset({"Обувь", "Одежда", "Галантерея и аксессуары", "Ювелирные изделия"})
SEED = 20260825
MANUAL_TUNE_FRAC = 0.10
MANUAL_EVAL_FRAC = 0.10

SCORE_W_PROBLEM = 0.50
SCORE_W_GRAY = 0.30
SCORE_W_TUNE = 0.20

DEFAULT_ALPHAS = [0.15, 0.20, 0.25, 0.30, 0.35]


def log(*a):
    print(*a, flush=True)


def union_find_components(frame: pd.DataFrame) -> np.ndarray:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        p = parent.setdefault(x, x)
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        parent[p] = p
        return p

    for a, b in zip(frame.id1.values, frame.id2.values):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return np.fromiter((find(int(i)) for i in frame.id1.values), dtype=np.int64, count=len(frame))


def component_split(frame: pd.DataFrame, tune_frac: float, eval_frac: float, seed: int) -> pd.DataFrame:
    out = frame.copy()
    out["_component"] = union_find_components(out)
    rng = np.random.RandomState(seed)
    split = {}
    for _, group in out.groupby("category", observed=True):
        arr = np.array(sorted(group._component.unique()), dtype=np.int64)
        rng.shuffle(arr)
        n_tune = max(1, int(round(len(arr) * tune_frac)))
        n_eval = max(1, int(round(len(arr) * eval_frac)))
        if n_tune + n_eval >= len(arr):
            n_tune = min(n_tune, max(0, len(arr) - 2))
            n_eval = min(n_eval, max(1, len(arr) - n_tune - 1))
        for c in arr[:n_tune]:
            split[int(c)] = "tune"
        for c in arr[n_tune:n_tune + n_eval]:
            split[int(c)] = "eval"
        for c in arr[n_tune + n_eval:]:
            split[int(c)] = "train"
    out["_split"] = out._component.map(split)
    return out


def blend_state_dict(sd_a: dict, sd_b: dict, alpha_a: float) -> dict:
    """theta = (1-alpha_a)*B + alpha_a*A  (alpha_a weight on step_400)."""
    out = {}
    for k in sd_a:
        if k not in sd_b:
            raise KeyError(f"missing key in B: {k}")
        ta, tb = sd_a[k], sd_b[k]
        if ta.dtype.is_floating_point:
            out[k] = ((1.0 - alpha_a) * tb.float() + alpha_a * ta.float()).to(ta.dtype)
        else:
            out[k] = ta
    return out


class PairDS(Dataset):
    def __init__(self, df, texts):
        self.a, self.b = df.id1.values, df.id2.values
        self.texts = texts

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.texts[int(self.a[i])], self.texts[int(self.b[i])]


@torch.no_grad()
def predict_probs(model, tok, texts: dict[int, str], df: pd.DataFrame, device, batch: int = 128) -> np.ndarray:
    model.eval()
    amp = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    def collate(rows):
        return tok([x[0] for x in rows], [x[1] for x in rows],
                   padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

    dl = DataLoader(PairDS(df, texts), batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)
    preds = []
    for enc in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", amp):
            preds.append(torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy())
    return np.concatenate(preds)


def macro_ap_cat(df: pd.DataFrame, p: np.ndarray) -> tuple[float, dict]:
    z = df.copy()
    z["p"] = p
    per = {}
    for cat, g in z.groupby("category", observed=True):
        if g.target.nunique() > 1:
            per[cat] = float(average_precision_score(g.target, g.p))
    return float(np.mean(list(per.values()))) if per else 0.0, per


def composite_score(problem_ap: float, gray_ap: float, tune_macro: float) -> float:
    return SCORE_W_PROBLEM * problem_ap + SCORE_W_GRAY * gray_ap + SCORE_W_TUNE * tune_macro


def load_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    gray = load_gray_holdout()
    human = pd.read_parquet(MATCHES_HUMAN)
    human["target"] = (human.target >= 0.5).astype(np.int8)
    cat_map = pd.read_parquet(ITEMS_HUMAN_PATH, columns=["id", "category"]).drop_duplicates("id")
    human = human.merge(cat_map.rename(columns={"id": "id1"}), on="id1", how="left")
    human["category"] = human.category.fillna("unknown").astype(str)
    human = component_split(human, MANUAL_TUNE_FRAC, MANUAL_EVAL_FRAC, SEED)
    manual_tune = human[human._split == "tune"].reset_index(drop=True)

    need = set(np.unique(np.concatenate([gray.id1.values, gray.id2.values])))
    need |= set(manual_tune.id1) | set(manual_tune.id2)
    log(f"loading {len(need):,} texts...")
    texts, cats = load_texts(need)
    gray = gray.copy()
    gray["category"] = [cats[int(i)] for i in gray.id1]
    manual_tune["category"] = [cats.get(int(i), "unknown") for i in manual_tune.id1]
    log(f"gray={len(gray):,} tune={len(manual_tune):,}")
    return gray, manual_tune, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", default=CKPT_A, help="high-gray checkpoint (weight alpha)")
    ap.add_argument("--ckpt-b", default=CKPT_B, help="high-problem checkpoint (weight 1-alpha)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-submit", action="store_true")
    args = ap.parse_args()

    ckpt_a, ckpt_b, out_dir = args.ckpt_a, args.ckpt_b, args.out_dir
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    os.makedirs(out_dir, exist_ok=True)

    log(f"checkpoint soup | A={ckpt_a} | B={ckpt_b}")
    log(f"blend: theta = (1-alpha)*B + alpha*A  | alphas={alphas}")
    log(f"OUT={out_dir}")

    sd_a = torch.load(ckpt_a, map_location="cpu", weights_only=True)
    sd_b = torch.load(ckpt_b, map_location="cpu", weights_only=True)

    gray, manual_tune, texts = load_eval_frames()
    prob_tune = manual_tune[manual_tune.category.isin(PROBLEM_CATS)]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows = []
    best_alpha = alphas[0]
    best_score = -1.0
    best_sd = None

    for alpha in alphas:
        t0 = time.time()
        blended = blend_state_dict(sd_a, sd_b, alpha)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
        model.load_state_dict(blended)
        model = model.to(device)

        gray_p = predict_probs(model, tok, texts, gray, device)
        gray_macro, _ = macro_ap_cat(gray, gray_p)

        tune_p = predict_probs(model, tok, texts, manual_tune, device)
        tune_macro, _ = macro_ap_cat(manual_tune, tune_p)

        if len(prob_tune):
            prob_p = predict_probs(model, tok, texts, prob_tune, device)
            problem_ap, _ = macro_ap_cat(prob_tune, prob_p)
        else:
            problem_ap = 0.0

        score = composite_score(problem_ap, gray_macro, tune_macro)
        row = {
            "alpha_step400": alpha,
            "weight_step2000": 1.0 - alpha,
            "gray_full": gray_macro,
            "problem_ap": problem_ap,
            "tune_macro": tune_macro,
            "composite_score": score,
            "elapsed_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        log(
            f"  alpha={alpha:.2f} (B={1-alpha:.2f}): gray={gray_macro:.4f} problem={problem_ap:.4f} "
            f"tune={tune_macro:.4f} score={score:.4f}  [{row['elapsed_s']}s]"
        )
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_sd = blended

        del model
        torch.cuda.empty_cache()
        gc.collect()

    json.dump(rows, open(f"{out_dir}/soup_grid.json", "w"), indent=2, ensure_ascii=False)

    log(f"\nBEST alpha={best_alpha:.2f} score={best_score:.4f}")
    torch.save(best_sd, f"{out_dir}/best.pt")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.load_state_dict(best_sd)
    export_dir = f"{out_dir}/export"
    export_fp16 = f"{out_dir}/export_fp16"
    model.save_pretrained(export_dir)
    tok.save_pretrained(export_dir)
    if os.path.exists(export_fp16):
        import shutil
        shutil.rmtree(export_fp16)
    import shutil
    shutil.copytree(export_dir, export_fp16)

    best_row = next(r for r in rows if r["alpha_step400"] == best_alpha)
    metrics = {
        "version": "checkpoint_soup",
        "blend": {
            "ckpt_a": ckpt_a,
            "ckpt_b": ckpt_b,
            "formula": "theta = (1-alpha)*B + alpha*A",
            "best_alpha": best_alpha,
            "best_weight_b": 1.0 - best_alpha,
        },
        "best_metrics": best_row,
        "grid": rows,
    }
    json.dump(metrics, open(f"{out_dir}/metrics.json", "w"), indent=2, ensure_ascii=False)
    log(f"saved {out_dir}/best.pt {export_fp16}/ metrics.json")

    if not args.skip_submit:
        import subprocess
        subprocess.run(
            ["python3", "build_bge_human_submit.py", "--run-dir", out_dir, "--full-only"],
            check=True,
        )


if __name__ == "__main__":
    main()
