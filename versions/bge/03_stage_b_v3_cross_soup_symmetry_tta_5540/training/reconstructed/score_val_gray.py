#!/usr/bin/env python3
"""Score BGE on holdout gray zone (0.2 < LLM target < 0.8) — LB proxy experiment."""
import argparse, gc, json, os, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from score_ensemble import (
    BGE_PATH, ITEMS_PATH, MATCHES_PATH, MAX_LEN, build_text, load_texts, macro_ap,
)

BGE_PATH_LOCAL = BGE_PATH


def load_gray_holdout():
    ml = pd.read_parquet(MATCHES_PATH)
    parent = {}

    def find(x):
        p = parent.setdefault(x, x)
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        parent[p] = p
        return p

    for a, b in zip(ml.id1.values, ml.id2.values):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comp = np.fromiter((find(i) for i in ml.id1.values), dtype=np.int64, count=len(ml))
    rng = np.random.RandomState(13)
    u = np.unique(comp)
    vs = set(u[rng.rand(len(u)) < 0.03].tolist())
    is_val = np.fromiter((c in vs for c in comp), dtype=bool, count=len(ml))
    gray = ml[is_val].copy()
    gray = gray[(gray.target > 0.2) & (gray.target < 0.8)]
    gray["target_soft"] = gray.target.astype(np.float32)
    gray["target"] = (gray.target >= 0.5).astype(np.int8)
    del ml, parent, comp
    gc.collect()
    return gray.reset_index(drop=True)


def free_gpus(max_used_mib=5000):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(line.split(", ")[0]) for line in out.strip().splitlines() if int(line.split(", ")[1]) <= max_used_mib]


class PairDS(Dataset):
    def __init__(self, df, texts):
        self.a, self.b = df.id1.values, df.id2.values
        self.texts = texts

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.texts[self.a[i]], self.texts[self.b[i]]


@torch.no_grad()
def score_bge(path, df, texts, device, batch=128):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(device).eval()
    cc = torch.cuda.get_device_capability(device)
    amp = torch.bfloat16 if cc[0] >= 8 else torch.float16

    def collate(rows):
        return tok([x[0] for x in rows], [x[1] for x in rows],
                   padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

    dl = DataLoader(PairDS(df, texts), batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)
    preds, t0 = [], time.time()
    for enc in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", amp):
            preds.append(torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy())
    dt = time.time() - t0
    print(f"BGE {len(df)/dt:.0f} pair/s on {device}", flush=True)
    return np.concatenate(preds)


def per_cat_ap(df, col="p_bge"):
    rows = []
    for cat, g in df.groupby("category", observed=True):
        ap = average_precision_score(g.target, g[col]) if g.target.nunique() > 1 else float("nan")
        rows.append({"category": cat, "n": len(g), "pos_rate": g.target.mean(), "ap": ap})
    return pd.DataFrame(rows).sort_values("category")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bge", default=BGE_PATH_LOCAL, help="BGE model path (export or export_fp16)")
    ap.add_argument("--out", default="./val_gray_bge_scores.parquet")
    ap.add_argument("--gpu", default="auto")
    args = ap.parse_args()

    gray = load_gray_holdout()
    print(f"gray zone: {len(gray):,} pairs, pos(bin)={gray.target.mean():.3f}, soft_mean={gray.target_soft.mean():.3f}", flush=True)

    need = np.unique(np.concatenate([gray.id1.values, gray.id2.values]))
    texts, cats = load_texts(set(need))
    gray = gray.copy()
    gray["category"] = [cats[i] for i in gray.id1]

    if args.gpu == "auto":
        gpus = free_gpus()
        if not gpus:
            raise SystemExit("no free GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        device = torch.device("cuda:0")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda:0")

    gray["p_bge"] = score_bge(args.bge, gray, texts, device)
    gray.to_parquet(args.out, index=False)

    macro = macro_ap(gray, "p_bge")
    cat = per_cat_ap(gray)
    cat.to_csv(args.out.replace(".parquet", "_per_cat.csv"), index=False)

    fashion = {"Обувь", "Одежда", "Галантерея и аксессуары", "Ювелирные изделия"}
    f = gray[gray.category.isin(fashion)]
    nf = gray[~gray.category.isin(fashion)]

    meta = {
        "bge_path": args.bge,
        "n": len(gray),
        "pos_rate_bin": float(gray.target.mean()),
        "pos_rate_soft_mean": float(gray.target_soft.mean()),
        "macro_ap_bge": float(macro),
        "macro_ap_fashion": float(macro_ap(f, "p_bge")) if len(f) else None,
        "macro_ap_non_fashion": float(macro_ap(nf, "p_bge")) if len(nf) else None,
        "fashion_cats": sorted(fashion),
    }
    json.dump(meta, open(args.out.replace(".parquet", "_meta.json"), "w"), indent=2)

    print(f"\nmacro AP (gray): {macro:.4f}", flush=True)
    print(f"  fashion (4 cat):     {meta['macro_ap_fashion']:.4f}  n={len(f):,}", flush=True)
    print(f"  non-fashion (16 cat): {meta['macro_ap_non_fashion']:.4f}  n={len(nf):,}", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
