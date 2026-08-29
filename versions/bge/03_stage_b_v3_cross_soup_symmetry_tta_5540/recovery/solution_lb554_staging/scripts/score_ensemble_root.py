#!/usr/bin/env python3
"""BGE + E5 rank-ensemble on holdout. Uses only GPUs with low memory use."""
import argparse, gc, json, os, re, subprocess, sys, threading, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
from ecup_v2.pair_text import build_text  # noqa: E402

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ITEMS_PATH = "./items.parquet"
MATCHES_PATH = "./matches_llm.parquet"
E5_PATH = "./e5base_320_run/export_fp16"
BGE_PATH = "./user_bge_m3_320_run/export_fp16"
MAX_LEN = 320


def free_gpus(max_used_mib=5000):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    ids = []
    for line in out.strip().splitlines():
        idx, used = line.split(", ")
        if int(used) <= max_used_mib:
            ids.append(int(idx))
    return ids


def load_holdout():
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
    holdout = ml[is_val].copy()
    holdout = holdout[(holdout.target <= 0.2) | (holdout.target >= 0.8)]
    holdout["target"] = (holdout.target >= 0.5).astype(np.int8)
    del ml, parent, comp
    gc.collect()
    return holdout.reset_index(drop=True)


def load_texts(need_ids):
    need = set(need_ids)
    texts, cats = {}, {}
    f = pq.ParquetFile(ITEMS_PATH)
    n_batches, hit = 0, 0
    t0 = time.time()
    for b in f.iter_batches(columns=["id", "name", "attributes", "category"], batch_size=500_000):
        n_batches += 1
        df = b.to_pandas()
        sub = df[df["id"].isin(need)]
        for i, n, a, c in sub.itertuples(index=False, name=None):
            texts[i] = build_text(n, a)
            cats[i] = c
        hit += len(sub)
        if n_batches % 5 == 0:
            print(f"  items scan: {hit:,}/{len(need):,} ids ({time.time()-t0:.0f}s)", flush=True)
        if hit >= len(need):
            break
    print(f"loaded {len(texts):,} texts in {time.time()-t0:.0f}s", flush=True)
    return texts, cats


class PairDS(Dataset):
    def __init__(self, df, texts):
        self.a = df.id1.values
        self.b = df.id2.values
        self.texts = texts

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.texts[self.a[i]], self.texts[self.b[i]]


@torch.no_grad()
def score_model(path, df, texts, device, batch=128):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
    model = model.to(device).eval()
    cc = torch.cuda.get_device_capability(device)
    amp = torch.bfloat16 if cc[0] >= 8 else torch.float16

    def collate(batch_rows):
        return tok([x[0] for x in batch_rows], [x[1] for x in batch_rows],
                   padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

    dl = DataLoader(PairDS(df, texts), batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)
    preds, t0, n = [], time.time(), len(df)
    for enc in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", amp):
            p = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        preds.append(p)
    dt = time.time() - t0
    del model, tok
    torch.cuda.empty_cache()
    print(f"  {path} gpu={device}: {n/dt:.0f} pair/s", flush=True)
    return np.concatenate(preds)


def macro_ap(df, col="p"):
    return float(df.groupby("category").apply(
        lambda g: average_precision_score(g.target, g[col]), include_groups=False).mean())


def rank_blend(df, w_bge):
    z = df.copy()
    z["rb"] = z.groupby("category")["p_bge"].rank(method="average", pct=True)
    z["re"] = z.groupby("category")["p_e5"].rank(method="average", pct=True)
    z["p"] = w_bge * z["rb"] + (1.0 - w_bge) * z["re"]
    return z


def tune_weights(df):
    best_w, best = 0.7, -1.0
    for w in np.arange(0.50, 0.96, 0.05):
        m = macro_ap(rank_blend(df, w))
        print(f"  w_bge={w:.2f} -> macro={m:.4f}", flush=True)
        if m > best:
            best, best_w = m, w
    return best_w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["fast", "full"], default="fast")
    ap.add_argument("--gpus", type=str, default="auto", help="comma ids or auto (<=5GB used)")
    ap.add_argument("--out", default="./ensemble_scores.parquet")
    args = ap.parse_args()

    if args.gpus == "auto":
        physical = free_gpus()
        if not physical:
            raise SystemExit("no free GPUs (all >5GB used)")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical[:2]))
        devices = [torch.device(f"cuda:{i}") for i in range(min(2, len(physical)))]
        print(f"free physical GPUs {physical} -> visible {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        devices = [torch.device(f"cuda:{i}") for i in range(len(args.gpus.split(",")))]

    holdout = load_holdout()
    print(f"holdout {len(holdout):,} pairs", flush=True)
    need_ids = np.unique(np.concatenate([holdout.id1.values, holdout.id2.values]))
    print(f"loading texts for {len(need_ids):,} unique items...", flush=True)
    texts, cats = load_texts(need_ids)
    holdout = holdout.copy()
    holdout["category"] = [cats[i] for i in holdout.id1]
    df = holdout.sample(frac=1, random_state=0).groupby("category").head(3000) if args.subset == "fast" else holdout
    print(f"scoring {len(df):,} pairs ({args.subset})", flush=True)

    scores = {}
    errs = {}

    def run(name, path, dev):
        try:
            scores[name] = score_model(path, df, texts, dev)
        except Exception as e:
            errs[name] = e

    threads = [
        threading.Thread(target=run, args=("e5", E5_PATH, devices[0])),
        threading.Thread(target=run, args=("bge", BGE_PATH, devices[min(1, len(devices) - 1)])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise RuntimeError(errs)

    out = df[["id1", "id2", "category", "target"]].copy()
    out["p_e5"] = scores["e5"]
    out["p_bge"] = scores["bge"]
    e5_m = macro_ap(out, "p_e5")
    bge_m = macro_ap(out, "p_bge")
    print(f"single: e5={e5_m:.4f} bge={bge_m:.4f}", flush=True)

    print("weight grid:", flush=True)
    w, ens_m = tune_weights(out)
    out = rank_blend(out, w)
    print(f"BEST w_bge={w:.2f} ensemble macro={ens_m:.4f} (delta vs bge {ens_m-bge_m:+.4f})", flush=True)
    out.to_parquet(args.out, index=False)
    meta = {"subset": args.subset, "n": len(out), "w_bge": w,
            "macro_e5": e5_m, "macro_bge": bge_m, "macro_ensemble": ens_m,
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    json.dump(meta, open(args.out.replace(".parquet", "_meta.json"), "w"), indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
