#!/usr/bin/env python3
"""BGE + E5 rank-ensemble on holdout. Uses only GPUs with low memory use."""
import argparse, gc, json, os, re, subprocess, threading, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
KEYS = ["бренд", "артикул", "партномер", "oem", "код", "модель", "размер",
        "цвет", "объем", "обьем", "вес", "тип", "материал", "количество"]

CYR2LAT = str.maketrans("аеорсухАЕОРСУХКМТВНЗЅІі", "aeopcyxAEOPCYXKMTBH3SIi")
LAT2CYR = str.maketrans("aeopcyxAEOPCYX", "аеорсухАЕОРСУХ")
_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мл|ml|л|l|мг|mg|г|g|гр|кг|kg|мм|mm|см|cm|м|мб|mb|гб|gb|тб|tb|вт|w|квт|kw|мач|mah)\b",
    re.IGNORECASE)
_UNIT_MUL = {"мл": ("ml", 1), "ml": ("ml", 1), "л": ("ml", 1000), "l": ("ml", 1000),
             "мг": ("g", 0.001), "mg": ("g", 0.001), "г": ("g", 1), "g": ("g", 1), "гр": ("g", 1),
             "кг": ("g", 1000), "kg": ("g", 1000), "мм": ("mm", 1), "mm": ("mm", 1),
             "см": ("mm", 10), "cm": ("mm", 10), "м": ("mm", 1000),
             "мб": ("gb", 0.001), "mb": ("gb", 0.001), "гб": ("gb", 1), "gb": ("gb", 1),
             "тб": ("gb", 1024), "tb": ("gb", 1024), "вт": ("w", 1), "w": ("w", 1),
             "квт": ("w", 1000), "kw": ("w", 1000), "мач": ("mah", 1), "mah": ("mah", 1)}
_QTY_RES = [re.compile(r"(\d+)\s*шт"), re.compile(r"набор\w*\s+из\s+(\d+)"),
            re.compile(r"(\d+)\s*(?:набор|упаков|комплект)\w*\s+по\s+(\d+)"),
            re.compile(r"[xх*](\d+)\b")]


def fix_homoglyphs(token):
    has_c = any("\u0400" <= ch <= "\u04ff" for ch in token)
    has_l = any(ch.isascii() and ch.isalpha() for ch in token)
    if not (has_c and has_l):
        return token
    n_c = sum("\u0400" <= ch <= "\u04ff" for ch in token)
    n_l = sum(ch.isascii() and ch.isalpha() for ch in token)
    return token.translate(CYR2LAT if n_l >= n_c else LAT2CYR)


def canon_units(text):
    out = set()
    for m in _UNIT_RE.finditer(text):
        unit, mul = _UNIT_MUL[m.group(2).lower()]
        v = float(m.group(1).replace(",", ".")) * mul
        out.add(f"{v:g}{unit}")
    return out


def total_qty(text):
    t = text.lower()
    m = _QTY_RES[2].search(t)
    if m:
        return int(m.group(1)) * int(m.group(2))
    for r in (_QTY_RES[0], _QTY_RES[1], _QTY_RES[3]):
        m = r.search(t)
        if m:
            return int(m.group(1))
    return None


def build_text(name, attributes):
    parts = [str(name) if name is not None else ""]
    try:
        attrs = json.loads(attributes) if isinstance(attributes, str) else {}
    except Exception:
        attrs = {}
    if isinstance(attrs, dict) and attrs:
        low = {str(k).lower(): str(v) for k, v in attrs.items() if v}
        picked, used = [], set()
        for w in KEYS:
            for k, v in low.items():
                if w in k and k not in used:
                    picked.append(f"{k}:{v}")
                    used.add(k)
        rest = [f"{k}:{v}" for k, v in low.items() if k not in used]
        parts.append(" ; ".join(picked + rest)[:520])
    base = " | ".join(parts)
    base = base.replace("ё", "е").replace("Ё", "Е")
    base = " ".join(fix_homoglyphs(t) for t in base.split())
    base = re.sub(r"[×хХ](?=\d)", "x", base)
    extras = []
    units = canon_units(base)
    if units:
        extras.append("ед: " + " ".join(sorted(units)[:12]))
    q = total_qty(base)
    if q and 1 < q <= 1000:
        extras.append(f"кол-во: {q}")
    return (base + (" | " + " | ".join(extras) if extras else ""))[:2000]


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
