"""Подготовка v3: фичи по ВСЕМ LLM-парам, чанками, с групповым val-сплитом.

Выход: artifacts/v3/llm_train_XX.parquet и artifacts/v3/llm_val.parquet
(колонки: target, category, 63 фичи).
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from multiprocessing import Pool

from src.features import featurize_pair, FEATURE_NAMES

H = "/Users/user/projects/хакатон"
OUT = "artifacts/v3"
CHUNK = 1_500_000
VAL_FRAC = 0.03
SEED = 13


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ml = pd.read_parquet(f"{H}/matches_llm.parquet")
    print(f"llm pairs: {len(ml):,}, {time.time()-t0:.0f}s", flush=True)

    # union-find по компонентам товаров
    t0 = time.time()
    parent = {}

    def find(x):
        p = parent.setdefault(x, x)
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        parent[x] = p
        return p

    for a, b in zip(ml.id1.values, ml.id2.values):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comp = np.fromiter((find(i) for i in ml.id1.values), dtype=np.int64, count=len(ml))
    rng = np.random.RandomState(SEED)
    uniq = np.unique(comp)
    val_set = set(uniq[rng.rand(len(uniq)) < VAL_FRAC].tolist())
    is_val = np.fromiter((c in val_set for c in comp), dtype=bool, count=len(ml))
    parent = None
    print(f"split: train={(~is_val).sum():,} val={is_val.sum():,}, {time.time()-t0:.0f}s", flush=True)

    ml["is_val"] = is_val

    def featurize_block(block: pd.DataFrame, tag: str):
        t = time.time()
        need = set(block.id1) | set(block.id2)
        recs = {}
        f = pq.ParquetFile(f"{H}/items.parquet")
        for b in f.iter_batches(columns=["id", "name", "attributes", "category"],
                                batch_size=500_000):
            df = b.to_pandas()
            df = df[df["id"].isin(need)]
            for i, n, a, c in df.itertuples(index=False, name=None):
                recs[i] = (n, a, c)
        rows = [(recs[r.id1][0], recs[r.id1][1], recs[r.id2][0], recs[r.id2][1])
                for r in block.itertuples()]
        with Pool(9) as p:
            feats = p.starmap(featurize_pair, rows, chunksize=500)
        out = pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)
        out.insert(0, "category", [recs[i][2] for i in block.id1])
        out.insert(0, "target", block.target.values)
        out.to_parquet(f"{OUT}/{tag}.parquet", index=False)
        print(f"{tag}: {len(out):,} пар, {time.time()-t:.0f}s", flush=True)
        del recs, rows, feats, out

    val_block = ml[ml.is_val]
    featurize_block(val_block, "llm_val")

    train_ml = ml[~ml.is_val].reset_index(drop=True)
    for ci, start in enumerate(range(0, len(train_ml), CHUNK)):
        featurize_block(train_ml.iloc[start:start + CHUNK], f"llm_train_{ci:02d}")

    print("done", flush=True)


if __name__ == "__main__":
    main()
