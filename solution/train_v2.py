"""v2: обучение на LLM-парах (soft labels) + ручные пары с повышенным весом.

Валидация двойная:
- val_manual: групповой holdout из ручных пар (как в v1, seed 42);
- val_llm: отложенные уверенные LLM-пары (target<=0.2 / >=0.8), товары
  не пересекаются с LLM-train (групповой сплит).
"""

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool as CbPool
from multiprocessing import Pool
from sklearn.metrics import average_precision_score

from src.features import featurize_pair, FEATURE_NAMES, MODEL_FEATURES

H = "/Users/user/projects/хакатон"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        self.parent[x] = p
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_split_mask(df, val_frac, seed):
    uf = UnionFind()
    for a, b in zip(df.id1.values, df.id2.values):
        uf.union(a, b)
    comp = np.array([uf.find(i) for i in df.id1.values])
    rng = np.random.RandomState(seed)
    uniq = np.unique(comp)
    val_comps = set(uniq[rng.rand(len(uniq)) < val_frac])
    return np.array([c in val_comps for c in comp])


def load_llm_items(need_ids):
    f = pq.ParquetFile(f"{H}/items.parquet")
    recs = {}
    for b in f.iter_batches(columns=["id", "name", "attributes", "category"], batch_size=500_000):
        df = b.to_pandas()
        df = df[df["id"].isin(need_ids)]
        for i, n, a, c in df.itertuples(index=False, name=None):
            recs[i] = (n, a, c)
    return recs


def featurize(pairs_iter, get1, get2, workers=10):
    rows = [(get1(r)[0], get1(r)[1], get2(r)[0], get2(r)[1]) for r in pairs_iter]
    with Pool(workers) as p:
        feats = p.starmap(featurize_pair, rows, chunksize=500)
    return np.vstack(feats)


def macro_ap(df):
    aps = df.groupby("category").apply(
        lambda g: average_precision_score(g.target, g.pred), include_groups=False)
    return float(aps.mean()), aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-sample", type=int, default=3_000_000)
    ap.add_argument("--manual-weight", type=float, default=5.0)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--model-out", default="artifacts/catboost_our_v2.cbm")
    args = ap.parse_args()

    # --- ручные пары: тот же сплит, что и v1 ---
    t0 = time.time()
    items_h = pd.read_parquet(f"{H}/items_human.parquet").set_index("id")
    m = pd.read_parquet(f"{H}/matches.parquet")
    m["category"] = items_h["category"].reindex(m.id1.values).values
    val_mask = group_split_mask(m, 0.2, 42)
    m_train, m_val = m[~val_mask], m[val_mask]
    print(f"manual: train={len(m_train)} val={len(m_val)}, {time.time()-t0:.0f}s", flush=True)

    # --- LLM-пары ---
    t0 = time.time()
    ml = pd.read_parquet(f"{H}/matches_llm.parquet")
    ml = ml.sample(min(args.llm_sample, len(ml)), random_state=7).reset_index(drop=True)
    llm_val_mask = group_split_mask(ml, 0.05, 13)
    ml_train = ml[~llm_val_mask].copy()
    ml_val = ml[llm_val_mask]
    ml_val = ml_val[(ml_val.target <= 0.2) | (ml_val.target >= 0.8)].copy()
    ml_val["target_bin"] = (ml_val.target >= 0.5).astype(int)
    print(f"llm: train={len(ml_train)} val={len(ml_val)}, {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    need = set(ml.id1) | set(ml.id2)
    recs = load_llm_items(need)
    print(f"items для llm-пар: {len(recs):,}, {time.time()-t0:.0f}s", flush=True)

    # --- фичи ---
    t0 = time.time()
    nh, ah = items_h["name"], items_h["attributes"]
    Xm_tr = featurize(m_train.itertuples(), lambda r: (nh[r.id1], ah[r.id1]),
                      lambda r: (nh[r.id2], ah[r.id2]))
    Xm_va = featurize(m_val.itertuples(), lambda r: (nh[r.id1], ah[r.id1]),
                      lambda r: (nh[r.id2], ah[r.id2]))
    print(f"фичи manual: {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    Xl_tr = featurize(ml_train.itertuples(), lambda r: recs[r.id1], lambda r: recs[r.id2])
    Xl_va = featurize(ml_val.itertuples(), lambda r: recs[r.id1], lambda r: recs[r.id2])
    print(f"фичи llm: {time.time()-t0:.0f}s", flush=True)

    ml_train["category"] = [recs[i][2] for i in ml_train.id1]
    ml_val["category"] = [recs[i][2] for i in ml_val.id1]

    # --- сборка трейна: LLM (soft) + manual (hard, вес выше) ---
    X_tr = np.vstack([Xl_tr, Xm_tr])
    cat_tr = np.concatenate([ml_train.category.values, m_train.category.values])
    y_tr = np.concatenate([ml_train.target.values, m_train.target.values])
    w_tr = np.concatenate([np.ones(len(ml_train)), np.full(len(m_train), args.manual_weight)])

    Xdf_tr = pd.DataFrame(X_tr, columns=FEATURE_NAMES)
    Xdf_tr.insert(0, "category", cat_tr)
    Xdf_va = pd.DataFrame(Xm_va, columns=FEATURE_NAMES)
    Xdf_va.insert(0, "category", m_val.category.values)

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=0.1,
        depth=8,
        loss_function="CrossEntropy",   # soft labels
        eval_metric="PRAUC",
        cat_features=["category"],
        early_stopping_rounds=300,
        random_seed=42,
        verbose=200,
        thread_count=10,
    )
    model.fit(
        CbPool(Xdf_tr[MODEL_FEATURES], y_tr, weight=w_tr, cat_features=["category"]),
        eval_set=CbPool(Xdf_va[MODEL_FEATURES], m_val.target.values, cat_features=["category"]),
    )

    # --- метрики ---
    m_val = m_val.assign(pred=model.predict_proba(Xdf_va[MODEL_FEATURES])[:, 1])
    mac, aps = macro_ap(m_val)
    print(f"\n[manual holdout] Macro PR-AUC = {mac:.4f}")
    print(aps.round(3).to_string())

    Xdf_lva = pd.DataFrame(Xl_va, columns=FEATURE_NAMES)
    Xdf_lva.insert(0, "category", ml_val.category.values)
    ml_val = ml_val.assign(pred=model.predict_proba(Xdf_lva[MODEL_FEATURES])[:, 1],
                           target=ml_val.target_bin)
    mac, aps = macro_ap(ml_val)
    print(f"\n[llm holdout, уверенные] Macro PR-AUC = {mac:.4f}")
    print(aps.round(3).to_string())

    model.save_model(args.model_out)
    print(f"\nмодель: {args.model_out}")


if __name__ == "__main__":
    main()
