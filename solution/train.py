"""Обучение v1: CatBoost на наших фичах, честный holdout без пересечения товаров.

Сплит групповой: пары связываются в компоненты (union-find по id товаров),
целые компоненты уходят либо в train, либо в val — ни один товар не попадает
в обе части, утечки через общие товары нет.
"""

import argparse
import time

import numpy as np
import pandas as pd
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


def group_split(m: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    uf = UnionFind()
    for id1, id2 in zip(m.id1.values, m.id2.values):
        uf.union(id1, id2)
    comp = np.array([uf.find(i) for i in m.id1.values])
    rng = np.random.RandomState(seed)
    uniq = np.unique(comp)
    val_comps = set(uniq[rng.rand(len(uniq)) < val_frac])
    is_val = np.array([c in val_comps for c in comp])
    return m[~is_val].copy(), m[is_val].copy()


def build_matrix(pairs: pd.DataFrame, items: pd.DataFrame, workers: int = 10):
    names = items["name"]
    attrs = items["attributes"]
    rows = [(names[r.id1], attrs[r.id1], names[r.id2], attrs[r.id2])
            for r in pairs.itertuples()]
    with Pool(workers) as p:
        feats = p.starmap(featurize_pair, rows, chunksize=500)
    X = pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)
    X.insert(0, "category", pairs["category"].values)
    return X


def macro_pr_auc(df: pd.DataFrame) -> tuple[float, pd.Series]:
    aps = df.groupby("category").apply(
        lambda g: average_precision_score(g.target, g.pred), include_groups=False)
    return float(aps.mean()), aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--model-out", default="artifacts/catboost_our_v1.cbm")
    args = ap.parse_args()

    t0 = time.time()
    items = pd.read_parquet(f"{H}/items_human.parquet").set_index("id")
    m = pd.read_parquet(f"{H}/matches.parquet")
    m["category"] = items["category"].reindex(m.id1.values).values

    train, val = group_split(m)
    print(f"split: train={len(train)} val={len(val)} "
          f"(позитивы {train.target.mean():.3f}/{val.target.mean():.3f}), {time.time()-t0:.0f}s")

    t0 = time.time()
    X_train = build_matrix(train, items)
    X_val = build_matrix(val, items)
    print(f"features: {X_train.shape} + {X_val.shape}, {time.time()-t0:.0f}s")

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=0.08,
        depth=8,
        loss_function="Logloss",
        eval_metric="PRAUC",
        cat_features=["category"],
        early_stopping_rounds=200,
        random_seed=42,
        verbose=200,
        thread_count=10,
    )
    model.fit(
        CbPool(X_train[MODEL_FEATURES], train.target.values, cat_features=["category"]),
        eval_set=CbPool(X_val[MODEL_FEATURES], val.target.values, cat_features=["category"]),
    )

    val = val.assign(pred=model.predict_proba(X_val[MODEL_FEATURES])[:, 1])
    macro, aps = macro_pr_auc(val)
    print(f"\nMacro PR-AUC (holdout): {macro:.4f}")
    print(aps.round(3).to_string())

    model.save_model(args.model_out)
    val[["id1", "id2", "target", "pred", "category"]].to_parquet("artifacts/val_preds_v1.parquet")
    print(f"\nмодель: {args.model_out}")


if __name__ == "__main__":
    main()
