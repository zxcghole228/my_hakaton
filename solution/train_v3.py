"""v3: CatBoost на всех 11.2M LLM-пар (фичи из prep_v3) + ручные с весом 5.

Early stopping — по бинаризованному уверенному LLM-val (ближе всего к LB).
Отчёт: manual holdout + llm holdout, обе macro PR-AUC.
"""

import argparse
import glob
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool as CbPool
from multiprocessing import Pool
from sklearn.metrics import average_precision_score

from src.features import featurize_pair, FEATURE_NAMES, MODEL_FEATURES
from train import group_split

H = "/Users/user/projects/хакатон"


def macro_ap(df):
    aps = df.groupby("category").apply(
        lambda g: average_precision_score(g.target, g.pred), include_groups=False)
    return float(aps.mean()), aps


def featurize_manual(pairs, items):
    names, attrs = items["name"], items["attributes"]
    rows = [(names[r.id1], attrs[r.id1], names[r.id2], attrs[r.id2])
            for r in pairs.itertuples()]
    with Pool(9) as p:
        feats = p.starmap(featurize_pair, rows, chunksize=500)
    return np.vstack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--manual-weight", type=float, default=5.0)
    ap.add_argument("--model-out", default="artifacts/catboost_our_v3.cbm")
    args = ap.parse_args()

    # LLM train
    t0 = time.time()
    parts = sorted(glob.glob("artifacts/v3/llm_train_*.parquet"))
    llm = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"llm train: {len(llm):,} из {len(parts)} чанков, {time.time()-t0:.0f}s", flush=True)

    # manual
    t0 = time.time()
    items_h = pd.read_parquet(f"{H}/items_human.parquet").set_index("id")
    m = pd.read_parquet(f"{H}/matches.parquet")
    m["category"] = items_h["category"].reindex(m.id1.values).values
    m_train, m_val = group_split(m)
    Xm_tr = featurize_manual(m_train, items_h)
    Xm_va = featurize_manual(m_val, items_h)
    print(f"manual: train={len(m_train)} val={len(m_val)}, {time.time()-t0:.0f}s", flush=True)

    # llm val (уверенные, бинаризованные)
    lval = pd.read_parquet("artifacts/v3/llm_val.parquet")
    lval = lval[(lval.target <= 0.2) | (lval.target >= 0.8)].reset_index(drop=True)
    lval["target"] = (lval["target"] >= 0.5).astype(int)
    print(f"llm val (уверенные): {len(lval):,}", flush=True)

    # сборка
    t0 = time.time()
    X_tr = np.vstack([llm[FEATURE_NAMES].to_numpy(dtype=np.float32), Xm_tr])
    cat_tr = np.concatenate([llm["category"].to_numpy(), m_train["category"].to_numpy()])
    y_tr = np.concatenate([llm["target"].to_numpy(), m_train["target"].to_numpy()])
    w_tr = np.concatenate([np.ones(len(llm)), np.full(len(m_train), args.manual_weight)])
    del llm
    Xdf_tr = pd.DataFrame(X_tr, columns=FEATURE_NAMES, copy=False)
    Xdf_tr.insert(0, "category", cat_tr)
    del X_tr
    print(f"train matrix: {Xdf_tr.shape}, {time.time()-t0:.0f}s", flush=True)

    Xdf_lva = lval[["category"] + FEATURE_NAMES]

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=8,
        loss_function="CrossEntropy",
        eval_metric="PRAUC",
        cat_features=["category"],
        early_stopping_rounds=400,
        random_seed=42,
        verbose=250,
        thread_count=9,
    )
    model.fit(
        CbPool(Xdf_tr[MODEL_FEATURES], y_tr, weight=w_tr, cat_features=["category"]),
        eval_set=CbPool(Xdf_lva[MODEL_FEATURES], lval["target"].values, cat_features=["category"]),
        use_best_model=True,
    )

    lval = lval.assign(pred=model.predict_proba(Xdf_lva[MODEL_FEATURES])[:, 1])
    mac, aps = macro_ap(lval)
    print(f"\n[llm holdout, уверенные] Macro PR-AUC = {mac:.4f}")
    print(aps.round(3).to_string())

    Xdf_mva = pd.DataFrame(Xm_va, columns=FEATURE_NAMES)
    Xdf_mva.insert(0, "category", m_val["category"].values)
    m_val = m_val.assign(pred=model.predict_proba(Xdf_mva[MODEL_FEATURES])[:, 1])
    mac, aps = macro_ap(m_val)
    print(f"\n[manual holdout] Macro PR-AUC = {mac:.4f}")
    print(aps.round(3).to_string())

    model.save_model(args.model_out)
    print(f"\nмодель: {args.model_out}, деревьев: {model.tree_count_}")


if __name__ == "__main__":
    main()
