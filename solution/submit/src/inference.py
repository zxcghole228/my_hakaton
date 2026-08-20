"""Инференс нашего решения (v2): фичи + CatBoost.

Гарантии по формату проверки:
- предсказание выдаётся для КАЖДОЙ пары из matches, в исходном порядке;
- отсутствующий в items товар не роняет решение (пустые name/attributes);
- выход: csv c колонками id1,id2,predict.
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if sys.platform.startswith("linux") and VENDOR.is_dir() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from .features import featurize_pair, FEATURE_NAMES, MODEL_FEATURES  # noqa: E402

WORKERS = 18


def _log(msg, t0=None):
    extra = f" ({time.perf_counter() - t0:.1f}s)" if t0 is not None else ""
    print(msg + extra, flush=True)


def run(items_path: str, matches_path: str, output_path: str, model_path: str) -> None:
    total = time.perf_counter()

    t = time.perf_counter()
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items = items.drop_duplicates("id").set_index("id")
    _log(f"[1/4] загружено: пар={len(matches):,}, товаров={len(items):,}", t)

    t = time.perf_counter()
    names = items["name"].to_dict()
    attrs = items["attributes"].to_dict()
    cats = items["category"].to_dict()
    rows = [
        (names.get(i1, ""), attrs.get(i1, ""), names.get(i2, ""), attrs.get(i2, ""))
        for i1, i2 in zip(matches["id1"].values, matches["id2"].values)
    ]
    pair_cat = [cats.get(i1, cats.get(i2, "")) for i1, i2 in
                zip(matches["id1"].values, matches["id2"].values)]
    _log(f"[2/4] пары собраны", t)

    t = time.perf_counter()
    with Pool(WORKERS) as p:
        feats = p.starmap(featurize_pair, rows, chunksize=500)
    X = pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)
    X.insert(0, "category", pair_cat)
    _log(f"[3/4] фичи: {X.shape}", t)

    t = time.perf_counter()
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(model_path)
    pred = model.predict_proba(X[MODEL_FEATURES])[:, 1]
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN/inf")

    out = pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"], "predict": pred})
    out.to_csv(output_path, index=False)
    _log(f"[4/4] сохранено {len(out):,} строк -> {output_path}", t)
    _log(f"итого", total)
