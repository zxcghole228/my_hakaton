"""Инференс cross-encoder ce-v1 (rubert-tiny2).

build_text ДОЛЖЕН побайтово совпадать с ноутбуком обучения
(ozon_ecup_crossencoder_v1.ipynb) — менять только синхронно.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

KEY_ORDER = ["бренд", "артикул", "партномер", "oem", "код", "модель", "размер",
             "цвет", "объем", "обьем", "вес", "тип", "материал", "количество"]

MAX_LEN = 160
BATCH = 1024


def build_text(name, attributes, max_attr_chars=260):
    parts = [str(name) if name is not None else ""]
    try:
        attrs = json.loads(attributes) if isinstance(attributes, str) else {}
    except Exception:
        attrs = {}
    if isinstance(attrs, dict) and attrs:
        low = {str(k).lower(): str(v) for k, v in attrs.items() if v}
        picked, used = [], set()
        for want in KEY_ORDER:
            for k, v in low.items():
                if want in k and k not in used:
                    picked.append(f"{k}:{v}"); used.add(k)
        rest = [f"{k}:{v}" for k, v in low.items() if k not in used]
        s = " ; ".join(picked + rest)[:max_attr_chars]
        parts.append(s)
    return " | ".join(parts)


def _log(msg, t0=None):
    extra = f" ({time.perf_counter() - t0:.1f}s)" if t0 is not None else ""
    print(msg + extra, flush=True)


def run(items_path: str, matches_path: str, output_path: str, model_dir: str) -> None:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    total = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    t = time.perf_counter()
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes"])
    items = items.drop_duplicates("id")
    texts = {i: build_text(n, a) for i, n, a in
             items[["id", "name", "attributes"]].itertuples(index=False, name=None)}
    _log(f"[1/3] пар={len(matches):,}, товаров={len(texts):,}, device={device}", t)

    t = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    if device == "cuda":
        model = model.half()

    id1 = matches["id1"].values
    id2 = matches["id2"].values
    n = len(matches)
    # сортировка по суммарной длине текстов — меньше паддинга в батчах
    lengths = np.fromiter((len(texts.get(a, "")) + len(texts.get(b, ""))
                           for a, b in zip(id1, id2)), dtype=np.int64, count=n)
    order = np.argsort(lengths, kind="stable")
    preds = np.empty(n, dtype=np.float64)

    with torch.inference_mode():
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            t1 = [texts.get(id1[i], "") for i in idx]
            t2 = [texts.get(id2[i], "") for i in idx]
            enc = tokenizer(t1, t2, padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1)
            preds[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    _log(f"[2/3] инференс {n:,} пар", t)

    if not np.isfinite(preds).all():
        raise ValueError("predictions contain NaN/inf")
    out = pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"], "predict": preds})
    out.to_csv(output_path, index=False)
    _log(f"[3/3] сохранено -> {output_path}", total)
