import json
import os
import re
from typing import List, Optional, Tuple

os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("RAYON_NUM_THREADS", "32")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 320
KEYS = [
    "бренд", "артикул", "партномер", "oem", "код", "модель", "размер",
    "цвет", "объем", "обьем", "вес", "тип", "материал", "количество",
]

CYR2LAT = str.maketrans("аеорсухАЕОРСУХКМТВНЗЅІі", "aeopcyxAEOPCYXKMTBH3SIi")
LAT2CYR = str.maketrans("aeopcyxAEOPCYX", "аеорсухАЕОРСУХ")
_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мл|ml|л|l|мг|mg|г|g|гр|кг|kg|мм|mm|см|cm|м|мб|mb|гб|gb|тб|tb|вт|w|квт|kw|мач|mah)\b",
    re.IGNORECASE,
)
_UNIT_MUL = {
    "мл": ("ml", 1), "ml": ("ml", 1), "л": ("ml", 1000), "l": ("ml", 1000),
    "мг": ("g", 0.001), "mg": ("g", 0.001), "г": ("g", 1), "g": ("g", 1), "гр": ("g", 1),
    "кг": ("g", 1000), "kg": ("g", 1000), "мм": ("mm", 1), "mm": ("mm", 1),
    "см": ("mm", 10), "cm": ("mm", 10), "м": ("mm", 1000),
    "мб": ("gb", 0.001), "mb": ("gb", 0.001), "гб": ("gb", 1), "gb": ("gb", 1),
    "тб": ("gb", 1024), "tb": ("gb", 1024), "вт": ("w", 1), "w": ("w", 1),
    "квт": ("w", 1000), "kw": ("w", 1000), "мач": ("mah", 1), "mah": ("mah", 1),
}
_QTY_RES = [
    re.compile(r"(\d+)\s*шт"),
    re.compile(r"набор\w*\s+из\s+(\d+)"),
    re.compile(r"(\d+)\s*(?:набор|упаков|комплект)\w*\s+по\s+(\d+)"),
    re.compile(r"[xх*](\d+)\b"),
]


def fix_homoglyphs(token: str) -> str:
    has_c = any("\u0400" <= ch <= "\u04ff" for ch in token)
    has_l = any(ch.isascii() and ch.isalpha() for ch in token)
    if not (has_c and has_l):
        return token
    n_c = sum("\u0400" <= ch <= "\u04ff" for ch in token)
    n_l = sum(ch.isascii() and ch.isalpha() for ch in token)
    return token.translate(CYR2LAT if n_l >= n_c else LAT2CYR)


def canon_units(text: str) -> set[str]:
    out = set()
    for m in _UNIT_RE.finditer(text):
        unit, mul = _UNIT_MUL[m.group(2).lower()]
        v = float(m.group(1).replace(",", ".")) * mul
        out.add(f"{v:g}{unit}")
    return out


def total_qty(text: str) -> Optional[int]:
    t = text.lower()
    m = _QTY_RES[2].search(t)
    if m:
        return int(m.group(1)) * int(m.group(2))
    for r in (_QTY_RES[0], _QTY_RES[1], _QTY_RES[3]):
        m = r.search(t)
        if m:
            return int(m.group(1))
    return None


def build_text(name, attributes) -> str:
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


def _get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_items_texts(data_path: str) -> dict[int, str]:
    df = pd.read_parquet(data_path, columns=["id", "name", "attributes"])
    texts = {}
    for i, n, a in df.itertuples(index=False, name=None):
        texts[int(i)] = build_text(n, a)
    return texts


def _build_pairs(match_df: pd.DataFrame, texts: dict[int, str]):
    pairs = []
    missing_items = 0
    empty_texts = 0

    for id1, id2 in zip(
        match_df.id1.values,
        match_df.id2.values,
    ):
        t1 = texts.get(int(id1))
        t2 = texts.get(int(id2))

        if t1 is None:
            missing_items += 1
            t1 = ""
        elif not t1:
            empty_texts += 1

        if t2 is None:
            missing_items += 1
            t2 = ""
        elif not t2:
            empty_texts += 1

        pairs.append((t1, t2))

    return pairs, missing_items, empty_texts


class _PairDS(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def _score_pairs(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float32)

    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    lengths = np.fromiter((len(p[0]) + len(p[1]) for p in pairs), dtype=np.int32, count=len(pairs))
    order = np.argsort(lengths)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    sorted_pairs = [pairs[i] for i in order]
    preds = np.empty(len(pairs), dtype=np.float32)

    def collate(batch_rows):
        enc_fwd = tokenizer(
            [x[0] for x in batch_rows],
            [x[1] for x in batch_rows],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc_rev = tokenizer(
            [x[1] for x in batch_rows],
            [x[0] for x in batch_rows],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return enc_fwd, enc_rev

    loader = DataLoader(_PairDS(sorted_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    offset = 0
    with torch.inference_mode():
        for enc_fwd, enc_rev in tqdm(loader, desc="Scoring pairs (symmetry TTA)"):
            enc_fwd = {k: v.to(device, non_blocking=True) for k, v in enc_fwd.items()}
            enc_rev = {k: v.to(device, non_blocking=True) for k, v in enc_rev.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda", dtype=amp):
                p_fwd = torch.sigmoid(model(**enc_fwd).logits.squeeze(-1).float())
                p_rev = torch.sigmoid(model(**enc_rev).logits.squeeze(-1).float())
                p = ((p_fwd + p_rev) * 0.5).cpu().numpy()
            n = len(p)
            preds[order[offset:offset + n]] = p
            offset += n
    return preds


def predict_pipeline(
    data_path: str,
    match_path: str,
    model_path: str,
    output_csv_path: str,
    device: Optional[torch.device] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    print("=== Stage-B v3 cross-soup | pair_text v1 | symmetry TTA ===")
    device = device or _get_best_device()
    print(f"device={device} batch={batch_size} max_len={MAX_LEN}")

    print("[1/4] Loading items...")
    texts = _load_items_texts(data_path)
    print(f"  items: {len(texts):,}")

    print("[2/4] Building pairs...")
    match_df = pd.read_parquet(match_path, columns=["id1", "id2"])
    pairs, missing_items, empty_texts = _build_pairs(
        match_df,
        texts,
    )
    print(f"  pairs: {len(pairs):,}")
    if missing_items or empty_texts:
        print(
            "  preserved rows with incomplete text: "
            f"missing item references={missing_items:,}; "
            f"empty item texts={empty_texts:,}"
        )

    print("[3/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        local_files_only=True,
    )
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[4/4] Inference (symmetry TTA: avg p(a,b) and p(b,a))...")
    predictions = _score_pairs(model, tokenizer, pairs, device, batch_size=batch_size)

    results_df = match_df.copy()
    results_df["predict"] = predictions
    results_df.to_csv(output_csv_path, index=False)
    print(f"saved {output_csv_path} ({len(results_df):,} rows)")
    return results_df
