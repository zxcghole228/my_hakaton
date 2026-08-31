import os
from typing import List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.pair_text import build_text  # pair_text v1 — frozen for LB 0.5522

MAX_LEN = 320


def _get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_items_texts(data_path: str) -> dict[int, str]:
    df = pd.read_parquet(data_path, columns=["id", "name", "attributes"])
    return {int(i): build_text(n, a) for i, n, a in df.itertuples(index=False, name=None)}


def _build_pairs(match_df: pd.DataFrame, texts: dict[int, str]):
    pairs, pair_ids = [], []
    for id1, id2 in zip(match_df.id1.values, match_df.id2.values):
        t1, t2 = texts.get(int(id1)), texts.get(int(id2))
        if t1 and t2:
            pairs.append((t1, t2))
            pair_ids.append((int(id1), int(id2)))
    return pairs, pair_ids


class _PairDS(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def _score_pairs(model, tokenizer, pairs, device, batch_size=128) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float32)
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    order = np.argsort([len(p[0]) + len(p[1]) for p in pairs])
    sorted_pairs = [pairs[i] for i in order]
    preds = np.empty(len(pairs), dtype=np.float32)

    def collate(batch_rows):
        return tokenizer(
            [x[0] for x in batch_rows], [x[1] for x in batch_rows],
            padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt",
        )

    loader = DataLoader(_PairDS(sorted_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate)
    offset = 0
    with torch.inference_mode():
        for enc in tqdm(loader, desc="Scoring pairs (no TTA)"):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device.type, enabled=device.type == "cuda", dtype=amp):
                p = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
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
    print("=== v2 soup submit | pair_text v1 | NO symmetry TTA ===")
    device = device or _get_best_device()
    texts = _load_items_texts(data_path)
    match_df = pd.read_parquet(match_path, columns=["id1", "id2"])
    pairs, pair_ids = _build_pairs(match_df, texts)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1).to(device).eval()
    predictions = _score_pairs(model, tokenizer, pairs, device, batch_size)
    out = pd.DataFrame({"id1": [a for a, _ in pair_ids], "id2": [b for _, b in pair_ids], "predict": predictions})
    out.to_csv(output_csv_path, index=False)
    print(f"saved {output_csv_path} ({len(out):,} rows)")
    return out
