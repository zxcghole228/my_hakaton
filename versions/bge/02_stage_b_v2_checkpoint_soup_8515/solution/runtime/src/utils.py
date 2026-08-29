import os
from pathlib import Path
from typing import List, Optional, Tuple


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.pair_text import build_text


MAX_LEN = 320


def _get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_items_texts(data_path: str) -> dict[int, str]:
    frame = pd.read_parquet(data_path, columns=["id", "name", "attributes"])
    return {
        int(item_id): build_text(name, attributes)
        for item_id, name, attributes in frame.itertuples(index=False, name=None)
    }


def _build_pairs(
    match_df: pd.DataFrame,
    texts: dict[int, str],
) -> tuple[List[Tuple[str, str]], int, int]:
    pairs: List[Tuple[str, str]] = []
    missing_items = 0
    empty_texts = 0

    for id1, id2 in zip(match_df.id1.values, match_df.id2.values):
        first = texts.get(int(id1))
        second = texts.get(int(id2))

        if first is None:
            missing_items += 1
            first = ""
        elif not first:
            empty_texts += 1

        if second is None:
            missing_items += 1
            second = ""
        elif not second:
            empty_texts += 1

        pairs.append((first, second))

    return pairs, missing_items, empty_texts


class _PairDS(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def _score_pairs(
    model,
    tokenizer,
    pairs,
    device,
    batch_size=128,
) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float32)

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    order = np.argsort(
        np.fromiter(
            (len(first) + len(second) for first, second in pairs),
            dtype=np.int64,
            count=len(pairs),
        ),
        kind="stable",
    )
    sorted_pairs = [pairs[index] for index in order]
    predictions = np.empty(len(pairs), dtype=np.float32)

    def collate(batch_rows):
        return tokenizer(
            [row[0] for row in batch_rows],
            [row[1] for row in batch_rows],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    loader = DataLoader(
        _PairDS(sorted_pairs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    offset = 0
    with torch.inference_mode():
        for encoded in tqdm(loader, desc="Scoring Stage-B v2 soup BGE pairs"):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(
                device.type,
                enabled=device.type == "cuda",
                dtype=amp_dtype,
            ):
                batch_predictions = (
                    torch.sigmoid(model(**encoded).logits.squeeze(-1).float())
                    .cpu()
                    .numpy()
                )
            size = len(batch_predictions)
            predictions[order[offset : offset + size]] = batch_predictions
            offset += size

    return predictions


def predict_pipeline(
    data_path: str,
    match_path: str,
    model_path: Path | str,
    output_csv_path: str,
    device: Optional[torch.device] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    print("=== Stage-B v2 checkpoint soup | pair_text v1 | single-pass inference ===")
    device = device or _get_best_device()
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    texts = _load_items_texts(data_path)
    match_df = pd.read_parquet(match_path, columns=["id1", "id2"])
    pairs, missing_items, empty_texts = _build_pairs(match_df, texts)
    if missing_items or empty_texts:
        print(
            "[warning] "
            f"missing item references={missing_items:,}; "
            f"empty item texts={empty_texts:,}. "
            "Rows are preserved and scored with empty text where necessary."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        local_files_only=True,
    ).to(device).eval()

    predictions = _score_pairs(
        model,
        tokenizer,
        pairs,
        device,
        batch_size,
    )
    if len(predictions) != len(match_df):
        raise RuntimeError(
            f"Prediction count mismatch: {len(predictions)} != {len(match_df)}"
        )

    output = match_df.copy()
    output["predict"] = predictions
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"saved {output_path} ({len(output):,} rows)")
    return output