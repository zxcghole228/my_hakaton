import contextlib
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from .preprocessing import load_required_item_texts


MAX_LEN = 192
OUTPUT_COLUMNS = ["id1", "id2", "predict"]


def _choose_device_and_batch_size() -> Tuple[torch.device, int]:
    override = os.environ.get("E5_BATCH_SIZE")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if memory_gb >= 60:
            batch_size = 512
        elif memory_gb >= 35:
            batch_size = 256
        elif memory_gb >= 14:
            batch_size = 64
        else:
            batch_size = 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        batch_size = 32
    else:
        device = torch.device("cpu")
        batch_size = 32

    if override is not None:
        batch_size = int(override)
        if batch_size <= 0:
            raise ValueError("E5_BATCH_SIZE must be a positive integer")

    return device, batch_size


def _load_tokenizer(model_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            use_fast=True,
        )
    except (KeyError, ValueError, OSError):
        # Backward-compatible fallback for images whose transformers version
        # predates the TokenizersBackend name written by transformers 5.
        return PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
            bos_token="<s>",
            cls_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            model_max_length=512,
        )


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()

    major, _ = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_matches(matches_path: Path) -> pd.DataFrame:
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    if list(matches.columns) != ["id1", "id2"]:
        raise ValueError("matches parquet must contain id1 and id2 columns")
    if matches[["id1", "id2"]].isna().any().any():
        raise ValueError("matches parquet contains null IDs")
    return matches


@torch.inference_mode()
def _predict(
    matches: pd.DataFrame,
    item_text: Dict[int, str],
    model_dir: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    print(f"[model] loading offline model from {model_dir}", flush=True)
    load_started = time.perf_counter()
    tokenizer = _load_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device).eval()
    print(
        f"[model] device={device} batch_size={batch_size} "
        f"load_time={time.perf_counter() - load_started:.2f}s",
        flush=True,
    )

    id1 = matches["id1"].to_numpy(copy=False)
    id2 = matches["id2"].to_numpy(copy=False)
    total = len(matches)
    predictions = np.empty(total, dtype=np.float32)

    # Length bucketing substantially reduces padding while the assignment back
    # to original indices preserves the exact input row order.
    lengths = np.fromiter(
        (len(item_text[a]) + len(item_text[b]) for a, b in zip(id1, id2)),
        dtype=np.int32,
        count=total,
    )
    order = np.argsort(lengths, kind="stable")
    report_every = max(1, (total + batch_size - 1) // batch_size // 20)
    started = time.perf_counter()

    for batch_number, offset in enumerate(range(0, total, batch_size), start=1):
        indices = order[offset : offset + batch_size]
        first_texts = [item_text[id1[index]] for index in indices]
        second_texts = [item_text[id2[index]] for index in indices]

        encoded = tokenizer(
            first_texts,
            second_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}

        with _autocast_context(device):
            logits = model(**encoded).logits.reshape(-1)

        predictions[indices] = torch.sigmoid(logits.float()).cpu().numpy()

        if batch_number % report_every == 0 or offset + len(indices) == total:
            done = min(offset + len(indices), total)
            elapsed = time.perf_counter() - started
            print(
                f"[predict] {done:,}/{total:,} pairs "
                f"({done / max(elapsed, 1e-9):.1f} pair/s)",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - started
    print(
        f"[predict] inference_time={elapsed:.2f}s "
        f"average_speed={total / max(elapsed, 1e-9):.1f} pair/s",
        flush=True,
    )
    return predictions


def _validate_output(result: pd.DataFrame, expected_rows: int) -> None:
    if list(result.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(f"Unexpected output columns: {list(result.columns)}")
    if len(result) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} output rows, got {len(result)}")
    if result.isna().any().any():
        raise RuntimeError("Output contains NaN values")
    values = result["predict"].to_numpy()
    if not len(values):
        return
    if not np.isfinite(values).all():
        raise RuntimeError("Output predictions contain non-finite values")
    if ((values < 0.0) | (values > 1.0)).any():
        raise RuntimeError("Output predictions are outside [0, 1]")


def predict_pipeline(
    items_path: Path,
    matches_path: Path,
    output_path: Path,
    model_dir: Path,
) -> pd.DataFrame:
    total_started = time.perf_counter()
    for path, label in ((items_path, "items"), (matches_path, "matches")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Offline model directory does not exist: {model_dir}")

    matches = _load_matches(matches_path)
    print(f"[matches] loaded {len(matches):,} pairs", flush=True)

    if matches.empty:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        required_ids = set(matches["id1"]).union(matches["id2"])
        item_text = load_required_item_texts(items_path, required_ids)
        device, batch_size = _choose_device_and_batch_size()
        predictions = _predict(matches, item_text, model_dir, device, batch_size)
        result = pd.DataFrame(
            {
                "id1": matches["id1"].to_numpy(copy=False),
                "id2": matches["id2"].to_numpy(copy=False),
                "predict": predictions,
            }
        )

    _validate_output(result, expected_rows=len(matches))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(
        f"[done] wrote {len(result):,} rows to {output_path} "
        f"in {time.perf_counter() - total_started:.2f}s",
        flush=True,
    )
    return result
