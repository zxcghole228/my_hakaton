import contextlib
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from .preprocessing import (
    ItemFeatures,
    MAX_ATTR_CHARS,
    VARIANT_NAMES,
    load_required_item_features,
    make_pair_signal,
)


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


def _load_tokenizer(tokenizer_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            use_fast=True,
        )
    except (KeyError, ValueError, OSError):
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
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


def _load_routing(model_dir: Path) -> Dict[str, str]:
    with (model_dir / "routing.json").open(encoding="utf-8") as file:
        payload = json.load(file)
    if int(payload.get("max_len", -1)) != MAX_LEN:
        raise ValueError("routing.json max_len does not match the V3 pipeline")
    if int(payload.get("max_attr_chars", -1)) != MAX_ATTR_CHARS:
        raise ValueError("routing.json max_attr_chars does not match the V3 pipeline")
    if tuple(payload.get("variant_names", ())) != VARIANT_NAMES:
        raise ValueError("routing.json variant_names do not match the V3 pipeline")
    routing = payload.get("routing")
    if not isinstance(routing, dict) or not routing:
        raise ValueError("routing.json does not contain a valid routing object")
    invalid = set(routing.values()) - {"base", "specialist"}
    if invalid:
        raise ValueError(f"routing.json contains invalid routes: {sorted(invalid)}")
    return routing


def _pair_texts(
    index: int,
    id1: np.ndarray,
    id2: np.ndarray,
    item_features: Dict[int, ItemFeatures],
    specialist: bool,
) -> Tuple[str, str]:
    first = item_features[id1[index]]
    second = item_features[id2[index]]
    if not specialist:
        return first.text, second.text
    signal = make_pair_signal(first.variant, second.variant)
    return first.text + signal, second.text + signal


@torch.inference_mode()
def _predict_route(
    route_name: str,
    model_path: Path,
    indices: np.ndarray,
    id1: np.ndarray,
    id2: np.ndarray,
    item_features: Dict[int, ItemFeatures],
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    specialist = route_name == "specialist"
    print(f"[model:{route_name}] loading offline model from {model_path}", flush=True)
    load_started = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.to(device).eval()
    print(
        f"[model:{route_name}] device={device} batch_size={batch_size} "
        f"load_time={time.perf_counter() - load_started:.2f}s",
        flush=True,
    )

    lengths = np.fromiter(
        (
            sum(
                map(
                    len,
                    _pair_texts(index, id1, id2, item_features, specialist),
                )
            )
            for index in indices
        ),
        dtype=np.int32,
        count=len(indices),
    )
    order = np.argsort(lengths, kind="stable")
    sorted_indices = indices[order]
    route_predictions = np.empty(len(indices), dtype=np.float32)
    route_positions = np.empty(len(indices), dtype=np.int64)
    route_positions[order] = np.arange(len(indices))

    report_every = max(1, (len(indices) + batch_size - 1) // batch_size // 20)
    started = time.perf_counter()
    sorted_predictions = np.empty(len(indices), dtype=np.float32)

    for batch_number, offset in enumerate(range(0, len(indices), batch_size), start=1):
        batch_indices = sorted_indices[offset : offset + batch_size]
        pairs = [
            _pair_texts(index, id1, id2, item_features, specialist)
            for index in batch_indices
        ]
        encoded = tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
        with _autocast_context(device):
            logits = model(**encoded).logits.reshape(-1)
        sorted_predictions[offset : offset + len(batch_indices)] = (
            torch.sigmoid(logits.float()).cpu().numpy()
        )

        if batch_number % report_every == 0 or offset + len(batch_indices) == len(indices):
            done = min(offset + len(batch_indices), len(indices))
            elapsed = time.perf_counter() - started
            print(
                f"[predict:{route_name}] {done:,}/{len(indices):,} pairs "
                f"({done / max(elapsed, 1e-9):.1f} pair/s)",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    route_predictions[:] = sorted_predictions[route_positions]
    elapsed = time.perf_counter() - started
    print(
        f"[predict:{route_name}] inference_time={elapsed:.2f}s "
        f"average_speed={len(indices) / max(elapsed, 1e-9):.1f} pair/s",
        flush=True,
    )

    model.to("cpu")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return route_predictions


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
    required_model_paths = [
        model_dir / "base_model",
        model_dir / "fashion_specialist",
        model_dir / "tokenizer",
        model_dir / "routing.json",
    ]
    missing_model_paths = [str(path) for path in required_model_paths if not path.exists()]
    if missing_model_paths:
        raise FileNotFoundError(f"Missing offline model paths: {missing_model_paths}")

    routing = _load_routing(model_dir)
    matches = _load_matches(matches_path)
    print(f"[matches] loaded {len(matches):,} pairs", flush=True)

    if matches.empty:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        required_ids = set(matches["id1"]).union(matches["id2"])
        item_features = load_required_item_features(items_path, required_ids)
        id1 = matches["id1"].to_numpy(copy=False)
        id2 = matches["id2"].to_numpy(copy=False)
        routes = np.asarray(
            [routing.get(item_features[item_id].category, "base") for item_id in id1],
            dtype=object,
        )
        base_indices = np.flatnonzero(routes == "base")
        specialist_indices = np.flatnonzero(routes == "specialist")
        print(
            f"[routing] base={len(base_indices):,} specialist={len(specialist_indices):,}",
            flush=True,
        )

        device, batch_size = _choose_device_and_batch_size()
        tokenizer = _load_tokenizer(model_dir / "tokenizer")
        predictions = np.full(len(matches), np.nan, dtype=np.float32)
        route_specs: Sequence[Tuple[str, Path, np.ndarray]] = (
            ("base", model_dir / "base_model", base_indices),
            ("specialist", model_dir / "fashion_specialist", specialist_indices),
        )
        for route_name, model_path, indices in route_specs:
            if not len(indices):
                continue
            predictions[indices] = _predict_route(
                route_name=route_name,
                model_path=model_path,
                indices=indices,
                id1=id1,
                id2=id2,
                item_features=item_features,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
            )

        result = pd.DataFrame(
            {
                "id1": id1,
                "id2": id2,
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
