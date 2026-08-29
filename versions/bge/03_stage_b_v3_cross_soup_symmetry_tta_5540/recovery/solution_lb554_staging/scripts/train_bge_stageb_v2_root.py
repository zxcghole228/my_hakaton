#!/usr/bin/env python3
"""
Stage-B v2 fine-tune — human-aligned + anti-forgetting + full-gray checkpoint sweep.

Changes vs train_bge_stageb.py (v1):
  - save every eval checkpoint (step_*.pt), post-train sweep on FULL gray (~45k)
  - select best: 0.5*problem_ap + 0.3*full_gray + 0.2*tune_macro
  - more LLM replay, fashion-heavy gray sample for fast eval
  - loss rebalance (more distill + rank)
  - oversample Обувь x2 in human train

  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_bge_stageb_v2.py
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import shutil
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import accuracy_score, average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_ensemble import build_text  # noqa: E402

ITEMS_PATH = os.environ.get("STAGE_B_ITEMS_PATH", "./items.parquet")
ITEMS_HUMAN_PATH = os.environ.get("STAGE_B_ITEMS_HUMAN_PATH", "./items_human.parquet")
MATCHES_HUMAN = os.environ.get("STAGE_B_MATCHES_HUMAN", "./matches.parquet")
MATCHES_LLM = os.environ.get("STAGE_B_MATCHES_LLM", "./matches_llm.parquet")
INIT_CKPT = os.environ.get("STAGE_B_INIT_CKPT", "./user_bge_m3_320_run/best.pt")
OUT_DIR = os.environ.get("STAGE_B_OUT_DIR", "./user_bge_stageb_v2_320_run")
MODEL_NAME = os.environ.get("STAGE_B_MODEL_NAME", "deepvk/USER-bge-m3")
TRUST_REMOTE_CODE = os.environ.get("STAGE_B_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")
if not TRUST_REMOTE_CODE and "reranker" in MODEL_NAME.lower():
    TRUST_REMOTE_CODE = True
_BATCH_OVERRIDE = os.environ.get("STAGE_B_BATCH")

PROBLEM_CATS = frozenset({"Обувь", "Одежда", "Галантерея и аксессуары", "Ювелирные изделия"})
SHOES_CAT = "Обувь"
MAX_LEN = 320
SEED = 20260825

MANUAL_TUNE_FRAC = 0.10
MANUAL_EVAL_FRAC = 0.10
LLM_POOL_PER_CAT = 12_000
LLM_HARD_PER_CAT = 8_000
LLM_ANCHOR_PER_CAT = 3_000
GRAY_PER_CAT = 500
GRAY_PER_CAT_FASHION = 2_000

MAX_EPOCHS = 2
UNFREEZE_LAST_N = 12
BACKBONE_LR = 5e-6
HEAD_LR = 2e-5
WARMUP_RATIO = 0.05
SWAP_P = 0.5

LABEL_W = 0.50
DISTILL_W = 0.20
RANK_W = 0.22
SYMM_W = 0.10
DISTILL_T = 1.5
RANK_MARGIN = 0.10
RANK_TEMP = 0.20

SCORE_W_PROBLEM = 0.50
SCORE_W_GRAY = 0.30
SCORE_W_TUNE = 0.20

USE_DDP = "LOCAL_RANK" in os.environ
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN = LOCAL_RANK == 0


def log(*a, **k):
    if IS_MAIN:
        print(*a, **k, flush=True)


if USE_DDP:
    dist.init_process_group("nccl", timeout=timedelta(hours=3))
    torch.cuda.set_device(LOCAL_RANK)

random.seed(SEED + LOCAL_RANK)
np.random.seed(SEED + LOCAL_RANK)
torch.manual_seed(SEED + LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}" if USE_DDP else "cuda")
if IS_MAIN:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)
if USE_DDP:
    dist.barrier()

CC = torch.cuda.get_device_capability(LOCAL_RANK if USE_DDP else 0)
AMP = torch.bfloat16 if CC[0] >= 8 else torch.float16
gpu_gb = torch.cuda.get_device_properties(LOCAL_RANK if USE_DDP else 0).total_memory / 1e9
if _BATCH_OVERRIDE:
    BATCH = int(_BATCH_OVERRIDE)
else:
    default_batch = 128 if gpu_gb > 70 else 96
    BATCH = 96 if "reranker" in MODEL_NAME.lower() and default_batch >= 128 else default_batch
ACCUM = 2


def composite_score(problem_ap: float, gray_ap: float, tune_macro: float) -> float:
    return SCORE_W_PROBLEM * problem_ap + SCORE_W_GRAY * gray_ap + SCORE_W_TUNE * tune_macro


def union_find_components(frame: pd.DataFrame) -> np.ndarray:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        p = parent.setdefault(x, x)
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        parent[p] = p
        return p

    for a, b in zip(frame.id1.values, frame.id2.values):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return np.fromiter((find(int(i)) for i in frame.id1.values), dtype=np.int64, count=len(frame))


def component_split(frame: pd.DataFrame, tune_frac: float, eval_frac: float, seed: int) -> pd.DataFrame:
    out = frame.copy()
    out["_component"] = union_find_components(out)
    rng = np.random.RandomState(seed)
    split = {}
    for cat, group in out.groupby("category", observed=True):
        arr = np.array(sorted(group._component.unique()), dtype=np.int64)
        rng.shuffle(arr)
        n_tune = max(1, int(round(len(arr) * tune_frac)))
        n_eval = max(1, int(round(len(arr) * eval_frac)))
        if n_tune + n_eval >= len(arr):
            n_tune = min(n_tune, max(0, len(arr) - 2))
            n_eval = min(n_eval, max(1, len(arr) - n_tune - 1))
        for c in arr[:n_tune]:
            split[int(c)] = "tune"
        for c in arr[n_tune:n_tune + n_eval]:
            split[int(c)] = "eval"
        for c in arr[n_tune + n_eval:]:
            split[int(c)] = "train"
    out["_split"] = out._component.map(split)
    assert out._split.notna().all()
    return out


def load_texts_mixed(need: set[int]) -> tuple[dict[int, str], dict[int, str]]:
    texts: dict[int, str] = {}
    cats: dict[int, str] = {}
    if os.path.exists(ITEMS_HUMAN_PATH):
        df = pd.read_parquet(ITEMS_HUMAN_PATH, columns=["id", "name", "attributes", "category"])
        sub = df[df.id.isin(need)]
        for i, n, a, c in sub.itertuples(index=False, name=None):
            texts[int(i)] = build_text(n, a)
            cats[int(i)] = str(c)
        log(f"items_human hit {len(texts):,}/{len(need):,}")
    missing = need - set(texts)
    if missing:
        f = pq.ParquetFile(ITEMS_PATH)
        for b in f.iter_batches(columns=["id", "name", "attributes", "category"], batch_size=500_000):
            df = b.to_pandas()
            sub = df[df.id.isin(missing)]
            for i, n, a, c in sub.itertuples(index=False, name=None):
                i = int(i)
                texts[i] = build_text(n, a)
                cats[i] = str(c)
                missing.discard(i)
            if not missing:
                break
        log(f"items.parquet scan; missing left {len(missing):,}; total texts {len(texts):,}")
    if missing:
        raise RuntimeError(f"missing texts for {len(missing)} ids (e.g. {next(iter(missing))})")
    return texts, cats


def llm_train_frame() -> pd.DataFrame:
    ml = pd.read_parquet(MATCHES_LLM)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        p = parent.setdefault(x, x)
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        parent[p] = p
        return p

    for a, b in zip(ml.id1.values, ml.id2.values):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    comp = np.fromiter((find(int(i)) for i in ml.id1.values), dtype=np.int64, count=len(ml))
    rng = np.random.RandomState(13)
    val_comps = set(np.unique(comp)[rng.rand(len(np.unique(comp))) < 0.03].tolist())
    train = ml[~np.fromiter((c in val_comps for c in comp), dtype=bool, count=len(ml))].copy()
    train["target"] = train.target.astype(np.float32)
    del ml, parent, comp
    gc.collect()
    return train.reset_index(drop=True)


def load_gray_frame() -> pd.DataFrame:
    ml = pd.read_parquet(MATCHES_LLM)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        p = parent.setdefault(x, x)
        while p != parent[p]:
            parent[p] = parent[parent[p]]
            p = parent[p]
        parent[p] = p
        return p

    for a, b in zip(ml.id1.values, ml.id2.values):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    comp = np.fromiter((find(int(i)) for i in ml.id1.values), dtype=np.int64, count=len(ml))
    rng = np.random.RandomState(13)
    val_comps = set(np.unique(comp)[rng.rand(len(np.unique(comp))) < 0.03].tolist())
    gray = ml[np.fromiter((c in val_comps for c in comp), dtype=bool, count=len(ml))].copy()
    gray = gray[(gray.target > 0.2) & (gray.target < 0.8)]
    gray["target"] = (gray.target >= 0.5).astype(np.int8)
    del ml, parent, comp
    gc.collect()
    return gray.reset_index(drop=True)


def attach_categories(frame: pd.DataFrame, cats: dict[int, str]) -> pd.DataFrame:
    out = frame.copy()
    out["category"] = [cats.get(int(i), "unknown") for i in out.id1]
    return out


def sample_gray(gray: pd.DataFrame, cats: dict[int, str]) -> pd.DataFrame:
    """Fast eval sample: more pairs for fashion problem cats."""
    gray = attach_categories(gray, cats)
    parts = []
    rng2 = np.random.RandomState(SEED + 7)
    for cat, g in gray.groupby("category", observed=True):
        cap = GRAY_PER_CAT_FASHION if cat in PROBLEM_CATS else GRAY_PER_CAT
        n = min(cap, len(g))
        parts.append(g.sample(n, random_state=rng2) if len(g) > n else g)
    return pd.concat(parts, ignore_index=True)


def sample_llm_replay(pool: pd.DataFrame, teacher: np.ndarray) -> pd.DataFrame:
    llm_train = pool.copy()
    llm_train["teacher_prob"] = teacher.clip(0.001, 0.999)
    llm_train["error"] = np.abs(llm_train.teacher_prob - llm_train.target)
    parts = []
    rng = np.random.RandomState(SEED + 11)
    for _, g in llm_train.groupby("category", observed=True):
        hard = g.nlargest(min(LLM_HARD_PER_CAT, len(g)), "error")
        used = set(map(tuple, hard[["id1", "id2"]].itertuples(index=False, name=None)))
        rest = g[~g.apply(lambda r: (r.id1, r.id2) in used, axis=1)]
        anchor = rest.sample(min(LLM_ANCHOR_PER_CAT, len(rest)), random_state=rng) if len(rest) else rest
        part = pd.concat([hard, anchor], ignore_index=True)
        part["source"] = "llm_hard"
        part.loc[part.index.isin(anchor.index), "source"] = "llm_anchor"
        parts.append(part)
    replay = pd.concat(parts, ignore_index=True)
    replay["label_target"] = replay.target.astype(np.float32)
    replay["label_weight"] = np.where(replay.source == "llm_hard", 1.5, 0.8).astype(np.float32)
    replay["distill_weight"] = np.where(replay.source == "llm_hard", 0.08, 1.0).astype(np.float32)
    return replay


def oversample_shoes(manual_train: pd.DataFrame) -> pd.DataFrame:
    shoes = manual_train[manual_train.category == SHOES_CAT]
    if shoes.empty:
        return manual_train
    out = pd.concat([manual_train, shoes], ignore_index=True)
    log(f"oversample {SHOES_CAT}: {len(manual_train):,} -> {len(out):,} human train rows")
    return out


def build_human_train(manual_train: pd.DataFrame, teacher: np.ndarray) -> pd.DataFrame:
    out = manual_train.copy()
    out["label_target"] = out.target.astype(np.float32)
    out["teacher_prob"] = teacher.clip(0.001, 0.999)
    err = np.abs(out.teacher_prob - out.label_target)
    out["label_weight"] = (3.5 + 2.0 * err).astype(np.float32)
    out["distill_weight"] = np.float32(0.02)
    out["source"] = "human"
    return out


def build_train_frame(human_part: pd.DataFrame, llm_part: pd.DataFrame) -> pd.DataFrame:
    cols = ["id1", "id2", "category", "label_target", "label_weight", "teacher_prob", "distill_weight", "source"]
    return pd.concat([human_part[cols], llm_part[cols]], ignore_index=True)


def build_epoch_order(frame: pd.DataFrame, epochs: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    by_cat = {cat: idx.values for cat, idx in frame.groupby("category", observed=True).groups.items()}
    cats = list(by_cat)
    order = []
    for _ep in range(epochs):
        ptr = {c: 0 for c in cats}
        sizes = {c: len(by_cat[c]) for c in cats}
        perm = {c: rng.permutation(s) for c, s in sizes.items()}
        remaining = sum(sizes.values())
        while remaining > 0:
            rng.shuffle(cats)
            for c in cats:
                if ptr[c] >= sizes[c]:
                    continue
                order.append(by_cat[c][perm[c][ptr[c]]])
                ptr[c] += 1
                remaining -= 1
    return np.array(order, dtype=np.int64)


def locate_encoder_layers(model: torch.nn.Module):
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        return model.roberta.encoder.layer
    if hasattr(model, "xlm_roberta") and hasattr(model.xlm_roberta, "encoder"):
        return model.xlm_roberta.encoder.layer
    raise RuntimeError("unknown encoder layout")


def configure_trainable(model: torch.nn.Module, unfreeze_last_n: int) -> None:
    for p in model.parameters():
        p.requires_grad = False
    layers = locate_encoder_layers(model)
    for layer in layers[-unfreeze_last_n:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def param_groups(model: torch.nn.Module):
    layers = locate_encoder_layers(model)
    backbone_ids = {id(p) for layer in layers for p in layer.parameters()}
    head_ids = {id(p) for p in model.classifier.parameters()}
    backbone, head = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif id(p) in backbone_ids:
            backbone.append(p)
    return [
        {"params": backbone, "lr": BACKBONE_LR},
        {"params": head, "lr": HEAD_LR},
    ]


@torch.no_grad()
def predict_probs(
    model, tok, texts: dict[int, str], frame: pd.DataFrame, bs: int = 128, tag: str = "",
) -> np.ndarray:
    model.eval()
    a, b = frame.id1.values, frame.id2.values
    preds = np.empty(len(frame), dtype=np.float32)
    n = len(frame)
    t0 = time.time()
    for i in range(0, n, bs):
        chunk = slice(i, i + bs)
        enc = tok(
            [texts[int(x)] for x in a[chunk]],
            [texts[int(x)] for x in b[chunk]],
            padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", AMP):
            p = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        preds[chunk] = p
        if tag and IS_MAIN and (i == 0 or (i // bs) % 200 == 0):
            done = min(i + bs, n)
            sps = done / max(time.time() - t0, 1e-6)
            log(f"  {tag}: {done:,}/{n:,} ({100 * done / n:.1f}%) {sps:.0f} pair/s")
    return preds


def macro_ap_cat(df: pd.DataFrame, p: np.ndarray) -> tuple[float, dict]:
    z = df.copy()
    z["p"] = p
    per = {}
    for cat, g in z.groupby("category", observed=True):
        ycol = "label_target" if "label_target" in g.columns else "target"
        if g[ycol].nunique() > 1:
            per[cat] = float(average_precision_score(g[ycol], g.p))
    return float(np.mean(list(per.values()))) if per else 0.0, per


class StageBDS(Dataset):
    def __init__(self, frame: pd.DataFrame, order: np.ndarray, texts: dict[int, str], cat_to_id: dict[str, int]):
        self.frame = frame
        self.order = order
        self.texts = texts
        self.cat_to_id = cat_to_id

    def __len__(self):
        return len(self.order)

    def __getitem__(self, i):
        row = self.frame.iloc[int(self.order[i])]
        id1, id2 = int(row.id1), int(row.id2)
        if random.random() < SWAP_P:
            id1, id2 = id2, id1
        return (
            self.texts[id1], self.texts[id2],
            float(row.label_target), float(row.label_weight),
            float(row.teacher_prob), float(row.distill_weight),
            self.cat_to_id[str(row.category)],
        )


def collate_batch(batch, tok):
    enc = tok([x[0] for x in batch], [x[1] for x in batch],
              padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    y = torch.tensor([x[2] for x in batch], dtype=torch.float32)
    lw = torch.tensor([x[3] for x in batch], dtype=torch.float32)
    tp = torch.tensor([x[4] for x in batch], dtype=torch.float32)
    dw = torch.tensor([x[5] for x in batch], dtype=torch.float32)
    cats = torch.tensor([x[6] for x in batch], dtype=torch.long)
    enc_rev = tok([x[1] for x in batch], [x[0] for x in batch],
                  padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc, enc_rev, y, lw, tp, dw, cats


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def rank_loss(logits: torch.Tensor, targets: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
    losses = []
    for cat in cats.unique():
        m = cats == cat
        if m.sum() < 2:
            continue
        pos = logits[m][targets[m] > 0.5]
        neg = logits[m][targets[m] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
        losses.append(F.softplus((RANK_MARGIN - diffs) / RANK_TEMP).mean() * RANK_TEMP)
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def combined_loss(logits, logits_rev, y, lw, tp, dw, cats):
    sup = weighted_mean(F.binary_cross_entropy_with_logits(logits, y, reduction="none"), lw)
    teacher_logit = torch.logit(tp.clamp(1e-4, 1 - 1e-4))
    soft_teacher = torch.sigmoid(teacher_logit / DISTILL_T)
    distill = weighted_mean(
        F.binary_cross_entropy_with_logits(logits / DISTILL_T, soft_teacher, reduction="none") * DISTILL_T ** 2,
        dw,
    )
    rank = rank_loss(logits, y, cats)
    symmetry = F.mse_loss(logits, logits_rev)
    total = LABEL_W * sup + DISTILL_W * distill + RANK_W * rank + SYMM_W * symmetry
    return total, (float(sup), float(distill), float(rank), float(symmetry))


def sweep_checkpoints(
    raw_model,
    tok,
    texts: dict[int, str],
    manual_tune: pd.DataFrame,
    gray_full: pd.DataFrame,
) -> tuple[str, list[dict]]:
    ckpt_dir = f"{OUT_DIR}/checkpoints"
    paths = sorted(glob.glob(f"{ckpt_dir}/step_*.pt"))
    last_path = f"{OUT_DIR}/last.pt"
    if os.path.exists(last_path):
        paths.append(last_path)

    if not paths:
        raise RuntimeError("no checkpoints to sweep")

    log(f"sweep {len(paths)} checkpoints on FULL gray ({len(gray_full):,} pairs)...")
    rows: list[dict] = []
    best_path = paths[0]
    best_score = -1.0

    prob_tune = manual_tune[manual_tune.category.isin(PROBLEM_CATS)]

    for path in paths:
        raw_model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        tune_p = predict_probs(raw_model, tok, texts, manual_tune.assign(label_target=manual_tune.target))
        tune_macro, _ = macro_ap_cat(manual_tune.assign(label_target=manual_tune.target), tune_p)
        if len(prob_tune):
            prob_p = predict_probs(raw_model, tok, texts, prob_tune.assign(label_target=prob_tune.target))
            problem_ap = macro_ap_cat(prob_tune.assign(label_target=prob_tune.target), prob_p)[0]
        else:
            problem_ap = 0.0
        gray_p = predict_probs(raw_model, tok, texts, gray_full.assign(label_target=gray_full.target))
        gray_macro, _ = macro_ap_cat(gray_full.assign(label_target=gray_full.target), gray_p)
        score = composite_score(problem_ap, gray_macro, tune_macro)
        step_tag = os.path.basename(path).replace(".pt", "")
        row = {
            "checkpoint": step_tag,
            "path": path,
            "tune_macro": tune_macro,
            "problem_ap": problem_ap,
            "gray_full": gray_macro,
            "composite_score": score,
        }
        rows.append(row)
        log(f"  {step_tag}: tune={tune_macro:.4f} problem={problem_ap:.4f} gray_full={gray_macro:.4f} score={score:.4f}")
        if score > best_score:
            best_score = score
            best_path = path

    sweep_path = f"{OUT_DIR}/checkpoint_sweep.json"
    json.dump(rows, open(sweep_path, "w"), indent=2, ensure_ascii=False)
    best_row = max(rows, key=lambda r: r["composite_score"])
    log(f"BEST sweep: {best_path} score={best_score:.4f} "
        f"(problem={best_row['problem_ap']:.4f} gray={best_row['gray_full']:.4f} tune={best_row['tune_macro']:.4f})")
    return best_path, rows


def export_best_model(raw_model, tok, best_path: str, sweep_rows: list[dict]) -> None:
    """Load sweep winner, persist best.pt + HuggingFace export (+ fp16 copy for submit)."""
    raw_model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    best_pt = f"{OUT_DIR}/best.pt"
    torch.save(raw_model.state_dict(), best_pt)
    best_ckpt_copy = f"{OUT_DIR}/checkpoints/best.pt"
    shutil.copy2(best_path, best_ckpt_copy)
    log(f"saved BEST weights -> {best_pt} (from {best_path})")

    export_dir = f"{OUT_DIR}/export"
    export_fp16 = f"{OUT_DIR}/export_fp16"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    if os.path.exists(export_fp16):
        shutil.rmtree(export_fp16)
    raw_model.save_pretrained(export_dir)
    tok.save_pretrained(export_dir)
    shutil.copytree(export_dir, export_fp16)
    log(f"export BEST -> {export_dir} and {export_fp16}")

    manifest = {
        "selected_checkpoint": best_path,
        "best_pt": best_pt,
        "export": export_dir,
        "export_fp16": export_fp16,
    }
    if sweep_rows:
        best_row = max(sweep_rows, key=lambda r: r["composite_score"])
        manifest["sweep_best"] = best_row
    json.dump(manifest, open(f"{OUT_DIR}/best_checkpoint.json", "w"), indent=2, ensure_ascii=False)


def ensure_teacher_cache(
    manual_train: pd.DataFrame,
    pool_ids: pd.DataFrame,
    texts: dict[int, str],
    cache_path: str,
) -> None:
    if os.path.exists(cache_path):
        log(f"teacher cache exists: {cache_path}")
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, trust_remote_code=TRUST_REMOTE_CODE,
    )
    base.load_state_dict(torch.load(INIT_CKPT, map_location="cpu", weights_only=True))
    base = base.to(device)
    log(f"scoring teacher probs for human train ({len(manual_train):,} pairs)...")
    teacher_human = predict_probs(base, tok, texts, manual_train, tag="human")
    log(f"scoring teacher probs for llm pool ({len(pool_ids):,} pairs)...")
    teacher_llm = predict_probs(base, tok, texts, pool_ids, tag="llm_pool")
    np.savez(cache_path, human_train=teacher_human, llm_pool=teacher_llm)
    del base
    torch.cuda.empty_cache()
    log(f"saved teacher cache -> {cache_path}")


def load_train_data():
    human = pd.read_parquet(MATCHES_HUMAN)
    human["target"] = (human.target >= 0.5).astype(np.float32)
    llm_train = llm_train_frame()

    cat_map = pd.read_parquet(ITEMS_HUMAN_PATH, columns=["id", "category"]).drop_duplicates("id")
    human = human.merge(cat_map.rename(columns={"id": "id1"}), on="id1", how="left")
    human["category"] = human.category.fillna("unknown").astype(str)
    human = component_split(human, MANUAL_TUNE_FRAC, MANUAL_EVAL_FRAC, SEED)
    manual_train = human[human._split == "train"].reset_index(drop=True)
    manual_tune = human[human._split == "tune"].reset_index(drop=True)
    manual_eval = human[human._split == "eval"].reset_index(drop=True)
    log("manual split:", {k: len(v) for k, v in [("train", manual_train), ("tune", manual_tune), ("eval", manual_eval)]})

    need = set(manual_train.id1) | set(manual_train.id2)
    need |= set(manual_tune.id1) | set(manual_tune.id2) | set(manual_eval.id1) | set(manual_eval.id2)
    llm_train["category"] = llm_train.id1.map(cat_map.set_index("id")["category"]).fillna("unknown").astype(str)
    pool_parts = []
    rng_pool = np.random.RandomState(SEED + 3)
    for _, g in llm_train.groupby("category", observed=True):
        pool_parts.append(g.sample(min(LLM_POOL_PER_CAT, len(g)), random_state=rng_pool))
    pool_ids = pd.concat(pool_parts, ignore_index=True)
    gray_frame = load_gray_frame()
    need |= set(pool_ids.id1) | set(pool_ids.id2)
    need |= set(gray_frame.id1) | set(gray_frame.id2)
    log(f"loading {len(need):,} texts...")
    texts, cats_full = load_texts_mixed(need)
    for df in (manual_train, manual_tune, manual_eval, llm_train):
        if "category" not in df or df.category.isna().any():
            df["category"] = [cats_full.get(int(i), "unknown") for i in df.id1]

    manual_train = oversample_shoes(manual_train)
    gray_sample = sample_gray(gray_frame, cats_full)
    gray_full = attach_categories(gray_frame, cats_full)
    log(f"gray: full={len(gray_full):,} sample={len(gray_sample):,}")
    return manual_train, manual_tune, manual_eval, pool_ids, texts, gray_sample, gray_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--skip-sweep", action="store_true", help="skip post-train full-gray checkpoint sweep")
    ap.add_argument(
        "--precompute-teacher",
        action="store_true",
        help="single-GPU: build teacher_probs.npz then exit (run before torchrun)",
    )
    args = ap.parse_args()

    if args.precompute_teacher:
        if USE_DDP:
            raise RuntimeError("Run --precompute-teacher with plain python on 1 GPU, not torchrun")
        log("PRECOMPUTE teacher cache (single GPU)")
        manual_train, _, _, pool_ids, texts, _, _ = load_train_data()
        ensure_teacher_cache(manual_train, pool_ids, texts, f"{OUT_DIR}/teacher_probs.npz")
        return

    log(f"STAGE-B v2 | gpus={WORLD_SIZE} batch={BATCH} accum={ACCUM} amp={AMP}")
    log(f"model={MODEL_NAME} init={INIT_CKPT} out={OUT_DIR}")
    log(f"loss w: label={LABEL_W} distill={DISTILL_W} rank={RANK_W} symm={SYMM_W}")
    log(f"LLM pool/hard/anchor={LLM_POOL_PER_CAT}/{LLM_HARD_PER_CAT}/{LLM_ANCHOR_PER_CAT}")
    log(f"gray sample caps: fashion={GRAY_PER_CAT_FASHION} other={GRAY_PER_CAT}")

    manual_train, manual_tune, manual_eval, pool_ids, texts, gray_sample, gray_full = load_train_data()

    cache_path = f"{OUT_DIR}/teacher_probs.npz"
    if IS_MAIN and not os.path.exists(cache_path):
        log("teacher cache missing — build with: python3 train_bge_stageb_v2.py --precompute-teacher")
        ensure_teacher_cache(manual_train, pool_ids, texts, cache_path)

    if USE_DDP:
        dist.barrier()

    cache = np.load(cache_path, allow_pickle=True)
    teacher_human = cache["human_train"]
    teacher_llm = cache["llm_pool"]
    if IS_MAIN:
        log("loaded teacher cache")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)

    llm_replay = sample_llm_replay(pool_ids, teacher_llm)
    human_part = build_human_train(manual_train, teacher_human)
    train_frame = build_train_frame(human_part, llm_replay)
    train_order = build_epoch_order(train_frame, args.epochs, SEED)
    log(f"train rows {len(train_frame):,} | ordered samples {len(train_order):,}")

    if USE_DDP:
        dist.barrier()

    cat_names = sorted(train_frame.category.unique())
    cat_to_id = {c: i for i, c in enumerate(cat_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.load_state_dict(torch.load(INIT_CKPT, map_location="cpu", weights_only=True))
    configure_trainable(model, UNFREEZE_LAST_N)
    model = model.to(device)
    raw_model = model
    if USE_DDP:
        model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)

    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in raw_model.parameters())
    log(f"trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    ds = StageBDS(train_frame, train_order, texts, cat_to_id)
    sampler = DistributedSampler(ds, shuffle=False) if USE_DDP else None
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, sampler=sampler,
                    num_workers=0, drop_last=True,
                    collate_fn=lambda b: collate_batch(b, tok))
    steps_total = max(1, (len(dl) // ACCUM) * args.epochs)
    opt = torch.optim.AdamW(param_groups(raw_model), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(steps_total * WARMUP_RATIO)), steps_total)
    scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)

    eval_log_path = f"{OUT_DIR}/checkpoint_evals.jsonl"

    @torch.no_grad()
    def eval_split(sub: pd.DataFrame, name: str) -> dict:
        raw_model.eval()
        p = predict_probs(raw_model, tok, texts, sub.assign(label_target=sub.target))
        y = sub.target.values.astype(int)
        macro, _ = macro_ap_cat(sub.assign(label_target=sub.target), p)
        prob = sub[sub.category.isin(PROBLEM_CATS)]
        pp = predict_probs(raw_model, tok, texts, prob.assign(label_target=prob.target)) if len(prob) else np.array([])
        prob_ap = macro_ap_cat(prob.assign(label_target=prob.target), pp)[0] if len(prob) else 0.0
        raw_model.train()
        return {
            "name": name,
            "macro_ap": macro,
            "problem_ap": prob_ap,
            "agree": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "pred_pos": float((p >= 0.5).mean()),
        }

    @torch.no_grad()
    def eval_gray(df: pd.DataFrame) -> float:
        raw_model.eval()
        p = predict_probs(raw_model, tok, texts, df.assign(label_target=df.target))
        m, _ = macro_ap_cat(df.assign(label_target=df.target), p)
        raw_model.train()
        return m

    step = 0
    t0 = time.time()

    if IS_MAIN:
        tune0 = eval_split(manual_tune, "tune")
        eval0 = eval_split(manual_eval, "eval")
        gray_sample0 = eval_gray(gray_sample)
        log(f"baseline tune AP={tune0['macro_ap']:.4f} problem={tune0['problem_ap']:.4f}")
        log(f"baseline eval AP={eval0['macro_ap']:.4f} gray_sample={gray_sample0:.4f}")

    raw_model.train()
    for ep in range(args.epochs):
        if sampler:
            sampler.set_epoch(ep)
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(dl):
            enc, enc_rev, y, lw, tp, dw, cats = batch
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            enc_rev = {k: v.to(device, non_blocking=True) for k, v in enc_rev.items()}
            y, lw, tp, dw, cats = [t.to(device) for t in (y, lw, tp, dw, cats)]
            with torch.autocast("cuda", AMP):
                logits = model(**enc).logits.squeeze(-1)
                logits_rev = model(**enc_rev).logits.squeeze(-1)
                loss, _parts = combined_loss(logits, logits_rev, y, lw, tp, dw, cats)
                loss = loss / ACCUM
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (bi + 1) % ACCUM == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in raw_model.parameters() if p.requires_grad], 1.0)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if IS_MAIN and step % 100 == 0:
                    sps = step * BATCH * ACCUM * WORLD_SIZE / max(time.time() - t0, 1e-6)
                    log(f"ep{ep} step {step}/{steps_total} loss={loss.item()*ACCUM:.4f} {sps:.0f} pair/s")
                if IS_MAIN and step % args.eval_every == 0:
                    tune_m = eval_split(manual_tune, "tune")
                    gray_sample_m = eval_gray(gray_sample)
                    score_sample = composite_score(tune_m["problem_ap"], gray_sample_m, tune_m["macro_ap"])
                    ckpt_path = f"{OUT_DIR}/checkpoints/step_{step:05d}.pt"
                    torch.save(raw_model.state_dict(), ckpt_path)
                    torch.save(raw_model.state_dict(), f"{OUT_DIR}/last.pt")
                    rec = {
                        "step": step,
                        "tune_macro": tune_m["macro_ap"],
                        "problem_ap": tune_m["problem_ap"],
                        "gray_sample": gray_sample_m,
                        "composite_sample": score_sample,
                        "checkpoint": ckpt_path,
                    }
                    with open(eval_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log(
                        f"  >> step {step}: tune={tune_m['macro_ap']:.4f} problem={tune_m['problem_ap']:.4f} "
                        f"gray_sample={gray_sample_m:.4f} score_sample={score_sample:.4f} -> saved {ckpt_path}"
                    )

    if USE_DDP:
        dist.barrier()

    if IS_MAIN:
        if not os.path.exists(f"{OUT_DIR}/last.pt"):
            torch.save(raw_model.state_dict(), f"{OUT_DIR}/last.pt")

        best_path = f"{OUT_DIR}/last.pt"
        sweep_rows: list[dict] = []
        if not args.skip_sweep:
            best_path, sweep_rows = sweep_checkpoints(raw_model, tok, texts, manual_tune, gray_full)
        else:
            log("skip-sweep: exporting last.pt (not recommended)")

        export_best_model(raw_model, tok, best_path, sweep_rows)
        tune_f = eval_split(manual_tune, "tune")
        eval_f = eval_split(manual_eval, "eval")
        gray_sample_f = eval_gray(gray_sample)
        gray_full_f = eval_gray(gray_full)

        metrics = {
            "version": "stageb_v2",
            "model": MODEL_NAME,
            "init": INIT_CKPT,
            "selected_checkpoint": best_path,
            "best_pt": f"{OUT_DIR}/best.pt",
            "export_fp16": f"{OUT_DIR}/export_fp16",
            "loss_weights": {"label": LABEL_W, "distill": DISTILL_W, "rank": RANK_W, "symm": SYMM_W},
            "llm_replay": {
                "pool_per_cat": LLM_POOL_PER_CAT,
                "hard_per_cat": LLM_HARD_PER_CAT,
                "anchor_per_cat": LLM_ANCHOR_PER_CAT,
            },
            "gray_caps": {"fashion": GRAY_PER_CAT_FASHION, "other": GRAY_PER_CAT},
            "score_weights": {
                "problem": SCORE_W_PROBLEM,
                "gray_full": SCORE_W_GRAY,
                "tune": SCORE_W_TUNE,
            },
            "final_tune": tune_f,
            "final_eval": eval_f,
            "final_gray_sample": gray_sample_f,
            "final_gray_full": gray_full_f,
            "composite_final": composite_score(tune_f["problem_ap"], gray_full_f, tune_f["macro_ap"]),
            "train_rows": int(len(train_frame)),
            "epochs": args.epochs,
            "sweep_n_checkpoints": len(sweep_rows),
        }
        if sweep_rows:
            best_row = max(sweep_rows, key=lambda r: r["composite_score"])
            metrics["sweep_best"] = best_row

        log(f"DONE BEST={best_path}")
        log(f"  tune={tune_f} eval={eval_f}")
        log(f"  gray_sample={gray_sample_f:.4f} gray_full={gray_full_f:.4f} composite={metrics['composite_final']:.4f}")

        json.dump(metrics, open(f"{OUT_DIR}/metrics.json", "w"), indent=2, ensure_ascii=False)
        log(f"metrics {OUT_DIR}/metrics.json | manifest {OUT_DIR}/best_checkpoint.json")

    if USE_DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
