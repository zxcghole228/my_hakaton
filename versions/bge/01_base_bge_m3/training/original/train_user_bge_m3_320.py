import os

# ===== КОНФИГ =====
ITEMS_PATH   = os.environ.get("STAGE_A_ITEMS_PATH", "./items.parquet")
MATCHES_PATH = os.environ.get("STAGE_A_MATCHES_PATH", "./matches_llm.parquet")
OUT_DIR      = os.environ.get(
    "STAGE_A_OUT_DIR",
    "./user_bge_m3_320_run",
)  # notebook задаёт отдельную папку, legacy CLI сохраняет прежний путь

MODEL_NAME = "deepvk/USER-bge-m3"
MAX_LEN    = 320
EPOCHS     = 2
LR         = 2e-5
WARMUP_FRAC = 0.05
SWAP_P     = 0.5
LOG_EVERY = int(os.environ.get("STAGE_A_LOG_EVERY", "250"))
EVAL_EVERY = int(os.environ.get("STAGE_A_EVAL_EVERY", "8000"))
TARGET_GLOBAL_BATCH = int(os.environ.get("STAGE_A_GLOBAL_BATCH", "512"))

import json, gc, time, random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SEED = 42
USE_DDP = "LOCAL_RANK" in os.environ
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN = LOCAL_RANK == 0

def log(*a, **k):
    if IS_MAIN:
        print(*a, **k, flush=True)

def record_history(event, **values):
    if IS_MAIN:
        row = {"event": event, **values}
        with open(f"{OUT_DIR}/training_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if USE_DDP:
    dist.init_process_group("nccl")
    torch.cuda.set_device(LOCAL_RANK)

random.seed(SEED + LOCAL_RANK)
np.random.seed(SEED + LOCAL_RANK)
torch.manual_seed(SEED + LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}" if USE_DDP else "cuda")
if IS_MAIN:
    os.makedirs(OUT_DIR, exist_ok=True)
if USE_DDP:
    dist.barrier()

CC = torch.cuda.get_device_capability(LOCAL_RANK if USE_DDP else 0)
AMP = torch.bfloat16 if CC[0] >= 8 else torch.float16
gpu_gb = torch.cuda.get_device_properties(LOCAL_RANK if USE_DDP else 0).total_memory / 1e9
BATCH = 128 if gpu_gb > 70 else (64 if gpu_gb > 45 else 32)  # 568M — меньше batch per GPU
ACCUM = max(1, (TARGET_GLOBAL_BATCH + BATCH * WORLD_SIZE - 1) // (BATCH * WORLD_SIZE))
EFFECTIVE_GLOBAL_BATCH = BATCH * WORLD_SIZE * ACCUM
log(f"gpus={WORLD_SIZE} {torch.cuda.get_device_name(LOCAL_RANK if USE_DDP else 0)} {gpu_gb:.0f}GB | amp={AMP} batch={BATCH} accum={ACCUM} global_batch={EFFECTIVE_GLOBAL_BATCH} | {MODEL_NAME}")
log(f"logging: loss every {LOG_EVERY} steps | fast-val every {EVAL_EVERY} steps")

# данные: групповой сплит seed 13 (общекомандный), тексты v1
ml = pd.read_parquet(MATCHES_PATH)
parent = {}
def find(x):
    p = parent.setdefault(x, x)
    while p != parent[p]:
        parent[p] = parent[parent[p]]; p = parent[p]
    parent[p] = p; return p
for a, b in zip(ml.id1.values, ml.id2.values):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
comp = np.fromiter((find(i) for i in ml.id1.values), dtype=np.int64, count=len(ml))
rng = np.random.RandomState(13)
u = np.unique(comp)
vs = set(u[rng.rand(len(u)) < 0.03].tolist())
is_val = np.fromiter((c in vs for c in comp), dtype=bool, count=len(ml))
train = ml[~is_val].reset_index(drop=True)
holdout = ml[is_val].copy()
holdout = holdout[(holdout.target <= 0.2) | (holdout.target >= 0.8)]
holdout["target"] = (holdout.target >= 0.5).astype(np.int8)
del ml, parent, comp; gc.collect()
log(f"train {len(train):,} | holdout {len(holdout):,} (ожидание 10,950,394 / 191,555)")

KEYS = ["бренд", "артикул", "партномер", "oem", "код", "модель", "размер",
        "цвет", "объем", "обьем", "вес", "тип", "материал", "количество"]

# --- жёсткий препроцессинг: добавки, не ломающие v1-каркас -------------------
import re as _re
CYR2LAT = str.maketrans("аеорсухАЕОРСУХКМТВНЗЅІі", "aeopcyxAEOPCYXKMTBH3SIi")
LAT2CYR = str.maketrans("aeopcyxAEOPCYX", "аеорсухАЕОРСУХ")
def fix_homoglyphs(token):
    has_c = any('\u0400' <= ch <= '\u04ff' for ch in token)
    has_l = any(ch.isascii() and ch.isalpha() for ch in token)
    if not (has_c and has_l):
        return token
    n_c = sum('\u0400' <= ch <= '\u04ff' for ch in token)
    n_l = sum(ch.isascii() and ch.isalpha() for ch in token)
    return token.translate(CYR2LAT if n_l >= n_c else LAT2CYR)

_UNIT_RE = _re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мл|ml|л|l|мг|mg|г|g|гр|кг|kg|мм|mm|см|cm|м|мб|mb|гб|gb|тб|tb|вт|w|квт|kw|мач|mah)\b",
    _re.IGNORECASE)
_UNIT_MUL = {"мл":("ml",1),"ml":("ml",1),"л":("ml",1000),"l":("ml",1000),
             "мг":("g",0.001),"mg":("g",0.001),"г":("g",1),"g":("g",1),"гр":("g",1),
             "кг":("g",1000),"kg":("g",1000),"мм":("mm",1),"mm":("mm",1),
             "см":("mm",10),"cm":("mm",10),"м":("mm",1000),
             "мб":("gb",0.001),"mb":("gb",0.001),"гб":("gb",1),"gb":("gb",1),
             "тб":("gb",1024),"tb":("gb",1024),"вт":("w",1),"w":("w",1),
             "квт":("w",1000),"kw":("w",1000),"мач":("mah",1),"mah":("mah",1)}
def canon_units(text):
    out = set()
    for m in _UNIT_RE.finditer(text):
        unit, mul = _UNIT_MUL[m.group(2).lower()]
        v = float(m.group(1).replace(",", ".")) * mul
        out.add(f"{v:g}{unit}")
    return out

_QTY_RES = [_re.compile(r"(\d+)\s*шт"), _re.compile(r"набор\w*\s+из\s+(\d+)"),
            _re.compile(r"(\d+)\s*(?:набор|упаков|комплект)\w*\s+по\s+(\d+)"),
            _re.compile(r"[xх*](\d+)\b")]
def total_qty(text):
    t = text.lower()
    m = _QTY_RES[2].search(t)
    if m:
        return int(m.group(1)) * int(m.group(2))
    for r in (_QTY_RES[0], _QTY_RES[1], _QTY_RES[3]):
        m = r.search(t)
        if m:
            return int(m.group(1))
    return None
def build_text(name, attributes):
    parts = [str(name) if name is not None else ""]
    try: attrs = json.loads(attributes) if isinstance(attributes, str) else {}
    except Exception: attrs = {}
    if isinstance(attrs, dict) and attrs:
        low = {str(k).lower(): str(v) for k, v in attrs.items() if v}
        picked, used = [], set()
        for w in KEYS:
            for k, v in low.items():
                if w in k and k not in used:
                    picked.append(f"{k}:{v}"); used.add(k)
        rest = [f"{k}:{v}" for k, v in low.items() if k not in used]
        parts.append(" ; ".join(picked + rest)[:520])   # 520 симв под seq 320
    base = " | ".join(parts)
    base = base.replace("ё", "е").replace("Ё", "Е")
    base = " ".join(fix_homoglyphs(t) for t in base.split())
    base = _re.sub(r"[×хХ](?=\d)", "x", base)
    extras = []
    units = canon_units(base)
    if units:
        extras.append("ед: " + " ".join(sorted(units)[:12]))
    q = total_qty(base)
    if q and 1 < q <= 1000:
        extras.append(f"кол-во: {q}")
    return (base + (" | " + " | ".join(extras) if extras else ""))[:2000]

texts, cats = {}, {}
f = pq.ParquetFile(ITEMS_PATH)
for b in f.iter_batches(columns=["id", "name", "attributes", "category"], batch_size=500_000):
    df = b.to_pandas()
    for i, n, a, c in df.itertuples(index=False, name=None):
        texts[i] = build_text(n, a); cats[i] = c
holdout["category"] = [cats[i] for i in holdout.id1]

# сбалансированная fast-валидация: до 3000 пар на категорию
hf = holdout.sample(frac=1, random_state=0).groupby("category").head(3000)
log(f"товаров: {len(texts):,}; fast-val: {len(hf):,}")

class DS(Dataset):
    def __init__(self, df, training=False):
        self.a, self.b = df.id1.values, df.id2.values
        self.y = df.target.values.astype(np.float32)
        self.training = training
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        a, b = self.a[i], self.b[i]
        if self.training and random.random() < SWAP_P:
            a, b = b, a
        return texts[a], texts[b], self.y[i]

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
raw_model = model
if USE_DDP:
    model = DDP(model, device_ids=[LOCAL_RANK])

def collate(batch):
    enc = tok([x[0] for x in batch], [x[1] for x in batch], padding=True,
              truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc, torch.tensor([x[2] for x in batch])

@torch.no_grad()
def macro_on(df, bs=256):
    raw_model.eval(); preds = []
    dl = DataLoader(DS(df), batch_size=bs, num_workers=0, shuffle=False,
                    collate_fn=lambda b: tok([x[0] for x in b], [x[1] for x in b],
                        padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"))
    for enc in dl:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", AMP):
            preds.append(torch.sigmoid(raw_model(**enc).logits.squeeze(-1).float()).cpu().numpy())
    z = df[["category", "target"]].copy(); z["p"] = np.concatenate(preds)
    raw_model.train()
    return float(z.groupby("category").apply(lambda g: average_precision_score(g.target, g.p)).mean())

train_ds = DS(train, training=True)
sampler = DistributedSampler(train_ds, shuffle=True) if USE_DDP else None
dl = DataLoader(train_ds, batch_size=BATCH, shuffle=(sampler is None), sampler=sampler,
                num_workers=0, drop_last=True, collate_fn=collate)
steps_total = len(dl) * EPOCHS // ACCUM
opt = torch.optim.AdamW(raw_model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(steps_total * WARMUP_FRAC), steps_total)
scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)

best = 0.0
step = 0
t0 = time.time()
raw_model.train()
for ep in range(EPOCHS):
    if sampler is not None:
        sampler.set_epoch(ep)
    for bi, (enc, y) in enumerate(dl):
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", AMP):
            per = F.binary_cross_entropy_with_logits(
                model(**enc).logits.squeeze(-1), y, reduction="none")
            w = 0.75 + 0.25 * (2 * y - 1).abs()          # confidence weighting
            loss = (per * w).sum() / w.sum().clamp_min(1e-6) / ACCUM
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (bi + 1) % ACCUM == 0:
            if scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True); sched.step(); step += 1
            if IS_MAIN and step % LOG_EVERY == 0:
                sps = step * BATCH * ACCUM * WORLD_SIZE / (time.time() - t0)
                loss_value = loss.item() * ACCUM
                log(f"[train] ep={ep} step={step}/{steps_total} loss={loss_value:.6f} speed={sps:.0f} pair/s")
                record_history(
                    "train",
                    epoch=ep,
                    step=step,
                    steps_total=steps_total,
                    loss=loss_value,
                    pairs_per_second=sps,
                )
            if IS_MAIN and step % EVAL_EVERY == 0:
                mac = macro_on(hf)
                log(f"[eval] step={step}/{steps_total} fast_val_macro={mac:.6f} best_before={best:.6f}")
                record_history(
                    "eval",
                    epoch=ep,
                    step=step,
                    steps_total=steps_total,
                    fast_val_macro=mac,
                    best_before=best,
                )
                torch.save(raw_model.state_dict(), f"{OUT_DIR}/last.pt")
                if mac > best:
                    best = mac
                    torch.save(raw_model.state_dict(), f"{OUT_DIR}/best.pt")
if IS_MAIN:
    log("обучение завершено; best fast-val:", best)

# финал: лучший чекпойнт -> полный holdout -> экспорт
if USE_DDP:
    dist.barrier()
if IS_MAIN:
    sd = torch.load(f"{OUT_DIR}/best.pt", map_location="cpu", weights_only=True)
    raw_model.load_state_dict(sd)
    full = macro_on(holdout, bs=256)
    log(f"ПОЛНЫЙ llm-holdout macro PR-AUC = {full:.4f}")
    log("ориентиры: e5-small 0.786, e5-base 0.841; калибровка LB ~= 0.6 x holdout")
    raw_model.save_pretrained(f"{OUT_DIR}/export")
    tok.save_pretrained(f"{OUT_DIR}/export")
    json.dump({"model": MODEL_NAME, "max_len": MAX_LEN, "epochs": EPOCHS, "swap": SWAP_P,
               "confidence_weighting": True, "full_llm_holdout_macro": full,
               "best_fastval": best, "world_size": WORLD_SIZE},
              open(f"{OUT_DIR}/metrics.json", "w"), indent=1)
    log("экспорт:", f"{OUT_DIR}/export", "| metrics.json готов")
if USE_DDP:
    dist.destroy_process_group()
