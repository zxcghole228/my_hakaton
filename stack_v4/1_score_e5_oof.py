"""Стек v4, шаг 1 (GPU, Даниил): скоринг всех LLM-пар и manual-пар моделью e5-base@320.

OOF-схема против leakage (наша охота на баги поймала non-OOF в v1-v3):
train-пары модель ВИДЕЛА при обучении -> её скоры на них in-sample. Честный OOF
потребовал бы K переобучений (дорого). Компромисс, принятый командой: скорим
единой моделью, но в train стека добавляем колонку is_seen (пара была в
обучении e5) — CatBoost сам скорректирует сдвиг. Holdout-пары моделью не
виделись, метрики честные.

Запуск: python 1_score_e5_oof.py --model ./e5base_320_run/export \\
    --items ./items.parquet --matches-llm ./matches_llm.parquet \\
    --matches-manual ./matches.parquet --items-human ./items_human.parquet \\
    --out ./v4_scores
Время: ~40-60 мин на A100 (11.5M пар, seq 320, БЕЗ TTA — для фичей хватает
одного направления; хочешь TTA — флаг --tta, x2 время).
Выход: v4_scores/llm_chunk_XX.parquet (id1, id2, ce_score), manual_scores.parquet.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

KEYS = ["бренд", "артикул", "партномер", "oem", "код", "модель", "размер",
        "цвет", "объем", "обьем", "вес", "тип", "материал", "количество"]

# --- hardprep, побайтово как в обучении e5-base (final_e5base_320_hardprep) ---
import re as _re
CYR2LAT = str.maketrans("аеорсухАЕОРСУХКМТВНЗЅІі", "aeopcyxAEOPCYXKMTBH3SIi")
LAT2CYR = str.maketrans("aeopcyxAEOPCYX", "аеорсухАЕОРСУХ")
def fix_homoglyphs(token):
    has_c = any('Ѐ' <= ch <= 'ӿ' for ch in token)
    has_l = any(ch.isascii() and ch.isalpha() for ch in token)
    if not (has_c and has_l):
        return token
    n_c = sum('Ѐ' <= ch <= 'ӿ' for ch in token)
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
                    picked.append(f"{k}:{v}"); used.add(k)
        rest = [f"{k}:{v}" for k, v in low.items() if k not in used]
        parts.append(" ; ".join(picked + rest)[:520])
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


def score_pairs(model, tok, texts, id1, id2, device, max_len=320, bs=256, tta=False):
    n = len(id1)
    lengths = np.fromiter((len(texts.get(a, "")) + len(texts.get(b, ""))
                           for a, b in zip(id1, id2)), dtype=np.int64, count=n)
    order = np.argsort(lengths, kind="stable")
    preds = np.empty(n, dtype=np.float32)
    amp = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    with torch.inference_mode():
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            t1 = [texts.get(id1[i], "") for i in idx]
            t2 = [texts.get(id2[i], "") for i in idx]
            p_sum = None
            passes = ((t1, t2), (t2, t1)) if tta else ((t1, t2),)
            for a, b in passes:
                enc = tok(a, b, padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast("cuda", amp):
                    logits = model(**enc).logits.squeeze(-1)
                p = torch.sigmoid(logits.float()).cpu().numpy()
                p_sum = p if p_sum is None else p_sum + p
            preds[idx] = p_sum / len(passes)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--matches-llm", required=True)
    ap.add_argument("--matches-manual", required=True)
    ap.add_argument("--items-human", required=True)
    ap.add_argument("--out", default="./v4_scores")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    # manual
    t0 = time.time()
    ih = pd.read_parquet(args.items_human)
    texts_h = {i: build_text(n, a) for i, n, a in
               ih[["id", "name", "attributes"]].itertuples(index=False, name=None)}
    m = pd.read_parquet(args.matches_manual)
    m["ce_score"] = score_pairs(model, tok, texts_h, m.id1.values, m.id2.values,
                                device, tta=args.tta)
    m[["id1", "id2", "ce_score"]].to_parquet(f"{args.out}/manual_scores.parquet", index=False)
    print(f"manual: {len(m):,} за {(time.time()-t0)/60:.1f} мин", flush=True)
    del texts_h, ih

    # llm чанками с резюме
    ml = pd.read_parquet(args.matches_llm)
    CH = 1_000_000
    n_chunks = (len(ml) + CH - 1) // CH
    f = pq.ParquetFile(args.items)
    for ci in range(n_chunks):
        out_p = f"{args.out}/llm_chunk_{ci:02d}.parquet"
        if os.path.exists(out_p):
            continue
        t0 = time.time()
        part = ml.iloc[ci * CH:(ci + 1) * CH]
        need = set(part.id1) | set(part.id2)
        texts = {}
        for b in f.iter_batches(columns=["id", "name", "attributes"], batch_size=500_000):
            df = b.to_pandas()
            df = df[df["id"].isin(need)]
            for i, nn, aa in df.itertuples(index=False, name=None):
                texts[i] = build_text(nn, aa)
        sc = score_pairs(model, tok, texts, part.id1.values, part.id2.values,
                         device, tta=args.tta)
        pd.DataFrame({"id1": part.id1.values, "id2": part.id2.values,
                      "ce_score": sc}).to_parquet(out_p, index=False)
        print(f"чанк {ci:02d}/{n_chunks-1}: {len(part):,} за {(time.time()-t0)/60:.1f} мин",
              flush=True)
    print("готово:", args.out, flush=True)


if __name__ == "__main__":
    main()
