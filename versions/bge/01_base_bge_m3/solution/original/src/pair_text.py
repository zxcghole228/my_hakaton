"""Frozen pair_text v1 — exact pipeline used for v2 soup (LB 0.5522) and symmetry TTA train/submit."""
from __future__ import annotations
import json
import re

MAX_LEN = 320
ATTR_CHARS_LIMIT = 520
PAIR_TEXT_VERSION = "v1"
KEYS = ["бренд", "артикул", "партномер", "oem", "код", "модель", "размер",
        "цвет", "объем", "обьем", "вес", "тип", "материал", "количество"]

CYR2LAT = str.maketrans("аеорсухАЕОРСУХКМТВНЗЅІі", "aeopcyxAEOPCYXKMTBH3SIi")
LAT2CYR = str.maketrans("aeopcyxAEOPCYX", "аеорсухАЕОРСУХ")
_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мл|ml|л|l|мг|mg|г|g|гр|кг|kg|мм|mm|см|cm|м|мб|mb|гб|gb|тб|tb|вт|w|квт|kw|мач|mah)\b",
    re.IGNORECASE)
_UNIT_MUL = {"мл": ("ml", 1), "ml": ("ml", 1), "л": ("ml", 1000), "l": ("ml", 1000),
             "мг": ("g", 0.001), "mg": ("g", 0.001), "г": ("g", 1), "g": ("g", 1), "гр": ("g", 1),
             "кг": ("g", 1000), "kg": ("g", 1000), "мм": ("mm", 1), "mm": ("mm", 1),
             "см": ("mm", 10), "cm": ("mm", 10), "м": ("mm", 1000),
             "мб": ("gb", 0.001), "mb": ("gb", 0.001), "гб": ("gb", 1), "gb": ("gb", 1),
             "тб": ("gb", 1024), "tb": ("gb", 1024), "вт": ("w", 1), "w": ("w", 1),
             "квт": ("w", 1000), "kw": ("w", 1000), "мач": ("mah", 1), "mah": ("mah", 1)}
_QTY_RES = [re.compile(r"(\d+)\s*шт"), re.compile(r"набор\w*\s+из\s+(\d+)"),
            re.compile(r"(\d+)\s*(?:набор|упаков|комплект)\w*\s+по\s+(\d+)"),
            re.compile(r"[xх*](\d+)\b")]


def fix_homoglyphs(token):
    has_c = any("\u0400" <= ch <= "\u04ff" for ch in token)
    has_l = any(ch.isascii() and ch.isalpha() for ch in token)
    if not (has_c and has_l):
        return token
    n_c = sum("\u0400" <= ch <= "\u04ff" for ch in token)
    n_l = sum(ch.isascii() and ch.isalpha() for ch in token)
    return token.translate(CYR2LAT if n_l >= n_c else LAT2CYR)


def canon_units(text):
    out = set()
    for m in _UNIT_RE.finditer(text):
        unit, mul = _UNIT_MUL[m.group(2).lower()]
        v = float(m.group(1).replace(",", ".")) * mul
        out.add(f"{v:g}{unit}")
    return out


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

