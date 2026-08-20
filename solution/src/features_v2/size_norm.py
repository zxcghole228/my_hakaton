"""Фича: нормализация размеров. 4 семейства: обувь, одежда, кольца, габариты.

Правила из разбора ошибок:
- обувь: RU/EU 30..50 (половинки), US 4..15 -> сравнение в двух шкалах
  (US ~ EU - 33 для ж / -33.5 для м; допуск +-1 из-за неоднозначности систем);
- одежда: буквенные XS..5XL <-> RU 38..62; джинсы wNN/lNN;
- кольца: 13..23.5 (в т.ч. суффикс артикула вида "_17,5" и "р.17,5");
- габариты: числа с мм/см/м -> мм, сортировка осей, сравнение множеств
  с допуском 2% (231х153х73 см == 2310x1530x730 мм).

API: size_features(name_a, attrs_a, name_b, attrs_b) -> tuple[12 float]
(по 3 на семейство: both, equal, conflict). Только stdlib.
"""
from __future__ import annotations

import json
import re

__all__ = ["size_features", "FEATURE_NAMES"]

FEATURE_NAMES = tuple(
    f"{fam}_{s}"
    for fam in ("shoe", "cloth", "ring", "dims")
    for s in ("both", "equal", "conflict")
)

_LETTER = {"xxs": 38, "xs": 40, "s": 42, "m": 46, "l": 48, "xl": 50,
           "xxl": 52, "2xl": 52, "3xl": 56, "4xl": 58, "5xl": 60}

_SIZE_KEY = re.compile(r"размер|росс.*размер|размер производителя|size")
_RING_KEY = re.compile(r"размер")
_MULT = re.compile(r"[×хx]", re.IGNORECASE)

_RE_SHOE_NUM = re.compile(r"\b(\d{2}(?:[.,]5)?)\b")
_RE_US = re.compile(r"\b(\d{1,2}(?:[.,]5)?)\s*us\b")
_RE_JEANS = re.compile(r"\bw\s?(\d{2})\s*[/\\ ]?\s*l\s?(\d{2})\b")
_RE_LETTER = re.compile(r"\b(xxs|xs|s|m|l|xl|xxl|[2-5]xl)\b")
_RE_RING = re.compile(r"(?:р\.|размер[:\s]|_)(1[3-9](?:[.,]5)?|2[0-3](?:[.,]5)?)\b")
_RE_DIM = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мм|см|м)\b|"
    r"(\d+(?:[.,]\d+)?)\s*[хx×]\s*(\d+(?:[.,]\d+)?)(?:\s*[хx×]\s*(\d+(?:[.,]\d+)?))?\s*(мм|см|м)?",
    re.IGNORECASE)


def _f(x):
    return float(str(x).replace(",", "."))


def _attrs(attrs):
    if isinstance(attrs, dict):
        return attrs
    if isinstance(attrs, str):
        try:
            d = json.loads(attrs)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _texts(name, attrs):
    a = _attrs(attrs)
    low = {str(k).lower(): str(v).lower() for k, v in a.items() if v}
    name_l = str(name or "").lower().replace("ё", "е")
    return name_l, low


def extract_shoe(name, attrs):
    """Возвращает set нормализованных EU-размеров (может быть пуст)."""
    name_l, low = _texts(name, attrs)
    out = set()
    for k, v in low.items():
        if _SIZE_KEY.search(k) and "одежд" not in k:
            m = _RE_US.search(v)
            if m:
                out.add(_f(m.group(1)) + 33.5)
                continue
            for m in _RE_SHOE_NUM.finditer(v):
                n = _f(m.group(1))
                if 30 <= n <= 50:
                    out.add(n)
    m = _RE_US.search(name_l)
    if m:
        out.add(_f(m.group(1)) + 33.5)
    return out


def extract_cloth(name, attrs):
    """Set в шкале RU."""
    name_l, low = _texts(name, attrs)
    out = set()
    for k, v in low.items():
        if _SIZE_KEY.search(k):
            for m in _RE_LETTER.finditer(v):
                out.add(float(_LETTER[m.group(1)]))
            m = _RE_JEANS.search(v)
            if m:
                out.add(1000 + _f(m.group(1)) * 10 + _f(m.group(2)) / 100)
            for m in _RE_SHOE_NUM.finditer(v):
                n = _f(m.group(1))
                if 38 <= n <= 64 and n == int(n):
                    out.add(n)
    m = _RE_JEANS.search(name_l)
    if m:
        out.add(1000 + _f(m.group(1)) * 10 + _f(m.group(2)) / 100)
    return out


_RE_RING_BARE = re.compile(r"\b(1[3-9](?:[.,]5)?|2[0-3](?:[.,]5)?)\b")


def extract_ring(name, attrs):
    name_l, low = _texts(name, attrs)
    out = set()
    for m in _RE_RING.finditer(name_l):
        out.add(_f(m.group(1)))
    for k, v in low.items():
        if "артикул" in k:
            for m in _RE_RING.finditer(v):
                out.add(_f(m.group(1)))
        elif _RING_KEY.search(k):
            for m in _RE_RING_BARE.finditer(v):
                out.add(_f(m.group(1)))
    return out


def extract_dims(name, attrs):
    """Отсортированный tuple габаритов в мм (или пустой)."""
    name_l, low = _texts(name, attrs)
    dims = []
    for src in [name_l] + list(low.values()):
        for m in _RE_DIM.finditer(src):
            if m.group(1):
                unit = m.group(2).lower()
                k = {"мм": 1, "см": 10, "м": 1000}[unit]
                dims.append(_f(m.group(1)) * k)
            elif m.group(3) and m.group(4):
                unit = (m.group(6) or "см").lower()
                k = {"мм": 1, "см": 10, "м": 1000}[unit]
                for g in (3, 4, 5):
                    if m.group(g):
                        dims.append(_f(m.group(g)) * k)
        if dims:
            break
    big = sorted((d for d in dims if d >= 20), reverse=True)[:3]
    return tuple(big)


def _cmp_sets(a, b, tol=0.0):
    both = bool(a and b)
    if not both:
        return 1.0 if both else 0.0, 0.0, 0.0
    if tol:
        eq = any(abs(x - y) <= tol for x in a for y in b)
    else:
        eq = bool(a & b)
    return 1.0, float(eq), float(not eq)


def _cmp_dims(da, db):
    both = bool(da and db)
    if not both:
        return 0.0, 0.0, 0.0
    if len(da) != len(db):
        da, db = da[:min(len(da), len(db))], db[:min(len(da), len(db))]
    eq = all(abs(x - y) <= 0.02 * max(x, y) for x, y in zip(da, db))
    return 1.0, float(eq), float(not eq)


def size_features(name_a, attrs_a, name_b, attrs_b):
    sa, sb = extract_shoe(name_a, attrs_a), extract_shoe(name_b, attrs_b)
    ca, cb = extract_cloth(name_a, attrs_a), extract_cloth(name_b, attrs_b)
    ra, rb = extract_ring(name_a, attrs_a), extract_ring(name_b, attrs_b)
    da, db = extract_dims(name_a, attrs_a), extract_dims(name_b, attrs_b)
    out = []
    out += _cmp_sets(sa, sb, tol=1.0)
    out += _cmp_sets(ca, cb, tol=0.0)
    out += _cmp_sets(ra, rb, tol=0.0)
    out += _cmp_dims(da, db)
    return tuple(out)
