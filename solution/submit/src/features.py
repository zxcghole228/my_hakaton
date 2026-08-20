"""Фичи пар товаров для нашего решения (v1).

Отличия от примерного catboost-решения организаторов:
- нормализация кодов/артикулов со схлопыванием разделителей (stellox_8500892sx == 8500892sx);
- кросс-сравнение кодов из имени и из атрибутов (артикул/партномер/oem/код товара);
- rapidfuzz вместо difflib (быстрее и точнее token_set/token_sort);
- вариантные атрибуты (размер/цвет/объем/вес) со своими конфликт-фичами.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, FrozenSet, Mapping, Tuple

import numpy as np
from rapidfuzz import fuzz

SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^0-9a-zа-я]+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
TOKEN_RE = re.compile(r"[0-9a-zа-я]+")
CODE_CHUNK_RE = re.compile(r"[0-9a-zа-я][0-9a-zа-я\-_./\\]{3,}", re.IGNORECASE)
SEPARATORS_RE = re.compile(r"[\s\-_./\\,;:()\[\]#№*'\"+]+")

# ключи атрибутов, где живут коды
CODE_KEYS = ("артикул", "партномер", "oem", "код товара", "код продукта", "штрихкод",
             "isbn", "mpn", "sku", "модель")
BRAND_KEYS = ("бренд", "brand", "производитель", "торговая марка")
VARIANT_KEYS = {
    "size": ("размер", "size", "российский размер", "размер производителя", "диагональ"),
    "color": ("цвет", "color", "оттенок"),
    "volume": ("объем", "обьем", "volume", "емкость", "вместимость"),
    "weight": ("вес", "weight", "масса"),
    "model": ("модель", "model"),
    "type": ("тип", "type", "вид"),
    "material": ("материал", "material", "состав"),
    "count": ("количество", "штук", "шт в упаковке", "количество в упаковке"),
}

NUM_BASE = 26  # число скалярных фич до блоков
VARIANT_NAMES = tuple(VARIANT_KEYS)

FEATURE_NAMES: list[str] = []


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).lower().replace("ё", "е").replace(",", ".")
    x = NON_ALNUM_RE.sub(" ", x)
    return SPACE_RE.sub(" ", x).strip()


def squash_code(x: str) -> str:
    """Код без разделителей: 'stellox_8500892-sx' -> 'stellox8500892sx'."""
    return SEPARATORS_RE.sub("", str(x).lower().replace("ё", "е"))


def extract_codes(raw_text: str) -> FrozenSet[str]:
    """Схлопнутые коды (буквы+цифры, длина >=5) из сырого текста."""
    out = set()
    for chunk in CODE_CHUNK_RE.findall(str(raw_text).lower()):
        sq = squash_code(chunk)
        if len(sq) >= 5 and any(c.isdigit() for c in sq):
            out.add(sq)
            # чисто цифровое ядро тоже полезно: 'sx8500892' ~ '8500892'
            digits = "".join(c for c in sq if c.isdigit())
            if len(digits) >= 6:
                out.add(digits)
    return frozenset(out)


def safe_json_loads(x: Any) -> Dict[str, str]:
    if not isinstance(x, str):
        return {}
    try:
        obj = json.loads(x)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {str(k).lower().replace("ё", "е").strip(): str(v) for k, v in obj.items()}


def _first_by_keys(attrs: Mapping[str, str], keys: tuple[str, ...]) -> str:
    for want in keys:
        for k, v in attrs.items():
            if want in k and v:
                return v
    return ""


def jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def overlap_min(a, b) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class Item:
    __slots__ = ("norm_name", "tokens", "name_numbers", "name_codes", "attrs",
                 "attr_keys", "attr_codes", "attr_numbers", "brand", "variants",
                 "all_codes")

    def __init__(self, name: Any, attributes: Any):
        raw_name = "" if name is None else str(name)
        self.norm_name = normalize_text(raw_name)
        self.tokens = frozenset(self.norm_name.split())
        self.name_numbers = frozenset(m.replace(",", ".") for m in NUMBER_RE.findall(self.norm_name))
        self.name_codes = extract_codes(raw_name)

        attrs = safe_json_loads(attributes)
        self.attrs = {k: normalize_text(v) for k, v in attrs.items()}
        self.attr_keys = frozenset(k for k in self.attrs)
        code_texts = []
        for want in CODE_KEYS:
            for k, v in attrs.items():
                if want in k and v:
                    code_texts.append(v)
        self.attr_codes = frozenset().union(*(extract_codes(t) for t in code_texts)) if code_texts else frozenset()
        nums = set()
        for v in self.attrs.values():
            nums.update(m.replace(",", ".") for m in NUMBER_RE.findall(v))
        self.attr_numbers = frozenset(nums)
        self.brand = normalize_text(_first_by_keys(attrs, BRAND_KEYS))
        self.variants = tuple(normalize_text(_first_by_keys(attrs, VARIANT_KEYS[n])) for n in VARIANT_NAMES)
        self.all_codes = self.name_codes | self.attr_codes


def _pair_block(v1: str, v2: str) -> tuple[float, float, float, float]:
    both = bool(v1 and v2)
    eq = both and v1 == v2
    return (float(both), float(eq), float(both and not eq),
            fuzz.ratio(v1, v2) / 100.0 if both else 0.0)


def build_features(i1: Item, i2: Item) -> np.ndarray:
    n1, n2 = i1.norm_name, i2.norm_name
    t1, t2 = i1.tokens, i2.tokens

    codes_union_match = jaccard(i1.all_codes, i2.all_codes)
    codes_any_common = float(bool(i1.all_codes & i2.all_codes))
    cross_12 = float(bool(i1.name_codes & i2.attr_codes))
    cross_21 = float(bool(i2.name_codes & i1.attr_codes))

    common_keys = i1.attr_keys & i2.attr_keys
    eq_cnt = sum(1 for k in common_keys if i1.attrs[k] == i2.attrs[k])
    conflict_cnt = len(common_keys) - eq_cnt

    feats = [
        float(n1 == n2 and bool(n1)),
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        jaccard(t1, t2),
        overlap_min(t1, t2),
        float(len(t1)), float(len(t2)), float(abs(len(t1) - len(t2))),
        min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
        # числа из имени
        jaccard(i1.name_numbers, i2.name_numbers),
        float(bool(i1.name_numbers and i2.name_numbers and not (i1.name_numbers & i2.name_numbers))),
        float(len(i1.name_numbers & i2.name_numbers)),
        # коды
        codes_union_match,
        codes_any_common,
        float(bool(i1.all_codes) and bool(i2.all_codes) and not (i1.all_codes & i2.all_codes)),
        cross_12, cross_21,
        float(len(i1.all_codes)), float(len(i2.all_codes)),
        # атрибуты
        float(len(common_keys)),
        jaccard(i1.attr_keys, i2.attr_keys),
        eq_cnt / len(common_keys) if common_keys else 0.0,
        conflict_cnt / len(common_keys) if common_keys else 0.0,
        jaccard(i1.attr_numbers, i2.attr_numbers),
        float(len(i1.attr_numbers & i2.attr_numbers)),
    ]
    assert len(feats) == NUM_BASE

    feats.extend(_pair_block(i1.brand, i2.brand))
    for idx in range(len(VARIANT_NAMES)):
        feats.extend(_pair_block(i1.variants[idx], i2.variants[idx]))

    return np.asarray(feats, dtype=np.float32)


def _make_feature_names() -> list[str]:
    base = [
        "name_exact", "name_ratio", "name_token_sort", "name_token_set",
        "name_jaccard", "name_overlap_min", "tokens1", "tokens2", "tokens_diff",
        "name_len_ratio",
        "nums_jaccard", "nums_disjoint", "nums_common",
        "codes_jaccard", "codes_any_common", "codes_disjoint",
        "codes_cross_12", "codes_cross_21", "codes_cnt1", "codes_cnt2",
        "attr_common_keys", "attr_key_jaccard", "attr_equal_ratio",
        "attr_conflict_ratio", "attr_nums_jaccard", "attr_nums_common",
    ]
    for block in ("brand",) + VARIANT_NAMES:
        base.extend(f"{block}_{s}" for s in ("both", "equal", "conflict", "sim"))
    return base


FEATURE_NAMES = _make_feature_names()
MODEL_FEATURES = ["category"] + FEATURE_NAMES


def featurize_pair(name1, attrs1, name2, attrs2) -> np.ndarray:
    return build_features(Item(name1, attrs1), Item(name2, attrs2))
