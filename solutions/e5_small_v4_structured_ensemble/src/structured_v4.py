import ctypes
import math
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


SPACE_RE = re.compile(r"\s+")
MULTIPLY_RE = re.compile(r"[×хХ]")
TOKEN_RE = re.compile(r"[0-9a-zа-я]+", re.I)
NUMBER_RE = re.compile(r"(?<![a-zа-я])\d+(?:[.,]\d+)?(?![a-zа-я])", re.I)
CODE_RE = re.compile(
    r"\b(?=[0-9a-zа-я_-]{4,}\b)(?=[0-9a-zа-я_-]*\d)(?=[0-9a-zа-я_-]*[a-zа-я])[0-9a-zа-я_-]+\b",
    re.I,
)

ALIASES = {
    "brand": ["бренд", "brand", "производитель"],
    "model": ["модель", "model"],
    "article": [
        "артикул",
        "sku",
        "oem",
        "партномер",
        "part number",
        "partnumber",
        "код производителя",
    ],
    "size": [
        "размер производителя",
        "размер обуви",
        "размер одежды",
        "размер",
        "size",
        "рост",
    ],
    "color": ["основной цвет", "цвет", "color", "расцветка"],
    "material": ["материал верха", "материал", "material"],
    "gender": ["пол", "gender"],
}

# IMPORTANT: this reproduces the exact color dictionary used in the V4 training notebook.
COLORS = {
    "черный": "черный",
    "чёрный": "черный",
    "белый": "белый",
    "серый": "серый",
    "серебристый": "серебристый",
    "красный": "красный",
    "бордовый": "бордовый",
    "синий": "синий",
    "голубой": "голубой",
    "зеленый": "зеленый",
    "зелёный": "зеленый",
    "желтый": "желтый",
    "жёлтый": "желтый",
    "оранжевый": "оранжевый",
    "розовый": "розовый",
    "фиолетовый": "фиолетовый",
    "бежевый": "бежевый",
    "коричневый": "коричневый",
    "золотой": "золотой",
    "золотистый": "золотой",
}

UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мл|ml|л|l|мг|mg|г|g|кг|kg|мм|mm|см|cm|м|m|вт|w|квт|kw|шт|pcs)\b",
    re.I,
)
SIZE_NAME_RE = re.compile(
    r"(?:размер|р-р|size)\s*[:=\-]?\s*"
    r"([0-9]{2,3}(?:[./-][0-9]{1,3})?|xxxs|xxs|xs|s|m|l|xl|xxl|xxxl)",
    re.I,
)

# Tuple layout exactly mirrors V4 training notebook.
I_NAME, I_TOK, I_NUM, I_CODE, I_UNIT, I_BRAND, I_MODEL, I_ARTICLE, I_SIZE, I_COLOR, I_MATERIAL, I_GENDER, I_KEYS, I_VALTOK, I_ACOUNT = range(15)

BASE_FEATURES = [
    "name_exact",
    "name_contains",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_len_ratio",
    "name_len_diff",
    "token_jaccard",
    "token_containment",
    "token_common",
    "token_union",
    "number_jaccard",
    "number_common",
    "number_union",
    "number_conflict",
    "number_equal",
    "code_jaccard",
    "code_common",
    "code_union",
    "code_conflict",
    "code_equal",
    "unit_jaccard",
    "unit_common",
    "unit_union",
    "unit_conflict",
    "unit_equal",
    "brand_status",
    "brand_present",
    "model_status",
    "model_present",
    "article_status",
    "article_present",
    "size_status",
    "size_present",
    "color_status",
    "color_present",
    "material_status",
    "material_present",
    "gender_status",
    "gender_present",
    "attr_key_jaccard",
    "attr_key_common",
    "attr_value_jaccard",
    "attr_value_common",
    "attr_count_diff",
    "attr_count_ratio",
]

WORD_HASH_FEATURES = 2**16
CHAR_HASH_FEATURES = 2**18

WORD_HASH = HashingVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    n_features=WORD_HASH_FEATURES,
    alternate_sign=False,
    norm="l2",
    lowercase=False,
)
CHAR_HASH = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    n_features=CHAR_HASH_FEATURES,
    alternate_sign=False,
    norm="l2",
    lowercase=False,
)

N_FLOAT_FEATURES = len(BASE_FEATURES) + 2


def norm(value) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("ё", "е")
    text = MULTIPLY_RE.sub("x", text).replace(",", ".")
    return SPACE_RE.sub(" ", text).strip()


def attrs_dict(value) -> Dict[str, str]:
    import json

    if isinstance(value, dict):
        obj = value
    elif isinstance(value, str):
        try:
            obj = json.loads(value)
        except Exception:
            obj = {}
    else:
        obj = {}
    if not isinstance(obj, dict):
        return {}

    out = {}
    for key, item_value in obj.items():
        normalized_key = norm(key)
        normalized_value = norm(item_value)
        if normalized_key and normalized_value:
            out[normalized_key] = normalized_value
    return out


def pick(attrs: Dict[str, str], aliases: Sequence[str]) -> str:
    for alias in aliases:
        for key, value in attrs.items():
            if alias in key:
                return value
    return ""


def compact(value, limit: int = 80) -> str:
    text = norm(value)
    text = re.sub(r"[^0-9a-zа-я.+/_\- ]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()[:limit]


def code_norm(value) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", compact(value))


def size_norm(value) -> str:
    text = compact(value, 60).replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text


def color_norm(value) -> str:
    text = compact(value, 60)
    found = [canonical for raw, canonical in COLORS.items() if raw.replace("ё", "е") in text]
    return "/".join(sorted(set(found))) if found else text


def fallback_size(name) -> str:
    match = SIZE_NAME_RE.search(norm(name))
    return size_norm(match.group(1)) if match else ""


def fallback_color(name) -> str:
    text = norm(name)
    found = [canonical for raw, canonical in COLORS.items() if raw.replace("ё", "е") in text]
    return "/".join(sorted(set(found)))


def unit_tokens(text: str):
    out = set()
    for raw, unit in UNIT_RE.findall(text):
        try:
            value = float(raw.replace(",", "."))
        except Exception:
            continue

        unit = unit.lower()
        if unit in {"мл", "ml"}:
            kind, base = "volume_ml", value
        elif unit in {"л", "l"}:
            kind, base = "volume_ml", value * 1000
        elif unit in {"мг", "mg"}:
            kind, base = "mass_g", value / 1000
        elif unit in {"г", "g"}:
            kind, base = "mass_g", value
        elif unit in {"кг", "kg"}:
            kind, base = "mass_g", value * 1000
        elif unit in {"мм", "mm"}:
            kind, base = "length_mm", value
        elif unit in {"см", "cm"}:
            kind, base = "length_mm", value * 10
        elif unit in {"м", "m"}:
            kind, base = "length_mm", value * 1000
        elif unit in {"вт", "w"}:
            kind, base = "power_w", value
        elif unit in {"квт", "kw"}:
            kind, base = "power_w", value * 1000
        else:
            kind, base = "count", value
        out.add(f"{kind}:{round(base, 4)}")
    return frozenset(out)


def compact_item(name, attributes) -> Tuple:
    normalized_name = norm(name)
    attrs = attrs_dict(attributes)
    values_text = " ".join(attrs.values())
    search_text = (normalized_name + " " + values_text[:1200]).strip()

    tokens = frozenset(token for token in TOKEN_RE.findall(normalized_name) if len(token) >= 2)
    numbers = frozenset(value.replace(",", ".") for value in NUMBER_RE.findall(search_text))
    codes = frozenset(code_norm(value) for value in CODE_RE.findall(search_text) if code_norm(value))

    selected = {key: pick(attrs, aliases) for key, aliases in ALIASES.items()}
    size = size_norm(selected["size"]) or fallback_size(normalized_name)
    color = color_norm(selected["color"]) or fallback_color(normalized_name)

    return (
        normalized_name,
        tokens,
        numbers,
        codes,
        unit_tokens(search_text),
        compact(selected["brand"]),
        code_norm(selected["model"]),
        code_norm(selected["article"]),
        size,
        color,
        compact(selected["material"]),
        compact(selected["gender"]),
        frozenset(attrs.keys()),
        frozenset(token for token in TOKEN_RE.findall(values_text[:1600]) if len(token) >= 2),
        len(attrs),
    )


def _jac(first, second) -> float:
    union = len(first | second)
    return len(first & second) / union if union else 0.0


def _contain(first, second) -> float:
    return len(first & second) / max(1, min(len(first), len(second))) if first and second else 0.0


def _eq(first, second) -> float:
    return -1.0 if not first or not second else float(first == second)


def _present(first, second) -> float:
    return float(bool(first) and bool(second))


def _conflict(first, second) -> float:
    return float(bool(first) and bool(second) and not (first & second))


def _eqset(first, second) -> float:
    return float(bool(first) and bool(second) and first == second)


def _sim(first: str, second: str) -> Tuple[float, float, float]:
    # V4 training ran with rapidfuzz=False, so SequenceMatcher was used for all 3 fields.
    ratio = SequenceMatcher(None, first, second).ratio()
    return ratio, ratio, ratio


def _sparse_cos(first, second) -> np.ndarray:
    return np.asarray(first.multiply(second).sum(axis=1)).ravel().astype(np.float32)


def _base_row(first: Tuple, second: Tuple) -> np.ndarray:
    n1, n2 = first[I_NAME], second[I_NAME]
    t1, t2 = first[I_TOK], second[I_TOK]
    nu1, nu2 = first[I_NUM], second[I_NUM]
    c1, c2 = first[I_CODE], second[I_CODE]
    u1, u2 = first[I_UNIT], second[I_UNIT]
    k1, k2 = first[I_KEYS], second[I_KEYS]
    v1, v2 = first[I_VALTOK], second[I_VALTOK]
    ratio, token_sort, token_set = _sim(n1, n2)
    ac1, ac2 = first[I_ACOUNT], second[I_ACOUNT]

    values = [
        float(bool(n1) and n1 == n2),
        float(bool(n1) and bool(n2) and (n1 in n2 or n2 in n1)),
        ratio,
        token_sort,
        token_set,
        min(len(n1), len(n2)) / max(1, max(len(n1), len(n2))),
        abs(len(n1) - len(n2)),
        _jac(t1, t2),
        _contain(t1, t2),
        len(t1 & t2),
        len(t1 | t2),
        _jac(nu1, nu2),
        len(nu1 & nu2),
        len(nu1 | nu2),
        _conflict(nu1, nu2),
        _eqset(nu1, nu2),
        _jac(c1, c2),
        len(c1 & c2),
        len(c1 | c2),
        _conflict(c1, c2),
        _eqset(c1, c2),
        _jac(u1, u2),
        len(u1 & u2),
        len(u1 | u2),
        _conflict(u1, u2),
        _eqset(u1, u2),
        _eq(first[I_BRAND], second[I_BRAND]),
        _present(first[I_BRAND], second[I_BRAND]),
        _eq(first[I_MODEL], second[I_MODEL]),
        _present(first[I_MODEL], second[I_MODEL]),
        _eq(first[I_ARTICLE], second[I_ARTICLE]),
        _present(first[I_ARTICLE], second[I_ARTICLE]),
        _eq(first[I_SIZE], second[I_SIZE]),
        _present(first[I_SIZE], second[I_SIZE]),
        _eq(first[I_COLOR], second[I_COLOR]),
        _present(first[I_COLOR], second[I_COLOR]),
        _eq(first[I_MATERIAL], second[I_MATERIAL]),
        _present(first[I_MATERIAL], second[I_MATERIAL]),
        _eq(first[I_GENDER], second[I_GENDER]),
        _present(first[I_GENDER], second[I_GENDER]),
        _jac(k1, k2),
        len(k1 & k2),
        _jac(v1, v2),
        len(v1 & v2),
        abs(ac1 - ac2),
        min(ac1, ac2) / max(1, max(ac1, ac2)),
    ]
    return np.asarray(values, dtype=np.float32)


class StructuredModel:
    def __init__(self, library_path: Path, n_threads: int | None = None):
        self.library_path = Path(library_path)
        if not self.library_path.is_file():
            raise FileNotFoundError(f"Structured V4 model library not found: {self.library_path}")
        self.library = ctypes.CDLL(str(self.library_path))
        self.predict_fn = self.library.predict_batch_v4
        self.predict_fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_uint,
        ]
        self.predict_fn.restype = None
        default_threads = min(os.cpu_count() or 1, 20)
        self.n_threads = int(n_threads or default_threads)

    def predict_raw(self, features: np.ndarray, categories: Sequence[str]) -> np.ndarray:
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != N_FLOAT_FEATURES:
            raise ValueError(
                f"Expected structured feature matrix (*, {N_FLOAT_FEATURES}), got {features.shape}"
            )
        n_rows = features.shape[0]
        if n_rows != len(categories):
            raise ValueError("Feature/category row count mismatch")

        encoded_categories = [str(category).encode("utf-8") for category in categories]
        category_array = (ctypes.c_char_p * n_rows)(*encoded_categories)
        output = np.empty(n_rows, dtype=np.float64)

        self.predict_fn(
            features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            category_array,
            ctypes.c_ulonglong(n_rows),
            ctypes.c_ulonglong(features.shape[1]),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_uint(self.n_threads),
        )
        return output

    def predict_proba(self, features: np.ndarray, categories: Sequence[str]) -> np.ndarray:
        raw = self.predict_raw(features, categories)
        # Stable sigmoid.
        positive = raw >= 0
        result = np.empty_like(raw)
        result[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
        exp_raw = np.exp(raw[~positive])
        result[~positive] = exp_raw / (1.0 + exp_raw)
        return result


def predict_structured_pairs(
    id1: np.ndarray,
    id2: np.ndarray,
    item_features,
    library_path: Path,
    chunk_size: int = 25_000,
) -> np.ndarray:
    if len(id1) != len(id2):
        raise ValueError("id1/id2 row count mismatch")

    n_rows = len(id1)
    output = np.empty(n_rows, dtype=np.float64)
    model = StructuredModel(library_path)
    started = time.perf_counter()

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        size = stop - start
        matrix = np.empty((size, N_FLOAT_FEATURES), dtype=np.float32)
        titles1 = []
        titles2 = []
        categories = []

        for local_index, global_index in enumerate(range(start, stop)):
            first_item = item_features[id1[global_index]]
            second_item = item_features[id2[global_index]]
            first = first_item.structured
            second = second_item.structured

            matrix[local_index, : len(BASE_FEATURES)] = _base_row(first, second)
            titles1.append(first[I_NAME])
            titles2.append(second[I_NAME])
            categories.append(str(first_item.category))

        word1 = WORD_HASH.transform(titles1)
        word2 = WORD_HASH.transform(titles2)
        char1 = CHAR_HASH.transform(titles1)
        char2 = CHAR_HASH.transform(titles2)
        matrix[:, len(BASE_FEATURES)] = _sparse_cos(word1, word2)
        matrix[:, len(BASE_FEATURES) + 1] = _sparse_cos(char1, char2)

        output[start:stop] = model.predict_proba(matrix, categories)

        done = stop
        elapsed = time.perf_counter() - started
        print(
            f"[predict:structured] {done:,}/{n_rows:,} pairs "
            f"({done / max(elapsed, 1e-9):.1f} pair/s)",
            flush=True,
        )

    return output
