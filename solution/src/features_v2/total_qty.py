"""Фича: суммарное количество единиц товара в паре (total quantity).

Парсит из имени и JSON-атрибутов полное количество единиц:
  "10 шт", "2 набора по 10 штук", "5 пар", "упаковка 100 шт",
  "12шт/уп ... 12 уп", "комплект из 3", "3 ед. товара",
  атрибуты "количество заводских упаковок", "количество в упаковке",
  "единиц в одном товаре", "число предметов", "комплектация" и т.п.

total = per_pack * packs. Числа-«не количества» (24 цвета, 20 листов,
40 мл, 0.7 мм, № 7, 12л) игнорируются.

Публичное API:
    extract_total_qty(name, attrs) -> (total: int, found: bool)
    quantity_features(name_a, attrs_a, name_b, attrs_b) -> tuple[8 float]

Только stdlib (re, json). Python 3.12.
"""
from __future__ import annotations

import json
import re

__all__ = ["extract_total_qty", "quantity_features", "FEATURE_NAMES"]

FEATURE_NAMES = (
    "qty_a",        # суммарное кол-во у A (капировано 1000)
    "qty_b",        # суммарное кол-во у B (капировано 1000)
    "qty_found_a",  # 1.0 если у A найден явный сигнал количества
    "qty_found_b",  # 1.0 если у B найден явный сигнал количества
    "qty_equal",    # 1.0 если qty_a == qty_b
    "qty_ratio",    # max/min, капировано 10.0 (1.0 при равенстве)
    "qty_multiple", # 1.0 если не равны, но большее кратно меньшему
    "qty_conflict", # 1.0 если qty_a != qty_b
)

_QTY_CAP = 1000

# --- регэкспы по имени / тексту ---------------------------------------------
# "N шт", "N штук", "6шт.", "12шт/уп", "10 штук в упаковке"
_RE_N_SHT = re.compile(r"(\d+)\s*шт(?:ук\w*|\.|\b)")
# "5 пар" (не матчится на "парковка": после "пар" требуется граница слова)
_RE_N_PAR = re.compile(r"(\d+)\s*пар(?:ы|а)?\b")
# "3 ед. товара", "3 единицы"
_RE_N_ED = re.compile(r"(\d+)\s*ед(?:иниц\w*|\.|\b)")
# "комплект из 3", "комплект 3" (число ПОСЛЕ слова)
_RE_KOMPLEKT_IZ = re.compile(r"комплект(?:\s+из)?\s+(\d+)\b")
# "2 набора", "12 уп", "3 упаковки", "12 комплектов" (число ПЕРЕД словом)
_RE_N_PACK = re.compile(
    r"(\d+)\s*(?:набор(?:а|ов)?\b|наб\.|упаковк\w*|уп\.|уп\b|комплект(?:а|ов)?\b)"
)
# "- упаковка 2" (без "шт" после числа, иначе это per-pack: "упаковка 100 шт")
_RE_PACK_N = re.compile(r"упаковк[аи]\s*[-—:]?\s*(\d+)\b(?!\s*шт)")

_RE_FIRST_INT = re.compile(r"\d+")

# --- ключи атрибутов ----------------------------------------------------------
# ключ количества: содержит один из этих маркеров...
_KEY_QTY_MARKERS = ("колич", "число", "единиц")
# ...но не должен содержать ни один из этих (это не количество единиц товара)
_KEY_BLOCK = (
    "цвет", "лист", "отделен", "секц", "разряд", "порци", "карман",
    "фломастер", "карандаш", "ручек", "кист", "слоев", "слоёв", "отверст",
)
# ключ означает число упаковок (иначе — штук в упаковке / в товаре)
_KEY_PACK_MARKERS = ("заводск", "упаковок", "наборов")
_VAL_BLOCK = ("лист", "цвет")


def _parse_text(text: str) -> tuple[int | None, int | None]:
    """Возвращает (per, packs) из свободного текста (имя или комплектация)."""
    per = None
    m = _RE_N_SHT.search(text)
    if m is None:
        m = _RE_N_PAR.search(text)
    if m is None:
        m = _RE_N_ED.search(text)
    if m is not None:
        v = int(m.group(1))
        if 0 < v <= 100000:
            per = v
    if per is None:
        m = _RE_KOMPLEKT_IZ.search(text)
        if m is not None:
            v = int(m.group(1))
            if 0 < v <= 100000:
                per = v

    packs = None
    m = _RE_N_PACK.search(text)
    if m is None:
        m = _RE_PACK_N.search(text)
    if m is not None:
        v = int(m.group(1))
        if 0 < v <= 10000:
            packs = v
    return per, packs


def _parse_attrs(attrs) -> tuple[int | None, int | None, int | None, int | None]:
    """Возвращает (per, packs, komp_per, komp_packs) из атрибутов."""
    if attrs is None:
        return None, None, None, None
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except (ValueError, TypeError):
            return None, None, None, None
    if not isinstance(attrs, dict):
        return None, None, None, None

    per = packs = komp_per = komp_packs = None
    for k, v in attrs.items():
        k_low = str(k).lower()
        if k_low == "комплектация":
            if komp_per is None and komp_packs is None:
                komp_per, komp_packs = _parse_text(str(v).lower().replace("ё", "е"))
            continue
        has_marker = False
        for mk in _KEY_QTY_MARKERS:
            if mk in k_low:
                has_marker = True
                break
        if not has_marker:
            continue
        blocked = False
        for b in _KEY_BLOCK:
            if b in k_low:
                blocked = True
                break
        if blocked:
            continue
        v_str = str(v).lower()
        skip_val = False
        for b in _VAL_BLOCK:
            if b in v_str:
                skip_val = True
                break
        if skip_val:
            continue
        m = _RE_FIRST_INT.search(v_str)
        if m is None:
            continue
        num = int(m.group(0))
        if not 0 < num <= 100000:
            continue
        is_pack = False
        for pk in _KEY_PACK_MARKERS:
            if pk in k_low:
                is_pack = True
                break
        if is_pack:
            if packs is None:
                packs = num
        else:
            if per is None:
                per = num
    return per, packs, komp_per, komp_packs


def extract_total_qty(name, attrs=None) -> tuple[int, bool]:
    """Суммарное количество единиц товара: (total, found).

    total = per_pack * packs; по умолчанию 1, если сигналов нет (found=False).
    Приоритет: имя > числовые атрибуты > атрибут "комплектация".
    Дубль-информация (например "2 штуки" в имени и
    "количество заводских упаковок"=2) не перемножается.
    """
    text = str(name or "").lower().replace("ё", "е")
    name_per, name_packs = _parse_text(text)
    attr_per, attr_packs, komp_per, komp_packs = _parse_attrs(attrs)

    per = name_per
    if per is None:
        per = attr_per
    if per is None:
        per = komp_per

    packs = name_packs
    if packs is None:
        cand = attr_packs if attr_packs is not None else komp_packs
        # защита от двойного учёта одной и той же информации:
        # per взят не из пары "по N", а число упаковок совпадает с ним
        if cand is not None and not (per is not None and name_packs is None and cand == per):
            packs = cand

    found = per is not None or packs is not None
    total = (per if per is not None else 1) * (packs if packs is not None else 1)
    if total > _QTY_CAP:
        total = _QTY_CAP
    return total, found


def quantity_features(name_a, attrs_a, name_b, attrs_b) -> tuple:
    """8 фичей сравнения суммарных количеств для пары товаров.

    Возвращает tuple float: см. FEATURE_NAMES
    (qty_a, qty_b, qty_found_a, qty_found_b,
     qty_equal, qty_ratio, qty_multiple, qty_conflict).
    """
    qa, fa = extract_total_qty(name_a, attrs_a)
    qb, fb = extract_total_qty(name_b, attrs_b)
    if qa == qb:
        equal, conflict, multiple, ratio = 1.0, 0.0, 0.0, 1.0
    else:
        equal, conflict = 0.0, 1.0
        hi, lo = (qa, qb) if qa > qb else (qb, qa)
        ratio = hi / lo
        if ratio > 10.0:
            ratio = 10.0
        multiple = 1.0 if hi % lo == 0 else 0.0
    return (
        float(qa), float(qb), float(fa), float(fb),
        equal, ratio, multiple, conflict,
    )
