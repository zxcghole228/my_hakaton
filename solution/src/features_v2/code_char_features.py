# -*- coding: utf-8 -*-
"""Посимвольное сравнение кодов/артикулов для матчинга товаров.

Идея: коды, отличающиеся суффиксом или 1-2 символами, — это РАЗНЫЕ модификации
(casio gd-100gb-1e vs gbd-100-1e, clt-k506l vs clt-k506s, a21-2 vs a21-5,
574 vs 5740), а усечение/длинный префикс продавца — тот же товар
(oas013 vs oas0136013).

Главная функция:

    code_pair_features(name_a, attrs_a, name_b, attrs_b) -> tuple[float, ...]

Вход: name — название товара, attrs — JSON-строка атрибутов (может быть "" или None).
Выход: 8 чисел:

    0 has_both      : 1.0 если у ОБЕИХ сторон найден хотя бы один код
    1 exact         : 1.0 если есть пара полностью совпадающих кодов
    2 prefix_trunc  : 1.0 если один код — собственный префикс другого и
                      "хвост" >= 3 символов (усечение продавца -> тот же товар)
    3 prefix_short  : 1.0 если один код — собственный префикс другого и
                      хвост 1-2 символа (другая модификация: 574 vs 5740)
    4 lev12         : 1.0 если лучшая пара кодов на расстоянии Левенштейна 1-2
                      (другая модификация: clt-k506l vs clt-k506s)
    5 best_prefix_frac : max по парам кодов доля общего префикса
                      (len(общий префикс)/max(len)); 0.0 если кодов нет
    6 conflict      : 1.0 если коды есть с обеих сторон, но нет ни точного
                      совпадения, ни усечения (сильный сигнал "разные товары")
    7 jaccard       : жаккар множеств кодов двух сторон

Только stdlib (re). Все регэкспы предкомпилированы, функция рассчитана на
миллионы вызовов.
"""

import re

__all__ = ["code_pair_features", "extract_codes"]

# токен: латиница/цифры/кириллица + внутренние -._  (слэш, запятая, скобки — разделители)
_TOK = re.compile(r"[a-z0-9а-яё][a-z0-9а-яё\-._]*")
_HAS_DIGIT = re.compile(r"\d")
_HAS_CYR = re.compile(r"[а-яё]")

# чистое измерение с единицей: 850w, 128gb, 5200mah, 4k, 2.4ghz ...
_MEAS_ALWAYS = re.compile(
    r"^\d+(?:[.,]\d+)?"
    r"(?:v|w|ah|mah|wh|hz|khz|mhz|ghz|mm|cm|km|ml|mb|gb|tb|kb|kg|bar|atm|bit|pin|ppi|nm|mp|fps|k)$"
)
# число с десятичной точкой + короткая единица: 10.8v, 2.4a, 1.35v
_MEAS_DEC = re.compile(r"^\d+[.,]\d+[a-z]{1,4}$")
# диапазон: 10.8-11.1v, 4-16x44 (дропаем только если есть десятичная точка,
# 'x' или буквенный хвост — иначе это может быть код вида 587706-421)
_RANGE = re.compile(r"^\d+(?:[.,]\d+)?(?:[-–—x]\d+(?:[.,]\d+)?)+[a-z]{0,4}$")
# габариты: 100x100, 2x16gb, 43x9x9
_DIMS = re.compile(r"^\d+(?:[.,]\d+)?(?:[x×]\d+(?:[.,]\d+)?)+[a-z]{0,4}$")
# технические стандарты, а не артикулы: usb2.0, ip68, wr50, ddr5, bt5.0 ...
_BLACKLIST = re.compile(
    r"^(?:usb|hdmi|bt|wi?fi|ipx?|wr|ddr|dvb[tcs]?2?|mp|sata|pcie?|vesa|lte|ios|win|atx|sim|rj)\d{0,4}$"
    r"|^h26[45]$"
)
_PURE_INT = re.compile(r"^\d+$")
_SEP_TRANS = str.maketrans("", "", "-._")

# ключи атрибутов, где живут артикулы/модели
_ATTR_CODE = re.compile(
    r'"(?:модель|наименование модели|артикул производителя|артикул sku|артикул|партномер|модель/исполнение)"'
    r'\s*:\s*"([^"]{1,80})"'
)

_MAX_CODES = 8


def _classify(tok):
    """-> ('strong'|'weak'|None, norm)"""
    if len(tok) < 3 or not _HAS_DIGIT.search(tok):
        return None, None
    if _HAS_CYR.search(tok):
        return None, None
    norm = tok.translate(_SEP_TRANS)
    if len(norm) < 3 or len(norm) > 25:
        return None, None
    if _PURE_INT.match(norm):
        if _PURE_INT.match(tok):
            # чистое число: слабый кандидат (574, 650547), но не год и не мелочь
            if 3 <= len(tok) <= 9 and not (1900 <= int(tok) <= 2099):
                return "weak", norm
            return None, None
        # цифры с разделителями: 587706-421 — код; 10.8 — измерение
        if len(norm) >= 5 and "." not in tok and "," not in tok:
            return "strong", norm
        return None, None
    # есть латинская буква
    if _MEAS_ALWAYS.match(tok) or _MEAS_DEC.match(tok) or _DIMS.match(tok):
        return None, None
    m = _RANGE.match(tok)
    if m and ("." in tok or "," in tok or "x" in tok or tok[-1].isalpha()):
        return None, None
    if _BLACKLIST.match(norm):
        return None, None
    return "strong", norm


def extract_codes(name, attrs):
    """Извлекает нормализованные коды из названия и значений модельных атрибутов.

    -> (strong: list[str], weak: list[str]) — без дубликатов, в порядке появления.
    """
    strong, weak = {}, {}
    if name:
        for tok in _TOK.findall(name.lower()):
            tok = tok.strip("-._")
            kind, norm = _classify(tok)
            if kind == "strong":
                strong[norm] = None
            elif kind == "weak":
                weak[norm] = None
    if attrs:
        for m in _ATTR_CODE.finditer(attrs.lower()):
            for tok in _TOK.findall(m.group(1)):
                tok = tok.strip("-._")
                kind, norm = _classify(tok)
                if kind == "strong":
                    strong[norm] = None
                elif kind == "weak":
                    weak[norm] = None
    return list(strong)[:_MAX_CODES], list(weak)[:_MAX_CODES]


def _lev_le2(a, b):
    """Расстояние Левенштейна с потолком 2 (возвращает 3, если больше)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la > lb:
        a, b, la, lb = b, a, lb, la
    if lb - la > 2:
        return 3
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        bj = b[j - 1]
        cur = [j]
        mn = j
        ap = prev
        for i in range(1, la + 1):
            c = prev[i] + 1
            d = cur[i - 1] + 1
            if d < c:
                c = d
            d = ap[i - 1] + (a[i - 1] != bj)
            if d < c:
                c = d
            cur.append(c)
            if c < mn:
                mn = c
        if mn > 2:
            return 3
        prev = cur
    return prev[la] if prev[la] <= 2 else 3


def _common_prefix_len(a, b):
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def code_pair_features(name_a, attrs_a, name_b, attrs_b):
    """Фичи посимвольного сравнения кодов для пары товаров. См. докстринг модуля."""
    sa, wa = extract_codes(name_a, attrs_a)
    sb, wb = extract_codes(name_b, attrs_b)
    # слабые (чисто числовые) коды используем, только если нет сильных
    ca = sa if sa else wa
    cb = sb if sb else wb
    if not ca or not cb:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    exact = 0.0
    trunc = 0.0
    pshort = 0.0
    lev12 = 0.0
    best_frac = 0.0

    for a in ca:
        la = len(a)
        for b in cb:
            if a == b:
                exact = 1.0
                best_frac = 1.0
                continue
            lb = len(b)
            p = _common_prefix_len(a, b)
            mx = la if la > lb else lb
            frac = p / mx
            if frac > best_frac:
                best_frac = frac
            if p == la or p == lb:  # один — собственный префикс другого
                ext = mx - p
                if ext <= 2:
                    pshort = 1.0
                elif p >= 4:
                    trunc = 1.0
            elif abs(la - lb) <= 2 and la >= 4 and lb >= 4:
                if _lev_le2(a, b) <= 2:
                    lev12 = 1.0

    conflict = 0.0 if (exact or trunc) else 1.0

    seta = set(ca)
    setb = set(cb)
    jac = len(seta & setb) / len(seta | setb)

    return (1.0, exact, trunc, pshort, lev12, round(best_frac, 4), conflict, round(jac, 4))
