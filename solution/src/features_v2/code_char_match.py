"""Посимвольное сравнение кодов/артикулов для матчинга товаров.

Идея: коды, отличающиеся суффиксом или 1-2 символами, — это РАЗНЫЕ модификации
(casio gd-100gb-1e vs gbd-100-1e, clt-k506l vs clt-k506s, a21-2 vs a21-5,
574 vs 5740, cs-ph3140x vs cs-ph3140, 28132 vs 28132np).
Но усечение / префикс-код продавца — тот же товар (oas013 vs oas0136013,
"4060" внутри "rtx4060ti").

Функция: code_pair_features(name_a, attrs_a, name_b, attrs_b) -> tuple из 9 float:

  0 both_have         оба товара содержат хотя бы один код
  1 exact_alpha       есть точное совпадение кода с буквами (сильный сигнал "тот же")
  2 exact_digit       есть точное совпадение чисто цифрового кода (слабый сигнал)
  3 prefix_short_ext  один код — префикс другого, довесок 1-2 символа
                      (разные модификации: 574/5740, cs-ph3140/cs-ph3140x)
  4 prefix_long_ext   один код — префикс другого, довесок >=3 символов
                      (усечение продавцом: oas013/oas0136013 — тот же товар)
  5 contained         короткий код (>=4 симв., >=40% длины) — подстрока длинного
                      не по префиксу ("4060" в "rtx4060ti" — тот же товар)
  6 near_miss         левенштейн 1-2 между буквенно-цифровыми кодами,
                      не префикс-пара (разные модификации: clt-k506l/clt-k506s)
  7 best_sim          максимум (1 - lev/maxlen) по всем парам кодов
  8 best_prefix_frac  максимум (длина общего префикса / maxlen) по всем парам

Только stdlib (re, json). Оптимизировано под миллионы вызовов:
предкомпилированные регэкспы, ранние выходы, короткие циклы.
"""

from __future__ import annotations

import json
import re

__all__ = ["code_pair_features", "extract_codes"]

# --- единицы измерения: "5200 mah", "10.8v", "128 гб", "6000стр" — не коды ---
_UNITS = [
    "mah", "ма·ч", "ма•ч", "мач", "wh", "втч", "квт", "kw", "ватт", "вт", "w",
    "ггц", "ghz", "мгц", "mhz", "кгц", "khz", "гц", "hz",
    "тб", "tb", "гб", "gb", "мб", "mb", "кб", "kb", "бит", "bit",
    "мм", "mm", "см", "cm", "км", "km", "дм", "m", "м",
    "кг", "kg", "мг", "mg", "гр", "г", "g",
    "мл", "ml", "л", "l",
    "шт", "штук", "страниц", "стр", "месяцев", "мес", "лет", "года", "год",
    "дюймов", "дюйма", "дюйм", "inch", "унц", "oz", "lb", "фунт",
    "бар", "bar", "атм", "atm", "ом", "ohm", "мкгн", "мкф", "uf", "uh",
    "ач", "ah", "в", "v", "а", "a", "к", "k",
]
_UNITS.sort(key=len, reverse=True)
_U = "(?:" + "|".join(_UNITS) + ")"
# диапазон с единицей: "10.8-11.1v", "46-52 см"
_RANGE_RE = re.compile(
    r"(?<![\w-])\d+(?:[.,]\d+)?\s*[-–—]\s*\d+(?:[.,]\d+)?\s*" + _U + r"(?![a-zа-яё0-9])"
)
# число с единицей: "5200 mah", "3в", "6000стр"; guard (?<![\w-]) не даёт
# отъедать хвосты кодов вида "gd-100gb-1e"
_SINGLE_RE = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)?\s*" + _U + r"(?![a-zа-яё0-9])")

# токен: буквенно-цифровые куски, склеенные только дефисом
# ('/' и '.' — разделители: списки совместимости "3140/3155/3160" не склеиваем)
_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*")

_DIGIT_RE = re.compile(r"\d")
_ALPHA_RE = re.compile(r"[a-z]")
_CYR_RE = re.compile(r"[а-яё]")
_DIM_RE = re.compile(r"^\d+(?:[хx]\d+)+[а-яёa-z]*$")  # габариты "366х76см"
_PURE_DIGIT_RE = re.compile(r"^\d+$")
_UNIT_RESIDUE_RE = re.compile(r"^\d+(?:[.,]\d+)?" + _U + r"$")

# ключи атрибутов, где живут артикулы/коды моделей
_ATTR_KEY_SUBSTR = ("артикул", "модел", "партномер", "part number", "парт-номер")

_MAX_CODES = 10
_ZEROS = (0.0,) * 9


def _codes_from_text(text, out):
    """Добавляет нормализованные коды из строки в dict out (код -> has_alpha)."""
    t = text.lower()
    t = _RANGE_RE.sub(" ", t)
    t = _SINGLE_RE.sub(" ", t)
    for m in _TOKEN_RE.finditer(t):
        tok = m.group()
        norm = tok.replace("-", "")
        n = len(norm)
        if n < 3 or n > 24:
            continue
        if norm in out:
            continue
        if not _DIGIT_RE.search(norm):
            continue
        if _DIM_RE.match(norm):
            continue
        if _UNIT_RESIDUE_RE.match(norm):
            continue
        has_alpha = bool(_ALPHA_RE.search(norm))
        if _CYR_RE.search(norm):
            # кириллица в коде — только если цифр >= 2 (отсеиваем "уп2", "тип1")
            if sum(c.isdigit() for c in norm) < 2:
                continue
            has_alpha = True
        elif not has_alpha:
            # чисто цифровой код: 3..9 знаков (артикулы, 574, 28132, 587706421)
            if n > 9:
                continue
        out[norm] = has_alpha


def extract_codes(name, attrs):
    """Коды из названия + значений атрибутов с ключами артикул/модель/партномер."""
    out = {}
    if name:
        _codes_from_text(name, out)
    if attrs:
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except (ValueError, TypeError):
                attrs = None
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if not isinstance(v, str):
                    continue
                kl = k.lower()
                for s in _ATTR_KEY_SUBSTR:
                    if s in kl:
                        _codes_from_text(v, out)
                        break
    if len(out) > _MAX_CODES:
        keep = sorted(out, key=len, reverse=True)[:_MAX_CODES]
        out = {k: out[k] for k in keep}
    return out


def _lev(a, b):
    """Расстояние Левенштейна; строки короткие (<=24)."""
    la, lb = len(a), len(b)
    if la > lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        bj = b[j - 1]
        cur = [j]
        append = cur.append
        for i in range(1, la + 1):
            if a[i - 1] == bj:
                append(prev[i - 1])
            else:
                x = prev[i]
                y = cur[i - 1]
                z = prev[i - 1]
                if y < x:
                    x = y
                if z < x:
                    x = z
                append(x + 1)
        prev = cur
    return prev[la]


def code_pair_features(name_a, attrs_a, name_b, attrs_b):
    """Фичи посимвольного сравнения кодов для пары товаров. См. докстринг модуля."""
    ca = extract_codes(name_a, attrs_a)
    if not ca:
        return _ZEROS
    cb = extract_codes(name_b, attrs_b)
    if not cb:
        return _ZEROS

    exact_alpha = 0.0
    exact_digit = 0.0
    prefix_short = 0.0
    prefix_long = 0.0
    contained = 0.0
    near_miss = 0.0
    best_sim = 0.0
    best_pf = 0.0

    for a, a_alpha in ca.items():
        la = len(a)
        for b, b_alpha in cb.items():
            if a == b:
                if a_alpha:
                    exact_alpha = 1.0
                else:
                    exact_digit = 1.0
                best_sim = 1.0
                best_pf = 1.0
                continue
            lb = len(b)
            mx = la if la > lb else lb
            if a.startswith(b) or b.startswith(a):
                ext = la - lb if la > lb else lb - la
                if ext <= 2:
                    prefix_short = 1.0
                else:
                    prefix_long = 1.0
                sim = 1.0 - ext / mx
                pf = (mx - ext) / mx
            else:
                if la < lb:
                    s, g = a, b
                else:
                    s, g = b, a
                if len(s) >= 4 and len(s) * 2.5 >= len(g) and s in g:
                    contained = 1.0
                d = _lev(a, b)
                if d <= 2 and a_alpha and b_alpha:
                    near_miss = 1.0
                sim = 1.0 - d / mx
                # общий префикс
                cp = 0
                lim = la if la < lb else lb
                while cp < lim and a[cp] == b[cp]:
                    cp += 1
                pf = cp / mx
            if sim > best_sim:
                best_sim = sim
            if pf > best_pf:
                best_pf = pf

    return (
        1.0,
        exact_alpha,
        exact_digit,
        prefix_short,
        prefix_long,
        contained,
        near_miss,
        best_sim,
        best_pf,
    )
