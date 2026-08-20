# -*- coding: utf-8 -*-
"""Фичи БРЕНДЫ-АЛИАСЫ для матчинга товаров.

brand_alias_features(name_a, attrs_a, name_b, attrs_b) -> (equal, conflict, missing)

- brand_equal_after_norm: 1.0 если бренды обеих сторон совпали после нормализации
  (транслит ru->en, словарь алиасов, срезание суббрендовых суффиксов, поиск бренда
  в названии другой стороны, если в атрибутах его нет).
- brand_conflict: 1.0 если оба бренда определены и различаются (и не суббренд один
  другого).
- brand_missing_any: 1.0 если хотя бы у одной стороны бренд не определён (в т.ч.
  заглушки "нет бренда", "не определен", "noname", "случай" и т.п. — трактуются
  как отсутствие, а НЕ как конфликт).

Только stdlib (re). Python 3.12. Рассчитано на миллионы вызовов:
всё предвычислено на уровне модуля, атрибуты можно передавать как JSON-строку
(бренд достаётся регэкспом без json.loads) или как dict.
"""
from __future__ import annotations

import re

__all__ = ["brand_alias_features", "canon_brand"]

# ---------------------------------------------------------------------------
# Нормализация текста
# ---------------------------------------------------------------------------

_NONWORD = re.compile(r"[^0-9a-zа-я]+")
_CYR = re.compile(r"[а-я]")

# ru -> en транслит (регулярные случаи: метта->metta, самсунг->samsung,
# красцветмет->krastsvetmet, луч->luch, гамма->gamma, хатбер->khatber->hatber)
_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})

# Заглушки: значение атрибута "бренд", означающее его отсутствие.
# Реальные примеры из файлов ошибок: "нет бренда", "не определен", "noname",
# "nobrand", "no name", "без бренда", "случай", "oem", пустая строка.
_STUBS = frozenset({
    "", "нет", "нет бренда", "без бренда", "бренд отсутствует",
    "не определен", "не определён", "не указан", "не указано", "нет данных",
    "неизвестный", "неизвестно", "отсутствует", "другой", "другие", "прочие",
    "прочее", "случай",
    "noname", "no name", "nobrand", "no brand", "non", "none", "no", "n a",
    "oem", "other", "others", "unknown", "generic", "china",
})

# Суббрендовые/линейковые хвосты: "adidas originals" ~ "adidas",
# "h&m kids" ~ "h&m", "adidas sportswear" ~ "adidas".
_GENERIC_TAIL = frozenset({
    "kids", "kid", "junior", "originals", "original", "official",
    "sportswear", "sport", "sports", "home", "collection", "professional",
    "shop", "store", "official store", "россия", "rus",
})

# Словарь алиасов: нормализованная поверхностная форма -> канон (без пробелов,
# латиница, после фолда kh->h). Собран по реальным парам из файлов ошибок
# (Мебель: метта~metta; Электроника: куасера~kyocera; Канцелярия:
# erich krause~erichkrause; Обувь: hoka one one~hoka, adidas sportswear~adidas;
# Одежда: h&m kids~h&m) + типовые транслит-написания.
_ALIAS_SRC = {
    # мебель / электроника
    "метта": "metta",
    "куасера": "kyocera", "киосера": "kyocera", "кайосера": "kyocera",
    "kyocera mita": "kyocera", "kyoceramita": "kyocera",
    "самсунг": "samsung", "сяоми": "xiaomi", "ксиоми": "xiaomi",
    "хуавей": "huawei", "хонор": "honor", "леново": "lenovo",
    "эппл": "apple", "апл": "apple",
    "кэнон": "canon", "канон": "canon",
    "hewlett packard": "hp", "хьюлет паккард": "hp",
    "эл джи": "lg", "элджи": "lg",
    # канцелярия
    "erich krause": "erichkrause", "эрих краузе": "erichkrause",
    "эрихкраузе": "erichkrause",
    "брауберг": "brauberg", "хатбер": "hatber",
    "фабер кастелл": "fabercastell", "faber castell": "fabercastell",
    # обувь / одежда / спорт
    "hoka one one": "hoka", "hokaoneone": "hoka", "хока": "hoka",
    "adidas originals": "adidas", "adidas sportswear": "adidas",
    "adidas kids": "adidas", "адидас": "adidas",
    "найк": "nike", "пума": "puma", "рибок": "reebok",
    "асикс": "asics", "конверс": "converse",
    "нью баланс": "newbalance", "new balance": "newbalance",
    "андер армор": "underarmour", "under armour": "underarmour",
    # ювелирка
    "красцветмет": "krastsvetmet", "соколов": "sokolov",
    # прочее
    "гринвей": "greenway", "лего": "lego", "джордан": "jordan",
    "луч": "luch", "гамма": "gamma", "гугл": "google",
}


def _norm(text: str) -> str:
    """лоуеркейс, ё->е, всё не-буквенно-цифровое -> пробел, схлопнуть."""
    return " ".join(_NONWORD.split(text.lower().replace("ё", "е")))


def _fold(latin: str) -> str:
    """Фолд латиницы для сравнения (х может писаться kh или h)."""
    return latin.replace("kh", "h") if "kh" in latin else latin


# Ключи алиасов доступны и в "пробельной", и в слитной форме.
_ALIAS: dict[str, str] = {}
for _k, _v in _ALIAS_SRC.items():
    _ALIAS[_k] = _v
    _ALIAS[_k.replace(" ", "")] = _v
del _k, _v


def canon_brand(raw: str) -> str:
    """Каноническая форма бренда: '' если это заглушка/пусто."""
    s = _norm(raw)
    if not s or s in _STUBS:
        return ""
    hit = _ALIAS.get(s)
    if hit is not None:
        return hit
    toks = s.split()
    while len(toks) > 1 and toks[-1] in _GENERIC_TAIL:
        toks.pop()
    d = "".join(toks)
    if len(d) < 2:
        d = s.replace(" ", "")
    hit = _ALIAS.get(d) or _ALIAS.get(" ".join(toks))
    if hit is not None:
        return hit
    if _CYR.search(d):
        d = _fold(d.translate(_TRANSLIT))
        return _ALIAS.get(d, d)
    return _fold(d)


# ---------------------------------------------------------------------------
# Извлечение бренда из атрибутов (JSON-строка или dict)
# ---------------------------------------------------------------------------

# Порядок ключей = приоритет.
_BRAND_KEYS = (
    "бренд в одежде и обуви",
    "бренд",
    "торговая марка",
    "brand",
)
_BRAND_RE = re.compile(
    r'"(?:бренд в одежде и обуви|бренд|торговая марка|brand)"\s*:\s*"([^"]*)"'
)


def _attr_brand(attrs) -> str:
    """Канонический бренд из атрибутов ('' если нет или заглушка)."""
    if attrs is None:
        return ""
    if isinstance(attrs, str):
        m = _BRAND_RE.search(attrs)
        return canon_brand(m.group(1)) if m else ""
    # dict-ветка
    for k in _BRAND_KEYS:
        v = attrs.get(k)
        if v:
            return canon_brand(v)
    return ""


# ---------------------------------------------------------------------------
# Поиск бренда в названии товара
# ---------------------------------------------------------------------------

def _in_name(brand: str, name: str) -> bool:
    """Есть ли канонический бренд `brand` в названии `name` (n-граммы 1..3)."""
    if not brand or not name:
        return False
    toks = _norm(name).split()
    if len(toks) > 24:
        toks = toks[:24]
    n = len(toks)
    blen = len(brand)
    for i in range(n):
        for j in (1, 2, 3):
            if i + j > n:
                break
            g = " ".join(toks[i:i + j])
            gd = g.replace(" ", "")
            # быстрый отсев: длина слитной n-граммы заведомо мала
            if len(gd) + 4 < blen and gd not in _ALIAS:
                continue
            c = canon_brand(g)
            if not c:
                continue
            if c == brand:
                return True
            if len(c) >= 3 and blen >= 3 and (
                c.startswith(brand) or brand.startswith(c)
            ):
                return True
    return False


# Лексикон известных брендов для детекции в названии, когда бренда нет в
# атрибутах ни у одной из сторон. Поверхностные формы собраны из файлов ошибок.
_LEXICON_SURFACES = sorted(
    {
        _norm(s)
        for s in (
        # алиасы + каноны
        *_ALIAS_SRC.keys(),
        *_ALIAS_SRC.values(),
        # бренды, встречавшиеся в названиях реальных пар-ошибок
        "metta", "метта", "kyocera",
        "erichkrause", "erich krause", "brauberg", "hatber", "calligrata",
        "attache", "cactus", "lomond", "colop", "centropen", "artfox",
        "faber castell", "lisik", "greenway",
        "mypads", "servicefull", "profiline", "sakura", "retech", "target",
        "hi black", "nv print", "g&g", "netac", "baseus", "iqzip", "huayu",
        "zeepdeep", "1st color", "element", "energizer",
        "nike", "adidas", "puma", "reebok", "asics", "converse", "jordan",
        "new balance", "hoka", "hoka one one", "under armour", "kappa",
        "hollywood hills", "el tempo", "enrico fantory", "t taccardi",
        "milana", "tendance", "kenka", "marko", "baden", "lowa", "geox",
        "h m", "kogankids", "candy kids", "indigo kids", "cool club",
        "луч", "гамма", "sokolov", "disney", "lego",
        )
    }
    - {""},
    key=len,
    reverse=True,
)


def _lex_pattern(surface: str) -> str:
    return re.escape(surface)


_LEXICON_RE = re.compile(
    r"(?<![0-9a-zа-я])(?:"
    + "|".join(_lex_pattern(s) for s in _LEXICON_SURFACES)
    + r")(?![0-9a-zа-я])"
)


def _lexicon_brand(name: str) -> str:
    """Самый ранний известный бренд в названии ('' если не найден)."""
    if not name:
        return ""
    m = _LEXICON_RE.search(_norm(name))
    return canon_brand(m.group(0)) if m else ""


# ---------------------------------------------------------------------------
# Основная фича
# ---------------------------------------------------------------------------

def brand_alias_features(name_a, attrs_a, name_b, attrs_b):
    """Фичи по брендам для пары товаров.

    name_*  : название товара (str)
    attrs_* : атрибуты — JSON-строка (можно обрезанную) или dict, или None

    Возвращает tuple из 3 float:
      (brand_equal_after_norm, brand_conflict, brand_missing_any)
    Ровно один из трёх равен 1.0.
    """
    ba = _attr_brand(attrs_a)
    bb = _attr_brand(attrs_b)
    if ba and not bb:
        bb = ba if _in_name(ba, name_b) else _lexicon_brand(name_b)
    elif bb and not ba:
        ba = bb if _in_name(bb, name_a) else _lexicon_brand(name_a)
    elif not ba and not bb:
        ba = _lexicon_brand(name_a)
        bb = _lexicon_brand(name_b)
        if ba and not bb and _in_name(ba, name_b):
            bb = ba
        elif bb and not ba and _in_name(bb, name_a):
            ba = bb
    if not ba or not bb:
        return (0.0, 0.0, 1.0)
    if ba == bb:
        return (1.0, 0.0, 0.0)
    # суббренд: один канон — префикс другого ("hokaoneone"/"hoka", "hmkids"/"hm")
    if len(ba) >= 3 and len(bb) >= 3 and (
        ba.startswith(bb) or bb.startswith(ba)
    ):
        return (1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0)
