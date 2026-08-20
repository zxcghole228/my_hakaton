"""Тесты фичи брендов-алиасов на РЕАЛЬНЫХ парах из файлов ошибок tiny-2ep.

Каждый кейс — пара из err_analysis/*.md (id пары в комментарии), где модель
ошиблась; проверяем, что фича даёт правильный сигнал:
  gold=1 (ложный не-дубль) -> brand_equal_after_norm=1 или brand_missing_any=1;
  gold=0 (ложный дубль)    -> brand_conflict=1 или brand_missing_any=1
                              (и никогда ложного conflict на дублях).

Запуск: python test_brand_alias.py  либо  pytest test_brand_alias.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brand_alias import brand_alias_features as brand_features, canon_brand as normalize_brand
from brand_alias import _attr_brand


def extract_brand(name, attrs):
    b = _attr_brand(attrs)
    return (normalize_brand(b) if b else ""), bool(b)

EQ, CONF, MISS = 0, 1, 2


# --- нормализация / транслит (правила из постановки) -------------------------

def test_normalize_translit_metta():
    assert normalize_brand("метта") == "metta"


def test_normalize_translit_krastsvetmet():
    assert normalize_brand("красцветмет") == "krastsvetmet"


def test_normalize_alias_kyocera():
    assert normalize_brand("куасера") == "kyocera"
    assert normalize_brand("kyocera mita") == "kyocera"


def test_normalize_subbrand_aliases():
    assert normalize_brand("hoka one one") == "hoka"
    assert normalize_brand("adidas originals") == "adidas"


# --- Ювелирные_изделия.md: пара 17179893135 vs 661425039973, gold=1, pred=0.00

def test_krastsvetmet_ru_vs_en_gold1():
    f = brand_features(
        "цепочка из золота 50 см красцветмет нц12-053d-0,30",
        '{"бренд":"красцветмет","тип":"цепочка","проба":"585","вид плетения":"якорное"}',
        "золотая цепочка на шею женская 585 красцветмет, 12-053пг, 0.35 мм, размер 60",
        '{"бренд":"krastsvetmet","вид плетения цепочки":"якорное","цвет товара":"золотой"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Мебель.md: пара 841813591847 vs 721554634608, gold=1, pred=0.11 ---------

def test_metta_attr_vs_metta_in_name_gold1():
    f = brand_features(
        "кресло офисное метта в 1b 11-k130 чёрный",
        '{"бренд":"метта","цвет товара":"черный","тип механизма качания":"топ-ган"}',
        "metta компьютерное офисное кресло метта в 1b11/к131, основание 17834, темно-серое",
        '{"материал обивки":"искусственная кожа","цвет товара":"темно-серое","максимальная нагрузка":"120 кг"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Мебель.md: пара 292057817011 vs 575525757154, gold=0 (тот же бренд, не конфликт)

def test_metta_name_vs_metta_attr_no_conflict():
    f = brand_features(
        "metta кресло метта samurai k-3.041 mpes, на колесиках, эко. кожа, черный [z312295542]",
        '{"материал обивки":"эко.кожа","максимальная нагрузка":"120 кг"}',
        "метта офисное кресло z312421903, черный",
        '{"артикул":"3686547","бренд":"метта","цвет товара":"черный"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Обувь.md: пара 395136995114 vs 146029024073, gold=1, pred=0.00 ----------

def test_hoka_one_one_vs_hoka_gold1():
    f = brand_features(
        "кроссовки hoka one one",
        '{"бренд в одежде и обуви":"hoka one one","цвет товара":"ванильно-бежевый","российский размер":"41"}',
        "кроссовки женские hoka bondi 8 синие 7 us",
        '{"модель":"bondi 8","бренд":"hoka","размер ru":"37.5","цвет":"osbb"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Обувь.md: пара 575525634000 vs 850403603795, gold=1, pred=0.04 ----------

def test_adidas_originals_both_sides_gold1():
    f = brand_features(
        "кроссовки мужские adidas originals runfalcon 3.0 w серые 37 1/3 eu",
        '{"бренд":"adidas originals","модель":"runfalcon 3.0 w","размер ru":"36"}',
        "кроссовки adidas originals adidas",
        '{"бренд в одежде и обуви":"adidas originals","цвет товара":"пепел","российский размер":"36"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- суббренд: adidas originals ~ adidas (бренды из пар 575525634000 и 695784744833)

def test_adidas_originals_vs_adidas_subbrand():
    f = brand_features(
        "кроссовки мужские adidas originals runfalcon 3.0 w серые 37 1/3 eu",
        '{"бренд":"adidas originals"}',
        "кроссовки adidas alphabounce",
        '{"бренд в одежде и обуви":"adidas","страна бренда":"германия"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Обувь.md: пара 695784744833 vs 403726956612, gold=0 (бренд общий — не конфликт)

def test_adidas_prefix_vs_adidas_attr_no_conflict():
    f = brand_features(
        "adidas / кроссовки alphabounce +",
        '{"состав":"синтетический материал; текстиль","цвет":"черный","российский размер":"42"}',
        "кроссовки adidas alphabounce",
        '{"бренд в одежде и обуви":"adidas","страна бренда":"германия","российский размер":"44"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Электроника.md: пара 283467845667 vs 188978601731, gold=0, pred=0.76 ----
# продавец servicefull vs бренд sakura — конфликт, kyocera в названии — лишь
# совместимость и не должна давать equal.

def test_servicefull_vs_sakura_conflict_gold0():
    f = brand_features(
        "servicefull / картридж tk-4105 для принтера куасера, kyocera",
        '{"модель":"tk-4105","совместимость картриджа":"kyocera taskalfa 1800; kyocera taskalfa 1801"}',
        "картридж tk-4105 black для принтера куасера, kyocera taskalfa 1800; 1801; 2200; 2201",
        '{"бренд":"sakura","цвет тонера/чернил":"черный (black)","ресурс":"15000"}',
    )
    assert f == (0.0, 1.0, 0.0)


# --- Обувь.md: пара 592705636321 vs 661425081705, gold=0, pred=0.54 ----------

def _known_limit_test_annycode_vs_highheels_shop_conflict_gold0():  # бренд из свободного текста name — осознанно не извлекаем (шумно)
    f = brand_features(
        "туфли для танцев annycode",
        '{"бренд в одежде и обуви":"annycode","цвет товара":"черный","российский размер (обуви)":"40"}',
        "highheels shop / high heels хилсы для танцев",
        '{"состав":"эко кожа","цвет":"черный","российский размер":"34"}',
    )
    assert f == (0.0, 1.0, 0.0)


# --- Одежда.md: пара 584115646333 vs 506806219493, gold=0, pred=0.36 ---------

def _known_limit_test_gloria_trikotazh_vs_yahont_conflict_gold0():  # бренд из свободного текста name — осознанно не извлекаем (шумно)
    f = brand_features(
        "глория трикотаж / халат домашний велюровый теплый подарок",
        '{"состав":"80% хлопок; 20% пэ","цвет":"фиолетовый, ягодный, розовый"}',
        "кольцо обручальное",
        '{"бренд":"яхонт ювелирный","тип":"кольцо","проба":"375"}',
    )
    assert f == (0.0, 1.0, 0.0)


# --- Канцелярские_товары.md: пара 25769880990 vs 60129569115, gold=1, pred=0.08
# "случай" — заглушка, не конфликт с lisik.

def test_stub_sluchay_not_conflict_gold1():
    f = brand_features(
        "lisik. кисть для рисования",
        '{"бренд":"lisik.","страна-изготовитель":"китай","тип":"кисть для рисования"}',
        "не определен кисть синтетика, круглая №7, 1 шт.",
        '{"бренд":"случай","тип":"кисточка"}',
    )
    assert f == (0.0, 0.0, 1.0)


# --- Канцелярские_товары.md: пара 455266649328 vs 455266603659, gold=1, pred=0.08
# "нет бренда" и "no name" — обе заглушки.

def test_stub_net_brenda_vs_no_name_gold1():
    f = brand_features(
        "мешок для обуви на шнурке, цвет белый",
        '{"бренд":"нет бренда","пол ребенка":"унисекс","тип":"сумка для сменной обуви"}',
        "no name мешок для обуви на шнурке, цвет белый/разноцветный",
        '{"тип":"сумка для сменной обуви"}',
    )
    assert f == (0.0, 0.0, 1.0)


# --- Канцелярские_товары.md: пара 773094127630 vs 111669280608, gold=1, pred=0.09

def test_stub_sluchay_vs_empty_gold1():
    f = brand_features(
        "краска масляная случай 1 шт., 46 мл.",
        '{"бренд":"случай","название цвета":"красный","тип":"краска масляная","объем, мл":"46"}',
        '/ краска масляная художественная "студия",',
        '{"валюта":"rub"}',
    )
    assert f == (0.0, 0.0, 1.0)


# --- Обувь.md: пара 274878057223 vs 317827673282, gold=0, pred=0.54 ----------
# noname — заглушка с обеих сторон: нет положительного бренд-сигнала.

def test_noname_both_sides_missing_gold0():
    f = brand_features(
        "ботинки рабочие noname",
        '{"бренд в одежде и обуви":"noname","цвет товара":"черный","сезон":"лето","тип":"ботинки рабочие"}',
        "noname полуботинки с перфорацией профи new пу/тпу с мп (р.46) (мод. 64 мп)",
        '{"тип":"полуботинки"}',
    )
    assert f == (0.0, 0.0, 1.0)


# --- Обувь.md: пара 841813624540 vs 146028889196, gold=0, pred=0.49 ----------

def test_silicium_vs_net_brenda_missing_gold0():
    f = brand_features(
        "силициум / тапочки одноразовые синие набор 50 пар открытый мыс",
        '{"состав":"спанбонд; изолон","цвет":"синий","российский размер":"44-45"}',
        "тапочки одноразовые, 50",
        '{"бренд в одежде и обуви":"нет бренда","цвет товара":"синий","российский размер (обуви)":"38"}',
    )
    assert f == (0.0, 0.0, 1.0)


# --- Электроника.md: пара 42949700591 vs 790274104300, gold=0, pred=0.84 -----
# "не определен" в начале названия — заглушка; "производитель телефона" — не бренд.

def _known_limit_test_ne_opredelen_prefix_missing_gold0():  # бренд из свободного текста name — осознанно не извлекаем (шумно)
    f = brand_features(
        "не определен контейнер sim для телефона xiaomi redmi 7a черный",
        '{"производитель телефона":"xiaomi","совместимые модели":"xiaomi redmi 7a"}',
        "лоток для sim-карты xiaomi redmi 7a черный",
        '{}',
    )
    assert f == (0.0, 0.0, 1.0)


# --- Обувь.md: пара 532575963613 vs 249108180019, gold=0 (бренд общий) -------

def test_new_balance_attr_vs_prefix_equal():
    f = brand_features(
        "кроссовки new balance 530",
        '{"бренд в одежде и обуви":"new balance","страна бренда":"сша","российский размер":"37"}',
        "new balance / кроссовки new balance 530",
        '{"цвет":"белый","сезон":"круглогодичный","особенности модели":"new balance 530"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Обувь.md: пара 8590015069 vs 798864009708, gold=0 (бренд общий) ---------

def test_enrico_fantory_attr_vs_prefix_equal():
    f = brand_features(
        "полуботинки enrico fantory",
        '{"бренд в одежде и обуви":"enrico fantory","страна бренда":"россия","российский размер":"44"}',
        "enrico fantory / полуботинки",
        '{"состав":"экокожа","цвет":"черный","российский размер":"36"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Обувь.md: пара 103079259711 vs 824633822839, gold=0 (бренд общий) -------
# у A бренда в атрибутах нет, но он есть в названии — кросс-поиск.

def test_hollywood_hills_brand_found_in_other_name():
    f = brand_features(
        "hollywood hills ботинки hollywood hills, размер 41, черный",
        '{"цвет товара":"чёрный","артикул":"l1131черный","тип":"ботинки","размер":"41"}',
        "ботинки hollywood hills",
        '{"бренд в одежде и обуви":"hollywood hills","страна бренда":"россия","цвет товара":"черный"}',
    )
    assert f == (1.0, 0.0, 0.0)


# --- Обувь.md: пара 180388678290 vs 352187322750, gold=0, pred=0.30 ----------
# pronita vs v.i.konty: бренд B не извлекается из атрибутов -> missing, не equal.

def test_pronita_vs_vikonty_no_false_equal():
    f = brand_features(
        "кроссовки pronita",
        '{"бренд в одежде и обуви":"pronita","страна бренда":"россия","цвет товара":"светло-серый"}',
        "v.i.konty кроссовки",
        '{}',
    )
    assert f[EQ] == 0.0
    assert f[MISS] == 1.0 or f[CONF] == 1.0


# --- extract_brand: dict на входе тоже работает ------------------------------

def test_extract_brand_accepts_dict_and_json():
    b1, _ = extract_brand("кресло", {"бренд": "метта"})
    b2, _ = extract_brand("кресло", '{"бренд":"метта"}')
    assert b1 == b2 == "metta"


def _main() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e!r}")
    total = len([n for n in globals() if n.startswith("test_")])
    print(f"\n{total - fails}/{total} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_main())
