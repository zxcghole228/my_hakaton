"""Тесты size_norm на паттернах из реальных ошибок tiny-2ep (err_analysis/*)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size_norm import size_features, extract_shoe, extract_ring, extract_dims

SHOE_B, SHOE_EQ, SHOE_C = 0, 1, 2
CLOTH_B, CLOTH_EQ, CLOTH_C = 3, 4, 5
RING_B, RING_EQ, RING_C = 6, 7, 8
DIM_B, DIM_EQ, DIM_C = 9, 10, 11

CASES = []

def case(fn):
    CASES.append(fn); return fn

# Обувь: лоферы 39 vs 44 (gold=0, модель дала 0.43) — конфликт
@case
def test_shoe_39_vs_44():
    f = size_features("лоферы enrico fantory", '{"российский размер (обуви)":"39"}',
                      "лоферы enrico fantory", '{"российский размер (обуви)":"44"}')
    assert f[SHOE_C] == 1.0, f

# Обувь: 9 us vs 41.5 ru — та же нога, не конфликт (gold=1, модель 0.05)
@case
def test_shoe_us_vs_ru():
    f = size_features("кроссовки nike, 9 us", '{}',
                      "кроссовки nike", '{"российский размер (обуви)":"42.5"}')
    assert f[SHOE_EQ] == 1.0, f

# Одежда: джинсы w34/l34 vs w36/l34 (gold=0, модель 0.74) — конфликт
@case
def test_jeans_w34_vs_w36():
    f = size_features("джинсы wrangler w34 l34", '{}',
                      "джинсы wrangler w36 l34", '{}')
    assert f[CLOTH_C] == 1.0, f

# Одежда: RU 46 == M (gold=1)
@case
def test_cloth_ru46_eq_m():
    f = size_features("футболка", '{"размер":"46"}',
                      "футболка", '{"размер":"m"}')
    assert f[CLOTH_EQ] == 1.0, f

# Кольца: sokolov _17,5 vs _17 (gold=0, модель 0.91) — конфликт
@case
def test_ring_17_5_vs_17():
    f = size_features("кольцо sokolov 94013490_17,5", '{}',
                      "кольцо sokolov 94013490_17", '{}')
    assert f[RING_C] == 1.0, f

# Кольца: р.17,5 в имени vs атрибут "размер: 17.5" (gold=1, модель низко)
@case
def test_ring_name_vs_attr():
    f = size_features("кольцо золотое р.17,5", '{}',
                      "кольцо золотое", '{"размер":"17.5"}')
    assert f[RING_EQ] == 1.0, f

# Габариты: мм vs см (gold=1) — 2310х1530х730 мм == 231х153х73 см
@case
def test_dims_mm_vs_cm():
    f = size_features("шкаф", '{"габариты":"2310x1530x730 мм"}',
                      "шкаф", '{"габариты":"231х153х73 см"}')
    assert f[DIM_EQ] == 1.0, f

# Габариты: шкаф 200 см vs 180 см (gold=0) — конфликт
@case
def test_dims_200_vs_180():
    f = size_features("шкаф торстейн 200 см", '{}',
                      "шкаф торстейн 180 см", '{}')
    assert f[DIM_C] == 1.0, f

# Ничего нет — нули, не конфликт
@case
def test_empty_no_signal():
    f = size_features("брошь", '{"валюта":"rub"}', "брошь", '{}')
    assert sum(f) == 0.0, f

# Обувь: допуск систем — 42 vs 42.5 не конфликт (разные сетки производителей)
@case
def test_shoe_tolerance():
    f = size_features("кеды", '{"размер":"42"}', "кеды", '{"размер":"42.5"}')
    assert f[SHOE_EQ] == 1.0, f


if __name__ == "__main__":
    ok = 0
    for fn in CASES:
        try:
            fn(); ok += 1; print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, e)
    print(f"\n{ok}/{len(CASES)} passed")
