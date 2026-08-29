"""Verify pair_text v1 matches frozen LB 0.5522 / symmetry TTA pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parent
REPO = LIB.parent.parent
sys.path.insert(0, str(LIB))

from pair_text_v1 import PAIR_TEXT_VERSION, ATTR_CHARS_LIMIT, KEYS, build_text  # noqa: E402


# Fashion stress case: many attrs — v1 may drop size under 520 cap (known issue)
FASHION_ITEM = {
    "name": "Кроссовки Nike Air Max 90",
    "attributes": json.dumps({
        "Бренд": "Nike",
        "Артикул": "CN8490-100",
        "Размер": "42",
        "Цвет": "белый",
        "Материал верха": "кожа",
        "Подкладка": "текстиль",
        "Стелька": "текстиль",
        "Подошва": "резина",
        "Сезон": "демисезон",
    }, ensure_ascii=False),
}

UNIT_ITEM = {
    "name": "Шампунь 500 мл",
    "attributes": json.dumps({"Объем": "500 мл", "Бренд": "Test"}, ensure_ascii=False),
}


def _attr_blob(text: str) -> str:
    return text.split(" | ", 1)[1] if " | " in text else ""


def verify_v1_contract() -> list[str]:
    errors: list[str] = []
    if PAIR_TEXT_VERSION != "v1":
        errors.append(f"expected PAIR_TEXT_VERSION=v1, got {PAIR_TEXT_VERSION}")
    if ATTR_CHARS_LIMIT != 520:
        errors.append(f"expected ATTR_CHARS_LIMIT=520, got {ATTR_CHARS_LIMIT}")
    if KEYS[0] != "бренд" or "размер" not in KEYS:
        errors.append(f"unexpected KEYS order: {KEYS[:8]}")
    return errors


def verify_build_text() -> list[str]:
    errors: list[str] = []
    t = build_text(FASHION_ITEM["name"], FASHION_ITEM["attributes"])
    if "Nike" not in t and "nike" not in t.lower():
        errors.append("fashion: brand missing")
    blob = _attr_blob(t)
    if len(blob) > 520:
        errors.append(f"attr blob exceeds 520: {len(blob)}")
    if "ед:" not in build_text(UNIT_ITEM["name"], UNIT_ITEM["attributes"]):
        errors.append("unit canon missing")
    # homoglyph
    hg = build_text("Test", json.dumps({"x": "128 GB"}, ensure_ascii=False))
    if "128" not in hg:
        errors.append("homoglyph/number missing")
    return errors


def compare_v1_vs_v2() -> dict:
    """Show why ecup_v2 pair_text must NOT be used for v2 soup / symmetry TTA."""
    sys.path.insert(0, str(REPO / "ecup_v2"))
    from pair_text import build_text as build_text_v2  # noqa: E402

    v1 = build_text(FASHION_ITEM["name"], FASHION_ITEM["attributes"])
    v2 = build_text_v2(FASHION_ITEM["name"], FASHION_ITEM["attributes"])
    b1, b2 = _attr_blob(v1), _attr_blob(v2)
    return {
        "v1_len": len(v1),
        "v2_len": len(v2),
        "v1_has_size_first": b1.lower().find("размер") < 30 if "размер" in b1.lower() else False,
        "v2_has_size_first": b2.lower().find("размер") < 30 if "размер" in b2.lower() else False,
        "texts_equal": v1 == v2,
        "v1_preview": v1[:200],
        "v2_preview": v2[:200],
    }


def verify_matches_score_ensemble() -> list[str]:
    """Byte-identical to final_4_models/scripts/score_ensemble.build_text."""
    scripts = REPO / "final_4_models" / "scripts"
    sys.path.insert(0, str(scripts))
    import score_ensemble  # noqa: E402

    cases = [
        FASHION_ITEM,
        UNIT_ITEM,
        {"name": "Test", "attributes": json.dumps({"x": "128 GB"}, ensure_ascii=False)},
        {"name": "", "attributes": "{}"},
    ]
    errors: list[str] = []
    for c in cases:
        t1 = build_text(c["name"], c["attributes"])
        t2 = score_ensemble.build_text(c["name"], c["attributes"])
        if t1 != t2:
            errors.append(f"score_ensemble mismatch for {c['name']!r}: v1={t1[:80]!r} se={t2[:80]!r}")
    return errors


def patch_score_ensemble_v1() -> None:
    """Force final_4_models/scripts/score_ensemble.py to use frozen v1 (train/blend)."""
    scripts = REPO / "final_4_models" / "scripts"
    sys.path.insert(0, str(scripts))
    import score_ensemble  # noqa: E402
    import pair_text_v1  # noqa: E402

    score_ensemble.build_text = pair_text_v1.build_text
    score_ensemble.KEYS = pair_text_v1.KEYS


def main() -> int:
    errors = verify_v1_contract() + verify_build_text() + verify_matches_score_ensemble()
    cmp = compare_v1_vs_v2()
    print("pair_text v1 verification")
    print(f"  version={PAIR_TEXT_VERSION} attr_limit={ATTR_CHARS_LIMIT}")
    print(f"  v1 vs v2 equal: {cmp['texts_equal']} (must be False for fashion stress)")
    print(f"  v2 size-first: {cmp['v2_has_size_first']} | v1 size-first: {cmp['v1_has_size_first']}")
    if errors:
        print("FAIL:")
        for e in errors:
            print(" ", e)
        return 1
    print("OK — use pair_text_v1 for v2 soup & symmetry TTA pipelines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
