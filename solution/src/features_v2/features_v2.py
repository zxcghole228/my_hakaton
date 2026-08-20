"""features_v2: объединение 4 фиче-модулей из разбора ошибок.

featurize_pair_v2(name1, attrs1, name2, attrs2) -> np.ndarray (32 фичи)
FEATURE_NAMES_V2 -- имена в том же порядке.

Состав: total_qty (8) + brand_alias (3) + code_char_match (9) + size_norm (12).
Все под-модули только stdlib; тесты: test_*.py рядом.
"""
from __future__ import annotations

import numpy as np

from total_qty import quantity_features, FEATURE_NAMES as QTY_NAMES
from brand_alias import brand_alias_features
from code_char_match import code_pair_features
from size_norm import size_features, FEATURE_NAMES as SIZE_NAMES

BRAND_NAMES = ("brandv2_equal", "brandv2_conflict", "brandv2_missing")
CODE_NAMES = ("codev2_exact", "codev2_prefix", "codev2_lev1_suffix", "codev2_lev2",
              "codev2_common_prefix_ratio", "codev2_both", "codev2_disjoint",
              "codev2_cnt_a", "codev2_cnt_b")

FEATURE_NAMES_V2 = tuple(QTY_NAMES) + BRAND_NAMES + tuple(CODE_NAMES) + tuple(SIZE_NAMES)


def featurize_pair_v2(name1, attrs1, name2, attrs2):
    out = []
    out += quantity_features(name1, attrs1, name2, attrs2)
    out += brand_alias_features(name1, attrs1, name2, attrs2)
    out += code_pair_features(name1, attrs1, name2, attrs2)
    out += size_features(name1, attrs1, name2, attrs2)
    return np.asarray(out, dtype=np.float32)
