import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .structured_v4 import compact_item


MAX_ATTR_CHARS = 460

SPACE_RE = re.compile(r"\s+")
MULTIPLY_RE = re.compile(r"[\u00d7\u0445\u0425]")

KEY_ORDER = [
    "\u0431\u0440\u0435\u043d\u0434",
    "brand",
    "\u0430\u0440\u0442\u0438\u043a\u0443\u043b",
    "\u043f\u0430\u0440\u0442\u043d\u043e\u043c\u0435\u0440",
    "part number",
    "partnumber",
    "oem",
    "\u043a\u043e\u0434",
    "sku",
    "\u043c\u043e\u0434\u0435\u043b\u044c",
    "model",
    "\u0440\u0430\u0437\u043c\u0435\u0440",
    "size",
    "\u0440\u043e\u0441\u0442",
    "\u043e\u0431\u0445\u0432\u0430\u0442",
    "\u043f\u043e\u043b",
    "gender",
    "\u0446\u0432\u0435\u0442",
    "color",
    "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
    "material",
    "\u0441\u0435\u0437\u043e\u043d",
    "\u043e\u0431\u044a\u0435\u043c",
    "\u043e\u0431\u044c\u0435\u043c",
    "volume",
    "\u0432\u0435\u0441",
    "weight",
    "\u0434\u043b\u0438\u043d\u0430",
    "\u0448\u0438\u0440\u0438\u043d\u0430",
    "\u0432\u044b\u0441\u043e\u0442\u0430",
    "\u043a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e",
    "\u043a\u043e\u043c\u043f\u043b\u0435\u043a\u0442\u0430\u0446\u0438\u044f",
    "\u0443\u043f\u0430\u043a\u043e\u0432",
    "\u0442\u0438\u043f",
    "type",
]

VARIANT_ALIASES = {
    "size": [
        "\u0440\u0430\u0437\u043c\u0435\u0440 \u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434\u0438\u0442\u0435\u043b\u044f",
        "\u0440\u0430\u0437\u043c\u0435\u0440 \u043e\u0431\u0443\u0432\u0438",
        "\u0440\u0430\u0437\u043c\u0435\u0440 \u043e\u0434\u0435\u0436\u0434\u044b",
        "\u0440\u0430\u0437\u043c\u0435\u0440",
        "size",
        "\u0440\u043e\u0441\u0442",
    ],
    "color": [
        "\u043e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0446\u0432\u0435\u0442",
        "\u0446\u0432\u0435\u0442",
        "color",
        "\u0440\u0430\u0441\u0446\u0432\u0435\u0442\u043a\u0430",
    ],
    "article": [
        "\u0430\u0440\u0442\u0438\u043a\u0443\u043b",
        "sku",
        "oem",
        "\u043f\u0430\u0440\u0442\u043d\u043e\u043c\u0435\u0440",
        "part number",
        "partnumber",
        "\u043a\u043e\u0434 \u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434\u0438\u0442\u0435\u043b\u044f",
    ],
    "model": ["\u043c\u043e\u0434\u0435\u043b\u044c", "model"],
    "gender": ["\u043f\u043e\u043b", "gender"],
    "material": [
        "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b \u0432\u0435\u0440\u0445\u0430",
        "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
        "material",
    ],
}

COMMON_COLORS = {
    "\u0447\u0435\u0440\u043d\u044b\u0439": "\u0447\u0435\u0440\u043d\u044b\u0439",
    "\u0447\u0451\u0440\u043d\u044b\u0439": "\u0447\u0435\u0440\u043d\u044b\u0439",
    "\u0431\u0435\u043b\u044b\u0439": "\u0431\u0435\u043b\u044b\u0439",
    "\u0441\u0435\u0440\u044b\u0439": "\u0441\u0435\u0440\u044b\u0439",
    "\u0441\u0435\u0440\u0435\u0431\u0440\u0438\u0441\u0442\u044b\u0439": "\u0441\u0435\u0440\u0435\u0431\u0440\u0438\u0441\u0442\u044b\u0439",
    "\u043a\u0440\u0430\u0441\u043d\u044b\u0439": "\u043a\u0440\u0430\u0441\u043d\u044b\u0439",
    "\u0431\u043e\u0440\u0434\u043e\u0432\u044b\u0439": "\u0431\u043e\u0440\u0434\u043e\u0432\u044b\u0439",
    "\u0441\u0438\u043d\u0438\u0439": "\u0441\u0438\u043d\u0438\u0439",
    "\u0433\u043e\u043b\u0443\u0431\u043e\u0439": "\u0433\u043e\u043b\u0443\u0431\u043e\u0439",
    "\u0437\u0435\u043b\u0435\u043d\u044b\u0439": "\u0437\u0435\u043b\u0435\u043d\u044b\u0439",
    "\u0437\u0435\u043b\u0451\u043d\u044b\u0439": "\u0437\u0435\u043b\u0435\u043d\u044b\u0439",
    "\u0436\u0435\u043b\u0442\u044b\u0439": "\u0436\u0435\u043b\u0442\u044b\u0439",
    "\u0436\u0451\u043b\u0442\u044b\u0439": "\u0436\u0435\u043b\u0442\u044b\u0439",
    "\u043e\u0440\u0430\u043d\u0436\u0435\u0432\u044b\u0439": "\u043e\u0440\u0430\u043d\u0436\u0435\u0432\u044b\u0439",
    "\u0440\u043e\u0437\u043e\u0432\u044b\u0439": "\u0440\u043e\u0437\u043e\u0432\u044b\u0439",
    "\u0444\u0438\u043e\u043b\u0435\u0442\u043e\u0432\u044b\u0439": "\u0444\u0438\u043e\u043b\u0435\u0442\u043e\u0432\u044b\u0439",
    "\u0431\u0435\u0436\u0435\u0432\u044b\u0439": "\u0431\u0435\u0436\u0435\u0432\u044b\u0439",
    "\u043a\u043e\u0440\u0438\u0447\u043d\u0435\u0432\u044b\u0439": "\u043a\u043e\u0440\u0438\u0447\u043d\u0435\u0432\u044b\u0439",
    "\u0437\u043e\u043b\u043e\u0442\u043e\u0439": "\u0437\u043e\u043b\u043e\u0442\u043e\u0439",
    "\u0437\u043e\u043b\u043e\u0442\u0438\u0441\u0442\u044b\u0439": "\u0437\u043e\u043b\u043e\u0442\u043e\u0439",
    "\u0440\u0430\u0437\u043d\u043e\u0446\u0432\u0435\u0442\u043d\u044b\u0439": "\u0440\u0430\u0437\u043d\u043e\u0446\u0432\u0435\u0442\u043d\u044b\u0439",
}

SIZE_NAME_RE = re.compile(
    r"(?:\u0440\u0430\u0437\u043c\u0435\u0440|\u0440-\u0440|size)\s*[:=\-]?\s*"
    r"([0-9]{2,3}(?:[./-][0-9]{1,3})?|"
    r"xxxs|xxs|xs|s|m|l|xl|xxl|xxxl)",
    flags=re.IGNORECASE,
)

VARIANT_NAMES = (
    "\u0440\u0430\u0437\u043c\u0435\u0440",
    "\u0446\u0432\u0435\u0442",
    "\u0430\u0440\u0442\u0438\u043a\u0443\u043b",
    "\u043c\u043e\u0434\u0435\u043b\u044c",
    "\u043f\u043e\u043b",
    "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
)


@dataclass(frozen=True)
class ItemFeatures:
    text: str
    variant: Tuple[str, str, str, str, str, str]
    category: str
    structured: Tuple


def normalize_piece(value) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("\u0451", "\u0435")
    text = MULTIPLY_RE.sub("x", text)
    text = text.replace(",", ".")
    return SPACE_RE.sub(" ", text).strip()


def safe_attrs(attributes) -> Dict[str, str]:
    if isinstance(attributes, dict):
        obj = attributes
    elif isinstance(attributes, str):
        try:
            obj = json.loads(attributes)
        except Exception:
            obj = {}
    else:
        obj = {}

    if not isinstance(obj, dict):
        return {}

    result = {}
    for key, value in obj.items():
        normalized_key = normalize_piece(key)
        normalized_value = normalize_piece(value)
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def build_text_v2(name, attributes, category, max_attr_chars: int = MAX_ATTR_CHARS) -> str:
    normalized_category = normalize_piece(category)
    normalized_name = normalize_piece(name)
    attrs = safe_attrs(attributes)

    picked = []
    used = set()
    for wanted_key in KEY_ORDER:
        for key, value in attrs.items():
            if key in used:
                continue
            if wanted_key in key:
                picked.append(f"{key}: {value}")
                used.add(key)

    rest = [f"{key}: {value}" for key, value in attrs.items() if key not in used]
    attr_text = " ; ".join(picked + rest)[:max_attr_chars]
    return (
        f"\u043a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f: {normalized_category} | "
        f"\u043d\u0430\u0437\u0432\u0430\u043d\u0438\u0435: {normalized_name} | "
        f"\u0430\u0442\u0440\u0438\u0431\u0443\u0442\u044b: {attr_text}"
    )


def pick_attr(attrs: Dict[str, str], aliases) -> str:
    for alias in aliases:
        for key, value in attrs.items():
            if alias in key:
                return value
    return ""


def canonical_compact(value) -> str:
    text = normalize_piece(value)
    if not text:
        return ""
    text = re.sub(r"[^0-9a-z\u0430-\u044f.+/_\- ]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()[:80]


def canonical_code(value) -> str:
    return re.sub(r"[^0-9a-z\u0430-\u044f]+", "", canonical_compact(value))


def canonical_size(value) -> str:
    text = canonical_compact(value)
    if not text:
        return ""
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text[:60]


def canonical_color(value) -> str:
    text = canonical_compact(value)
    if not text:
        return ""
    found = []
    for raw, canonical in COMMON_COLORS.items():
        if raw.replace("\u0451", "\u0435") in text:
            found.append(canonical)
    if found:
        return "/".join(sorted(set(found)))
    return text[:60]


def fallback_size_from_name(name) -> str:
    match = SIZE_NAME_RE.search(normalize_piece(name))
    return canonical_size(match.group(1)) if match else ""


def fallback_color_from_name(name) -> str:
    normalized_name = normalize_piece(name)
    found = []
    for raw, canonical in COMMON_COLORS.items():
        if raw.replace("\u0451", "\u0435") in normalized_name:
            found.append(canonical)
    return "/".join(sorted(set(found)))


def extract_variant_tuple(name, attributes) -> Tuple[str, str, str, str, str, str]:
    attrs = safe_attrs(attributes)
    size = canonical_size(pick_attr(attrs, VARIANT_ALIASES["size"]))
    if not size:
        size = fallback_size_from_name(name)

    color = canonical_color(pick_attr(attrs, VARIANT_ALIASES["color"]))
    if not color:
        color = fallback_color_from_name(name)

    article = canonical_code(pick_attr(attrs, VARIANT_ALIASES["article"]))
    model = canonical_code(pick_attr(attrs, VARIANT_ALIASES["model"]))
    gender = canonical_compact(pick_attr(attrs, VARIANT_ALIASES["gender"]))
    material = canonical_compact(pick_attr(attrs, VARIANT_ALIASES["material"]))
    return size, color, article, model, gender, material


def compare_variant_value(first: str, second: str) -> str:
    if not first or not second:
        return "\u043d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445"
    if first == second:
        return "\u0441\u043e\u0432\u043f\u0430\u0434\u0430\u0435\u0442"

    first_parts = {value for value in re.split(r"[/|; ]+", first) if value}
    second_parts = {value for value in re.split(r"[/|; ]+", second) if value}
    if first_parts and second_parts and first_parts.intersection(second_parts):
        return "\u0447\u0430\u0441\u0442\u0438\u0447\u043d\u043e \u0441\u043e\u0432\u043f\u0430\u0434\u0430\u0435\u0442"
    return "\u0440\u0430\u0437\u043b\u0438\u0447\u0430\u0435\u0442\u0441\u044f"


def make_pair_signal(
    first: Tuple[str, str, str, str, str, str],
    second: Tuple[str, str, str, str, str, str],
) -> str:
    parts = []
    for name, first_value, second_value in zip(VARIANT_NAMES, first, second):
        status = compare_variant_value(first_value, second_value)
        if name in {
            "\u0440\u0430\u0437\u043c\u0435\u0440",
            "\u0446\u0432\u0435\u0442",
            "\u0430\u0440\u0442\u0438\u043a\u0443\u043b",
            "\u043c\u043e\u0434\u0435\u043b\u044c",
        } and first_value and second_value:
            parts.append(f"{name}: {first_value} vs {second_value} => {status}")
        else:
            parts.append(f"{name}: {status}")
    return " | \u0441\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u0432\u0430\u0440\u0438\u0430\u043d\u0442\u043e\u0432: " + " ; ".join(parts)


def load_required_item_features(
    items_path: Path,
    required_ids: Iterable[int],
    batch_size: int = 400_000,
) -> Dict[int, ItemFeatures]:
    required: Set[int] = set(required_ids)
    if not required:
        return {}

    parquet = pq.ParquetFile(items_path)
    expected_columns = {"id", "name", "attributes", "category"}
    missing_columns = expected_columns - set(parquet.schema_arrow.names)
    if missing_columns:
        raise ValueError(f"items parquet is missing columns: {sorted(missing_columns)}")

    id_type = parquet.schema_arrow.field("id").type
    try:
        value_set = pa.array(list(required), type=id_type)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
        raise ValueError(f"Item and match ID types are incompatible: {error}") from error

    features: Dict[int, ItemFeatures] = {}
    started = time.perf_counter()
    for batch_number, batch in enumerate(
        parquet.iter_batches(
            columns=["id", "name", "attributes", "category"],
            batch_size=batch_size,
            use_threads=True,
        ),
        start=1,
    ):
        id_column = batch.column(batch.schema.get_field_index("id"))
        selected = batch.filter(pc.is_in(id_column, value_set=value_set))
        if selected.num_rows:
            frame = selected.to_pandas()
            for item_id, name, attributes, category in frame.itertuples(index=False, name=None):
                features[item_id] = ItemFeatures(
                    text=build_text_v2(name, attributes, category),
                    variant=extract_variant_tuple(name, attributes),
                    category=category,
                    structured=compact_item(name, attributes),
                )

        if batch_number % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[items] batches={batch_number} "
                f"found={len(features):,}/{len(required):,} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if len(features) == len(required):
            break

    missing_ids = required - set(features)
    if missing_ids:
        examples = sorted(missing_ids)[:10]
        raise ValueError(
            f"items parquet does not contain {len(missing_ids)} required IDs; "
            f"examples: {examples}"
        )

    print(
        f"[items] loaded {len(features):,} required items in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return features
