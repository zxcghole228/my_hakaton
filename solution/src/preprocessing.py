import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Set

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


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


def load_required_item_texts(
    items_path: Path,
    required_ids: Iterable[int],
    batch_size: int = 400_000,
) -> Dict[int, str]:
    """Scan the item parquet once and materialize text only for requested IDs."""
    required: Set[int] = set(required_ids)
    if not required:
        return {}

    parquet = pq.ParquetFile(items_path)
    expected_columns = {"id", "name", "attributes", "category"}
    available_columns = set(parquet.schema_arrow.names)
    missing_columns = expected_columns - available_columns
    if missing_columns:
        raise ValueError(f"items parquet is missing columns: {sorted(missing_columns)}")

    id_type = parquet.schema_arrow.field("id").type
    try:
        value_set = pa.array(list(required), type=id_type)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
        raise ValueError(f"Item and match ID types are incompatible: {error}") from error

    item_text: Dict[int, str] = {}
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
        mask = pc.is_in(id_column, value_set=value_set)
        selected = batch.filter(mask)

        if selected.num_rows:
            frame = selected.to_pandas()
            for item_id, name, attributes, category in frame.itertuples(index=False, name=None):
                item_text[item_id] = build_text_v2(name, attributes, category)

        if batch_number % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[items] batches={batch_number} "
                f"found={len(item_text):,}/{len(required):,} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if len(item_text) == len(required):
            break

    missing_ids = required - set(item_text)
    if missing_ids:
        examples = sorted(missing_ids)[:10]
        raise ValueError(
            f"items parquet does not contain {len(missing_ids)} required IDs; "
            f"examples: {examples}"
        )

    elapsed = time.perf_counter() - started
    print(f"[items] loaded {len(item_text):,} required items in {elapsed:.2f}s", flush=True)
    return item_text
