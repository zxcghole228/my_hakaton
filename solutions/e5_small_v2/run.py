import argparse
import os
from pathlib import Path


# The evaluator has no network access. Set offline mode before importing
# transformers through src.pipeline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from src.pipeline import predict_pipeline


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models" / "e5_small_macro_v2_30k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E-CUP 2026 Ozon matching inference")
    parser.add_argument("--items_path", required=True, help="Path to test items parquet")
    parser.add_argument("--matches_path", required=True, help="Path to test matches parquet")
    parser.add_argument("--output_path", required=True, help="Path to output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_pipeline(
        items_path=Path(args.items_path),
        matches_path=Path(args.matches_path),
        output_path=Path(args.output_path),
        model_dir=MODEL_DIR,
    )


if __name__ == "__main__":
    main()
