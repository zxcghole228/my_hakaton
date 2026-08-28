import argparse
import os
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.utils import predict_pipeline


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "user-bge-m3-baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline USER-BGE-M3 inference for Ozon E-CUP 2026"
    )
    parser.add_argument(
        "--items_path",
        "--items-path",
        dest="items_path",
        required=True,
    )
    parser.add_argument(
        "--matches_path",
        "--matches-path",
        dest="matches_path",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        "-o",
        dest="output_path",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=128,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_pipeline(
        data_path=args.items_path,
        match_path=args.matches_path,
        model_path=MODEL_PATH,
        output_csv_path=args.output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()