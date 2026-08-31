import argparse
from pathlib import Path

from src.utils import predict_pipeline

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "user-bge-m3-v3-cross-soup-tta"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-B v3 cross-soup USER-BGE-M3 inference "
            "with symmetry TTA for Ozon E-CUP 2026"
        )
    )
    parser.add_argument(
        "--items-path",
        "--items_path",
        dest="items_path",
        required=True,
    )
    parser.add_argument(
        "--matches-path",
        "--matches_path",
        dest="matches_path",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    predict_pipeline(
        data_path=args.items_path,
        match_path=args.matches_path,
        model_path=str(MODEL_PATH),
        output_csv_path=args.output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
