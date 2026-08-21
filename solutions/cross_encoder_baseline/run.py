import argparse
from pathlib import Path

from src.utils import predict_pipeline


ROOT_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = ROOT_DIR / "baseline_logreg_l12.joblib"
MODEL_CE_PATH = ROOT_DIR / "models" / "cross-encoder-ms-marco-MiniLM-L12-v2"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", required=True, help="Output CSV path")
    parser.add_argument("--items_path", required=True, help="Test items parquet path")
    parser.add_argument("--matches_path", required=True, help="Test matches parquet path")
    args = parser.parse_args()

    items_path = args.items_path
    matches_path = args.matches_path
    output_path = args.output_path

    # full pipeline in one go
    predict_pipeline(
        data_path=items_path,
        match_path=matches_path,
        model_path=str(MODEL_CE_PATH),
        logreg_path=str(CLASSIFIER_PATH),
        output_csv_path=output_path,
        batch_size=512,
    )



if __name__ == "__main__":
    main()
