import argparse

from src.utils import predict_pipeline

MODEL_PATH = "models/user-bge-human-ft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--items_path", type=str, default=None)
    parser.add_argument("--matches_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    predict_pipeline(
        data_path=args.items_path,
        match_path=args.matches_path,
        model_path=MODEL_PATH,
        output_csv_path=args.output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
