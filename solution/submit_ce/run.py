import argparse

from src.inference import run

MODEL_DIR = "ce_v1_final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", dest="items_path", required=True)
    parser.add_argument("--matches_path", "--matches-path", dest="matches_path", required=True)
    parser.add_argument("--output_path", "--output-path", "-o", dest="output_path", required=True)
    args = parser.parse_args()

    run(args.items_path, args.matches_path, args.output_path, MODEL_DIR)


if __name__ == "__main__":
    main()
