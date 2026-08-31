#!/usr/bin/env python3
"""Build the corrected Stage-B v2 checkpoint-soup solution archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = VERSION_ROOT / "solution" / "runtime"
DEFAULT_MODEL_DIR = (
    VERSION_ROOT
    / "solution"
    / "original"
    / "models"
    / "user-bge-human-ft"
)

SUBMIT_DIR_NAME = "matching-bge-human-ft-v2-soup"

EXPECTED_MODEL_SHA256 = (
    "d51bd8b0170a4a0d307d803050fa8aa1"
    "bc525fd2780e8b4faef6a13e8c6e3d93"
)

RUNTIME_FILES = (
    "metadata.json",
    "run.py",
    "src/__init__.py",
    "src/pair_text.py",
    "src/utils.py",
)

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def require_files(base: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [
        relative
        for relative in relative_paths
        if not (base / relative).is_file()
    ]

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            f"Missing required files in {base}:\n{formatted}"
        )


def prepare_solution(
    destination: Path,
    model_dir: Path,
) -> None:
    require_files(RUNTIME_DIR, RUNTIME_FILES)
    require_files(model_dir, MODEL_FILES)

    model_path = model_dir / "model.safetensors"
    model_digest = sha256(model_path)

    if model_digest != EXPECTED_MODEL_SHA256:
        raise SystemExit(
            "Unexpected model.safetensors SHA-256:\n"
            f"expected={EXPECTED_MODEL_SHA256}\n"
            f"actual={model_digest}\n"
            f"path={model_path}"
        )

    destination.mkdir(parents=True)

    shutil.copy2(
        RUNTIME_DIR / "metadata.json",
        destination / "metadata.json",
    )
    shutil.copy2(
        RUNTIME_DIR / "run.py",
        destination / "run.py",
    )

    shutil.copytree(
        RUNTIME_DIR / "src",
        destination / "src",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".DS_Store",
        ),
    )

    destination_model = (
        destination
        / "models"
        / "user-bge-m3-v2-soup"
    )
    destination_model.mkdir(parents=True)

    for filename in MODEL_FILES:
        shutil.copy2(
            model_dir / filename,
            destination_model / filename,
        )


def write_zip(
    solution_dir: Path,
    output_zip: Path,
) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for path in sorted(solution_dir.rglob("*")):
            if not path.is_file():
                continue

            relative = path.relative_to(solution_dir)
            archive_name = Path(SUBMIT_DIR_NAME) / relative
            archive.write(path, archive_name.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the corrected Stage-B v2 checkpoint-soup "
            "solution archive"
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the exported Hugging Face model",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        required=True,
        help="Destination ZIP archive",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output ZIP if it already exists",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    output_zip = args.output_zip.expanduser().resolve()

    if output_zip.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_zip}\n"
                "Use --overwrite to replace it."
            )
        output_zip.unlink()

    with tempfile.TemporaryDirectory(
        prefix="bge_v2_soup_"
    ) as temporary:
        temporary_root = Path(temporary)
        solution_dir = temporary_root / SUBMIT_DIR_NAME

        prepare_solution(solution_dir, model_dir)
        write_zip(solution_dir, output_zip)

    print(f"created: {output_zip}")
    print(f"size_bytes: {output_zip.stat().st_size}")
    print(f"sha256: {sha256(output_zip)}")


if __name__ == "__main__":
    main()