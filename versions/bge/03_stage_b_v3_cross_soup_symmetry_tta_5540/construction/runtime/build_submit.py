#!/usr/bin/env python3
"""Build the corrected Stage-B v3 cross-soup symmetry-TTA solution."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parents[2]
RUNTIME_DIR = VERSION_DIR / "solution" / "runtime"
DEFAULT_MODEL_DIR = (
    VERSION_DIR
    / "solution"
    / "original"
    / "models"
    / "user-bge-human-ft"
)

ARCHIVE_ROOT = "matching-bge-human-ft-v3-soup-tta"
MODEL_DESTINATION = "user-bge-m3-v3-cross-soup-tta"

EXPECTED_MODEL_SHA256 = (
    "b9d98750751a2442a946f6d9188fa3cda251df4d818ad0b23fa13abf223469da"
)

RUNTIME_FILES = (
    "metadata.json",
    "README.md",
    "run.py",
    "src/__init__.py",
    "src/utils.py",
)

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)

ZIP_TIMESTAMP = (2026, 8, 29, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(
    base: Path,
    relative_paths: tuple[str, ...],
) -> None:
    missing = [
        relative
        for relative in relative_paths
        if not (base / relative).is_file()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {item}"
            for item in missing
        )
        raise SystemExit(
            f"Missing required files under {base}:\n"
            f"{formatted}"
        )


def copy_files(
    source: Path,
    destination: Path,
    relative_paths: tuple[str, ...],
) -> None:
    for relative in relative_paths:
        source_path = source / relative
        destination_path = destination / relative

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_path,
            destination_path,
        )


def write_deterministic_zip(
    source_root: Path,
    output_zip: Path,
) -> None:
    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for path in sorted(
            item
            for item in source_root.rglob("*")
            if item.is_file()
        ):
            relative = path.relative_to(source_root)

            archive_name = (
                Path(ARCHIVE_ROOT) / relative
            ).as_posix()

            info = zipfile.ZipInfo(
                archive_name,
                date_time=ZIP_TIMESTAMP,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16

            with path.open("rb") as source_file:
                with archive.open(
                    info,
                    mode="w",
                ) as target_file:
                    shutil.copyfileobj(
                        source_file,
                        target_file,
                        length=1 << 20,
                    )


def build(
    model_dir: Path,
    output_zip: Path,
    overwrite: bool,
) -> None:
    require_files(
        RUNTIME_DIR,
        RUNTIME_FILES,
    )
    require_files(
        model_dir,
        MODEL_FILES,
    )

    model_path = model_dir / "model.safetensors"
    model_digest = sha256(model_path)

    if model_digest != EXPECTED_MODEL_SHA256:
        raise SystemExit(
            "Unexpected model.safetensors SHA-256:\n"
            f"expected={EXPECTED_MODEL_SHA256}\n"
            f"actual={model_digest}"
        )

    if output_zip.exists() and not overwrite:
        raise SystemExit(
            f"Output already exists: {output_zip}\n"
            "Pass --overwrite to replace it."
        )

    output_zip.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_zip.exists():
        output_zip.unlink()

    with tempfile.TemporaryDirectory(
        prefix="bge-v3-soup-tta-"
    ) as temporary_directory:
        staging = (
            Path(temporary_directory)
            / "solution"
        )

        copy_files(
            RUNTIME_DIR,
            staging,
            RUNTIME_FILES,
        )

        copy_files(
            model_dir,
            staging
            / "models"
            / MODEL_DESTINATION,
            MODEL_FILES,
        )

        write_deterministic_zip(
            staging,
            output_zip,
        )

    print(f"created={output_zip}")
    print(
        f"size_bytes={output_zip.stat().st_size}"
    )
    print(f"sha256={sha256(output_zip)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the corrected Stage-B v3 "
            "cross-soup symmetry-TTA solution archive"
        )
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory containing the exported "
            "Hugging Face model"
        ),
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build(
        model_dir=args.model_dir.resolve(),
        output_zip=args.output_zip.resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()