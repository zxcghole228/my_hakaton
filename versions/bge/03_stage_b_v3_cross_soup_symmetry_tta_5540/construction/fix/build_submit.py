#!/usr/bin/env python3
"""Build a flat V3 submitted-solution ZIP from the fixed runtime and model."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


EXPECTED_MODEL_SHA256 = (
    "b9d98750751a2442a946f6d9188fa3cda251df4d818ad0b23fa13abf223469da"
)
ZIP_TIMESTAMP = (2026, 8, 30, 0, 0, 0)
RUNTIME_FILES = ("metadata.json", "run.py", "src/utils.py")
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(root: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise SystemExit(
            f"Missing files under {root}:\n" + "\n".join(f"- {x}" for x in missing)
        )


def add_directory(archive: zipfile.ZipFile, name: str) -> None:
    info = zipfile.ZipInfo(name.rstrip("/") + "/", ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (0o40755 << 16) | 0x10
    archive.writestr(info, b"")


def add_file(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with source.open("rb") as src, archive.open(info, "w") as dst:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            dst.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    runtime = args.runtime_dir.expanduser().resolve()
    model = args.model_dir.expanduser().resolve()
    output = args.output_zip.expanduser().resolve()
    require_files(runtime, RUNTIME_FILES)
    require_files(model, MODEL_FILES)

    metadata = json.loads((runtime / "metadata.json").read_text(encoding="utf-8"))
    if metadata != {
        "image": "odsai/ecup26-matching-baseline:1.0",
        "entry_point": "python -u run.py",
    }:
        raise SystemExit("Unexpected metadata.json")

    model_hash = sha256(model / "model.safetensors")
    if model_hash != EXPECTED_MODEL_SHA256:
        raise SystemExit(
            f"Unexpected model SHA-256:\nexpected={EXPECTED_MODEL_SHA256}\nactual={model_hash}"
        )

    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        add_file(archive, runtime / "metadata.json", "metadata.json")
        add_file(archive, runtime / "run.py", "run.py")
        add_directory(archive, "src")
        add_file(archive, runtime / "src/utils.py", "src/utils.py")
        add_directory(archive, "models")
        add_directory(archive, "models/user-bge-human-ft")
        for filename in MODEL_FILES:
            add_file(
                archive,
                model / filename,
                f"models/user-bge-human-ft/{filename}",
            )

    print(f"created={output}")
    print(f"size_bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")
    print(f"model_sha256={model_hash}")


if __name__ == "__main__":
    main()
