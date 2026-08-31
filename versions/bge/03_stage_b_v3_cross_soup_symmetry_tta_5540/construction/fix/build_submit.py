#!/usr/bin/env python3
"""Build the exact fixed V3 solution payload as a flat ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


CANDIDATE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_RUNTIME_DIR = (
    CANDIDATE_DIR
    / "construction"
    / "fix"
    / "runtime"
)

DEFAULT_MODEL_DIR = (
    CANDIDATE_DIR
    / "solution"
    / "original"
    / "models"
    / "user-bge-human-ft"
)

ZIP_TIMESTAMP = (2026, 8, 30, 0, 0, 0)

EXPECTED_METADATA = {
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py",
}

EXPECTED_RUNTIME_SHA256 = {
    "metadata.json":
        "8649f720ef0a69c1312813e85c965b2d09591b7dc7914375c2a6cb3e4056007a",
    "run.py":
        "7a1b80007a6310ec77b3b199fa329959a9c89aceb20a7344a3e6b7e88b0a77b2",
    "src/utils.py":
        "f43d6d86a27f7b95c9895ccd4e1cee7228ae65e5155e951c1bed7ec194f44e01",
}

EXPECTED_MODEL_SHA256 = {
    "config.json":
        "dce5292d40b9ccc26008bec2663cf8a972e89ac9b95616df100751abe5f5adc8",
    "model.safetensors":
        "b9d98750751a2442a946f6d9188fa3cda251df4d818ad0b23fa13abf223469da",
    "tokenizer.json":
        "a103899553362761806e7cb54eb43dab2d61684bf6e1ea8ff4b1a5ff3621a0aa",
    "tokenizer_config.json":
        "fc9b2761c4e3aa73907a422b17949afd1d64aee91a0461b28f55a1f016e90fdb",
}

ARCHIVE_NAMES = (
    "metadata.json",
    "run.py",
    "src/",
    "src/utils.py",
    "models/",
    "models/user-bge-human-ft/",
    "models/user-bge-human-ft/config.json",
    "models/user-bge-human-ft/model.safetensors",
    "models/user-bge-human-ft/tokenizer.json",
    "models/user-bge-human-ft/tokenizer_config.json",
)


def stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()

    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)

    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_sha256(stream)


def verify_files(
    root: Path,
    expected: dict[str, str],
    label: str,
) -> None:
    missing = [
        relative_name
        for relative_name in expected
        if not (root / relative_name).is_file()
    ]

    if missing:
        raise SystemExit(
            f"Missing {label} files under {root}:\n"
            + "\n".join(f"- {name}" for name in missing)
        )

    for relative_name, expected_sha in expected.items():
        path = root / relative_name
        actual_sha = sha256(path)

        if actual_sha != expected_sha:
            raise SystemExit(
                f"Unexpected SHA-256 for {path}:\n"
                f"expected={expected_sha}\n"
                f"actual={actual_sha}"
            )


def make_info(
    name: str,
    mode: int,
    directory: bool = False,
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.create_system = 3

    if directory:
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = (mode << 16) | 0x10
    else:
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = mode << 16

    return info


def add_directory(
    archive: zipfile.ZipFile,
    name: str,
) -> None:
    archive.writestr(
        make_info(
            name.rstrip("/") + "/",
            0o40755,
            directory=True,
        ),
        b"",
    )


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: str,
) -> None:
    info = make_info(
        archive_name,
        0o100644,
    )

    with (
        source.open("rb") as source_stream,
        archive.open(info, "w") as archive_stream,
    ):
        for chunk in iter(
            lambda: source_stream.read(1 << 20),
            b"",
        ):
            archive_stream.write(chunk)


def expected_archive_hashes() -> dict[str, str]:
    result = dict(EXPECTED_RUNTIME_SHA256)

    for name, expected_sha in EXPECTED_MODEL_SHA256.items():
        result[f"models/user-bge-human-ft/{name}"] = (
            expected_sha
        )

    return result


def validate_archive(path: Path) -> None:
    expected_hashes = expected_archive_hashes()

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)

        if names != ARCHIVE_NAMES:
            raise SystemExit(
                "Unexpected archive layout:\n"
                f"expected={ARCHIVE_NAMES}\n"
                f"actual={names}"
            )

        if len(names) != len(set(names)):
            raise SystemExit(
                "Duplicate entries found in output archive"
            )

        unsafe = [
            name
            for name in names
            if (
                PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
            )
        ]

        if unsafe:
            raise SystemExit(
                f"Unsafe paths in output archive: {unsafe}"
            )

        for archive_name, expected_sha in expected_hashes.items():
            with archive.open(archive_name) as stream:
                actual_sha = stream_sha256(stream)

            if actual_sha != expected_sha:
                raise SystemExit(
                    "Unexpected archived file SHA-256:\n"
                    f"file={archive_name}\n"
                    f"expected={expected_sha}\n"
                    f"actual={actual_sha}"
                )

        metadata = json.loads(
            archive.read("metadata.json").decode("utf-8")
        )

        if metadata != EXPECTED_METADATA:
            raise SystemExit(
                "Unexpected metadata.json in output archive"
            )


def build_archive(
    runtime: Path,
    model: Path,
    output: Path,
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            add_file(
                archive,
                runtime / "metadata.json",
                "metadata.json",
            )
            add_file(
                archive,
                runtime / "run.py",
                "run.py",
            )

            add_directory(archive, "src")
            add_file(
                archive,
                runtime / "src/utils.py",
                "src/utils.py",
            )

            add_directory(archive, "models")
            add_directory(
                archive,
                "models/user-bge-human-ft",
            )

            for filename in EXPECTED_MODEL_SHA256:
                add_file(
                    archive,
                    model / filename,
                    (
                        "models/user-bge-human-ft/"
                        f"{filename}"
                    ),
                )

        validate_archive(temporary)
        os.replace(temporary, output)

    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact fixed V3 inference payload"
        )
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
        help=(
            "Canonical fixed runtime directory "
            f"(default: {DEFAULT_RUNTIME_DIR})"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory containing the exact V3 model files "
            f"(default: {DEFAULT_MODEL_DIR})"
        ),
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    runtime = args.runtime_dir.expanduser().resolve()
    model = args.model_dir.expanduser().resolve()
    output = args.output_zip.expanduser().resolve()

    if output.suffix.lower() != ".zip":
        raise SystemExit(
            f"Output must have .zip suffix: {output}"
        )

    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output}; "
            "pass --overwrite"
        )

    verify_files(
        runtime,
        EXPECTED_RUNTIME_SHA256,
        "runtime",
    )
    verify_files(
        model,
        EXPECTED_MODEL_SHA256,
        "model",
    )

    metadata = json.loads(
        (runtime / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    if metadata != EXPECTED_METADATA:
        raise SystemExit("Unexpected metadata.json")

    build_archive(
        runtime,
        model,
        output,
    )

    print(f"created={output}")
    print(f"size_bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")
    print(
        "model_sha256="
        + EXPECTED_MODEL_SHA256["model.safetensors"]
    )
    print("archive_validation=ok")


if __name__ == "__main__":
    main()
