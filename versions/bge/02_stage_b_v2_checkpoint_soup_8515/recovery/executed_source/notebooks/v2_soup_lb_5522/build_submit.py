#!/usr/bin/env python3
"""Pack v2 soup submit (pair_text v1, NO symmetry TTA)."""
import argparse
import hashlib
import os
import shutil
import zipfile
from pathlib import Path

NB = Path(__file__).resolve().parent
REPO = NB.parent.parent
LIB = REPO / "notebooks" / "lib"
SRC_DIR = NB / "submit" / "matching-bge-human-ft"
STAGE = REPO / "submit_staging" / "matching-bge-human-ft-v2-soup"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_pair_text() -> None:
    dst = SRC_DIR / "src" / "pair_text.py"
    shutil.copy2(LIB / "pair_text_v1.py", dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for root, _, files in os.walk(folder):
            for name in files:
                full = Path(root) / name
                arc = full.relative_to(folder.parent)
                zf.write(full, arc.as_posix())


def build(run_dir: Path) -> None:
    sync_pair_text()
    model_src = run_dir / "export_fp16"
    if not model_src.exists():
        raise SystemExit(f"missing {model_src}")
    copy_tree(SRC_DIR, STAGE)
    copy_tree(model_src, STAGE / "models" / "user-bge-human-ft")
    zip_path = run_dir / "matching-bge-human-ft-submit.zip"
    write_zip(STAGE, zip_path)
    print(f"created {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"sha256: {sha256(zip_path)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    build(ap.parse_args().run_dir.resolve())
