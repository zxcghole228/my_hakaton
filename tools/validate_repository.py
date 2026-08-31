#!/usr/bin/env python3
"""Validate the repository without importing ML dependencies.

The checks are intentionally limited to repository structure, metadata,
checksums, source syntax, and the static inference contract. Large local
weights are optional and are not hashed unless --include-local-artifacts is
requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BGE_ROOT = ROOT / "versions" / "bge"
REGISTRY_PATH = BGE_ROOT / "registry.json"

REQUIRED_REPOSITORY_FILES = (
    "README.md",
    "artifacts/README.md",
    "data/README.md",
    "docs/COMPLIANCE.md",
    "docs/THIRD_PARTY_LICENSES.md",
    "requirements/README.md",
    "requirements/training-cu124.txt",
    "versions/bge/README.md",
    "versions/bge/registry.json",
)

FORBIDDEN_TRACKED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".csv",
    ".parquet",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".zip",
}
MAX_TRACKED_FILE_SIZE = 100 * 1024 * 1024

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
}


class Validator:
    def __init__(self, *, include_local_artifacts: bool, quiet: bool) -> None:
        self.include_local_artifacts = include_local_artifacts
        self.quiet = quiet
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checked = 0
        self.tracked = self._tracked_files()

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def check(self, condition: bool, message: str) -> None:
        self.checked += 1
        if not condition:
            self.error(message)

    def _tracked_files(self) -> set[Path]:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return {
                Path(value.decode("utf-8"))
                for value in result.stdout.split(b"\0")
                if value
            }

        # Useful for validating a source archive that has no .git directory.
        excluded_parts = {".git", ".venv", "__pycache__"}
        return {
            path.relative_to(ROOT)
            for path in ROOT.rglob("*")
            if path.is_file()
            and not excluded_parts.intersection(path.relative_to(ROOT).parts)
        }

    @staticmethod
    def _json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_required_files(self) -> None:
        for relative in REQUIRED_REPOSITORY_FILES:
            path = ROOT / relative
            self.check(path.is_file(), f"Required file is missing: {relative}")
            if path.is_file():
                self.check(path.stat().st_size > 0, f"Required file is empty: {relative}")

    def validate_tracked_files(self) -> None:
        for relative in sorted(self.tracked):
            path = ROOT / relative
            parts = relative.parts
            self.check(
                ".DS_Store" not in parts,
                f"macOS metadata is tracked: {relative}",
            )
            self.check(
                "__pycache__" not in parts,
                f"Python cache directory is tracked: {relative}",
            )
            self.check(
                path.suffix.lower() not in FORBIDDEN_TRACKED_SUFFIXES,
                f"Generated, data, archive, or weight file is tracked: {relative}",
            )
            if path.is_file():
                self.check(
                    path.stat().st_size <= MAX_TRACKED_FILE_SIZE,
                    f"Tracked file exceeds 100 MiB: {relative}",
                )

    def validate_structured_files(self) -> None:
        for relative in sorted(self.tracked):
            path = ROOT / relative
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in {".json", ".ipynb"}:
                try:
                    value = self._json(path)
                    self.checked += 1
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    self.error(f"Invalid JSON: {relative}: {exc}")
                    continue
                if suffix == ".ipynb":
                    self.check(
                        isinstance(value, dict) and isinstance(value.get("cells"), list),
                        f"Notebook has no cells list: {relative}",
                    )
            elif suffix == ".jsonl":
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError) as exc:
                    self.error(f"Cannot read JSONL: {relative}: {exc}")
                    continue
                for line_number, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                        self.checked += 1
                    except json.JSONDecodeError as exc:
                        self.error(
                            f"Invalid JSONL: {relative}:{line_number}: {exc}"
                        )

    def validate_python(self) -> None:
        for relative in sorted(self.tracked):
            if relative.suffix.lower() != ".py":
                continue
            path = ROOT / relative
            try:
                source = path.read_text(encoding="utf-8")
                compile(source, str(relative), "exec")
                self.checked += 1
            except (OSError, UnicodeError, SyntaxError) as exc:
                self.error(f"Invalid Python source: {relative}: {exc}")

    def validate_markdown(self) -> None:
        for relative in sorted(self.tracked):
            if relative.suffix.lower() != ".md":
                continue
            path = ROOT / relative
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                self.error(f"Cannot read Markdown: {relative}: {exc}")
                continue
            self.check(
                text.count("```") % 2 == 0,
                f"Unbalanced fenced code blocks: {relative}",
            )

    def validate_secrets(self) -> None:
        text_suffixes = {".json", ".jsonl", ".md", ".py", ".txt", ".yml", ".yaml"}
        for relative in sorted(self.tracked):
            path = ROOT / relative
            if path.suffix.lower() not in text_suffixes or not path.is_file():
                continue
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                self.check(
                    pattern.search(text) is None,
                    f"Possible {label} in tracked file: {relative}",
                )

    def validate_source_manifest(self, version_dir: Path) -> None:
        manifest_path = version_dir / "training" / "source_manifest.json"
        if not manifest_path.is_file():
            self.error(f"Source manifest is missing: {manifest_path.relative_to(ROOT)}")
            return
        manifest = self._json(manifest_path)
        self.check(manifest.get("schema_version") == 1, f"Bad source schema: {manifest_path}")
        files = manifest.get("files")
        self.check(isinstance(files, list) and bool(files), f"Empty source manifest: {manifest_path}")
        if not isinstance(files, list):
            return
        for entry in files:
            relative = entry.get("path")
            expected = entry.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                self.error(f"Malformed source entry in {manifest_path}: {entry!r}")
                continue
            path = version_dir / relative
            self.check(path.is_file(), f"Manifest source is missing: {path.relative_to(ROOT)}")
            if not path.is_file():
                continue
            self.check(
                self._sha256(path) == expected,
                f"Source SHA-256 mismatch: {path.relative_to(ROOT)}",
            )
            size = entry.get("size_bytes")
            if isinstance(size, int):
                self.check(
                    path.stat().st_size == size,
                    f"Source size mismatch: {path.relative_to(ROOT)}",
                )

    def validate_artifact_manifest(self, version_dir: Path, version_id: str) -> None:
        manifest_path = version_dir / "artifacts" / "manifest.json"
        if not manifest_path.is_file():
            self.error(f"Artifact manifest is missing: {manifest_path.relative_to(ROOT)}")
            return
        manifest = self._json(manifest_path)
        self.check(manifest.get("schema_version") == 1, f"Bad artifact schema: {manifest_path}")
        self.check(manifest.get("version_id") == version_id, f"Artifact version id mismatch: {manifest_path}")
        entries = manifest.get("artifacts")
        self.check(isinstance(entries, list) and bool(entries), f"Empty artifact manifest: {manifest_path}")
        if not isinstance(entries, list):
            return
        for entry in entries:
            relative = entry.get("path")
            if not isinstance(relative, str):
                continue
            path = version_dir / relative
            root_relative = path.relative_to(ROOT)
            committed = entry.get("committed_to_git") is True
            if committed:
                self.check(path.is_file(), f"Tracked artifact is missing: {root_relative}")
                self.check(root_relative in self.tracked, f"Artifact marked tracked but absent from Git: {root_relative}")
            else:
                self.check(root_relative not in self.tracked, f"Artifact marked untracked but committed: {root_relative}")

            should_hash = committed or self.include_local_artifacts
            if not path.is_file() or not should_hash:
                continue
            expected_size = entry.get("size_bytes")
            expected_hash = entry.get("sha256")
            if isinstance(expected_size, int):
                self.check(path.stat().st_size == expected_size, f"Artifact size mismatch: {root_relative}")
            if isinstance(expected_hash, str):
                self.check(self._sha256(path) == expected_hash, f"Artifact SHA-256 mismatch: {root_relative}")

    def validate_metadata(self, path: Path) -> None:
        self.check(path.is_file(), f"metadata.json is missing: {path.relative_to(ROOT)}")
        if not path.is_file():
            return
        metadata = self._json(path)
        self.check(isinstance(metadata.get("image"), str) and bool(metadata["image"]), f"Missing image: {path}")
        self.check(metadata.get("entry_point") == "python -u run.py", f"Unexpected entry_point: {path}")

    def validate_runtime(self, version_dir: Path, *, symmetry_tta: bool) -> None:
        runtime = version_dir / "solution" / "runtime"
        run_path = runtime / "run.py"
        utils_path = runtime / "src" / "utils.py"
        self.validate_metadata(runtime / "metadata.json")
        self.check(run_path.is_file(), f"Runtime run.py is missing: {run_path.relative_to(ROOT)}")
        self.check(utils_path.is_file(), f"Runtime utils.py is missing: {utils_path.relative_to(ROOT)}")
        if not run_path.is_file() or not utils_path.is_file():
            return
        run_text = run_path.read_text(encoding="utf-8")
        utils_text = utils_path.read_text(encoding="utf-8")
        for flag in ("--items-path", "--items_path", "--matches-path", "--matches_path", "--output-path", "--output_path"):
            self.check(flag in run_text, f"Runtime flag {flag} is missing: {run_path.relative_to(ROOT)}")
        self.check(utils_text.count("local_files_only=True") >= 2, f"Runtime is not local-only: {utils_path.relative_to(ROOT)}")
        self.check("predict" in utils_text, f"Runtime output column is undocumented in code: {utils_path.relative_to(ROOT)}")
        self.check(
            "match_df.copy()" in utils_text,
            f"Runtime does not preserve the input frame: {utils_path.relative_to(ROOT)}",
        )
        self.check(
            "if not t1 or not t2" not in utils_text,
            f"Runtime may drop incomplete pairs: {utils_path.relative_to(ROOT)}",
        )
        if symmetry_tta:
            self.check("symmetry TTA" in utils_text, f"Symmetry TTA marker is missing: {utils_path.relative_to(ROOT)}")

    def validate_registry(self) -> None:
        if not REGISTRY_PATH.is_file():
            self.error("versions/bge/registry.json is missing")
            return
        registry = self._json(REGISTRY_PATH)
        self.check(registry.get("schema_version") == 1, "Unsupported registry schema")
        self.check(registry.get("family") == "bge", "Unexpected registry family")
        candidates = registry.get("candidates")
        self.check(isinstance(candidates, list) and len(candidates) == 3, "Registry must contain exactly three candidates")
        if not isinstance(candidates, list):
            return

        ids = [entry.get("id") for entry in candidates]
        paths = [entry.get("path") for entry in candidates]
        numbers = [entry.get("candidate_number") for entry in candidates]
        self.check(len(ids) == len(set(ids)), "Candidate ids are not unique")
        self.check(len(paths) == len(set(paths)), "Candidate paths are not unique")
        self.check(numbers == [1, 2, 3], "Candidate numbers must be ordered as 1, 2, 3")

        selection = registry.get("final_selection", {})
        selected = selection.get("selected_candidate_ids", [])
        maximum = selection.get("maximum_selected_solutions")
        self.check(maximum == 2, "At most two final solutions must be selectable")
        self.check(isinstance(selected, list), "selected_candidate_ids must be a list")
        if not isinstance(selected, list):
            selected = []
        self.check(len(selected) <= 2, "More than two final solutions are selected")
        self.check(len(selected) == len(set(selected)), "Final selection contains duplicate ids")
        self.check(set(selected).issubset(set(ids)), "Final selection contains an unknown candidate")

        for entry in candidates:
            candidate_path = entry.get("path")
            candidate_id = entry.get("id")
            if not isinstance(candidate_path, str) or not isinstance(candidate_id, str):
                self.error(f"Malformed candidate registry entry: {entry!r}")
                continue
            version_dir = BGE_ROOT / candidate_path
            version_path = version_dir / "version.json"
            self.check(version_dir.is_dir(), f"Candidate directory is missing: {candidate_path}")
            self.check(version_path.is_file(), f"Candidate version.json is missing: {candidate_path}")
            if not version_path.is_file():
                continue
            version = self._json(version_path)
            self.check(version.get("id") == candidate_id, f"Candidate id mismatch: {candidate_path}")
            self.check(version.get("family") == "bge", f"Candidate family mismatch: {candidate_path}")
            self.check(
                version.get("base_model", {}).get("name") == registry.get("base_model", {}).get("name"),
                f"Base model mismatch: {candidate_path}",
            )
            self.check(
                version.get("base_model", {}).get("license") == "apache-2.0",
                f"Base model license is missing or inconsistent: {candidate_path}",
            )
            self.check(
                version.get("selection", {}).get("candidate_number") == entry.get("candidate_number"),
                f"Candidate number mismatch: {candidate_path}",
            )
            score = version.get("leaderboard", {}).get("public_macro_ap")
            self.check(
                isinstance(score, (int, float))
                and math.isclose(float(score), float(entry.get("public_macro_ap")), rel_tol=0, abs_tol=1e-15),
                f"Leaderboard score mismatch: {candidate_path}",
            )
            weights = version.get("weights", {})
            self.check(weights.get("sha256") == entry.get("weight_sha256"), f"Weight hash mismatch: {candidate_path}")
            self.check(weights.get("size_bytes") == entry.get("weight_size_bytes"), f"Weight size mismatch: {candidate_path}")
            self.check(
                version.get("solution", {}).get("output_columns") == ["id1", "id2", "predict"],
                f"Output contract mismatch: {candidate_path}",
            )
            self.check((version_dir / "README.md").is_file(), f"Candidate README is missing: {candidate_path}")
            self.validate_metadata(version_dir / "solution" / "original" / "metadata.json")
            self.validate_runtime(version_dir, symmetry_tta=entry.get("symmetry_tta") is True)
            self.validate_source_manifest(version_dir)
            self.validate_artifact_manifest(version_dir, candidate_id)

            selected_here = candidate_id in selected
            recorded = version.get("selection", {}).get("final_selected")
            if selected:
                self.check(recorded is selected_here, f"Final selection flag mismatch: {candidate_path}")
            else:
                self.check(recorded is None, f"Pending candidate must have final_selected=null: {candidate_path}")

        if selected:
            self.check(
                selection.get("status") not in {"pending_platform_confirmation", None},
                "Selection has candidates but is still marked pending",
            )
            self.check((ROOT / "final" / "selection.json").is_file(), "Final selection requires final/selection.json")
        else:
            self.check(
                selection.get("status") == "pending_platform_confirmation",
                "Empty final selection must be marked pending_platform_confirmation",
            )

    def run(self) -> int:
        self.validate_required_files()
        self.validate_tracked_files()
        self.validate_structured_files()
        self.validate_python()
        self.validate_markdown()
        self.validate_secrets()
        self.validate_registry()

        for warning in self.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        for error in self.errors:
            print(f"ERROR: {error}", file=sys.stderr)

        if self.errors:
            print(
                f"repository_validation_failed: {len(self.errors)} error(s), "
                f"{len(self.warnings)} warning(s), {self.checked} checks",
                file=sys.stderr,
            )
            return 1
        if not self.quiet:
            print(
                f"repository_validation_ok: {self.checked} checks, "
                f"{len(self.warnings)} warning(s), 3 candidates"
            )
        return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-local-artifacts",
        action="store_true",
        help="Also hash ignored local artifacts such as model.safetensors.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only errors.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return Validator(
        include_local_artifacts=args.include_local_artifacts,
        quiet=args.quiet,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
