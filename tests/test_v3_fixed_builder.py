from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CANDIDATE = (
    ROOT
    / "versions"
    / "bge"
    / "03_stage_b_v3_cross_soup_symmetry_tta_5540"
)

FIXED_RUNTIME = (
    CANDIDATE
    / "construction"
    / "fix"
    / "runtime"
)

BUILDER = (
    CANDIDATE
    / "construction"
    / "fix"
    / "build_submit.py"
)

MODEL_DIR = (
    CANDIDATE
    / "solution"
    / "original"
    / "models"
    / "user-bge-human-ft"
)

EXPECTED_RUNTIME = {
    "metadata.json":
        "8649f720ef0a69c1312813e85c965b2d09591b7dc7914375c2a6cb3e4056007a",
    "run.py":
        "7a1b80007a6310ec77b3b199fa329959a9c89aceb20a7344a3e6b7e88b0a77b2",
    "src/utils.py":
        "f43d6d86a27f7b95c9895ccd4e1cee7228ae65e5155e951c1bed7ec194f44e01",
}

EXPECTED_MODEL_AUXILIARY = {
    "config.json":
        "dce5292d40b9ccc26008bec2663cf8a972e89ac9b95616df100751abe5f5adc8",
    "tokenizer.json":
        "a103899553362761806e7cb54eb43dab2d61684bf6e1ea8ff4b1a5ff3621a0aa",
    "tokenizer_config.json":
        "fc9b2761c4e3aa73907a422b17949afd1d64aee91a0461b28f55a1f016e90fdb",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1 << 20),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "v3_fixed_builder",
        BUILDER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load fixed V3 builder")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V3FixedBuilderTest(unittest.TestCase):
    def test_exact_runtime_hashes(self):
        for relative_name, expected_sha in (
            EXPECTED_RUNTIME.items()
        ):
            path = FIXED_RUNTIME / relative_name
            self.assertTrue(path.is_file(), path)
            self.assertEqual(
                file_sha256(path),
                expected_sha,
                path,
            )

    def test_exact_model_auxiliary_hashes(self):
        for relative_name, expected_sha in (
            EXPECTED_MODEL_AUXILIARY.items()
        ):
            path = MODEL_DIR / relative_name
            self.assertTrue(path.is_file(), path)
            self.assertEqual(
                file_sha256(path),
                expected_sha,
                path,
            )

    def test_metadata_and_run_contract(self):
        metadata = json.loads(
            (
                FIXED_RUNTIME
                / "metadata.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            metadata,
            {
                "image":
                    "odsai/ecup26-matching-baseline:1.0",
                "entry_point": "python -u run.py",
            },
        )

        run_source = (
            FIXED_RUNTIME
            / "run.py"
        ).read_text(encoding="utf-8")

        ast.parse(run_source)
        self.assertIn(
            "models/user-bge-human-ft",
            run_source,
        )

    def test_empty_rows_order_and_duplicates(self):
        utils_source = (
            FIXED_RUNTIME
            / "src/utils.py"
        ).read_text(encoding="utf-8")

        module = ast.parse(utils_source)

        function = next(
            node
            for node in module.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_build_pairs"
            )
        )

        isolated_module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                function,
            ],
            type_ignores=[],
        )

        namespace = {}

        exec(
            compile(
                ast.fix_missing_locations(
                    isolated_module
                ),
                "<fixed-utils>",
                "exec",
            ),
            namespace,
        )

        class Column:
            def __init__(self, values):
                self.values = values

        class Matches:
            def __init__(self, pairs):
                self.id1 = Column(
                    [left for left, _ in pairs]
                )
                self.id2 = Column(
                    [right for _, right in pairs]
                )

        input_pairs = [
            (1, 2),
            (2, 3),
            (999, 3),
            (1, 2),
            (2, 2),
        ]

        texts = {
            1: "normal-a",
            2: "",
            3: "normal-b",
        }

        pairs, pair_ids = namespace[
            "_build_pairs"
        ](
            Matches(input_pairs),
            texts,
        )

        self.assertEqual(pair_ids, input_pairs)
        self.assertEqual(len(pairs), len(input_pairs))
        self.assertEqual(
            pairs,
            [
                ("normal-a", ""),
                ("", "normal-b"),
                ("", "normal-b"),
                ("normal-a", ""),
                ("", ""),
            ],
        )

    def test_builder_constants_match_manifest(self):
        builder = load_builder()

        self.assertEqual(
            builder.EXPECTED_RUNTIME_SHA256,
            EXPECTED_RUNTIME,
        )

        for relative_name, expected_sha in (
            EXPECTED_MODEL_AUXILIARY.items()
        ):
            self.assertEqual(
                builder.EXPECTED_MODEL_SHA256[
                    relative_name
                ],
                expected_sha,
            )

        self.assertEqual(
            builder.EXPECTED_MODEL_SHA256[
                "model.safetensors"
            ],
            "b9d98750751a2442a946f6d9188fa3cda251df4d818ad0b23fa13abf223469da",
        )


if __name__ == "__main__":
    unittest.main()
