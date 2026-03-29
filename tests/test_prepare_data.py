"""Data preparation tests."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.adapters import prepare_datasets
from src.data.io import read_json, read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PrepareDataTests(unittest.TestCase):
    def test_prepare_datasets_writes_expected_splits(self) -> None:
        raw_paths = {
            "codereviewer": str(PROJECT_ROOT / "sample_data/raw/codereviewer.jsonl"),
            "codereviewqa": str(PROJECT_ROOT / "sample_data/raw/codereviewqa.jsonl"),
            "codocbench": str(PROJECT_ROOT / "sample_data/raw/codocbench.jsonl"),
        }

        with TemporaryDirectory() as temp_dir:
            prepared_dir = Path(temp_dir) / "prepared"
            summary = prepare_datasets(raw_paths=raw_paths, prepared_dir=prepared_dir)

            train_rows = read_jsonl(prepared_dir / "train.jsonl")
            validation_rows = read_jsonl(prepared_dir / "validation.jsonl")
            test_rows = read_jsonl(prepared_dir / "test.jsonl")
            demo_row = read_json(prepared_dir / "demo.json")

        self.assertEqual(summary["counts"]["train"], 5)
        self.assertEqual(len(train_rows), 5)
        self.assertEqual(len(validation_rows), 3)
        self.assertEqual(len(test_rows), 1)
        self.assertEqual(demo_row["example_id"], "cr-demo-001")


if __name__ == "__main__":
    unittest.main()
