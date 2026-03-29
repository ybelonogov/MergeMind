"""Model baseline tests."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.adapters import prepare_datasets
from src.data.io import read_jsonl
from src.data.schema import MRExample
from src.inference.pipeline import run_inference
from src.models.baseline import RetrievalGenerator, Reranker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineModelTests(unittest.TestCase):
    def test_retrieval_generator_and_reranker_find_relevant_comment(self) -> None:
        raw_paths = {
            "codereviewer": str(PROJECT_ROOT / "sample_data/raw/codereviewer.jsonl"),
            "codereviewqa": str(PROJECT_ROOT / "sample_data/raw/codereviewqa.jsonl"),
            "codocbench": str(PROJECT_ROOT / "sample_data/raw/codocbench.jsonl"),
        }

        with TemporaryDirectory() as temp_dir:
            prepared_dir = Path(temp_dir) / "prepared"
            prepare_datasets(raw_paths=raw_paths, prepared_dir=prepared_dir)
            train_examples = [MRExample.from_dict(row) for row in read_jsonl(prepared_dir / "train.jsonl")]
            validation_examples = [MRExample.from_dict(row) for row in read_jsonl(prepared_dir / "validation.jsonl")]

        target_example = next(example for example in validation_examples if example.example_id == "cr-val-001")

        generator = RetrievalGenerator.fit(train_examples)
        reranker = Reranker.from_config(
            {
                "retrieval_weight": 0.65,
                "identifier_weight": 0.2,
                "file_weight": 0.1,
                "action_weight": 0.08,
                "hedge_penalty": 0.08,
                "verbosity_penalty": 0.05,
                "short_penalty": 0.04,
            }
        )

        predictions = run_inference(target_example, generator, reranker, top_n=3)

        self.assertTrue(predictions)
        self.assertIn("guard", predictions[0].text.lower())
        self.assertIn("ratio", predictions[0].text.lower())


if __name__ == "__main__":
    unittest.main()
