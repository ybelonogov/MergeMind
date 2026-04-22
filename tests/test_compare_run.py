"""Run comparison report tests."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class CompareRunScriptTests(unittest.TestCase):
    def test_compare_run_prints_mode_table(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            prepared_dir = base_dir / "prepared"
            runs_dir = base_dir / "runs"
            config_path = base_dir / "base.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "paths:",
                        f"  prepared_dir: {prepared_dir}",
                        f"  runs_dir: {runs_dir}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_jsonl(
                prepared_dir / "validation.jsonl",
                [
                    {
                        "source_dataset": "CodeReviewer",
                        "example_id": "mr-1",
                        "split": "validation",
                        "repo": "demo/repo",
                        "title": "Guard cart",
                        "description": "Handle empty carts.",
                        "diff": "diff --git a/cart.py b/cart.py\n+if not items:\n+    return 0\n",
                        "changed_files": [{"path": "cart.py"}],
                        "gold_comments": [{"text": "Guard empty carts."}],
                    }
                ],
            )
            for mode, prediction, valid_alt in [
                ("baseline_retrieval_logistic", "Use the old guard.", 0.0),
                ("qwen35_full_with_rewriter", "Add a guard for empty carts.", 0.8),
            ]:
                _write_jsonl(
                    runs_dir / "demo_run" / mode / "predictions.jsonl",
                    [
                        {
                            "example_id": "mr-1",
                            "source_dataset": "CodeReviewer",
                            "best_similarity_at_k": 0.5,
                            "hit_at_k": 1,
                            "judge_score": 0.8,
                            "judge": {
                                "gold_alignment_score": 0.5,
                                "valid_alternative_score": valid_alt,
                                "groundedness": 0.9,
                                "usefulness": 0.8,
                                "reason": "Useful.",
                            },
                            "gold_comments": ["Guard empty carts."],
                            "predictions": [{"text": prediction, "essence": "Empty cart guard"}],
                        }
                    ],
                )

            result = subprocess.run(
                [
                    "python",
                    "scripts/compare_run.py",
                    "--config",
                    str(config_path),
                    "--run",
                    "demo_run",
                    "--modes",
                    "baseline_retrieval_logistic",
                    "qwen35_rewriter",
                    "--limit",
                    "1",
                    "--diff-lines",
                    "4",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("MergeMind Run Comparison", result.stdout)
        self.assertIn("baseline_retrieval_logistic", result.stdout)
        self.assertIn("qwen35_full_with_rewriter", result.stdout)
        self.assertIn("Valid alt", result.stdout)
        self.assertIn("Add a guard for empty carts.", result.stdout)
        self.assertIn("```diff", result.stdout)


if __name__ == "__main__":
    unittest.main()
