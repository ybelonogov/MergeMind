"""Human-readable prediction inspection tests."""

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


class InspectPredictionsScriptTests(unittest.TestCase):
    def test_inspect_predictions_prints_human_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            prepared_dir = base_dir / "prepared"
            runs_dir = base_dir / "runs"
            mode_dir = runs_dir / "demo_run" / "qwen35_generator_qwen35_reranker"
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
            _write_jsonl(
                mode_dir / "predictions.jsonl",
                [
                    {
                        "example_id": "mr-1",
                        "source_dataset": "CodeReviewer",
                        "top1_similarity": 0.5,
                        "best_similarity_at_k": 0.5,
                        "hit_at_k": 1,
                        "gold_comments": ["Guard empty carts."],
                        "predictions": [
                            {
                                "text": "Add a guard for empty carts before checkout.",
                                "generator_score": 0.9,
                                "reranker_score": 0.8,
                                "source_example_id": "mr-1",
                                "evidence": [
                                    "llm_generator",
                                    "reason=The diff touches empty cart handling.",
                                    "usefulness=0.900",
                                ],
                            }
                        ],
                        "judge_score": 0.7,
                    }
                ],
            )

            result = subprocess.run(
                [
                    "python",
                    "scripts/inspect_predictions.py",
                    "--config",
                    str(config_path),
                    "--run",
                    "demo_run",
                    "--limit",
                    "1",
                    "--diff-lines",
                    "5",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("MergeMind Prediction Inspection", result.stdout)
        self.assertIn("Guard empty carts.", result.stdout)
        self.assertIn("Add a guard for empty carts before checkout.", result.stdout)
        self.assertIn("The diff touches empty cart handling.", result.stdout)
        self.assertIn("```diff", result.stdout)


if __name__ == "__main__":
    unittest.main()
