from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.compare_swe_ci_runs import build_comparison, render_markdown


class CompareSweCiRunsTests(unittest.TestCase):
    def test_builds_iteration_and_gap_deltas(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            baseline = base / "baseline"
            assisted = base / "assisted"
            baseline.mkdir()
            assisted.mkdir()
            (baseline / "task_results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "task-1",
                                "status": "success",
                                "duration_seconds": 10,
                                "metrics": {"actual_iterations": 3, "final_gap": 2, "best_gap": 2},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (assisted / "task_results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "task-1",
                                "status": "success",
                                "duration_seconds": 12,
                                "metrics": {
                                    "actual_iterations": 2,
                                    "final_gap": 0,
                                    "best_gap": 0,
                                    "mergemind_assist_comment_count": 4,
                                    "mergemind_assist_revision_count": 2,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows, summary = build_comparison(baseline, assisted)
            markdown = render_markdown(baseline, assisted, rows, summary)

        self.assertEqual(rows[0]["iteration_delta"], -1)
        self.assertEqual(rows[0]["final_gap_delta"], -2)
        self.assertEqual(summary["assisted_comment_count"], 4)
        self.assertIn("task-1", markdown)


if __name__ == "__main__":
    unittest.main()
