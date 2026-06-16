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
                                "metrics": {
                                    "actual_iterations": 3,
                                    "gap_sequence": [5, 4, 2, 2],
                                    "final_gap": 2,
                                    "best_gap": 2,
                                    "total_tokens": 100,
                                    "official_evoscore": 0.25,
                                    "failed_test_nodeids_by_iteration": [
                                        ["a", "b", "c", "d", "e"],
                                        ["a", "b", "c", "d"],
                                        ["a", "b"],
                                        ["a", "b"],
                                    ],
                                },
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
                                    "gap_sequence": [5, 2, 0],
                                    "final_gap": 0,
                                    "best_gap": 0,
                                    "total_tokens": 150,
                                    "official_evoscore": 0.75,
                                    "mergemind_review_tokens": 50,
                                    "llm_call_count": 3,
                                    "mergemind_assist_comment_count": 4,
                                    "mergemind_assist_revision_count": 2,
                                    "failed_test_nodeids_by_iteration": [
                                        ["a", "b", "c", "d", "e"],
                                        ["a", "b"],
                                        [],
                                    ],
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
        self.assertEqual(rows[0]["baseline_iterations_to_best_gap"], 2)
        self.assertEqual(rows[0]["assisted_iterations_to_best_gap"], 2)
        self.assertEqual(rows[0]["iterations_to_best_gap_delta"], 0)
        self.assertEqual(rows[0]["result_label"], "improved")
        self.assertEqual(rows[0]["first_iter_to_same_gap"], 1)
        self.assertEqual(rows[0]["same_gap_iteration_delta"], -1)
        self.assertEqual(rows[0]["failed_set_jaccard_vs_baseline"], 0.0)
        self.assertEqual(rows[0]["fixed_failure_count"], 2)
        self.assertEqual(rows[0]["new_failure_count"], 0)
        self.assertEqual(rows[0]["tokens_per_gap_delta"], 75)
        self.assertEqual(rows[0]["tokens_per_fixed_failure"], 75)
        self.assertEqual(rows[0]["official_evoscore_delta"], 0.5)
        self.assertEqual(summary["assisted_comment_count"], 4)
        self.assertEqual(summary["improved_count"], 1)
        self.assertEqual(summary["worse_count"], 0)
        self.assertEqual(summary["mean_iterations_to_best_gap_delta"], 0)
        self.assertEqual(summary["mean_official_evoscore_delta"], 0.5)
        self.assertEqual(summary["baseline_total_tokens"], 100)
        self.assertEqual(summary["assisted_total_tokens"], 150)
        self.assertEqual(summary["assisted_review_tokens"], 50)
        self.assertIn("task-1", markdown)
        self.assertIn("failed_jaccard", markdown)
        self.assertIn("improved/worse/unchanged/incomplete: 1 / 0 / 0 / 0", markdown)
        self.assertIn("mean official EvoScore delta: 0.500", markdown)

    def test_invalid_assisted_final_gap_is_not_counted_as_improvement(self) -> None:
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
                                "metrics": {
                                    "actual_iterations": 3,
                                    "gap_sequence": [5, 11],
                                    "final_gap": 11,
                                    "invalid_iteration_count": 0,
                                    "failed_test_nodeids_by_iteration": [["a"], ["a", "b"]],
                                },
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
                                "metrics": {
                                    "actual_iterations": 3,
                                    "gap_sequence": [5, 5, -1],
                                    "final_gap": -1,
                                    "invalid_iteration_count": 1,
                                    "failed_test_nodeids_by_iteration": [["a"], ["a"], []],
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows, summary = build_comparison(baseline, assisted)
            markdown = render_markdown(baseline, assisted, rows, summary)

        self.assertIsNone(rows[0]["assisted_final_gap"])
        self.assertFalse(rows[0]["assisted_final_gap_valid"])
        self.assertIsNone(rows[0]["final_gap_delta"])
        self.assertEqual(rows[0]["result_label"], "worse")
        self.assertIsNone(summary["mean_final_gap_delta"])
        self.assertEqual(summary["invalid_final_gap_count"], 1)
        self.assertEqual(summary["assisted_invalid_iteration_count"], 1)
        self.assertEqual(summary["worse_count"], 1)
        self.assertIn("invalid assisted final gaps: 1", markdown)

    def test_same_gap_with_new_failed_tests_is_worse(self) -> None:
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
                                "metrics": {
                                    "gap_sequence": [3, 2],
                                    "final_gap": 2,
                                    "best_gap": 2,
                                    "failed_test_nodeids_by_iteration": [["a", "b", "c"], ["a", "b"]],
                                },
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
                                "metrics": {
                                    "gap_sequence": [3, 2],
                                    "final_gap": 2,
                                    "best_gap": 2,
                                    "failed_test_nodeids_by_iteration": [["a", "b", "c"], ["a", "d"]],
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows, summary = build_comparison(baseline, assisted)

        self.assertEqual(rows[0]["result_label"], "worse")
        self.assertEqual(rows[0]["new_failure_count"], 1)
        self.assertFalse(rows[0]["same_gap_same_tests"])
        self.assertEqual(summary["worse_count"], 1)


if __name__ == "__main__":
    unittest.main()
