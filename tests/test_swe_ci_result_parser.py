from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.result_parser import parse_swe_ci_result
from src.validation.swe_ci.schemas import SweCiTaskRunResult


def _process_result(status: str = "success", exit_code: int | None = 0) -> SweCiTaskRunResult:
    return SweCiTaskRunResult(
        task_id="task-1",
        status=status,  # type: ignore[arg-type]
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        exit_code=exit_code,
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        events_path="events.jsonl",
        metrics={},
        error_message="",
    )


class SweCiResultParserTests(unittest.TestCase):
    def test_missing_result_file_is_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            parsed = parse_swe_ci_result(_process_result(), Path(tmp) / "outputs")

        self.assertEqual(parsed.status, "failed")
        self.assertEqual(parsed.error_message, "Could not locate SWE-CI result file")

    def test_success_result_file_is_success(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            (output_dir / "result.json").write_text(json.dumps({"success": True}), encoding="utf-8")

            parsed = parse_swe_ci_result(_process_result(), output_dir)

        self.assertEqual(parsed.status, "success")
        self.assertIn("swe_ci_result_file", parsed.metrics)

    def test_official_iteration_file_is_success_signal(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp) / "SWE-CI"
            iteration_file = repo / "experiments" / "exp-1" / "task-1" / "iteration.jsonl"
            iteration_file.parent.mkdir(parents=True)
            iteration_file.write_text(
                "\n".join(
                    [
                        json.dumps({"gap": 5}),
                        json.dumps(
                            {
                                "gap": 3,
                                "architect": {"input_tokens": 10, "output_tokens": 5},
                                "programmer": {"input_tokens": 20, "output_tokens": 7},
                                "mergemind_review": {
                                    "status": "success",
                                    "comment_count": 2,
                                    "comments_path": str(repo / "comments.json"),
                                },
                            }
                        ),
                        json.dumps({"gap": 0, "programmer_revision": {"input_tokens": 1}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (repo / "comments.json").write_text(
                json.dumps(
                    {
                        "llm_stats": {
                            "total_tokens": 33,
                            "llm_call_count": 3,
                            "parse_error_rate": 0.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            for index, failed_nodeids in enumerate(
                [
                    ["tests/test_a.py::test_a", "tests/test_b.py::test_b"],
                    ["tests/test_b.py::test_b"],
                    [],
                ]
            ):
                report_dir = iteration_file.parent / f"2026-01-01-00-00-0{index}"
                report_dir.mkdir()
                (report_dir / "test_report.json").write_text(
                    json.dumps(
                        {
                            "tests": [
                                {"nodeid": nodeid, "outcome": "failed", "call": {"outcome": "failed"}}
                                for nodeid in failed_nodeids
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            parsed = parse_swe_ci_result(
                _process_result(),
                Path(tmp) / "outputs",
                swe_ci_repo_path=repo,
                experiment_name="exp-1",
            )

        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.metrics["swe_ci_iteration_count"], 3)
        self.assertEqual(parsed.metrics["actual_iterations"], 2)
        self.assertEqual(parsed.metrics["final_gap"], 0)
        self.assertEqual(parsed.metrics["best_gap"], 0)
        self.assertEqual(parsed.metrics["mergemind_assist_comment_count"], 2)
        self.assertEqual(parsed.metrics["mergemind_assist_revision_count"], 1)
        self.assertEqual(parsed.metrics["failed_test_counts_by_iteration"], [2, 1, 0])
        self.assertEqual(parsed.metrics["coding_tokens"], 42)
        self.assertEqual(parsed.metrics["revision_tokens"], 1)
        self.assertEqual(parsed.metrics["mergemind_review_tokens"], 33)
        self.assertEqual(parsed.metrics["total_tokens"], 76)
        self.assertEqual(parsed.metrics["llm_call_count"], 3)
        self.assertEqual(parsed.metrics["tokens_per_successful_revision"], 76)
        self.assertIn("swe_ci_iteration_file", parsed.metrics)

    def test_failed_test_reports_skip_invalid_gap_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = Path(tmp) / "SWE-CI"
            iteration_file = repo / "experiments" / "exp-1" / "task-1" / "iteration.jsonl"
            iteration_file.parent.mkdir(parents=True)
            iteration_file.write_text(
                "\n".join(
                    [
                        json.dumps({"gap": 2}),
                        json.dumps({"gap": -1}),
                        json.dumps({"gap": 1}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            for name, failed_nodeids in [
                ("2026-01-01-00-00-00", ["tests/test_a.py::test_a", "tests/test_b.py::test_b"]),
                ("2026-01-01-00-01-00", ["tests/test_b.py::test_b"]),
            ]:
                report_dir = iteration_file.parent / name
                report_dir.mkdir()
                (report_dir / "test_report.json").write_text(
                    json.dumps(
                        {
                            "tests": [
                                {"nodeid": nodeid, "outcome": "failed", "call": {"outcome": "failed"}}
                                for nodeid in failed_nodeids
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            parsed = parse_swe_ci_result(
                _process_result(),
                Path(tmp) / "outputs",
                swe_ci_repo_path=repo,
                experiment_name="exp-1",
            )

        self.assertEqual(parsed.metrics["failed_test_counts_by_iteration"], [2, 0, 1])
        self.assertEqual(parsed.metrics["failed_test_nodeids_by_iteration"][2], ["tests/test_b.py::test_b"])


if __name__ == "__main__":
    unittest.main()
