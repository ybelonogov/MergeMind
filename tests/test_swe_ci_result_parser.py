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
            iteration_file.write_text("{}\n{}\n", encoding="utf-8")

            parsed = parse_swe_ci_result(
                _process_result(),
                Path(tmp) / "outputs",
                swe_ci_repo_path=repo,
                experiment_name="exp-1",
            )

        self.assertEqual(parsed.status, "success")
        self.assertEqual(parsed.metrics["swe_ci_iteration_count"], 2)
        self.assertIn("swe_ci_iteration_file", parsed.metrics)


if __name__ == "__main__":
    unittest.main()
