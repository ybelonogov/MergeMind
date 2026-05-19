from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.reporter import write_report
from src.validation.swe_ci.schemas import SweCiTaskRunResult


class SweCiReporterTests(unittest.TestCase):
    def test_writes_metrics_and_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = SweCiTaskRunResult(
                task_id="task-1",
                status="failed",
                started_at="2026-01-01T00:00:00+00:00",
                finished_at="2026-01-01T00:00:01+00:00",
                duration_seconds=1.0,
                exit_code=1,
                stdout_path=str(run_dir / "logs" / "task-1" / "stdout.log"),
                stderr_path=str(run_dir / "logs" / "task-1" / "stderr.log"),
                events_path=str(run_dir / "logs" / "task-1" / "events.jsonl"),
                metrics={"phase": "swe_ci.evaluate", "mergemind_review": {"status": "success", "comment_count": 2}},
                error_message="Could not locate SWE-CI result file",
            )

            metrics = write_report(run_dir, "run-1", [result])
            metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            summary = (run_dir / "summary.md").read_text(encoding="utf-8")
            task_results_exists = (run_dir / "task_results.json").exists()

            self.assertEqual(metrics["failed"], 1)
            self.assertEqual(metrics["mergemind_reviewed"], 1)
            self.assertEqual(metrics["mergemind_comment_count"], 2)
            self.assertEqual(metrics_payload["task_count"], 1)
            self.assertEqual(metrics_payload["mergemind_reviewed"], 1)
            self.assertIn("task-1", summary)
            self.assertIn("MergeMind comments: 2", summary)
            self.assertIn("Could not locate SWE-CI result file", summary)
            self.assertTrue(task_results_exists)


if __name__ == "__main__":
    unittest.main()
