from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.monitoring.agent import (
    _read_json_list,
    collect_swe_ci_runs,
    render_chronicle,
    render_presentation_markdown,
    write_monitoring_artifacts,
)


class MonitoringAgentTests(unittest.TestCase):
    def test_read_json_list_accepts_task_results_wrapper(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "task_results.json"
            path.write_text(json.dumps({"results": [{"task_id": "task-1"}]}), encoding="utf-8")

            rows = _read_json_list(path)

        self.assertEqual(rows, [{"task_id": "task-1"}])

    def test_collect_swe_ci_runs_reads_metrics_and_errors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "artifacts" / "swe_ci_runs" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(json.dumps({"timeout": 1}), encoding="utf-8")
            (run_dir / "summary.md").write_text("# summary", encoding="utf-8")
            (run_dir / "task_results.json").write_text(
                json.dumps({"results": [{"task_id": "task-1", "error_message": "Process timed out."}]}),
                encoding="utf-8",
            )

            runs = collect_swe_ci_runs(root)

        self.assertEqual(runs[0]["run_id"], "run-1")
        self.assertEqual(runs[0]["metrics"]["timeout"], 1)
        self.assertEqual(runs[0]["latest_errors"][0]["task_id"], "task-1")

    def test_writes_chronicle_dashboard_and_presentation(self) -> None:
        snapshot = {
            "snapshot_id": "20260101T000000Z",
            "created_at": "2026-01-01T00:00:00+00:00",
            "git": {"branch": "## main...origin/main", "dirty_file_count": 0, "latest_commits": ["abc commit"]},
            "dashboard": {
                "lmstudio": {"ok": True, "configured_model": "qwen", "base_url": "http://localhost:1234/v1"},
                "gpu": {"gpus": [{"name": "GPU", "utilization_gpu": 5, "memory_used_mb": 100, "memory_total_mb": 1000}]},
                "runs": [
                    {
                        "run_id": "ab-1",
                        "modes": [
                            {
                                "mode": "qwen35_rewriter",
                                "model_id": "qwen",
                                "hit_rate_at_k": 0.3,
                                "best_similarity_at_k": 0.2,
                                "judge_score": 0.8,
                                "avg_total_wall_latency_sec": 1.2,
                                "status": "completed",
                            }
                        ],
                    }
                ],
            },
            "swe_ci_runs": [],
            "latest_files": [],
            "tests": {"skipped": True},
        }
        with TemporaryDirectory() as tmp:
            paths = write_monitoring_artifacts(snapshot, output_dir=Path(tmp))
            chronicle = Path(paths["chronicle"]).read_text(encoding="utf-8")
            dashboard = Path(paths["dashboard"]).read_text(encoding="utf-8")
            presentation = Path(paths["presentation"]).read_text(encoding="utf-8")

        self.assertIn("A/B Runs", chronicle)
        self.assertIn("qwen35_rewriter", dashboard)
        self.assertIn("Что уже собрано", presentation)
        self.assertIn("qwen35_rewriter", render_chronicle(snapshot))
        self.assertIn("MergeMind", render_presentation_markdown(snapshot))


if __name__ == "__main__":
    unittest.main()
