from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.analyze_swe_ci_requirements import analyze_run, render_markdown


class AnalyzeSweCiRequirementsTests(unittest.TestCase):
    def test_analyzes_requirement_drift_and_review_injection(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            task_id = "owner__repo__base__target"
            task_dir = run_dir / "workdirs" / "assisted_swe_ci" / "experiments" / "exp-1" / task_id
            task_dir.mkdir(parents=True)
            iteration_file = task_dir / "iteration.jsonl"
            iteration_file.write_text(
                "\n".join(
                    [
                        json.dumps({"gap": 2, "pytest": {"passed": 10}}),
                        json.dumps(
                            {
                                "gap": 1,
                                "pytest": {"passed": 11},
                                "mergemind_review": {
                                    "epoch": 1,
                                    "status": "success",
                                    "comment_count": 1,
                                    "apply_revision": True,
                                    "target_sha_used_for_review": False,
                                },
                                "programmer_revision": {"changed_files": ["src/app.py"]},
                            }
                        ),
                        json.dumps(
                            {
                                "gap": 1,
                                "pytest": {"passed": 11},
                                "mergemind_review": {
                                    "epoch": 2,
                                    "status": "skipped",
                                    "comment_count": 0,
                                    "apply_revision": False,
                                    "target_sha_used_for_review": False,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            for name, text in [
                ("2026-01-01-00-00-01", "<requirements><a /></requirements>"),
                ("2026-01-01-00-00-02", "<requirements><b /></requirements>"),
            ]:
                req_dir = task_dir / name
                req_dir.mkdir()
                (req_dir / "requirement.xml").write_text(text, encoding="utf-8")
            (run_dir / "task_results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": task_id,
                                "metrics": {
                                    "swe_ci_iteration_file": str(iteration_file),
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = analyze_run(run_dir)
            markdown = render_markdown(payload)

        task = payload["tasks"][0]
        self.assertEqual(task["status"], "ok")
        self.assertEqual(task["requirement_file_count"], 2)
        self.assertEqual(task["iterations"][0]["requirement_path"], "")
        self.assertIsNone(task["iterations"][1]["requirement_changed_vs_previous"])
        self.assertTrue(task["iterations"][2]["requirement_changed_vs_previous"])
        self.assertEqual(task["iterations"][1]["mergemind_review_status"], "success")
        self.assertTrue(task["iterations"][1]["programmer_revision_present"])
        self.assertFalse(task["iterations"][1]["target_sha_used_for_review"])
        self.assertIn("/app/mergemind_review.md", markdown)

    def test_reports_missing_iteration_file(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "task_results.json").write_text(
                json.dumps({"results": [{"task_id": "task-1", "metrics": {}}]}),
                encoding="utf-8",
            )

            payload = analyze_run(run_dir)

        self.assertEqual(payload["tasks"][0]["status"], "missing_iteration_file")


if __name__ == "__main__":
    unittest.main()
