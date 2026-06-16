from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.combine_swe_ci_run_chunks import main


def _write_chunk(path: Path, task_id: str) -> None:
    path.mkdir(parents=True)
    (path / "task_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": task_id,
                        "status": "success",
                        "started_at": "",
                        "finished_at": "",
                        "duration_seconds": 1,
                        "exit_code": 0,
                        "stdout_path": str(path / "stdout.log"),
                        "stderr_path": str(path / "stderr.log"),
                        "events_path": str(path / "events.jsonl"),
                        "metrics": {"actual_iterations": 1, "final_gap": 0, "best_gap": 0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class CombineSweCiRunChunksTests(unittest.TestCase):
    def test_combines_chunk_task_results(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_chunk(base / "chunk_01", "task-1")
            _write_chunk(base / "chunk_02", "task-2")
            output_dir = base / "combined"

            exit_code = main(
                [
                    "--chunks-parent",
                    str(base),
                    "--output-dir",
                    str(output_dir),
                    "--run-id",
                    "combined",
                ]
            )

            payload = json.loads((output_dir / "task_results.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["task_id"] for row in payload["results"]], ["task-1", "task-2"])


if __name__ == "__main__":
    unittest.main()
