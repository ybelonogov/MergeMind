from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_swe_ci_pair_manifest import select_tasks, write_chunks, write_manifest
from src.validation.swe_ci.schemas import SweCiTask


def _task(task_id: str, gap: int) -> SweCiTask:
    return SweCiTask(
        task_id=task_id,
        repo_name="owner/repo",
        repo_url="https://github.com/owner/repo",
        current_sha="abc",
        target_sha="def",
        image_sha="sha256:image",
        test_gap=str(gap),
    )


class PrepareSweCiPairManifestTests(unittest.TestCase):
    def test_selects_unique_tasks_with_gap_filter_and_writes_chunks(self) -> None:
        source = [_task("task-1", 1), _task("task-2", 5), _task("task-2", 5), _task("task-3", 9), _task("task-4", 12)]

        selected = select_tasks(
            source,
            limit=2,
            excluded_task_ids={"task-1"},
            min_gap=5,
            max_gap=10,
        )

        self.assertEqual([task.task_id for task in selected], ["task-2", "task-3"])

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest = output_dir / "pair.jsonl"
            write_manifest(selected, manifest)
            chunk_paths = write_chunks(selected, output_dir, "pair", chunk_size=1)

            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], ["task-2", "task-3"])
        self.assertEqual([path.name for path in chunk_paths], ["pair_chunk_01.jsonl", "pair_chunk_02.jsonl"])


if __name__ == "__main__":
    unittest.main()
