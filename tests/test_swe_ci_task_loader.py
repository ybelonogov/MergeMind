from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.task_loader import SweCiTaskLoaderError, load_swe_ci_tasks


class SweCiTaskLoaderTests(unittest.TestCase):
    def test_load_valid_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "repo_name": "owner/repo",
                        "repo_url": "https://github.com/owner/repo",
                        "current_sha": "abc",
                        "target_sha": "def",
                        "image_sha": "sha256:image",
                        "test_gap": {"failed": 3},
                        "splitting": "smoke",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            tasks = load_swe_ci_tasks(path)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "task-1")
        self.assertEqual(tasks[0].repo_url, "https://github.com/owner/repo")
        self.assertEqual(tasks[0].metadata["splitting"], "smoke")

    def test_accepts_url_alias_for_repo_url(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "task-1",
                            "repo_name": "owner/repo",
                            "url": "https://github.com/owner/repo",
                            "current_sha": "abc",
                            "target_sha": "def",
                            "image_sha": "sha256:image",
                            "test_gap": ["test_a"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            tasks = load_swe_ci_tasks(path)

        self.assertEqual(tasks[0].repo_url, "https://github.com/owner/repo")

    def test_missing_required_field_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "repo_url": "https://github.com/owner/repo",
                        "current_sha": "abc",
                        "target_sha": "def",
                        "image_sha": "sha256:image",
                        "test_gap": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SweCiTaskLoaderError, "repo_name"):
                load_swe_ci_tasks(path)

    def test_duplicate_task_id_fails(self) -> None:
        row = {
            "task_id": "task-1",
            "repo_name": "owner/repo",
            "repo_url": "https://github.com/owner/repo",
            "current_sha": "abc",
            "target_sha": "def",
            "image_sha": "sha256:image",
            "test_gap": {},
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(SweCiTaskLoaderError, "Duplicate"):
                load_swe_ci_tasks(path)

    def test_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            rows = []
            for index in range(3):
                rows.append(
                    {
                        "task_id": f"task-{index}",
                        "repo_name": "owner/repo",
                        "repo_url": "https://github.com/owner/repo",
                        "current_sha": "abc",
                        "target_sha": "def",
                        "image_sha": "sha256:image",
                        "test_gap": {},
                    }
                )
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            tasks = load_swe_ci_tasks(path, limit=1)

        self.assertEqual([task.task_id for task in tasks], ["task-0"])


if __name__ == "__main__":
    unittest.main()
