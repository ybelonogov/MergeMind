from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.dataset import SweCiDatasetError, prepare_task_dataset_root
from src.validation.swe_ci.schemas import SweCiRunConfig, SweCiTask


def _config(tmp: str) -> SweCiRunConfig:
    return SweCiRunConfig(
        swe_ci_repo_path=Path(tmp) / "SWE-CI",
        tasks_path=Path(tmp) / "tasks.jsonl",
        output_dir=Path(tmp) / "runs",
        limit=1,
        max_iterations=1,
        timeout_seconds=60,
        mode="baseline",
        run_id="run-1",
        splitting="lite_smoke",
    )


def _task() -> SweCiTask:
    return SweCiTask(
        task_id="owner__repo__abc__def",
        repo_name="owner/repo",
        repo_url="https://github.com/owner/repo.git",
        current_sha="abc",
        target_sha="def",
        image_sha="sha256:image",
        test_gap="3",
        metadata={"licence": "MIT", "code_sha": "sha256:code"},
    )


class SweCiDatasetTests(unittest.TestCase):
    def test_prepare_task_dataset_root_writes_single_task_split(self) -> None:
        with TemporaryDirectory() as tmp:
            config = _config(tmp)
            source = config.swe_ci_repo_path / "data" / _task().task_id
            source.mkdir(parents=True)
            (source / "code.zip").write_bytes(b"code")
            (source / "image.tar.gz").write_bytes(b"image")

            dataset_root = prepare_task_dataset_root(config, _task(), Path(tmp) / "run")
            metadata_path = dataset_root / "metadata" / "lite_smoke.csv"

            with metadata_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task_id"], _task().task_id)
            self.assertTrue((dataset_root / "data" / _task().task_id / "code.zip").exists())

    def test_missing_source_data_fails_before_swe_ci_run(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SweCiDatasetError, "task data is missing"):
                prepare_task_dataset_root(_config(tmp), _task(), Path(tmp) / "run")


if __name__ == "__main__":
    unittest.main()
