"""Prepare per-task SWE-CI dataset roots for controlled wrapper runs."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .config import task_dataset_dir
from .schemas import SweCiRunConfig, SweCiTask

_CSV_FIELDS = (
    "task_id",
    "repo_name",
    "url",
    "licence",
    "current_sha",
    "target_sha",
    "test_gap",
    "image_sha",
    "code_sha",
)


class SweCiDatasetError(RuntimeError):
    """Raised when a SWE-CI task dataset root cannot be prepared."""


def _source_data_dir(config: SweCiRunConfig, task: SweCiTask) -> Path:
    data_dir = task.metadata.get("data_dir")
    if data_dir:
        return Path(str(data_dir)).expanduser().resolve()
    return config.swe_ci_repo_path / "data" / task.task_id


def _link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    try:
        destination.symlink_to(source, target_is_directory=True)
        if not (destination / "code.zip").exists():
            destination.unlink()
            shutil.copytree(source, destination)
    except OSError:
        shutil.copytree(source, destination)


def _write_single_task_metadata(path: Path, config: SweCiRunConfig, task: SweCiTask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task_id": task.task_id,
        "repo_name": task.repo_name,
        "url": task.repo_url,
        "licence": str(task.metadata.get("licence", "")),
        "current_sha": task.current_sha,
        "target_sha": task.target_sha,
        "test_gap": str(task.test_gap),
        "image_sha": task.image_sha,
        "code_sha": str(task.metadata.get("code_sha", "")),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        writer.writerow(row)


def prepare_task_dataset_root(config: SweCiRunConfig, task: SweCiTask, run_dir: str | Path) -> Path:
    """Create a minimal SWE-CI save_root_dir containing exactly one task."""

    source = _source_data_dir(config, task)
    if not source.exists():
        raise SweCiDatasetError(
            "SWE-CI task data is missing: "
            f"{source}. Run `PYTHONPATH=src python -m swe_ci.download --splitting {config.splitting}` "
            "or download/copy this task's data folder before running the benchmark."
        )
    if not (source / "code.zip").is_file() or not (source / "image.tar.gz").is_file():
        raise SweCiDatasetError(f"SWE-CI task data is incomplete: expected code.zip and image.tar.gz in {source}.")

    dataset_root = task_dataset_dir(run_dir, task)
    split = str(config.splitting or task.metadata.get("splitting") or "default")
    _write_single_task_metadata(dataset_root / "metadata" / f"{split}.csv", config, task)
    (dataset_root / "data").mkdir(parents=True, exist_ok=True)
    _link_or_copy_tree(source, dataset_root / "data" / task.task_id)
    return dataset_root
