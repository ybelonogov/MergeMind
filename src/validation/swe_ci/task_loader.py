"""Load SWE-CI task manifests without running benchmark code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import SweCiTask

_REQUIRED_FIELDS = ("task_id", "repo_name", "current_sha", "target_sha", "image_sha", "test_gap")
_KNOWN_FIELDS = set(_REQUIRED_FIELDS) | {"repo_url", "url"}


class SweCiTaskLoaderError(ValueError):
    """Raised when a SWE-CI task manifest is malformed."""


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SweCiTaskLoaderError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise SweCiTaskLoaderError(f"{path}:{line_number} must contain a JSON object.")
                rows.append(value)
        return rows

    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SweCiTaskLoaderError(f"{path} is not valid JSON: {exc}") from exc
        if isinstance(payload, list):
            if not all(isinstance(row, dict) for row in payload):
                raise SweCiTaskLoaderError(f"{path} must contain JSON objects.")
            return list(payload)
        if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            tasks = payload["tasks"]
            if not all(isinstance(row, dict) for row in tasks):
                raise SweCiTaskLoaderError(f"{path} field 'tasks' must contain JSON objects.")
            return list(tasks)
        raise SweCiTaskLoaderError(f"{path} must be a JSON array or an object with a 'tasks' array.")

    raise SweCiTaskLoaderError(f"Unsupported tasks file extension '{path.suffix}'. Use .jsonl or .json.")


def _require_string(row: dict[str, Any], field_name: str, row_number: int) -> str:
    value = row.get(field_name)
    if value is None or str(value).strip() == "":
        raise SweCiTaskLoaderError(f"Task row {row_number} is missing required field '{field_name}'.")
    return str(value)


def _normalize_task(row: dict[str, Any], row_number: int) -> SweCiTask:
    for field_name in _REQUIRED_FIELDS:
        if field_name not in row:
            raise SweCiTaskLoaderError(f"Task row {row_number} is missing required field '{field_name}'.")

    repo_url = row.get("repo_url", row.get("url"))
    if repo_url is None or str(repo_url).strip() == "":
        raise SweCiTaskLoaderError("Task row {row_number} is missing required field 'repo_url' or 'url'.".format(row_number=row_number))

    metadata = {key: value for key, value in row.items() if key not in _KNOWN_FIELDS}
    return SweCiTask(
        task_id=_require_string(row, "task_id", row_number),
        repo_name=_require_string(row, "repo_name", row_number),
        repo_url=str(repo_url),
        current_sha=_require_string(row, "current_sha", row_number),
        target_sha=_require_string(row, "target_sha", row_number),
        image_sha=_require_string(row, "image_sha", row_number),
        test_gap=row["test_gap"],
        metadata=metadata,
    )


def load_swe_ci_tasks(path: str | Path, limit: int | None = None) -> list[SweCiTask]:
    tasks_path = Path(path)
    if not tasks_path.exists():
        raise SweCiTaskLoaderError(f"SWE-CI tasks file does not exist: {tasks_path}")
    if limit is not None and limit < 0:
        raise SweCiTaskLoaderError("limit must be >= 0 when provided.")

    rows = _read_rows(tasks_path)
    if limit is not None:
        rows = rows[:limit]

    tasks = [_normalize_task(row, index) for index, row in enumerate(rows, start=1)]
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise SweCiTaskLoaderError(f"Duplicate SWE-CI task_id: {task.task_id}")
        seen.add(task.task_id)
    return tasks
