"""Parse real SWE-CI output artifacts when they are available."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import SweCiTaskRunResult

RESULT_FILE_NAMES = (
    "result.json",
    "results.json",
    "summary.json",
    "metrics.json",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def locate_swe_ci_result_file(task_output_dir: str | Path) -> Path | None:
    output_dir = Path(task_output_dir)
    if not output_dir.exists():
        return None
    for file_name in RESULT_FILE_NAMES:
        matches = sorted(output_dir.rglob(file_name))
        if matches:
            return matches[0]
    return None


def locate_swe_ci_iteration_file(
    *,
    swe_ci_repo_path: str | Path | None,
    experiment_name: str | None,
    task_id: str,
) -> Path | None:
    if not swe_ci_repo_path or not experiment_name:
        return None
    iteration_file = Path(swe_ci_repo_path) / "experiments" / experiment_name / task_id / "iteration.jsonl"
    return iteration_file if iteration_file.exists() else None


def _count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _infer_success(payload: dict[str, Any], process_result: SweCiTaskRunResult) -> bool | None:
    for key in ("success", "passed", "pass"):
        if isinstance(payload.get(key), bool):
            return bool(payload[key])
    status = str(payload.get("status", "")).lower()
    if status in {"success", "passed", "pass"}:
        return True
    if status in {"failed", "failure", "error", "timeout"}:
        return False
    if process_result.status == "timeout":
        return False
    return None


def parse_swe_ci_result(
    process_result: SweCiTaskRunResult,
    task_output_dir: str | Path,
    *,
    swe_ci_repo_path: str | Path | None = None,
    experiment_name: str | None = None,
) -> SweCiTaskRunResult:
    if process_result.status == "timeout":
        return process_result

    result_file = locate_swe_ci_result_file(task_output_dir)
    metrics = dict(process_result.metrics)
    metrics["swe_ci_output_dir"] = str(task_output_dir)
    if result_file is None:
        iteration_file = locate_swe_ci_iteration_file(
            swe_ci_repo_path=swe_ci_repo_path,
            experiment_name=experiment_name,
            task_id=process_result.task_id,
        )
        if iteration_file is not None:
            metrics["swe_ci_iteration_file"] = str(iteration_file)
            metrics["swe_ci_iteration_count"] = _count_jsonl_rows(iteration_file)
            status = "success" if process_result.exit_code == 0 else "failed"
            return SweCiTaskRunResult(
                task_id=process_result.task_id,
                status=status,  # type: ignore[arg-type]
                started_at=process_result.started_at,
                finished_at=process_result.finished_at,
                duration_seconds=process_result.duration_seconds,
                exit_code=process_result.exit_code,
                stdout_path=process_result.stdout_path,
                stderr_path=process_result.stderr_path,
                events_path=process_result.events_path,
                metrics=metrics,
                error_message="" if status == "success" else process_result.error_message,
            )
        if process_result.error_message:
            metrics["process_error_message"] = process_result.error_message
        return SweCiTaskRunResult(
            task_id=process_result.task_id,
            status="failed",
            started_at=process_result.started_at,
            finished_at=process_result.finished_at,
            duration_seconds=process_result.duration_seconds,
            exit_code=process_result.exit_code,
            stdout_path=process_result.stdout_path,
            stderr_path=process_result.stderr_path,
            events_path=process_result.events_path,
            metrics=metrics,
            error_message="Could not locate SWE-CI result file",
        )

    payload = _load_json(result_file)
    metrics["swe_ci_result_file"] = str(result_file)
    if payload is not None:
        metrics["swe_ci_result"] = payload
    success = _infer_success(payload or {}, process_result)
    if success is None:
        return SweCiTaskRunResult(
            task_id=process_result.task_id,
            status="failed",
            started_at=process_result.started_at,
            finished_at=process_result.finished_at,
            duration_seconds=process_result.duration_seconds,
            exit_code=process_result.exit_code,
            stdout_path=process_result.stdout_path,
            stderr_path=process_result.stderr_path,
            events_path=process_result.events_path,
            metrics=metrics,
            error_message="Could not infer SWE-CI task status from result file",
        )

    status = "success" if success and process_result.exit_code == 0 else "failed"
    error_message = "" if status == "success" else process_result.error_message or "SWE-CI result indicates failure"
    return SweCiTaskRunResult(
        task_id=process_result.task_id,
        status=status,  # type: ignore[arg-type]
        started_at=process_result.started_at,
        finished_at=process_result.finished_at,
        duration_seconds=process_result.duration_seconds,
        exit_code=process_result.exit_code,
        stdout_path=process_result.stdout_path,
        stderr_path=process_result.stderr_path,
        events_path=process_result.events_path,
        metrics=metrics,
        error_message=error_message,
    )
