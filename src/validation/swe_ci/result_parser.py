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


def _phase_failed(payload: Any) -> bool:
    return isinstance(payload, dict) and str(payload.get("outcome", "")).lower() not in {"", "passed", "skipped"}


def _test_failed(test: dict[str, Any]) -> bool:
    if str(test.get("outcome", "")).lower() not in {"", "passed"}:
        return True
    return any(_phase_failed(test.get(phase)) for phase in ("setup", "call", "teardown"))


def _failed_nodeids_from_report(path: Path) -> list[str]:
    payload = _load_json(path)
    if payload is None:
        return []
    tests = payload.get("tests", [])
    if not isinstance(tests, list):
        return []
    nodeids: set[str] = set()
    for test in tests:
        if not isinstance(test, dict) or not _test_failed(test):
            continue
        nodeid = str(test.get("nodeid", "")).strip()
        if nodeid:
            nodeids.add(nodeid)
    return sorted(nodeids)


def _iteration_report_paths(iteration_file: Path, gaps: list[int]) -> list[Path | None]:
    task_dir = iteration_file.parent
    report_dirs = [
        path.parent
        for path in task_dir.glob("*/test_report.json")
        if path.parent.name not in {"target"}
    ]
    report_dirs.sort(key=lambda path: path.name)
    reports = [path / "test_report.json" for path in report_dirs]
    current_report = task_dir / "current" / "test_report.json"
    if current_report.exists() and current_report not in reports:
        reports.insert(0, current_report)
    output: list[Path | None] = []
    report_index = 0
    for gap in gaps:
        if gap < 0:
            output.append(None)
            continue
        if report_index < len(reports):
            output.append(reports[report_index])
            report_index += 1
        else:
            output.append(None)
    return output


def _tokens_for_agent(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    total = payload.get("total_tokens")
    if isinstance(total, int):
        return total
    input_tokens = int(payload.get("input_tokens", 0) or payload.get("prompt_tokens", 0) or 0)
    output_tokens = int(payload.get("output_tokens", 0) or payload.get("completion_tokens", 0) or 0)
    return input_tokens + output_tokens


def _review_llm_stats(review: dict[str, Any]) -> dict[str, Any]:
    comments_path = review.get("comments_path")
    if not comments_path:
        return {}
    payload = _load_json(Path(str(comments_path)))
    stats = payload.get("llm_stats", {}) if isinstance(payload, dict) else {}
    return stats if isinstance(stats, dict) else {}


def summarize_iteration_file(path: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        rows = []

    gaps: list[int] = []
    for row in rows:
        try:
            gaps.append(int(row.get("gap")))
        except (TypeError, ValueError):
            continue
    nonnegative_gaps = [gap for gap in gaps if gap >= 0]
    regressions = 0
    previous: int | None = None
    for gap in nonnegative_gaps:
        if previous is not None and gap > previous:
            regressions += 1
        previous = gap
    reviews = [row.get("mergemind_review") for row in rows if isinstance(row.get("mergemind_review"), dict)]
    revisions = [row.get("programmer_revision") for row in rows if isinstance(row.get("programmer_revision"), dict)]
    report_paths = _iteration_report_paths(Path(path), gaps)
    failed_test_nodeids_by_iteration = [
        _failed_nodeids_from_report(report_path) if report_path is not None else []
        for report_path in report_paths
    ]
    coding_tokens = sum(
        _tokens_for_agent(row.get("architect")) + _tokens_for_agent(row.get("programmer"))
        for row in rows
    )
    revision_tokens = sum(_tokens_for_agent(row.get("programmer_revision")) for row in rows)
    review_stats = [_review_llm_stats(review) for review in reviews]
    mergemind_review_tokens = sum(int(stats.get("total_tokens", 0) or 0) for stats in review_stats)
    llm_call_count = sum(int(stats.get("llm_call_count", 0) or 0) for stats in review_stats)
    parse_error_count = sum(
        float(stats.get("parse_error_rate", 0.0) or 0.0) * int(stats.get("llm_call_count", 0) or 0)
        for stats in review_stats
    )
    total_tokens = coding_tokens + revision_tokens + mergemind_review_tokens
    successful_revision_count = len(revisions)
    return {
        "swe_ci_iteration_count": len(rows),
        "actual_iterations": max(0, len(rows) - 1),
        "gap_sequence": gaps,
        "initial_gap": gaps[0] if gaps else None,
        "final_gap": gaps[-1] if gaps else None,
        "best_gap": min(nonnegative_gaps) if nonnegative_gaps else None,
        "gap_zero": any(gap == 0 for gap in gaps),
        "regressions_count": regressions,
        "invalid_iteration_count": sum(1 for gap in gaps if gap < 0),
        "mergemind_assist_review_count": len(reviews),
        "mergemind_assist_success_count": sum(1 for review in reviews if review.get("status") == "success"),
        "mergemind_assist_comment_count": sum(int(review.get("comment_count", 0) or 0) for review in reviews),
        "mergemind_assist_revision_count": len(revisions),
        "failed_test_nodeids_by_iteration": failed_test_nodeids_by_iteration,
        "failed_test_counts_by_iteration": [len(nodeids) for nodeids in failed_test_nodeids_by_iteration],
        "coding_tokens": coding_tokens,
        "revision_tokens": revision_tokens,
        "mergemind_review_tokens": mergemind_review_tokens,
        "total_tokens": total_tokens,
        "llm_call_count": llm_call_count,
        "parse_error_rate": parse_error_count / llm_call_count if llm_call_count else 0.0,
        "tokens_per_successful_revision": (
            total_tokens / successful_revision_count if successful_revision_count else None
        ),
    }


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
    iteration_file = locate_swe_ci_iteration_file(
        swe_ci_repo_path=swe_ci_repo_path,
        experiment_name=experiment_name,
        task_id=process_result.task_id,
    )
    if iteration_file is not None:
        metrics["swe_ci_iteration_file"] = str(iteration_file)
        metrics.update(summarize_iteration_file(iteration_file))
    if result_file is None:
        if iteration_file is not None:
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
