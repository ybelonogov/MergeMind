"""Run artifact writer for SWE-CI validation."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.io import write_json

from .schemas import SweCiRunConfig, SweCiTask, SweCiTaskRunResult

_SECRET_KEYS = {"api_key", "hf_token", "token", "password", "secret"}


def _redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***" if key.lower() in _SECRET_KEYS else _redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


def run_dir_for(config: SweCiRunConfig) -> Path:
    return config.output_dir / config.run_id


def write_run_inputs(run_dir: str | Path, config: SweCiRunConfig, tasks: list[SweCiTask]) -> None:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "run_config.json", _redact_payload(config.to_dict()))
    write_json(directory / "tasks.json", {"tasks": [_redact_payload(task.to_dict()) for task in tasks]})


def append_run_event(run_dir: str | Path, payload: dict[str, Any]) -> None:
    event_path = Path(run_dir) / "events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def compute_metrics(results: list[SweCiTaskRunResult]) -> dict[str, Any]:
    counts = Counter(result.status for result in results)
    durations = [result.duration_seconds for result in results]
    task_count = len(results)
    review_payloads = [
        result.metrics.get("mergemind_review")
        for result in results
        if isinstance(result.metrics.get("mergemind_review"), dict)
    ]
    review_status_counts = Counter(str(payload.get("status", "")) for payload in review_payloads)
    return {
        "task_count": task_count,
        "success": counts.get("success", 0),
        "failed": counts.get("failed", 0),
        "timeout": counts.get("timeout", 0),
        "skipped": counts.get("skipped", 0),
        "pass_rate": counts.get("success", 0) / task_count if task_count else 0.0,
        "average_duration_seconds": sum(durations) / task_count if task_count else 0.0,
        "total_duration_seconds": sum(durations),
        "mergemind_reviewed": review_status_counts.get("success", 0),
        "mergemind_review_skipped": review_status_counts.get("skipped", 0),
        "mergemind_comment_count": sum(int(payload.get("comment_count", 0)) for payload in review_payloads),
    }


def _relative_link(run_dir: Path, path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(run_dir.resolve())).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def render_summary(run_id: str, run_dir: Path, results: list[SweCiTaskRunResult], metrics: dict[str, Any]) -> str:
    lines = [
        f"# SWE-CI Run Summary: {run_id}",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Tasks: {metrics['task_count']}",
        f"- Success: {metrics['success']}",
        f"- Failed: {metrics['failed']}",
        f"- Timeout: {metrics['timeout']}",
        f"- Skipped: {metrics['skipped']}",
        f"- Pass rate: {metrics['pass_rate']:.3f}",
        f"- Average duration seconds: {metrics['average_duration_seconds']:.3f}",
        f"- MergeMind reviewed tasks: {metrics['mergemind_reviewed']}",
        f"- MergeMind skipped reviews: {metrics['mergemind_review_skipped']}",
        f"- MergeMind comments: {metrics['mergemind_comment_count']}",
        "",
        "## Tasks",
        "",
        "| task_id | status | duration_sec | exit_code | mergemind_review | comments | stdout | stderr | error |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for result in results:
        stdout_link = _relative_link(run_dir, result.stdout_path)
        stderr_link = _relative_link(run_dir, result.stderr_path)
        review = result.metrics.get("mergemind_review") if isinstance(result.metrics, dict) else None
        review_status = str(review.get("status", "")) if isinstance(review, dict) else ""
        review_comments = int(review.get("comment_count", 0)) if isinstance(review, dict) else 0
        error = result.error_message.replace("|", "\\|") if result.error_message else ""
        lines.append(
            f"| {result.task_id} | {result.status} | {result.duration_seconds:.3f} | "
            f"{'' if result.exit_code is None else result.exit_code} | "
            f"{review_status} | {review_comments} | [stdout]({stdout_link}) | [stderr]({stderr_link}) | {error} |"
        )

    errors = [result for result in results if result.error_message]
    lines.extend(["", "## Errors", ""])
    if errors:
        for result in errors:
            lines.append(f"- `{result.task_id}`: {result.error_message}")
    else:
        lines.append("- No task errors were reported.")

    conclusion = "At least one SWE-CI task failed; inspect logs before using this run as a benchmark signal."
    if results and metrics["success"] == metrics["task_count"]:
        conclusion = "All SWE-CI tasks reported success."
    elif not results:
        conclusion = "No SWE-CI tasks were executed."
    lines.extend(["", "## Conclusion", "", conclusion, ""])
    return "\n".join(lines)


def write_report(run_dir: str | Path, run_id: str, results: list[SweCiTaskRunResult]) -> dict[str, Any]:
    directory = Path(run_dir)
    metrics = compute_metrics(results)
    write_json(directory / "task_results.json", {"results": [result.to_dict() for result in results]})
    write_json(directory / "metrics.json", metrics)
    (directory / "summary.md").write_text(render_summary(run_id, directory, results, metrics), encoding="utf-8")
    return metrics
