"""Compare SWE-CI baseline and MergeMind-assisted run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


_bootstrap_path()


def _load_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "task_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a result list or a 'results' list.")
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("task_id"):
            results[str(row["task_id"])] = row
    return results


def _metric(row: dict[str, Any], name: str) -> Any:
    metrics = row.get("metrics", {})
    return metrics.get(name) if isinstance(metrics, dict) else None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_comparison(baseline_dir: Path, assisted_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = _load_results(baseline_dir)
    assisted = _load_results(assisted_dir)
    task_ids = sorted(set(baseline) | set(assisted))
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        left = baseline.get(task_id, {})
        right = assisted.get(task_id, {})
        baseline_iterations = _number(_metric(left, "actual_iterations"))
        assisted_iterations = _number(_metric(right, "actual_iterations"))
        baseline_final_gap = _number(_metric(left, "final_gap"))
        assisted_final_gap = _number(_metric(right, "final_gap"))
        rows.append(
            {
                "task_id": task_id,
                "baseline_status": left.get("status", ""),
                "assisted_status": right.get("status", ""),
                "baseline_iterations": baseline_iterations,
                "assisted_iterations": assisted_iterations,
                "iteration_delta": (
                    assisted_iterations - baseline_iterations
                    if baseline_iterations is not None and assisted_iterations is not None
                    else None
                ),
                "baseline_final_gap": baseline_final_gap,
                "assisted_final_gap": assisted_final_gap,
                "final_gap_delta": (
                    assisted_final_gap - baseline_final_gap
                    if baseline_final_gap is not None and assisted_final_gap is not None
                    else None
                ),
                "baseline_best_gap": _number(_metric(left, "best_gap")),
                "assisted_best_gap": _number(_metric(right, "best_gap")),
                "assisted_comments": _metric(right, "mergemind_assist_comment_count"),
                "assisted_revisions": _metric(right, "mergemind_assist_revision_count"),
                "baseline_duration": _number(left.get("duration_seconds")),
                "assisted_duration": _number(right.get("duration_seconds")),
            }
        )

    iteration_deltas = [row["iteration_delta"] for row in rows if row["iteration_delta"] is not None]
    final_gap_deltas = [row["final_gap_delta"] for row in rows if row["final_gap_delta"] is not None]
    summary = {
        "task_count": len(rows),
        "compared_task_count": len(iteration_deltas),
        "mean_iteration_delta": sum(iteration_deltas) / len(iteration_deltas) if iteration_deltas else None,
        "mean_final_gap_delta": sum(final_gap_deltas) / len(final_gap_deltas) if final_gap_deltas else None,
        "assisted_comment_count": sum(int(row["assisted_comments"] or 0) for row in rows),
        "assisted_revision_count": sum(int(row["assisted_revisions"] or 0) for row in rows),
    }
    return rows, summary


def render_markdown(baseline_dir: Path, assisted_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# SWE-CI Baseline vs MergeMind Assisted",
        "",
        f"- baseline: `{baseline_dir}`",
        f"- assisted: `{assisted_dir}`",
        f"- tasks: {summary['task_count']}",
        f"- compared tasks: {summary['compared_task_count']}",
        f"- mean iteration delta: {_fmt(summary['mean_iteration_delta'])}",
        f"- mean final gap delta: {_fmt(summary['mean_final_gap_delta'])}",
        f"- assisted comments: {summary['assisted_comment_count']}",
        f"- assisted revisions: {summary['assisted_revision_count']}",
        "",
        "Negative deltas mean the assisted run used fewer iterations or ended with a smaller gap.",
        "",
        "| task_id | baseline | assisted | base_iter | assist_iter | iter_delta | base_gap | assist_gap | gap_delta | comments | revisions |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {task_id} | {baseline_status} | {assisted_status} | {baseline_iterations} | "
            "{assisted_iterations} | {iteration_delta} | {baseline_final_gap} | {assisted_final_gap} | "
            "{final_gap_delta} | {assisted_comments} | {assisted_revisions} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SWE-CI baseline and MergeMind-assisted runs.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--assisted-run-dir", required=True)
    parser.add_argument("--output", default="", help="Optional markdown output path.")
    parser.add_argument("--json-output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_run_dir).resolve()
    assisted_dir = Path(args.assisted_run_dir).resolve()
    rows, summary = build_comparison(baseline_dir, assisted_dir)
    markdown = render_markdown(baseline_dir, assisted_dir, rows, summary)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
