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


def _int_number(value: Any) -> int | None:
    numeric = _number(value)
    return int(numeric) if numeric is not None else None


def _valid_gap(value: Any) -> int | None:
    gap = _int_number(value)
    return gap if gap is not None and gap >= 0 else None


def _gap_sequence(row: dict[str, Any]) -> list[int]:
    value = _metric(row, "gap_sequence")
    if not isinstance(value, list):
        return []
    gaps: list[int] = []
    for item in value:
        try:
            gaps.append(int(item))
        except (TypeError, ValueError):
            continue
    return gaps


def _first_iter_to_gap(gaps: list[int], target_gap: int | None, *, same_or_lower: bool) -> int | None:
    if target_gap is None:
        return None
    for index, gap in enumerate(gaps):
        if gap < 0:
            continue
        if gap == target_gap or (same_or_lower and gap <= target_gap):
            return index
    return None


def _final_failed_set(row: dict[str, Any]) -> set[str] | None:
    value = _metric(row, "failed_test_nodeids_by_iteration")
    if not isinstance(value, list) or not value:
        return None
    final_value = value[-1]
    if not isinstance(final_value, list):
        return None
    return {str(item) for item in final_value}


def _jaccard(left: set[str] | None, right: set[str] | None) -> float | None:
    if left is None or right is None:
        return None
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _delta(left: int | None, right: int | None) -> int | None:
    return right - left if left is not None and right is not None else None


def _first_iter_to_best_gap(gaps: list[int]) -> int | None:
    valid_gaps = [gap for gap in gaps if gap >= 0]
    if not valid_gaps:
        return None
    best_gap = min(valid_gaps)
    return _first_iter_to_gap(gaps, best_gap, same_or_lower=False)


def _classify_result(row: dict[str, Any]) -> str:
    if not row["baseline_status"] or not row["assisted_status"]:
        return "incomplete"

    baseline_best = row["baseline_best_gap"]
    assisted_best = row["assisted_best_gap"]
    baseline_final = row["baseline_final_gap"]
    assisted_final = row["assisted_final_gap"]
    baseline_best_iter = row["baseline_iterations_to_best_gap"]
    assisted_best_iter = row["assisted_iterations_to_best_gap"]
    new_failures = int(row["new_failure_count"] or 0)

    if assisted_final is None and baseline_final is not None:
        return "worse"
    if baseline_best is not None and assisted_best is not None and assisted_best < baseline_best:
        return "improved"
    if baseline_final is not None and assisted_final is not None and assisted_final < baseline_final:
        return "improved"
    if (
        baseline_best is not None
        and assisted_best is not None
        and assisted_best == baseline_best
        and baseline_best_iter is not None
        and assisted_best_iter is not None
        and assisted_best_iter < baseline_best_iter
        and new_failures == 0
        and (baseline_final is None or assisted_final is None or assisted_final <= baseline_final)
    ):
        return "improved"

    if baseline_best is not None and assisted_best is not None and assisted_best > baseline_best:
        return "worse"
    if baseline_final is not None and assisted_final is not None and assisted_final > baseline_final:
        return "worse"
    if baseline_final == assisted_final and new_failures > 0:
        return "worse"
    if row["assisted_invalid_iteration_count"] > row["baseline_invalid_iteration_count"]:
        return "worse"
    return "unchanged"


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
        baseline_final_gap_raw = _int_number(_metric(left, "final_gap"))
        assisted_final_gap_raw = _int_number(_metric(right, "final_gap"))
        baseline_final_gap = _valid_gap(_metric(left, "final_gap"))
        assisted_final_gap = _valid_gap(_metric(right, "final_gap"))
        baseline_gaps = _gap_sequence(left)
        assisted_gaps = _gap_sequence(right)
        baseline_iterations_to_best = _first_iter_to_best_gap(baseline_gaps)
        assisted_iterations_to_best = _first_iter_to_best_gap(assisted_gaps)
        baseline_first_same = _first_iter_to_gap(baseline_gaps, baseline_final_gap, same_or_lower=False)
        assisted_first_same = _first_iter_to_gap(assisted_gaps, baseline_final_gap, same_or_lower=False)
        baseline_first_same_or_lower = _first_iter_to_gap(baseline_gaps, baseline_final_gap, same_or_lower=True)
        assisted_first_same_or_lower = _first_iter_to_gap(assisted_gaps, baseline_final_gap, same_or_lower=True)
        baseline_failed_set = _final_failed_set(left)
        assisted_failed_set = _final_failed_set(right)
        fixed_failures = len(baseline_failed_set - assisted_failed_set) if baseline_failed_set is not None and assisted_failed_set is not None else None
        new_failures = len(assisted_failed_set - baseline_failed_set) if baseline_failed_set is not None and assisted_failed_set is not None else None
        assisted_total_tokens = _number(_metric(right, "total_tokens"))
        baseline_official_evoscore = _number(_metric(left, "official_evoscore"))
        assisted_official_evoscore = _number(_metric(right, "official_evoscore"))
        baseline_official_solved_rate = _number(_metric(left, "official_solved_rate"))
        assisted_official_solved_rate = _number(_metric(right, "official_solved_rate"))
        row = {
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
            "baseline_final_gap_raw": baseline_final_gap_raw,
            "assisted_final_gap_raw": assisted_final_gap_raw,
            "baseline_final_gap_valid": baseline_final_gap is not None,
            "assisted_final_gap_valid": assisted_final_gap is not None,
            "baseline_invalid_iteration_count": _int_number(_metric(left, "invalid_iteration_count")) or 0,
            "assisted_invalid_iteration_count": _int_number(_metric(right, "invalid_iteration_count")) or 0,
            "final_gap_delta": (
                assisted_final_gap - baseline_final_gap
                if baseline_final_gap is not None and assisted_final_gap is not None
                else None
            ),
            "baseline_best_gap": _number(_metric(left, "best_gap")),
            "assisted_best_gap": _number(_metric(right, "best_gap")),
            "baseline_gap_sequence": baseline_gaps,
            "assisted_gap_sequence": assisted_gaps,
            "baseline_iterations_to_best_gap": baseline_iterations_to_best,
            "assisted_iterations_to_best_gap": assisted_iterations_to_best,
            "iterations_to_best_gap_delta": _delta(baseline_iterations_to_best, assisted_iterations_to_best),
            "baseline_first_iter_to_final_gap": baseline_first_same,
            "first_iter_to_same_gap": assisted_first_same,
            "same_gap_iteration_delta": _delta(baseline_first_same, assisted_first_same),
            "baseline_first_iter_to_same_or_lower_gap": baseline_first_same_or_lower,
            "first_iter_to_same_or_lower_gap": assisted_first_same_or_lower,
            "same_or_lower_gap_iteration_delta": _delta(baseline_first_same_or_lower, assisted_first_same_or_lower),
            "failed_set_jaccard_vs_baseline": _jaccard(baseline_failed_set, assisted_failed_set),
            "new_failure_count": new_failures,
            "fixed_failure_count": fixed_failures,
            "same_gap_same_tests": (
                baseline_final_gap == assisted_final_gap
                and baseline_failed_set is not None
                and assisted_failed_set is not None
                and baseline_failed_set == assisted_failed_set
            ),
            "assisted_comments": _metric(right, "mergemind_assist_comment_count"),
            "assisted_revisions": _metric(right, "mergemind_assist_revision_count"),
            "baseline_total_tokens": _number(_metric(left, "total_tokens")),
            "assisted_total_tokens": assisted_total_tokens,
            "assisted_review_tokens": _number(_metric(right, "mergemind_review_tokens")),
            "assisted_llm_call_count": _number(_metric(right, "llm_call_count")),
            "baseline_official_evoscore": baseline_official_evoscore,
            "assisted_official_evoscore": assisted_official_evoscore,
            "official_evoscore_delta": (
                assisted_official_evoscore - baseline_official_evoscore
                if assisted_official_evoscore is not None and baseline_official_evoscore is not None
                else None
            ),
            "baseline_official_solved_rate": baseline_official_solved_rate,
            "assisted_official_solved_rate": assisted_official_solved_rate,
            "official_solved_rate_delta": (
                assisted_official_solved_rate - baseline_official_solved_rate
                if assisted_official_solved_rate is not None and baseline_official_solved_rate is not None
                else None
            ),
            "baseline_official_zero_regression": _number(_metric(left, "official_zero_regression")),
            "assisted_official_zero_regression": _number(_metric(right, "official_zero_regression")),
            "tokens_per_gap_delta": (
                assisted_total_tokens / abs(assisted_final_gap - baseline_final_gap)
                if assisted_total_tokens is not None
                and baseline_final_gap is not None
                and assisted_final_gap is not None
                and assisted_final_gap < baseline_final_gap
                else None
            ),
            "tokens_per_fixed_failure": (
                assisted_total_tokens / fixed_failures
                if assisted_total_tokens is not None and fixed_failures
                else None
            ),
            "baseline_duration": _number(left.get("duration_seconds")),
            "assisted_duration": _number(right.get("duration_seconds")),
        }
        row["result_label"] = _classify_result(row)
        rows.append(row)

    iteration_deltas = [row["iteration_delta"] for row in rows if row["iteration_delta"] is not None]
    final_gap_deltas = [row["final_gap_delta"] for row in rows if row["final_gap_delta"] is not None]
    same_or_lower_deltas = [
        row["same_or_lower_gap_iteration_delta"]
        for row in rows
        if row["same_or_lower_gap_iteration_delta"] is not None
    ]
    jaccards = [
        row["failed_set_jaccard_vs_baseline"]
        for row in rows
        if row["failed_set_jaccard_vs_baseline"] is not None
    ]
    evoscore_deltas = [
        row["official_evoscore_delta"]
        for row in rows
        if row["official_evoscore_delta"] is not None
    ]
    solved_rate_deltas = [
        row["official_solved_rate_delta"]
        for row in rows
        if row["official_solved_rate_delta"] is not None
    ]
    best_iter_deltas = [
        row["iterations_to_best_gap_delta"]
        for row in rows
        if row["iterations_to_best_gap_delta"] is not None
    ]
    result_counts = {
        label: sum(1 for row in rows if row["result_label"] == label)
        for label in ("improved", "worse", "unchanged", "incomplete")
    }
    summary = {
        "task_count": len(rows),
        "compared_task_count": len(iteration_deltas),
        "mean_iteration_delta": sum(iteration_deltas) / len(iteration_deltas) if iteration_deltas else None,
        "mean_final_gap_delta": sum(final_gap_deltas) / len(final_gap_deltas) if final_gap_deltas else None,
        "mean_same_or_lower_gap_iteration_delta": (
            sum(same_or_lower_deltas) / len(same_or_lower_deltas) if same_or_lower_deltas else None
        ),
        "mean_iterations_to_best_gap_delta": (
            sum(best_iter_deltas) / len(best_iter_deltas) if best_iter_deltas else None
        ),
        "mean_failed_set_jaccard_vs_baseline": sum(jaccards) / len(jaccards) if jaccards else None,
        "mean_official_evoscore_delta": sum(evoscore_deltas) / len(evoscore_deltas) if evoscore_deltas else None,
        "mean_official_solved_rate_delta": (
            sum(solved_rate_deltas) / len(solved_rate_deltas) if solved_rate_deltas else None
        ),
        "new_failure_count": sum(int(row["new_failure_count"] or 0) for row in rows),
        "fixed_failure_count": sum(int(row["fixed_failure_count"] or 0) for row in rows),
        "assisted_comment_count": sum(int(row["assisted_comments"] or 0) for row in rows),
        "assisted_revision_count": sum(int(row["assisted_revisions"] or 0) for row in rows),
        "baseline_total_tokens": sum(float(row["baseline_total_tokens"] or 0) for row in rows),
        "assisted_total_tokens": sum(float(row["assisted_total_tokens"] or 0) for row in rows),
        "assisted_review_tokens": sum(float(row["assisted_review_tokens"] or 0) for row in rows),
        "assisted_llm_call_count": sum(float(row["assisted_llm_call_count"] or 0) for row in rows),
        "invalid_final_gap_count": sum(1 for row in rows if not row["assisted_final_gap_valid"]),
        "assisted_invalid_iteration_count": sum(int(row["assisted_invalid_iteration_count"] or 0) for row in rows),
        **{f"{label}_count": count for label, count in result_counts.items()},
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
        f"- mean same/lower gap iteration delta: {_fmt(summary['mean_same_or_lower_gap_iteration_delta'])}",
        f"- mean iterations-to-best-gap delta: {_fmt(summary['mean_iterations_to_best_gap_delta'])}",
        f"- mean failed-set Jaccard vs baseline: {_fmt(summary['mean_failed_set_jaccard_vs_baseline'])}",
        f"- mean official EvoScore delta: {_fmt(summary['mean_official_evoscore_delta'])}",
        f"- mean official solved-rate delta: {_fmt(summary['mean_official_solved_rate_delta'])}",
        f"- improved/worse/unchanged/incomplete: {summary['improved_count']} / {summary['worse_count']} / {summary['unchanged_count']} / {summary['incomplete_count']}",
        f"- fixed failures: {summary['fixed_failure_count']}",
        f"- new failures: {summary['new_failure_count']}",
        f"- assisted comments: {summary['assisted_comment_count']}",
        f"- assisted revisions: {summary['assisted_revision_count']}",
        f"- baseline total tokens: {_fmt(summary['baseline_total_tokens'])}",
        f"- assisted total tokens: {_fmt(summary['assisted_total_tokens'])}",
        f"- assisted review tokens: {_fmt(summary['assisted_review_tokens'])}",
        f"- assisted LLM calls: {_fmt(summary['assisted_llm_call_count'])}",
        f"- invalid assisted final gaps: {summary['invalid_final_gap_count']}",
        f"- assisted invalid iterations: {summary['assisted_invalid_iteration_count']}",
        "",
        "Negative deltas mean the assisted run used fewer iterations or ended with a smaller valid gap.",
        "Rows with invalid final gaps are excluded from final-gap delta averages.",
        "",
        "| task_id | label | baseline | assisted | base_iter | assist_iter | iter_delta | base_gap | assist_gap | gap_delta | base_best_iter | assist_best_iter | best_iter_delta | evoscore_delta | solved_delta | invalid_iters | first_same | first_same/lower | failed_jaccard | fixed | new | same_tests | tokens | review_tokens | comments | revisions |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {task_id} | {result_label} | {baseline_status} | {assisted_status} | {baseline_iterations} | "
            "{assisted_iterations} | {iteration_delta} | {baseline_final_gap} | {assisted_final_gap} | "
            "{final_gap_delta} | {baseline_iterations_to_best_gap} | {assisted_iterations_to_best_gap} | "
            "{iterations_to_best_gap_delta} | {official_evoscore_delta} | {official_solved_rate_delta} | "
            "{assisted_invalid_iteration_count} | "
            "{first_iter_to_same_gap} | {first_iter_to_same_or_lower_gap} | "
            "{failed_set_jaccard_vs_baseline} | {fixed_failure_count} | {new_failure_count} | "
            "{same_gap_same_tests} | {assisted_total_tokens} | {assisted_review_tokens} | "
            "{assisted_comments} | {assisted_revisions} |".format(
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
