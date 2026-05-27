"""Summarize a SWE-CI baseline against multiple MergeMind-assisted runs."""

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

from scripts.compare_swe_ci_runs import build_comparison  # noqa: E402


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_grid_summary(baseline_dir: Path, assisted_dirs: list[Path]) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for assisted_dir in assisted_dirs:
        rows, summary = build_comparison(baseline_dir, assisted_dir)
        configs.append(
            {
                "name": assisted_dir.name,
                "assisted_dir": str(assisted_dir),
                "summary": summary,
                "rows": rows,
            }
        )
    configs.sort(
        key=lambda item: (
            item["summary"].get("mean_final_gap_delta") if item["summary"].get("mean_final_gap_delta") is not None else 9999,
            item["summary"].get("mean_same_or_lower_gap_iteration_delta")
            if item["summary"].get("mean_same_or_lower_gap_iteration_delta") is not None
            else 9999,
            item["summary"].get("assisted_total_tokens") or 0,
        )
    )
    return {"baseline_dir": str(baseline_dir), "configs": configs}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SWE-CI Caveman Grid Summary",
        "",
        f"- baseline: `{payload['baseline_dir']}`",
        "",
        "| config | tasks | mean_gap_delta | evoscore_delta | invalid_final | invalid_iters | same/lower_iter_delta | failed_jaccard | fixed | new | comments | revisions | tokens | review_tokens | llm_calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["configs"]:
        summary = item["summary"]
        lines.append(
            "| {name} | {task_count} | {gap_delta} | {evoscore_delta} | {invalid_final} | {invalid_iters} | {iter_delta} | {jaccard} | {fixed} | {new} | "
            "{comments} | {revisions} | {tokens} | {review_tokens} | {calls} |".format(
                name=item["name"],
                task_count=_fmt(summary.get("task_count")),
                gap_delta=_fmt(summary.get("mean_final_gap_delta")),
                evoscore_delta=_fmt(summary.get("mean_official_evoscore_delta")),
                invalid_final=_fmt(summary.get("invalid_final_gap_count")),
                invalid_iters=_fmt(summary.get("assisted_invalid_iteration_count")),
                iter_delta=_fmt(summary.get("mean_same_or_lower_gap_iteration_delta")),
                jaccard=_fmt(summary.get("mean_failed_set_jaccard_vs_baseline")),
                fixed=_fmt(summary.get("fixed_failure_count")),
                new=_fmt(summary.get("new_failure_count")),
                comments=_fmt(summary.get("assisted_comment_count")),
                revisions=_fmt(summary.get("assisted_revision_count")),
                tokens=_fmt(summary.get("assisted_total_tokens")),
                review_tokens=_fmt(summary.get("assisted_review_tokens")),
                calls=_fmt(summary.get("assisted_llm_call_count")),
            )
        )
    lines.extend(["", "## Winner Rule", ""])
    if payload["configs"]:
        best = payload["configs"][0]
        lines.append(
            "The table is sorted by lower mean final gap delta, then lower same/lower-gap iteration delta, "
            "then lower token usage. Inspect per-task rows before declaring a real winner."
        )
        lines.append(f"Current top-ranked config by this rule: `{best['name']}`.")
    else:
        lines.append("No assisted runs were provided.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a SWE-CI caveman prompt grid.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--assisted-run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_run_dir).resolve()
    assisted_dirs = [Path(item).resolve() for item in args.assisted_run_dirs]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_grid_summary(baseline_dir, assisted_dirs)
    (output_dir / "summary.md").write_text(render_markdown(payload), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
