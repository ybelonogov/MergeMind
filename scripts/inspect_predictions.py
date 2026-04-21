"""Render MergeMind predictions as a human-readable review report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.config import load_config, resolve_path
from src.data.io import iter_jsonl, read_json, read_jsonl
from src.data.schema import MRExample
from src.inference.factory import canonical_pipeline_mode


def _resolve_mode_dir(runs_dir: Path, run_id: str, mode: str = "") -> Path:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if mode:
        mode_dir = run_dir / canonical_pipeline_mode(mode)
        if not mode_dir.exists():
            raise FileNotFoundError(f"Pipeline mode directory not found: {mode_dir}")
        return mode_dir

    candidates = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and ((path / "predictions.jsonl").exists() or (path / "summary.json").exists())
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No prediction artifacts found under: {run_dir}")

    modes = ", ".join(path.name for path in candidates)
    raise ValueError(f"Run has multiple modes; pass --mode. Available modes: {modes}")


def _load_records(mode_dir: Path) -> list[dict[str, Any]]:
    predictions_path = mode_dir / "predictions.jsonl"
    if predictions_path.exists():
        return read_jsonl(predictions_path)

    summary_path = mode_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        return list(summary.get("examples", []))

    raise FileNotFoundError(f"No predictions.jsonl or summary.json found in: {mode_dir}")


def _load_examples_by_id(prepared_dir: Path, example_ids: set[str]) -> dict[str, MRExample]:
    examples: dict[str, MRExample] = {}
    for file_name in ("validation.jsonl", "test.jsonl", "train.jsonl"):
        path = prepared_dir / file_name
        for row in iter_jsonl(path):
            example_id = str(row.get("example_id", ""))
            if example_id in example_ids and example_id not in examples:
                examples[example_id] = MRExample.from_dict(row)
            if len(examples) == len(example_ids):
                return examples

    demo_path = prepared_dir / "demo.json"
    if demo_path.exists():
        row = read_json(demo_path)
        example_id = str(row.get("example_id", ""))
        if example_id in example_ids and example_id not in examples:
            examples[example_id] = MRExample.from_dict(row)
    return examples


def _clip_lines(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    clipped = lines[:max_lines]
    suffix = "" if len(lines) <= max_lines else f"\n... clipped {len(lines) - max_lines} lines ..."
    return "\n".join(clipped) + suffix


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _split_evidence(evidence: list[str]) -> tuple[list[str], list[str]]:
    reasons = []
    signals = []
    for item in evidence:
        if item.startswith("reason="):
            reasons.append(item.removeprefix("reason="))
        elif "=" in item:
            signals.append(item)
    return reasons, signals


def _render_prediction(prediction: dict[str, Any], index: int) -> list[str]:
    evidence = list(prediction.get("evidence", []))
    reasons, signals = _split_evidence(evidence)
    lines = [
        f"### Prediction {index}",
        f"- generator_score: {_format_score(prediction.get('generator_score'))}",
        f"- reranker_score: {_format_score(prediction.get('reranker_score'))}",
        f"- source_example_id: {prediction.get('source_example_id', '')}",
        "",
        str(prediction.get("text", "")).strip(),
    ]
    if reasons:
        lines.extend(["", "Reasons:"])
        lines.extend(f"- {reason}" for reason in reasons)
    if signals:
        lines.extend(["", "Signals:"])
        lines.extend(f"- {signal}" for signal in signals)
    return lines


def render_report(
    records: list[dict[str, Any]],
    examples_by_id: dict[str, MRExample],
    limit: int,
    start: int = 0,
    diff_lines: int = 80,
) -> str:
    selected = records[start : start + limit if limit > 0 else None]
    report_lines = [
        "# MergeMind Prediction Inspection",
        "",
        f"Records shown: {len(selected)}",
    ]

    for record_index, record in enumerate(selected, start=start + 1):
        example_id = str(record.get("example_id", ""))
        example = examples_by_id.get(example_id)
        predictions = list(record.get("predictions", []))
        gold_comments = list(record.get("gold_comments", []))

        report_lines.extend(
            [
                "",
                "---",
                "",
                f"## MR {record_index}: {example_id}",
                f"- source_dataset: {record.get('source_dataset', '')}",
                f"- hit_at_k: {record.get('hit_at_k', 0)}",
                f"- top1_similarity: {_format_score(record.get('top1_similarity'))}",
                f"- best_similarity_at_k: {_format_score(record.get('best_similarity_at_k'))}",
                f"- judge_score: {_format_score(record.get('judge_score'))}",
            ]
        )

        if example is not None:
            changed_paths = ", ".join(changed_file.path for changed_file in example.changed_files[:8])
            report_lines.extend(
                [
                    f"- title: {example.title}",
                    f"- repo: {example.repo}",
                    f"- changed_files: {changed_paths}",
                ]
            )

        report_lines.extend(["", "### Gold comments"])
        if gold_comments:
            report_lines.extend(f"- {comment}" for comment in gold_comments)
        else:
            report_lines.append("- <none>")

        report_lines.extend(["", "### Predictions"])
        if predictions:
            for index, prediction in enumerate(predictions, start=1):
                report_lines.extend(["", *_render_prediction(prediction, index)])
        else:
            report_lines.append("- <none>")

        if example is not None and diff_lines > 0:
            report_lines.extend(["", "### Diff excerpt", "", "```diff"])
            report_lines.append(_clip_lines(example.diff, diff_lines))
            report_lines.append("```")

    return "\n".join(report_lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MergeMind predictions in a human-readable format.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--run", required=True, help="Run id under artifacts/runs.")
    parser.add_argument("--mode", default="", help="Pipeline mode. Required when a run contains multiple modes.")
    parser.add_argument("--limit", type=int, default=5, help="Number of MR records to show. Use 0 for all.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based record offset.")
    parser.add_argument("--diff-lines", type=int, default=80, help="Diff lines to show per MR. Use 0 to hide diff.")
    parser.add_argument("--output", default="", help="Optional path to write the report.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    runs_dir = resolve_path(PROJECT_ROOT, config["paths"].get("runs_dir", "artifacts/runs"))
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    mode_dir = _resolve_mode_dir(runs_dir, args.run, args.mode)
    records = _load_records(mode_dir)
    limit = max(args.limit, 0)
    start = max(args.start, 0)
    selected = records[start : start + limit if limit > 0 else None]
    example_ids = {str(record.get("example_id", "")) for record in selected if record.get("example_id")}
    examples_by_id = _load_examples_by_id(prepared_dir, example_ids)
    report = render_report(records, examples_by_id, limit=limit, start=start, diff_lines=args.diff_lines)

    if args.output:
        output_path = resolve_path(PROJECT_ROOT, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"[inspect_predictions] wrote {output_path}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
