"""Compare predictions from multiple pipeline modes in one run."""

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


def _clip(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    clipped = lines[:max_lines]
    suffix = "" if len(lines) <= max_lines else f"\n... clipped {len(lines) - max_lines} lines ..."
    return "\n".join(clipped) + suffix


def _format_score(value: Any) -> str:
    if value in {"", None}:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _escape_table(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _load_records(mode_dir: Path) -> list[dict[str, Any]]:
    predictions_path = mode_dir / "predictions.jsonl"
    if predictions_path.exists():
        return read_jsonl(predictions_path)
    summary_path = mode_dir / "summary.json"
    if summary_path.exists():
        return list(read_json(summary_path).get("examples", []))
    raise FileNotFoundError(f"No predictions found in {mode_dir}")


def _discover_modes(run_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and ((path / "predictions.jsonl").exists() or (path / "summary.json").exists())
    )


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
    return examples


def _judge_field(record: dict[str, Any], field_name: str) -> Any:
    judge = record.get("judge", {})
    if isinstance(judge, dict) and field_name in judge:
        return judge.get(field_name)
    return record.get(field_name, "")


def _top_prediction(record: dict[str, Any]) -> str:
    predictions = list(record.get("predictions", []))
    if not predictions:
        return "<none>"
    prediction = predictions[0]
    essence = str(prediction.get("essence", "")).strip()
    text = str(prediction.get("text", "")).strip()
    if essence:
        return f"{essence}: {text}"
    return text


def render_comparison_report(
    run_id: str,
    records_by_mode: dict[str, list[dict[str, Any]]],
    examples_by_id: dict[str, MRExample],
    limit: int,
    diff_lines: int,
) -> str:
    mode_names = list(records_by_mode.keys())
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for record in records_by_mode[mode_names[0]]:
        example_id = str(record.get("example_id", ""))
        if example_id and example_id not in seen:
            ordered_ids.append(example_id)
            seen.add(example_id)
        if limit > 0 and len(ordered_ids) >= limit:
            break

    records_by_id = {
        mode: {str(record.get("example_id", "")): record for record in records}
        for mode, records in records_by_mode.items()
    }
    lines = [
        "# MergeMind Run Comparison",
        "",
        f"Run: `{run_id}`",
        f"Modes: {', '.join(f'`{mode}`' for mode in mode_names)}",
        f"Records shown: {len(ordered_ids)}",
    ]

    for index, example_id in enumerate(ordered_ids, start=1):
        example = examples_by_id.get(example_id)
        first_record = records_by_id[mode_names[0]].get(example_id, {})
        gold_comments = list(first_record.get("gold_comments", []))
        if not gold_comments and example is not None:
            gold_comments = [comment.text for comment in example.gold_comments if comment.text]

        lines.extend(
            [
                "",
                "---",
                "",
                f"## MR {index}: {example_id}",
            ]
        )
        if example is not None:
            changed_files = ", ".join(changed_file.path for changed_file in example.changed_files[:8])
            lines.extend(
                [
                    f"- title: {example.title}",
                    f"- repo: {example.repo}",
                    f"- changed_files: {changed_files}",
                ]
            )
        lines.append(f"- gold: {_escape_table(' | '.join(gold_comments[:2]) or '<none>')}")
        lines.extend(
            [
                "",
                "| Mode | Hit | Best sim | Judge | Gold align | Valid alt | Top prediction |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for mode in mode_names:
            record = records_by_id[mode].get(example_id, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{mode}`",
                        str(record.get("hit_at_k", "-")),
                        _format_score(record.get("best_similarity_at_k")),
                        _format_score(record.get("judge_score")),
                        _format_score(_judge_field(record, "gold_alignment_score")),
                        _format_score(_judge_field(record, "valid_alternative_score")),
                        _escape_table(_top_prediction(record))[:220],
                    ]
                )
                + " |"
            )

        if example is not None and diff_lines > 0:
            lines.extend(["", "### Diff excerpt", "", "```diff", _clip(example.diff, diff_lines), "```"])

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pipeline predictions inside one run.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--run", required=True, help="Run id under artifacts/runs.")
    parser.add_argument("--modes", nargs="*", default=None, help="Optional pipeline modes to compare.")
    parser.add_argument("--limit", type=int, default=20, help="Number of MR records to show. Use 0 for all.")
    parser.add_argument("--diff-lines", type=int, default=40, help="Diff lines to show per MR. Use 0 to hide diff.")
    parser.add_argument("--output", default="", help="Optional output markdown path.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    runs_dir = resolve_path(PROJECT_ROOT, config["paths"].get("runs_dir", "artifacts/runs"))
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    run_dir = runs_dir / args.run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    modes = [canonical_pipeline_mode(mode) for mode in args.modes] if args.modes else _discover_modes(run_dir)
    if not modes:
        raise FileNotFoundError(f"No pipeline mode artifacts found in {run_dir}")

    records_by_mode = {mode: _load_records(run_dir / mode) for mode in modes}
    selected_ids = {
        str(record.get("example_id", ""))
        for records in records_by_mode.values()
        for record in records[: args.limit if args.limit > 0 else None]
        if record.get("example_id")
    }
    examples_by_id = _load_examples_by_id(prepared_dir, selected_ids)
    report = render_comparison_report(
        run_id=args.run,
        records_by_mode=records_by_mode,
        examples_by_id=examples_by_id,
        limit=max(args.limit, 0),
        diff_lines=max(args.diff_lines, 0),
    )

    if args.output:
        output_path = resolve_path(PROJECT_ROOT, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"[compare_run] wrote {output_path}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
