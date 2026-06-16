"""Prepare fixed SWE-CI manifests and chunks for paired A/B runs."""

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

from src.validation.swe_ci.schemas import SweCiTask  # noqa: E402
from src.validation.swe_ci.task_loader import load_swe_ci_tasks  # noqa: E402


def _gap_as_int(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("gap", "failed", "count", "test_gap"):
            if key in value:
                return _gap_as_int(value[key])
        return None
    if isinstance(value, list):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_row(task: SweCiTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "repo_name": task.repo_name,
        "repo_url": task.repo_url,
        "current_sha": task.current_sha,
        "target_sha": task.target_sha,
        "image_sha": task.image_sha,
        "test_gap": task.test_gap,
        **task.metadata,
    }


def select_tasks(
    source_tasks: list[SweCiTask],
    *,
    limit: int,
    excluded_task_ids: set[str] | None = None,
    min_gap: int | None = None,
    max_gap: int | None = None,
) -> list[SweCiTask]:
    excluded_task_ids = excluded_task_ids or set()
    selected: list[SweCiTask] = []
    seen: set[str] = set()
    for task in source_tasks:
        if task.task_id in seen or task.task_id in excluded_task_ids:
            continue
        seen.add(task.task_id)
        gap = _gap_as_int(task.test_gap)
        if min_gap is not None and (gap is None or gap < min_gap):
            continue
        if max_gap is not None and (gap is None or gap > max_gap):
            continue
        selected.append(task)
        if len(selected) >= limit:
            break
    return selected


def write_manifest(tasks: list[SweCiTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_task_row(task) for task in tasks]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def write_chunks(tasks: list[SweCiTask], output_dir: Path, stem: str, chunk_size: int) -> list[Path]:
    chunk_paths: list[Path] = []
    for index in range(0, len(tasks), chunk_size):
        chunk_number = index // chunk_size + 1
        chunk_path = output_dir / f"{stem}_chunk_{chunk_number:02d}.jsonl"
        write_manifest(tasks[index : index + chunk_size], chunk_path)
        chunk_paths.append(chunk_path)
    return chunk_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare fixed SWE-CI Pair30 manifests.")
    parser.add_argument("--source-tasks-path", required=True, help="Source SWE-CI task manifest.")
    parser.add_argument("--output-dir", default="configs", help="Directory for selected manifest and chunks.")
    parser.add_argument("--output-stem", default="swe_ci_nir_pair30_tasks", help="Output manifest filename stem.")
    parser.add_argument("--limit", type=int, default=30, help="Number of tasks to select.")
    parser.add_argument("--chunk-size", type=int, default=5, help="Tasks per chunk file.")
    parser.add_argument("--exclude-tasks-path", default="", help="Optional manifest with task ids to exclude.")
    parser.add_argument("--min-gap", type=int, default=None, help="Optional lower bound for initial test gap.")
    parser.add_argument("--max-gap", type=int, default=None, help="Optional upper bound for initial test gap.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected task ids without writing files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    source_tasks = load_swe_ci_tasks(args.source_tasks_path)
    excluded: set[str] = set()
    if args.exclude_tasks_path:
        excluded = {task.task_id for task in load_swe_ci_tasks(args.exclude_tasks_path)}
    selected = select_tasks(
        source_tasks,
        limit=args.limit,
        excluded_task_ids=excluded,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
    )
    if len(selected) < args.limit:
        raise ValueError(f"Only selected {len(selected)} tasks, expected {args.limit}.")

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / f"{args.output_stem}.jsonl"
    chunk_paths = [output_dir / f"{args.output_stem}_chunk_{index:02d}.jsonl" for index in range(1, (len(selected) - 1) // args.chunk_size + 2)]
    if args.dry_run:
        print(json.dumps({"manifest": str(manifest_path), "chunks": [str(path) for path in chunk_paths], "task_ids": [task.task_id for task in selected]}, indent=2))
        return 0

    write_manifest(selected, manifest_path)
    chunk_paths = write_chunks(selected, output_dir, args.output_stem, args.chunk_size)
    info = {
        "source_tasks_path": str(Path(args.source_tasks_path).resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "chunk_paths": [str(path.resolve()) for path in chunk_paths],
        "limit": args.limit,
        "chunk_size": args.chunk_size,
        "min_gap": args.min_gap,
        "max_gap": args.max_gap,
        "excluded_task_count": len(excluded),
        "task_ids": [task.task_id for task in selected],
    }
    (output_dir / f"{args.output_stem}_manifest_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
