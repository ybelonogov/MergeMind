"""Combine chunked SWE-CI run artifacts into one synthetic run directory."""

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

from src.validation.swe_ci.reporter import write_report  # noqa: E402
from src.validation.swe_ci.schemas import SweCiTaskRunResult  # noqa: E402


def _load_chunk_results(path: Path) -> list[dict[str, Any]]:
    task_results_path = path / "task_results.json"
    payload = json.loads(task_results_path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload)
    if not isinstance(rows, list):
        raise ValueError(f"{task_results_path} must contain a result list or a 'results' list.")
    return [row for row in rows if isinstance(row, dict) and row.get("task_id")]


def _to_result(row: dict[str, Any]) -> SweCiTaskRunResult:
    return SweCiTaskRunResult(
        task_id=str(row["task_id"]),
        status=row.get("status", "failed"),
        started_at=str(row.get("started_at", "")),
        finished_at=str(row.get("finished_at", "")),
        duration_seconds=float(row.get("duration_seconds", 0.0) or 0.0),
        exit_code=row.get("exit_code"),
        stdout_path=str(row.get("stdout_path", "")),
        stderr_path=str(row.get("stderr_path", "")),
        events_path=str(row.get("events_path", "")),
        metrics=dict(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}),
        error_message=str(row.get("error_message", "")),
    )


def collect_chunk_dirs(args: argparse.Namespace) -> list[Path]:
    chunk_dirs = [Path(path).resolve() for path in args.chunk_run_dir]
    if args.chunks_parent:
        parent = Path(args.chunks_parent).resolve()
        chunk_dirs.extend(sorted(path for path in parent.glob(args.glob) if path.is_dir()))
    if not chunk_dirs:
        raise ValueError("Provide at least one --chunk-run-dir or --chunks-parent.")
    return chunk_dirs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine chunked SWE-CI run directories.")
    parser.add_argument("--chunk-run-dir", action="append", default=[], help="Chunk run directory containing task_results.json.")
    parser.add_argument("--chunks-parent", default="", help="Optional parent directory containing chunk run dirs.")
    parser.add_argument("--glob", default="chunk_*", help="Glob used with --chunks-parent.")
    parser.add_argument("--output-dir", required=True, help="Combined run directory to write.")
    parser.add_argument("--run-id", required=True, help="Run id for the combined summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    chunk_dirs = collect_chunk_dirs(args)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk_dir in chunk_dirs:
        for row in _load_chunk_results(chunk_dir):
            task_id = str(row["task_id"])
            if task_id in seen:
                raise ValueError(f"Duplicate task_id across chunks: {task_id}")
            seen.add(task_id)
            rows.append(row)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = write_report(output_dir, args.run_id, [_to_result(row) for row in rows])
    (output_dir / "combined_chunks.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "chunk_run_dirs": [str(path) for path in chunk_dirs],
                "task_count": len(rows),
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "task_count": len(rows), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
