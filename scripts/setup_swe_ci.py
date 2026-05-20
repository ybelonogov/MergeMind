"""Check whether a local SWE-CI checkout is ready for MergeMind validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.validation.swe_ci.config import validate_run_config
from src.validation.swe_ci.schemas import SweCiRunConfig
from src.validation.swe_ci.task_loader import SweCiTaskLoaderError, load_swe_ci_tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local SWE-CI setup for MergeMind.")
    parser.add_argument("--swe-ci-repo-path", required=True, help="Path to the cloned SWE-CI repository.")
    parser.add_argument("--tasks-path", required=True, help="Path to SWE-CI task manifest (.jsonl or .json).")
    parser.add_argument("--output-dir", required=True, help="Directory for MergeMind SWE-CI run artifacts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional task loading limit for validation.")
    parser.add_argument("--max-iterations", type=int, default=3, help="SWE-CI max evolution iterations for validation.")
    parser.add_argument("--timeout-seconds", type=int, default=7200, help="Per-task timeout for validation.")
    parser.add_argument("--splitting", default="default", help="SWE-CI dataset split expected by the task manifest.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = SweCiRunConfig(
        swe_ci_repo_path=Path(args.swe_ci_repo_path).resolve(),
        tasks_path=Path(args.tasks_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        limit=args.limit,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
        mode="baseline",
        run_id="setup_check",
        splitting=args.splitting,
    )

    errors = validate_run_config(config)
    if not errors:
        try:
            tasks = load_swe_ci_tasks(config.tasks_path, limit=args.limit)
        except SweCiTaskLoaderError as exc:
            errors.append(str(exc))
        else:
            print(f"[setup_swe_ci] Loaded {len(tasks)} task(s) from {config.tasks_path}")

    if errors:
        print("[setup_swe_ci] SWE-CI setup is not ready:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("[setup_swe_ci] OK")
    print(f"[setup_swe_ci] SWE-CI repo: {config.swe_ci_repo_path}")
    print(f"[setup_swe_ci] Tasks: {config.tasks_path}")
    print(f"[setup_swe_ci] Output dir: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
