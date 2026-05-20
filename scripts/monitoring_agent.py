"""Generate MergeMind monitoring chronicle, static dashboard and presentation notes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.monitoring.agent import collect_monitoring_snapshot, sleep_until_next_interval, write_monitoring_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MergeMind monitoring report agent.")
    parser.add_argument("--config", default="configs/base.yaml", help="Project config path.")
    parser.add_argument("--output-dir", default="artifacts/monitoring", help="Directory for monitoring artifacts.")
    parser.add_argument("--limit", type=int, default=8, help="Number of latest runs/artifacts to include.")
    parser.add_argument("--run-tests", action="store_true", help="Run the unit test suite before writing the snapshot.")
    parser.add_argument("--watch", action="store_true", help="Keep writing snapshots periodically.")
    parser.add_argument("--interval-seconds", type=int, default=300, help="Watch interval.")
    parser.add_argument("--no-append", action="store_true", help="Do not append to cumulative chronicle.md.")
    return parser.parse_args()


def _write_once(args: argparse.Namespace) -> dict[str, str]:
    snapshot = collect_monitoring_snapshot(
        project_root=PROJECT_ROOT,
        config_path=(PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config),
        run_tests=args.run_tests,
        limit=args.limit,
    )
    return write_monitoring_artifacts(
        snapshot,
        output_dir=(PROJECT_ROOT / args.output_dir).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir),
        append_chronicle=not args.no_append,
    )


def main() -> int:
    args = _parse_args()
    if args.interval_seconds <= 0:
        print("[monitoring_agent] --interval-seconds must be positive.", file=sys.stderr)
        return 1

    while True:
        started_at = time.time()
        paths = _write_once(args)
        print("[monitoring_agent] wrote artifacts:")
        for name, path in paths.items():
            print(f"- {name}: {path}")
        if not args.watch:
            break
        sleep_until_next_interval(started_at, args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
