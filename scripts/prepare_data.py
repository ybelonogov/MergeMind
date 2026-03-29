"""Prepare datasets for MergeMind MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.config import load_config, resolve_path
from src.data.adapters import prepare_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MergeMind datasets.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    raw_paths = {
        name: str(resolve_path(PROJECT_ROOT, path))
        for name, path in config["paths"]["raw"].items()
    }
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    summary = prepare_datasets(raw_paths=raw_paths, prepared_dir=prepared_dir)

    print(f"[prepare_data] prepared_dir={prepared_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
