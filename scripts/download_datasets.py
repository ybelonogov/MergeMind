"""Download real datasets for MergeMind."""

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

from src.config import load_config
from src.data.download import download_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real datasets for MergeMind.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    manifest = download_datasets(config=config, project_root=PROJECT_ROOT, force=args.force)

    print("[download_datasets] completed")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
