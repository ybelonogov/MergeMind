"""CLI wrapper for host-side MergeMind SWE-CI assisted review."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


_bootstrap_path()

from src.validation.swe_ci.assist_helper import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
