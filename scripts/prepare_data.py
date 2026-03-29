"""Prepare datasets for MergeMind MVP."""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print(f"[prepare_data] project_root={project_root}")
    print("[prepare_data] TODO: load, normalize, and export datasets")


if __name__ == "__main__":
    main()
