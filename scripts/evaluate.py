"""Evaluate MergeMind baseline outputs."""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print(f"[evaluate] project_root={project_root}")
    print("[evaluate] TODO: implement offline metrics and evaluation pipeline")


if __name__ == "__main__":
    main()
