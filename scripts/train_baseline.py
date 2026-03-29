"""Train baseline model pipeline for MergeMind MVP."""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print(f"[train_baseline] project_root={project_root}")
    print("[train_baseline] TODO: implement generator and reranker baseline")


if __name__ == "__main__":
    main()
