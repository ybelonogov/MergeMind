"""Run a demo inference on one MR example."""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print(f"[demo_mr] project_root={project_root}")
    print("[demo_mr] TODO: load one MR example and print candidate comments")


if __name__ == "__main__":
    main()
