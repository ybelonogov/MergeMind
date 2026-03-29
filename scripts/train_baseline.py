"""Train baseline model pipeline for MergeMind MVP."""

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
from src.data.io import read_jsonl, write_json
from src.data.schema import MRExample
from src.models.baseline import RetrievalGenerator, Reranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MergeMind baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    model_dir = resolve_path(PROJECT_ROOT, config["paths"]["model_dir"])
    train_examples = [MRExample.from_dict(row) for row in read_jsonl(prepared_dir / "train.jsonl")]

    generator = RetrievalGenerator.fit(train_examples)
    reranker = Reranker.from_config(config["model"]["reranker"])

    generator_path = model_dir / "generator.pkl"
    reranker_path = model_dir / "reranker.json"
    generator.save(generator_path)
    reranker.save(reranker_path)
    write_json(
        model_dir / "manifest.json",
        {
            "generator_path": str(generator_path),
            "reranker_path": str(reranker_path),
            "train_examples": len(train_examples),
        },
    )

    print(f"[train_baseline] model_dir={model_dir}")
    print(json.dumps({"train_examples": len(train_examples)}, indent=2))


if __name__ == "__main__":
    main()
