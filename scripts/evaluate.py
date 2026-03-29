"""Evaluate MergeMind baseline outputs."""

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
from src.data.io import read_json, read_jsonl, write_json, write_jsonl
from src.data.schema import MRExample
from src.models.baseline import RetrievalGenerator, Reranker
from src.validation.metrics import evaluate_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MergeMind baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    model_dir = resolve_path(PROJECT_ROOT, config["paths"]["model_dir"])
    evaluation_dir = resolve_path(PROJECT_ROOT, config["paths"]["evaluation_dir"])

    generator = RetrievalGenerator.load(model_dir / "generator.pkl")
    reranker = Reranker.load(model_dir / "reranker.pkl")

    validation_examples = [MRExample.from_dict(row) for row in read_jsonl(prepared_dir / "validation.jsonl")]
    test_examples = [MRExample.from_dict(row) for row in read_jsonl(prepared_dir / "test.jsonl")]
    all_examples = validation_examples + test_examples

    summary = evaluate_examples(
        examples=all_examples,
        generator=generator,
        reranker=reranker,
        top_n=int(config["validation"]["top_n"]),
        similarity_threshold=float(config["validation"]["similarity_threshold"]),
        use_llm_judge=bool(config["validation"]["use_llm_judge"]),
        llm_judge_model=str(config["validation"].get("llm_judge_model", "")),
        llm_judge_max_examples=int(config["validation"].get("llm_judge_max_examples", 25)),
    )

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_json(evaluation_dir / "summary.json", summary)
    write_jsonl(evaluation_dir / "predictions.jsonl", summary["examples"])
    manifest = read_json(prepared_dir / "manifest.json")

    print(f"[evaluate] evaluation_dir={evaluation_dir}")
    print(
        json.dumps(
            {
                "example_count": summary["example_count"],
                "top1_similarity": summary["top1_similarity"],
                "best_similarity_at_k": summary["best_similarity_at_k"],
                "hit_rate_at_k": summary["hit_rate_at_k"],
                "mrr_at_k": summary["mrr_at_k"],
                "judge_backend": summary["judge_backend"],
                "data_manifest": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
