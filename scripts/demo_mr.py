"""Run a demo inference on one MR example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.config import load_config, resolve_path
from src.data.io import read_json
from src.data.schema import MRExample
from src.inference.factory import BASELINE_MODE, build_pipeline_components, canonical_pipeline_mode
from src.inference.pipeline import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MergeMind demo inference.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--input", default=None, help="Optional path to a prepared MR example JSON.")
    parser.add_argument("--pipeline", default=BASELINE_MODE, help="Pipeline mode to run.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    input_path = resolve_path(PROJECT_ROOT, args.input) if args.input else prepared_dir / "demo.json"
    pipeline_mode = canonical_pipeline_mode(args.pipeline)

    example = MRExample.from_dict(read_json(input_path))
    generator, reranker, _ = build_pipeline_components(pipeline_mode, config, PROJECT_ROOT)
    predictions = run_inference(example, generator, reranker, top_n=int(config["demo"]["top_n"]))

    print(f"[demo_mr] example_id={example.example_id}")
    print(f"[demo_mr] pipeline={pipeline_mode}")
    print(f"[demo_mr] title={example.title}")
    for index, prediction in enumerate(predictions, start=1):
        evidence = "; ".join(prediction.evidence)
        print(f"{index}. score={prediction.reranker_score:.3f} | {prediction.text}")
        print(f"   evidence: {evidence}")


if __name__ == "__main__":
    main()
