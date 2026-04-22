"""Evaluate MergeMind baseline and local LLM pipeline outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
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
from src.inference.factory import (
    BASELINE_MODE,
    build_pipeline_components,
    canonical_pipeline_mode,
    pipeline_uses_llm_judge,
    resolve_profile_limit,
)
from src.validation.metrics import OpenAICompatibleLLMJudge
from src.validation.metrics import evaluate_examples


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - git metadata is optional for local evaluation.
        return ""


def _assert_llm_ready(llm_client: object) -> None:
    try:
        status = llm_client.health_check()
    except Exception as exc:  # noqa: BLE001 - fail fast with a local-server hint.
        raise RuntimeError("LM Studio local server is not reachable. Start it and load the Qwen model.") from exc
    if not status["model_available"]:
        available = ", ".join(status["available_models"]) or "<none>"
        raise RuntimeError(
            f"Configured model '{status['configured_model']}' is not available at "
            f"{status['base_url']}. Available models: {available}"
        )


def _append_progress(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MergeMind baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--pipeline", default=BASELINE_MODE, help="Pipeline mode to evaluate.")
    parser.add_argument("--profile", default="", help="Evaluation profile: smoke, main, or full.")
    parser.add_argument("--limit", type=int, default=None, help="Override number of validation/test examples.")
    parser.add_argument("--run-id", default="", help="Optional run id for artifacts/runs output.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    evaluation_dir = resolve_path(PROJECT_ROOT, config["paths"]["evaluation_dir"])
    runs_dir = resolve_path(PROJECT_ROOT, config["paths"].get("runs_dir", "artifacts/runs"))
    pipeline_mode = canonical_pipeline_mode(args.pipeline)

    validation_examples = [MRExample.from_dict(row) for row in read_jsonl(prepared_dir / "validation.jsonl")]
    test_examples = [MRExample.from_dict(row) for row in read_jsonl(prepared_dir / "test.jsonl")]
    all_examples = validation_examples + test_examples
    limit = resolve_profile_limit(config, profile=args.profile, explicit_limit=args.limit)
    if limit > 0:
        all_examples = all_examples[:limit]

    generator, reranker, llm_client = build_pipeline_components(pipeline_mode, config, PROJECT_ROOT)
    if llm_client is not None:
        _assert_llm_ready(llm_client)
    judge = None
    judge_backend = ""
    if pipeline_uses_llm_judge(pipeline_mode):
        llm_config = dict(config.get("llm", {}))
        if llm_client is None:
            _, _, llm_client = build_pipeline_components(pipeline_mode, config, PROJECT_ROOT)
        judge = OpenAICompatibleLLMJudge(
            client=llm_client,
            temperature=float(llm_config.get("temperature_judge", 0.0)),
            max_tokens=int(llm_config.get("max_tokens_judge", 400)),
        )
        judge_backend = "local_llm"

    started_at = datetime.now(timezone.utc)
    if args.run_id or pipeline_mode != BASELINE_MODE or args.profile or args.limit is not None:
        run_id = args.run_id or f"evaluate_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
        output_dir = runs_dir / run_id / pipeline_mode
    else:
        output_dir = evaluation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    def progress_callback(event: dict) -> None:
        event["pipeline_mode"] = pipeline_mode
        event["profile"] = args.profile or "all"
        event["run_id"] = output_dir.parent.name if output_dir.parent != evaluation_dir.parent else "evaluation"
        _append_progress(progress_path, event)

    summary = evaluate_examples(
        examples=all_examples,
        generator=generator,
        reranker=reranker,
        top_n=int(config["validation"]["top_n"]),
        similarity_threshold=float(config["validation"]["similarity_threshold"]),
        use_llm_judge=bool(config["validation"]["use_llm_judge"]),
        llm_judge_model=str(config["validation"].get("llm_judge_model", "")),
        llm_judge_max_examples=int(config["validation"].get("llm_judge_max_examples", 25)),
        judge_override=judge,
        judge_backend_override=judge_backend,
        progress_callback=progress_callback,
    )
    summary["pipeline_mode"] = pipeline_mode
    summary["profile"] = args.profile or "all"
    summary["example_limit"] = limit

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", summary["examples"])
    write_json(output_dir / "config_snapshot.json", config)
    write_json(
        output_dir / "run_manifest.json",
        {
            "pipeline_mode": pipeline_mode,
            "profile": args.profile or "all",
            "example_limit": limit,
            "git_commit": _git_commit(),
            "model_id": dict(config.get("llm", {})).get("model", ""),
            "base_url": dict(config.get("llm", {})).get("base_url", ""),
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    manifest = read_json(prepared_dir / "manifest.json")

    print(f"[evaluate] evaluation_dir={output_dir}")
    print(
        json.dumps(
            {
                "example_count": summary["example_count"],
                "pipeline_mode": summary["pipeline_mode"],
                "profile": summary["profile"],
                "top1_similarity": summary["top1_similarity"],
                "best_similarity_at_k": summary["best_similarity_at_k"],
                "hit_rate_at_k": summary["hit_rate_at_k"],
                "mrr_at_k": summary["mrr_at_k"],
                "judge_backend": summary["judge_backend"],
                "judge_score": summary.get("judge_score", 0.0),
                "avg_latency_sec": summary["avg_latency_sec"],
                "total_tokens": summary["total_tokens"],
                "tokens_per_second": summary["tokens_per_second"],
                "cache_hit_rate": summary["cache_hit_rate"],
                "parse_error_rate": summary["parse_error_rate"],
                "data_manifest": manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
