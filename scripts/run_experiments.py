"""Run local MergeMind experiment modes and write an A/B metrics table."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.config import load_config, resolve_path
from src.data.io import read_jsonl, write_json, write_jsonl
from src.data.schema import MRExample
from src.inference.factory import (
    QWEN_FULL_JUDGE_MODE,
    build_pipeline_components,
    canonical_pipeline_mode,
    pipeline_uses_llm,
    resolve_profile_limit,
)
from src.validation.metrics import OpenAICompatibleLLMJudge, evaluate_examples


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - git is optional metadata for local runs.
        return ""


def _load_eval_examples(config: dict[str, Any], limit: int) -> list[MRExample]:
    prepared_dir = resolve_path(PROJECT_ROOT, config["paths"]["prepared_dir"])
    rows = read_jsonl(prepared_dir / "validation.jsonl") + read_jsonl(prepared_dir / "test.jsonl")
    if limit > 0:
        rows = rows[:limit]
    return [MRExample.from_dict(row) for row in rows]


def _metrics_row(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pipeline_mode",
        "profile",
        "example_count",
        "top1_similarity",
        "best_similarity_at_k",
        "hit_rate_at_k",
        "mrr_at_k",
        "judge_backend",
        "judge_score",
        "avg_latency_sec",
        "p95_latency_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens_per_second",
        "cache_hit_rate",
        "parse_error_rate",
        "fallback_rate",
        "avg_candidates_per_mr",
    ]
    return {key: summary.get(key, 0.0) for key in keys}


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


def _append_progress(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MergeMind local LLM experiments.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--profile", default="smoke", help="Evaluation profile: smoke, main, or full.")
    parser.add_argument("--limit", type=int, default=None, help="Override profile example count.")
    parser.add_argument("--run-id", default="", help="Optional run id.")
    parser.add_argument("--modes", nargs="*", default=None, help="Optional explicit pipeline modes.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    limit = resolve_profile_limit(config, profile=args.profile, explicit_limit=args.limit)
    examples = _load_eval_examples(config, limit)
    modes = args.modes or list(config.get("experiments", {}).get("default_modes", []))
    if not modes:
        raise ValueError("No experiment modes configured.")

    started_at = datetime.now(timezone.utc)
    run_id = args.run_id or f"run_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    runs_dir = resolve_path(PROJECT_ROOT, config["paths"].get("runs_dir", "artifacts/runs"))
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_table: list[dict[str, Any]] = []
    llm_config = dict(config.get("llm", {}))
    needs_llm = any(pipeline_uses_llm(mode) for mode in modes)
    if needs_llm:
        _, _, llm_client = build_pipeline_components(
            canonical_pipeline_mode(next(mode for mode in modes if pipeline_uses_llm(mode))),
            config,
            PROJECT_ROOT,
        )
        _assert_llm_ready(llm_client)

    for mode in modes:
        pipeline_mode = canonical_pipeline_mode(mode)
        generator, reranker, llm_client = build_pipeline_components(pipeline_mode, config, PROJECT_ROOT)
        judge = None
        judge_backend = ""
        if pipeline_mode == QWEN_FULL_JUDGE_MODE:
            judge = OpenAICompatibleLLMJudge(
                client=llm_client,
                temperature=float(llm_config.get("temperature_judge", 0.0)),
                max_tokens=int(llm_config.get("max_tokens_judge", 400)),
            )
            judge_backend = "local_llm"

        mode_dir = run_dir / pipeline_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        progress_path = mode_dir / "progress.jsonl"
        if progress_path.exists():
            progress_path.unlink()

        def progress_callback(event: dict[str, Any], current_mode: str = pipeline_mode) -> None:
            event["pipeline_mode"] = current_mode
            event["profile"] = args.profile
            event["run_id"] = run_id
            _append_progress(progress_path, event)

        summary = evaluate_examples(
            examples=examples,
            generator=generator,
            reranker=reranker,
            top_n=int(config["validation"]["top_n"]),
            similarity_threshold=float(config["validation"]["similarity_threshold"]),
            use_llm_judge=False,
            llm_judge_max_examples=int(config["validation"].get("llm_judge_max_examples", 25)),
            judge_override=judge,
            judge_backend_override=judge_backend,
            progress_callback=progress_callback,
        )
        summary["pipeline_mode"] = pipeline_mode
        summary["profile"] = args.profile
        summary["example_limit"] = limit

        write_json(mode_dir / "summary.json", summary)
        write_jsonl(mode_dir / "predictions.jsonl", summary["examples"])
        write_json(mode_dir / "config_snapshot.json", config)
        write_json(
            mode_dir / "run_manifest.json",
            {
                "pipeline_mode": pipeline_mode,
                "profile": args.profile,
                "example_limit": limit,
                "git_commit": _git_commit(),
                "model_id": llm_config.get("model", ""),
                "base_url": llm_config.get("base_url", ""),
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        metrics_table.append(_metrics_row(summary))

    write_json(run_dir / "metrics_table.json", {"run_id": run_id, "rows": metrics_table})
    print(f"[run_experiments] run_dir={run_dir}")
    print(json.dumps({"run_id": run_id, "rows": metrics_table}, indent=2))


if __name__ == "__main__":
    main()
