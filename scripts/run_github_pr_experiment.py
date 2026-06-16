"""Run reproducible MergeMind pipeline comparisons on curated GitHub PRs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from scripts.review_github_pr import _deterministic_metrics, _render_report
from src.config import apply_llm_provider, load_config, load_dotenv, resolve_path
from src.data.github import GitHubClientError, fetch_github_pr_example
from src.data.io import write_json
from src.inference.factory import build_pipeline_components, canonical_pipeline_mode
from src.inference.pipeline import run_inference
from src.models.llm import build_llm_client
from src.validation.metrics import OpenAICompatibleLLMJudge


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional local package.
        raise SystemExit("YAML manifest support requires PyYAML; use JSON or install PyYAML.") from exc
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise SystemExit("Manifest must be a mapping.")
    return payload


def _manifest_prs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    prs = manifest.get("prs", [])
    if not isinstance(prs, list) or not prs:
        raise SystemExit("Manifest must include a non-empty 'prs' list.")
    output: list[dict[str, Any]] = []
    for item in prs:
        if isinstance(item, str):
            output.append({"url": item})
        elif isinstance(item, dict) and item.get("url"):
            output.append(dict(item))
        else:
            raise SystemExit("Each manifest PR must be a URL string or object with 'url'.")
    return output


def _manifest_pipelines(manifest: dict[str, Any]) -> list[str]:
    pipelines = manifest.get("pipelines", [])
    if not isinstance(pipelines, list) or not pipelines:
        raise SystemExit("Manifest must include a non-empty 'pipelines' list.")
    return [canonical_pipeline_mode(str(item)) for item in pipelines]


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _render_summary(run_id: str, rows: list[dict[str, Any]], command: list[str]) -> str:
    lines = [
        f"# GitHub PR Experiment: {run_id}",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "## Aggregate",
        "",
    ]
    by_pipeline: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pipeline.setdefault(str(row["pipeline"]), []).append(row)
    lines.append("| pipeline | PRs | ok | failed | comments | gold PRs | hit@k | avg judge | avg grounded | avg useful | tokens | latency sec | fallbacks |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pipeline, items in sorted(by_pipeline.items()):
        count = len(items)
        ok_items = [item for item in items if item.get("status", "success") == "success"]
        ok_count = len(ok_items)
        failed_count = count - ok_count
        comments = sum(int(item.get("comment_count", 0)) for item in ok_items)
        gold_prs = sum(1 for item in ok_items if int(item.get("gold_comment_count", 0)) > 0)
        hit_rate = sum(float(item.get("hit_at_k", 0)) for item in ok_items) / ok_count if ok_count else 0.0
        judge_values = [float(item.get("judge_score", 0.0)) for item in ok_items if item.get("judge_enabled")]
        grounded_values = [float(item.get("groundedness", 0.0)) for item in ok_items if item.get("judge_enabled")]
        useful_values = [float(item.get("usefulness", 0.0)) for item in ok_items if item.get("judge_enabled")]
        avg_judge = sum(judge_values) / len(judge_values) if judge_values else 0.0
        avg_grounded = sum(grounded_values) / len(grounded_values) if grounded_values else 0.0
        avg_useful = sum(useful_values) / len(useful_values) if useful_values else 0.0
        tokens = sum(int(item.get("total_tokens", 0)) for item in ok_items)
        latency = sum(float(item.get("total_wall_latency_sec", 0.0)) for item in ok_items)
        fallbacks = sum(int(item.get("fallback_count", 0)) for item in ok_items)
        lines.append(
            "| {pipeline} | {count} | {ok_count} | {failed_count} | {comments} | {gold_prs} | {hit_rate:.3f} | {avg_judge:.3f} | "
            "{avg_grounded:.3f} | {avg_useful:.3f} | {tokens} | {latency:.3f} | {fallbacks} |".format(
                pipeline=pipeline,
                count=count,
                ok_count=ok_count,
                failed_count=failed_count,
                comments=comments,
                gold_prs=gold_prs,
                hit_rate=hit_rate,
                avg_judge=avg_judge,
                avg_grounded=avg_grounded,
                avg_useful=avg_useful,
                tokens=tokens,
                latency=latency,
                fallbacks=fallbacks,
            )
        )

    lines.extend(
        [
            "",
            "## PR Results",
            "",
            "| PR | pipeline | status | gold | comments | hit@k | best sim | judge | grounded | useful | tokens | latency | notes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        notes = str(row.get("manual_notes") or row.get("notes") or row.get("error_message") or "").replace("|", "\\|")
        lines.append(
            "| [{title}]({url}) | {pipeline} | {status} | {gold} | {comments} | {hit} | {sim} | {judge} | "
            "{grounded} | {useful} | {tokens} | {latency} | {notes} |".format(
                title=str(row.get("title") or row.get("url")),
                url=row["url"],
                pipeline=row["pipeline"],
                status=row.get("status", "success"),
                gold=row.get("gold_comment_count", 0),
                comments=row.get("comment_count", 0),
                hit=row.get("hit_at_k", 0),
                sim=_format_float(row.get("best_similarity_at_k")),
                judge=_format_float(row.get("judge_score")),
                grounded=_format_float(row.get("groundedness")),
                useful=_format_float(row.get("usefulness")),
                tokens=row.get("total_tokens", 0),
                latency=_format_float(row.get("total_wall_latency_sec")),
                notes=notes,
            )
        )
    return "\n".join(lines) + "\n"


def _run_one(
    *,
    config: dict[str, Any],
    output_root: Path,
    pr_item: dict[str, Any],
    example: Any,
    pr_ref: Any,
    pipeline: str,
    judge_enabled: bool,
    limit_comments: int,
    max_repository_files: int,
    diff_lines: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    generator, reranker, shared_client = build_pipeline_components(pipeline, config, PROJECT_ROOT)
    inference_started = time.perf_counter()
    predictions = run_inference(example, generator, reranker, top_n=limit_comments)
    inference_latency = time.perf_counter() - inference_started
    gold_comments = [comment.text for comment in example.gold_comments if comment.text]
    deterministic = _deterministic_metrics(
        predictions=predictions,
        gold_comments=gold_comments,
        threshold=float(config.get("validation", {}).get("similarity_threshold", 0.35)),
    )

    judge_result: dict[str, Any] = {}
    judge_latency = 0.0
    judge_client = shared_client
    judge = None
    if judge_enabled:
        judge_client = judge_client or build_llm_client(config, PROJECT_ROOT)
        llm_config = dict(config.get("llm", {}))
        judge = OpenAICompatibleLLMJudge(
            judge_client,
            temperature=float(llm_config.get("temperature_judge", 0.0)),
            max_tokens=int(llm_config.get("max_tokens_judge", 400)),
        )
        judge_started = time.perf_counter()
        judge_result = judge.evaluate(predictions, gold_comments, example)
        judge_latency = time.perf_counter() - judge_started

    output_dir = output_root / pr_ref.safe_id / _safe_segment(pipeline)
    total_wall_latency = time.perf_counter() - started
    llm_metrics = judge_client.stats() if judge_client is not None else {}
    fallback_count = int(getattr(generator, "fallback_count", 0))
    fallback_count += int(getattr(reranker, "fallback_count", 0))
    fallback_count += int(getattr(judge, "fallback_count", 0)) if judge is not None else 0
    evaluation = {
        "pr": {
            "url": pr_item["url"],
            "example_id": example.example_id,
            "repo": example.repo,
            "title": example.title,
            "gold_comment_count": len(gold_comments),
            "notes": pr_item.get("notes", ""),
            "manual_notes": pr_item.get("manual_notes", ""),
        },
        "pipeline": pipeline,
        "judge_enabled": judge_enabled,
        "judge_mode": "gold" if gold_comments else "no_gold",
        "deterministic": deterministic,
        "judge": judge_result,
        "runtime": {
            "inference_latency_sec": inference_latency,
            "judge_latency_sec": judge_latency,
            "total_wall_latency_sec": total_wall_latency,
        },
        "llm": llm_metrics,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count,
    }
    write_json(output_dir / "example.json", example.to_dict())
    write_json(
        output_dir / "predictions.json",
        {
            "pr": evaluation["pr"],
            "pipeline": pipeline,
            "predictions": [prediction.to_dict() for prediction in predictions],
        },
    )
    write_json(output_dir / "evaluation.json", evaluation)
    (output_dir / "report.md").write_text(
        _render_report(
            example=example,
            url=str(pr_item["url"]),
            pipeline_mode=pipeline,
            predictions=predictions,
            evaluation=evaluation,
            diff_lines=diff_lines,
        ),
        encoding="utf-8",
    )
    return {
        "status": "success",
        "url": pr_item["url"],
        "repo": example.repo,
        "title": example.title,
        "pipeline": pipeline,
        "artifact_dir": str(output_dir),
        "notes": pr_item.get("notes", ""),
        "manual_notes": pr_item.get("manual_notes", ""),
        "gold_comment_count": len(gold_comments),
        "comment_count": len(predictions),
        "hit_at_k": deterministic.get("hit_at_k", 0),
        "best_similarity_at_k": deterministic.get("best_similarity_at_k", 0.0),
        "judge_enabled": judge_enabled,
        "judge_score": judge_result.get("judge_score", 0.0),
        "groundedness": judge_result.get("groundedness", 0.0),
        "usefulness": judge_result.get("usefulness", 0.0),
        "total_tokens": int(llm_metrics.get("total_tokens", 0) or 0),
        "llm_call_count": int(llm_metrics.get("llm_call_count", 0) or 0),
        "parse_error_rate": float(llm_metrics.get("parse_error_rate", 0.0) or 0.0),
        "fallback_count": fallback_count,
        "total_wall_latency_sec": total_wall_latency,
    }


def _failed_row(pr_item: dict[str, Any], pipeline: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "url": pr_item["url"],
        "repo": "",
        "title": str(pr_item.get("url", "")),
        "pipeline": pipeline,
        "artifact_dir": "",
        "notes": pr_item.get("notes", ""),
        "manual_notes": pr_item.get("manual_notes", ""),
        "gold_comment_count": 0,
        "comment_count": 0,
        "hit_at_k": 0,
        "best_similarity_at_k": 0.0,
        "judge_enabled": False,
        "judge_score": 0.0,
        "groundedness": 0.0,
        "usefulness": 0.0,
        "total_tokens": 0,
        "llm_call_count": 0,
        "parse_error_rate": 0.0,
        "fallback_count": 0,
        "total_wall_latency_sec": 0.0,
        "error_message": str(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MergeMind pipeline comparisons on curated GitHub PRs.")
    parser.add_argument("--manifest", required=True, help="JSON or YAML PR experiment manifest.")
    parser.add_argument("--config", default="configs/base.yaml", help="MergeMind config path.")
    parser.add_argument("--run-id", default="", help="Override manifest run_id.")
    parser.add_argument("--output-dir", default="artifacts/github_pr_experiments", help="Experiment output root.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned PR/pipeline matrix.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    prs = _manifest_prs(manifest)
    pipelines = _manifest_pipelines(manifest)
    run_id = args.run_id or str(manifest.get("run_id") or f"pr_experiment_{int(time.time())}")
    llm_provider = str(manifest.get("llm_provider", ""))
    judge_enabled = bool(manifest.get("judge", False))
    limit_comments = int(manifest.get("limit_comments", 3))
    max_repository_files = int(manifest.get("max_repository_files", 20))
    diff_lines = int(manifest.get("diff_lines", 80))

    if args.dry_run:
        print(f"run_id={run_id}")
        print(f"llm_provider={llm_provider or '<config default>'}")
        print(f"judge={judge_enabled}")
        for pr in prs:
            for pipeline in pipelines:
                print(f"{pr['url']} :: {pipeline}")
        return

    config = apply_llm_provider(load_config(PROJECT_ROOT / args.config), llm_provider)
    output_root = resolve_path(PROJECT_ROOT, args.output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "config_snapshot.json", config)
    write_json(
        output_root / "command.json",
        {
            "argv": sys.argv,
            "manifest": str(manifest_path),
            "config": args.config,
            "run_id": run_id,
        },
    )

    rows: list[dict[str, Any]] = []
    for pr in prs:
        try:
            example, pr_ref = fetch_github_pr_example(str(pr["url"]), max_repository_files=max_repository_files)
        except (GitHubClientError, ValueError) as error:
            print(f"[pr_experiment] failed to fetch {pr['url']}: {error}")
            for pipeline in pipelines:
                rows.append(_failed_row(pr, pipeline, error))
            continue
        for pipeline in pipelines:
            print(f"[pr_experiment] {pr['url']} :: {pipeline}")
            rows.append(
                _run_one(
                    config=config,
                    output_root=output_root,
                    pr_item=pr,
                    example=example,
                    pr_ref=pr_ref,
                    pipeline=pipeline,
                    judge_enabled=judge_enabled,
                    limit_comments=limit_comments,
                    max_repository_files=max_repository_files,
                    diff_lines=diff_lines,
                )
            )

    summary = {
        "run_id": run_id,
        "manifest": str(manifest_path),
        "llm_provider": llm_provider,
        "judge_enabled": judge_enabled,
        "pr_count": len(prs),
        "pipeline_count": len(pipelines),
        "rows": rows,
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(_render_summary(run_id, rows, sys.argv), encoding="utf-8")
    print(f"[pr_experiment] artifacts={output_root}")


if __name__ == "__main__":
    main()
