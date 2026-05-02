"""Run MergeMind review inference on a live GitHub Pull Request."""

from __future__ import annotations

import argparse
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.config import apply_llm_provider, load_config, load_dotenv, resolve_path
from src.data.github import GitHubClientError, fetch_github_pr_example
from src.data.io import write_json
from src.inference.factory import BASELINE_MODE, build_pipeline_components, canonical_pipeline_mode
from src.inference.pipeline import run_inference
from src.models.llm import OpenAICompatibleLLMClient, build_llm_client
from src.validation.metrics import OpenAICompatibleLLMJudge


def _artifact_dir(config: dict, safe_id: str) -> Path:
    configured = config.get("paths", {}).get("github_pr_dir", "artifacts/github_pr")
    return resolve_path(PROJECT_ROOT, configured) / safe_id


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()


def _deterministic_metrics(predictions: list[Any], gold_comments: list[str], threshold: float) -> dict[str, Any]:
    top1_similarity = 0.0
    best_similarity = 0.0
    first_hit_rank = 0
    for rank, prediction in enumerate(predictions, start=1):
        current = max((_text_similarity(prediction.text, gold) for gold in gold_comments), default=0.0)
        best_similarity = max(best_similarity, current)
        if rank == 1:
            top1_similarity = current
        if not first_hit_rank and current >= threshold:
            first_hit_rank = rank
    return {
        "top1_similarity": top1_similarity,
        "best_similarity_at_k": best_similarity,
        "hit_at_k": int(best_similarity >= threshold) if gold_comments else 0,
        "mrr_at_k": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "similarity_threshold": threshold,
    }


def _llm_metrics(client: OpenAICompatibleLLMClient | None) -> dict[str, Any]:
    if client is None:
        return {}
    return client.stats()


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _render_report(
    *,
    example: Any,
    url: str,
    pipeline_mode: str,
    predictions: list[Any],
    evaluation: dict[str, Any],
    diff_lines: int,
) -> str:
    changed_paths = ", ".join(changed_file.path for changed_file in example.changed_files[:12])
    gold_comments = [comment.text for comment in example.gold_comments if comment.text]
    lines = [
        "# MergeMind GitHub PR Review",
        "",
        f"- url: {url}",
        f"- example_id: {example.example_id}",
        f"- repo: {example.repo}",
        f"- title: {example.title}",
        f"- pipeline: {pipeline_mode}",
        f"- changed_files: {changed_paths or '<none>'}",
        f"- gold_comment_count: {len(gold_comments)}",
        "",
        "## Metrics",
        "",
        f"- judge_score: {_format_score(evaluation.get('judge', {}).get('judge_score'))}",
        f"- gold_alignment_score: {_format_score(evaluation.get('judge', {}).get('gold_alignment_score'))}",
        f"- valid_alternative_score: {_format_score(evaluation.get('judge', {}).get('valid_alternative_score'))}",
        f"- groundedness: {_format_score(evaluation.get('judge', {}).get('groundedness'))}",
        f"- usefulness: {_format_score(evaluation.get('judge', {}).get('usefulness'))}",
        f"- best_similarity_at_k: {_format_score(evaluation.get('deterministic', {}).get('best_similarity_at_k'))}",
        f"- hit_at_k: {evaluation.get('deterministic', {}).get('hit_at_k', 0)}",
        f"- inference_latency_sec: {_format_score(evaluation.get('runtime', {}).get('inference_latency_sec'))}",
        f"- judge_latency_sec: {_format_score(evaluation.get('runtime', {}).get('judge_latency_sec'))}",
        f"- total_wall_latency_sec: {_format_score(evaluation.get('runtime', {}).get('total_wall_latency_sec'))}",
        "",
    ]
    reason = str(evaluation.get("judge", {}).get("reason", "")).strip()
    if reason:
        lines.extend(["## Judge reason", "", reason, ""])

    lines.extend(["## Gold comments"])
    if gold_comments:
        lines.extend(f"- {comment}" for comment in gold_comments[:10])
    else:
        lines.append("- <none>; judge is running in no-gold live PR mode")

    lines.extend(["", "## Predictions"])
    for index, prediction in enumerate(predictions, start=1):
        lines.extend(
            [
                "",
                f"### Prediction {index}",
                f"- reranker_score: {_format_score(prediction.reranker_score)}",
                f"- generator_score: {_format_score(prediction.generator_score)}",
            ]
        )
        if prediction.essence:
            lines.append(f"- essence: {prediction.essence}")
        if prediction.severity:
            lines.append(f"- severity: {prediction.severity}")
        lines.extend(["", prediction.text])
        if prediction.evidence:
            lines.extend(["", "Evidence:"])
            lines.extend(f"- {item}" for item in prediction.evidence)

    if diff_lines > 0:
        diff_excerpt = "\n".join(example.diff.splitlines()[:diff_lines])
        lines.extend(["", "## Diff excerpt", "", "```diff", diff_excerpt, "```"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a GitHub Pull Request with MergeMind.")
    parser.add_argument("--url", required=True, help="GitHub PR URL, e.g. https://github.com/OWNER/REPO/pull/123.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--pipeline", default=BASELINE_MODE, help="Pipeline mode to run.")
    parser.add_argument("--llm-provider", default="", help="Optional provider from llm_providers, e.g. local_qwen36_27b_iq2.")
    parser.add_argument("--limit-comments", type=int, default=None, help="Override number of final comments.")
    parser.add_argument("--max-repository-files", type=int, default=20, help="Max changed files to fetch as repository context.")
    parser.add_argument("--judge", action="store_true", help="Run local LLM judge and save evaluation metrics.")
    parser.add_argument("--diff-lines", type=int, default=80, help="Diff lines to include in report.md.")
    args = parser.parse_args()

    script_started_at = time.perf_counter()
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config(PROJECT_ROOT / args.config)
    config = apply_llm_provider(config, args.llm_provider)
    pipeline_mode = canonical_pipeline_mode(args.pipeline)
    top_n = args.limit_comments if args.limit_comments is not None else int(config.get("demo", {}).get("top_n", 3))

    try:
        example, pr_ref = fetch_github_pr_example(args.url, max_repository_files=args.max_repository_files)
    except (GitHubClientError, ValueError) as error:
        raise SystemExit(f"[review_github_pr] failed to fetch PR: {error}") from error

    generator, reranker, shared_client = build_pipeline_components(pipeline_mode, config, PROJECT_ROOT)
    inference_started_at = time.perf_counter()
    predictions = run_inference(example, generator, reranker, top_n=top_n)
    inference_latency = time.perf_counter() - inference_started_at

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
    if args.judge:
        judge_client = judge_client or build_llm_client(config, PROJECT_ROOT)
        llm_config = dict(config.get("llm", {}))
        judge = OpenAICompatibleLLMJudge(
            judge_client,
            temperature=float(llm_config.get("temperature_judge", 0.0)),
            max_tokens=int(llm_config.get("max_tokens_judge", 400)),
        )
        judge_started_at = time.perf_counter()
        judge_result = judge.evaluate(predictions, gold_comments, example)
        judge_latency = time.perf_counter() - judge_started_at

    output_dir = _artifact_dir(config, pr_ref.safe_id)
    total_wall_latency = time.perf_counter() - script_started_at
    llm_metrics = _llm_metrics(judge_client)
    fallback_count = int(getattr(generator, "fallback_count", 0))
    fallback_count += int(getattr(reranker, "fallback_count", 0))
    fallback_count += int(getattr(judge, "fallback_count", 0)) if judge is not None else 0
    evaluation = {
        "pr": {
            "url": args.url,
            "example_id": example.example_id,
            "repo": example.repo,
            "title": example.title,
            "gold_comment_count": len(gold_comments),
        },
        "pipeline": pipeline_mode,
        "judge_enabled": args.judge,
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
            "pr": {
                "url": args.url,
                "example_id": example.example_id,
                "repo": example.repo,
                "title": example.title,
            },
            "pipeline": pipeline_mode,
            "predictions": [prediction.to_dict() for prediction in predictions],
        },
    )
    write_json(output_dir / "evaluation.json", evaluation)
    (output_dir / "report.md").write_text(
        _render_report(
            example=example,
            url=args.url,
            pipeline_mode=pipeline_mode,
            predictions=predictions,
            evaluation=evaluation,
            diff_lines=max(args.diff_lines, 0),
        ),
        encoding="utf-8",
    )

    changed_paths = [changed_file.path for changed_file in example.changed_files]
    print(f"[review_github_pr] example_id={example.example_id}")
    print(f"[review_github_pr] pipeline={pipeline_mode}")
    print(f"[review_github_pr] title={example.title}")
    print(f"[review_github_pr] changed_files={', '.join(changed_paths) or '<none>'}")
    print(f"[review_github_pr] gold_comment_count={len(gold_comments)}")
    print(f"[review_github_pr] artifacts={output_dir}")
    if args.judge:
        print(
            "[review_github_pr] judge="
            f"{_format_score(judge_result.get('judge_score'))} "
            f"gold_alignment={_format_score(judge_result.get('gold_alignment_score'))} "
            f"valid_alternative={_format_score(judge_result.get('valid_alternative_score'))} "
            f"groundedness={_format_score(judge_result.get('groundedness'))} "
            f"usefulness={_format_score(judge_result.get('usefulness'))}"
        )
        print(f"[review_github_pr] judge_reason={judge_result.get('reason', '')}")
    print(
        "[review_github_pr] metrics="
        f"best_similarity={_format_score(deterministic.get('best_similarity_at_k'))} "
        f"hit_at_k={deterministic.get('hit_at_k', 0)} "
        f"inference_latency={_format_score(inference_latency)}s "
        f"judge_latency={_format_score(judge_latency)}s "
        f"tokens={int(llm_metrics.get('total_tokens', 0)) if llm_metrics else 0}"
    )
    for index, prediction in enumerate(predictions, start=1):
        evidence = "; ".join(prediction.evidence)
        print(f"{index}. score={prediction.reranker_score:.3f} | {prediction.text}")
        if prediction.essence or prediction.severity:
            print(f"   essence={prediction.essence or '<none>'} | severity={prediction.severity or '<none>'}")
        if prediction.original_text and prediction.original_text != prediction.text:
            print(f"   original: {prediction.original_text}")
        print(f"   evidence: {evidence}")


if __name__ == "__main__":
    main()
