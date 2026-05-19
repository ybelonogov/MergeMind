"""MergeMind review sidecar for SWE-CI agent patches."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import apply_llm_provider, load_config, resolve_path
from src.context.processing import enrich_example
from src.data.io import write_json
from src.data.schema import CandidateComment, MRExample
from src.inference.factory import build_pipeline_components, canonical_pipeline_mode, pipeline_uses_llm_judge
from src.inference.pipeline import run_inference
from src.models.llm import build_llm_client
from src.validation.metrics import OpenAICompatibleLLMJudge

from .schemas import SweCiRunConfig, SweCiTask, SweCiTaskRunResult

PATCH_FILE_NAMES = (
    "patch.diff",
    "patch.patch",
    "agent.patch",
    "agent.diff",
    "model.patch",
    "model.diff",
    "prediction.patch",
    "prediction.diff",
    "output.patch",
    "output.diff",
)
PATCH_SUFFIXES = (".patch", ".diff")
PATCH_JSON_KEYS = (
    "patch",
    "diff",
    "model_patch",
    "agent_patch",
    "prediction_patch",
    "generated_patch",
)
MAX_PATCH_BYTES = 2_000_000


@dataclass(frozen=True)
class AgentPatch:
    text: str
    source_path: str
    source_type: str


def looks_like_unified_diff(text: str) -> bool:
    return "diff --git " in text or ("@@ " in text and "--- " in text and "+++ " in text)


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_PATCH_BYTES:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _candidate_patch_files(task_output_dir: Path) -> list[Path]:
    if not task_output_dir.exists():
        return []
    paths: list[Path] = []
    for path in task_output_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in PATCH_FILE_NAMES or path.suffix.lower() in PATCH_SUFFIXES:
            paths.append(path)
    return sorted(paths, key=lambda item: (item.name.lower() not in PATCH_FILE_NAMES, len(str(item)), str(item)))


def _walk_json_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PATCH_JSON_KEYS and isinstance(item, str):
                strings.append(item)
            else:
                strings.extend(_walk_json_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(_walk_json_strings(item))
    return strings


def find_agent_patch(task_output_dir: str | Path) -> AgentPatch | None:
    output_dir = Path(task_output_dir)
    for path in _candidate_patch_files(output_dir):
        text = _read_text(path)
        if looks_like_unified_diff(text):
            return AgentPatch(text=text, source_path=str(path), source_type="file")

    if not output_dir.exists():
        return None
    for path in sorted(output_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for text in _walk_json_strings(payload):
            if looks_like_unified_diff(text):
                return AgentPatch(text=text, source_path=str(path), source_type="json")
    return None


def build_review_example(task: SweCiTask, patch: AgentPatch) -> MRExample:
    """Build an MRExample from an agent patch without exposing target_sha."""

    return MRExample(
        source_dataset="swe-ci-agent-patch",
        example_id=f"swe-ci:{task.task_id}:agent_patch",
        split=str(task.metadata.get("splitting", "swe-ci")),
        repo=task.repo_name,
        title=f"SWE-CI agent patch review for {task.task_id}",
        description="\n".join(
            [
                "Review the patch produced by the SWE-CI coding-agent.",
                f"Repository: {task.repo_url}",
                f"Base commit: {task.current_sha}",
                "Gold target commit is intentionally omitted from this review context.",
            ]
        ),
        diff=patch.text,
        ci_signals={"swe_ci_task_id": task.task_id},
        metadata={
            "swe_ci_task_id": task.task_id,
            "repo_url": task.repo_url,
            "current_sha": task.current_sha,
            "image_sha": task.image_sha,
            "patch_source_path": patch.source_path,
            "patch_source_type": patch.source_type,
            "target_sha_used_for_review": False,
        },
    )


def _render_review_markdown(
    task: SweCiTask,
    patch: AgentPatch | None,
    predictions: list[CandidateComment],
    judge_result: dict[str, Any],
    status: str,
    error_message: str,
) -> str:
    lines = [
        f"# MergeMind SWE-CI Review: {task.task_id}",
        "",
        f"- status: {status}",
        f"- repo: {task.repo_name}",
        f"- repo_url: {task.repo_url}",
        f"- current_sha: {task.current_sha}",
        "- target_sha_used_for_review: false",
    ]
    if patch is not None:
        lines.extend([f"- patch_source: {patch.source_path}", f"- patch_source_type: {patch.source_type}"])
    if error_message:
        lines.extend(["", "## Error", "", error_message])
    if judge_result:
        lines.extend(
            [
                "",
                "## Judge",
                "",
                f"- judge_score: {judge_result.get('judge_score', 0.0)}",
                f"- valid_alternative_score: {judge_result.get('valid_alternative_score', 0.0)}",
                f"- groundedness: {judge_result.get('groundedness', 0.0)}",
                f"- usefulness: {judge_result.get('usefulness', 0.0)}",
                f"- reason: {judge_result.get('reason', '')}",
            ]
        )
    lines.extend(["", "## Comments", ""])
    if not predictions:
        lines.append("- No MergeMind comments were generated.")
    for index, prediction in enumerate(predictions, start=1):
        lines.extend(
            [
                f"### Comment {index}",
                "",
                prediction.text,
                "",
                f"- score: {prediction.reranker_score:.3f}",
                f"- severity: {prediction.severity or 'n/a'}",
                f"- essence: {prediction.essence or 'n/a'}",
                "",
            ]
        )
    return "\n".join(lines)


def write_mergemind_review_artifacts(
    task_output_dir: str | Path,
    task: SweCiTask,
    patch: AgentPatch | None,
    example: MRExample | None,
    predictions: list[CandidateComment],
    judge_result: dict[str, Any] | None = None,
    status: str = "success",
    error_message: str = "",
    runtime: dict[str, Any] | None = None,
    llm_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(task_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comments_path = output_dir / "mergemind_comments.json"
    review_path = output_dir / "mergemind_review.md"
    example_path = output_dir / "mergemind_example.json"

    if example is not None:
        write_json(example_path, example.to_dict())

    payload = {
        "status": status,
        "error_message": error_message,
        "task_id": task.task_id,
        "repo_name": task.repo_name,
        "repo_url": task.repo_url,
        "current_sha": task.current_sha,
        "target_sha_used_for_review": False,
        "patch_source_path": patch.source_path if patch else "",
        "patch_source_type": patch.source_type if patch else "",
        "comment_count": len(predictions),
        "comments": [prediction.to_dict() for prediction in predictions],
        "judge": judge_result or {},
        "runtime": runtime or {},
        "llm_stats": llm_stats or {},
    }
    write_json(comments_path, payload)
    review_path.write_text(
        _render_review_markdown(
            task=task,
            patch=patch,
            predictions=predictions,
            judge_result=judge_result or {},
            status=status,
            error_message=error_message,
        ),
        encoding="utf-8",
    )
    return {
        "status": status,
        "error_message": error_message,
        "comments_path": str(comments_path),
        "review_path": str(review_path),
        "example_path": str(example_path) if example is not None else "",
        "comment_count": len(predictions),
        "target_sha_used_for_review": False,
    }


def run_mergemind_patch_review(
    config: SweCiRunConfig,
    task: SweCiTask,
    task_result: SweCiTaskRunResult,
    task_output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    task_dir = Path(task_output_dir)
    patch = find_agent_patch(task_dir)
    if patch is None:
        return write_mergemind_review_artifacts(
            task_output_dir=task_dir,
            task=task,
            patch=None,
            example=None,
            predictions=[],
            status="skipped",
            error_message="Could not locate a coding-agent patch/diff in SWE-CI outputs.",
            runtime={"swe_ci_status": task_result.status},
        )

    started_at = time.perf_counter()
    project_root_path = Path(project_root)
    config_path = config.mergemind_config_path or (project_root_path / "configs" / "base.yaml")
    mergemind_config = load_config(config_path)
    mergemind_config = apply_llm_provider(mergemind_config, config.mergemind_llm_provider)
    example = enrich_example(build_review_example(task, patch))

    generator, reranker, shared_client = build_pipeline_components(
        config.mergemind_pipeline,
        mergemind_config,
        project_root_path,
    )
    predictions = run_inference(example, generator, reranker, top_n=config.mergemind_top_n)
    judge_result: dict[str, Any] = {}
    if pipeline_uses_llm_judge(config.mergemind_pipeline):
        llm_config = dict(mergemind_config.get("llm", {}))
        judge_client = shared_client or build_llm_client(mergemind_config, project_root_path)
        judge = OpenAICompatibleLLMJudge(
            judge_client,
            temperature=float(llm_config.get("temperature_judge", 0.0)),
            max_tokens=int(llm_config.get("max_tokens_judge", 400)),
        )
        judge_result = judge.evaluate(predictions, [], example)
        shared_client = judge_client

    runtime = {
        "swe_ci_status": task_result.status,
        "review_latency_seconds": time.perf_counter() - started_at,
    }
    llm_stats = shared_client.stats() if shared_client is not None else {}
    return write_mergemind_review_artifacts(
        task_output_dir=task_dir,
        task=task,
        patch=patch,
        example=example,
        predictions=predictions,
        judge_result=judge_result,
        runtime=runtime,
        llm_stats=llm_stats,
    )
