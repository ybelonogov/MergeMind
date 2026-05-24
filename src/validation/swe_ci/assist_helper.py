"""Host-side MergeMind helper for SWE-CI assisted iterations."""

from __future__ import annotations

import argparse
import difflib
import json
import time
from pathlib import Path
from typing import Any

from src.config import apply_llm_provider, load_config
from src.context.processing import enrich_example
from src.data.io import write_json
from src.data.schema import CandidateComment, MRExample
from src.inference.factory import build_pipeline_components, pipeline_uses_llm_judge
from src.inference.pipeline import run_inference
from src.models.llm import build_llm_client
from src.validation.metrics import OpenAICompatibleLLMJudge

_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg"}
_MAX_FILE_BYTES = 500_000


def _is_test_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def _iter_source_files(root: Path) -> dict[Path, Path]:
    if not root.exists():
        return {}
    files: dict[Path, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        if _is_test_path(relative):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        files[relative] = path
    return files


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    if path.stat().st_size > _MAX_FILE_BYTES:
        return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if b"\x00" in raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="ignore")


def _test_failed(test: dict[str, Any]) -> bool:
    if str(test.get("outcome", "")).lower() not in {"", "passed"}:
        return True
    for phase in ("setup", "call", "teardown"):
        payload = test.get(phase)
        if isinstance(payload, dict) and str(payload.get("outcome", "")).lower() not in {"", "passed", "skipped"}:
            return True
    return False


def _load_failed_nodeids(report_path: Path) -> list[str]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tests = payload.get("tests", []) if isinstance(payload, dict) else []
    if not isinstance(tests, list):
        return []
    nodeids: list[str] = []
    for test in tests:
        if not isinstance(test, dict) or not _test_failed(test):
            continue
        nodeid = str(test.get("nodeid", "")).strip()
        if nodeid:
            nodeids.append(nodeid)
    return sorted(set(nodeids))


def build_previous_failure_context(before_code_dir: str | Path, *, max_nodeids: int = 30) -> str:
    """Return a compact summary of visible pytest failures before this revision."""

    report_path = Path(before_code_dir).parent / "test_report.json"
    if not report_path.exists():
        return ""
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    failed_nodeids = _load_failed_nodeids(report_path)
    lines = ["Visible previous pytest failures before this revision:"]
    if isinstance(summary, dict) and summary:
        lines.append(
            "summary: "
            + ", ".join(f"{key}={summary[key]}" for key in sorted(summary) if key in {"failed", "error", "passed", "total", "collected"})
        )
    if failed_nodeids:
        lines.append("failed_nodeids:")
        for nodeid in failed_nodeids[:max_nodeids]:
            lines.append(f"- {nodeid}")
        remaining = len(failed_nodeids) - max_nodeids
        if remaining > 0:
            lines.append(f"- ... {remaining} more")
    return "\n".join(lines).strip()


def build_code_diff(before_code_dir: str | Path, after_code_dir: str | Path) -> str:
    """Build a unified diff for source changes, excluding test files."""

    before_root = Path(before_code_dir)
    after_root = Path(after_code_dir)
    before_files = _iter_source_files(before_root)
    after_files = _iter_source_files(after_root)
    chunks: list[str] = []
    for relative in sorted(set(before_files) | set(after_files), key=lambda item: item.as_posix()):
        before_text = _read_text(before_files.get(relative))
        after_text = _read_text(after_files.get(relative))
        if before_text == after_text:
            continue
        rel_posix = relative.as_posix()
        chunks.append(f"diff --git a/{rel_posix} b/{rel_posix}")
        chunks.extend(
            difflib.unified_diff(
                before_text.splitlines(),
                after_text.splitlines(),
                fromfile=f"a/{rel_posix}",
                tofile=f"b/{rel_posix}",
                lineterm="",
            )
        )
    return "\n".join(chunks) + ("\n" if chunks else "")


def build_assist_example(
    *,
    task_id: str,
    repo_name: str,
    repo_url: str,
    current_sha: str,
    image_sha: str,
    requirement_text: str,
    diff_text: str,
    failure_context_text: str = "",
) -> MRExample:
    """Create an MRExample for a programmer patch without exposing oracle fields."""

    return MRExample(
        source_dataset="swe-ci-assisted-diff",
        example_id=f"swe-ci:{task_id}:assisted_patch",
        split="swe-ci-assisted",
        repo=repo_name,
        title=f"SWE-CI assisted review for {task_id}",
        description="\n".join(
            [
                "Review the patch produced by the SWE-CI programmer before pytest is executed.",
                f"Repository: {repo_url}",
                f"Base commit: {current_sha}",
                "Hidden solution data is intentionally omitted from this review context.",
                "",
                "Architect requirement:",
                requirement_text.strip(),
                "",
                failure_context_text.strip(),
            ]
        ).strip(),
        diff=diff_text,
        ci_signals={"swe_ci_task_id": task_id, "previous_failure_context": failure_context_text},
        metadata={
            "swe_ci_task_id": task_id,
            "repo_url": repo_url,
            "current_sha": current_sha,
            "image_sha": image_sha,
            "target_sha_used_for_review": False,
        },
    )


def render_review_markdown(predictions: list[CandidateComment], status: str, error_message: str = "") -> str:
    lines = [
        "# MergeMind Review Guidance",
        "",
        "Use these review comments to revise the current code before pytest is executed.",
        "Do not edit tests. Keep the change minimal and aligned with requirement.xml.",
        "Ignore any comment that is already addressed or would require broad unrelated rewrites.",
        "",
        f"- status: {status}",
        "- target_sha_used_for_review: false",
        "",
        "## Comments",
        "",
    ]
    if error_message:
        lines.extend(["## Error", "", error_message, ""])
    if not predictions:
        lines.append("- No actionable MergeMind comments were generated.")
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


def _write_result(output_dir: Path, payload: dict[str, Any], predictions: list[CandidateComment] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "mergemind_review.md"
    comments_path = output_dir / "mergemind_comments.json"
    result_path = output_dir / "assist_result.json"
    payload = {
        "target_sha_used_for_review": False,
        "review_path": str(review_path),
        "comments_path": str(comments_path),
        "result_path": str(result_path),
        **payload,
    }
    predictions = predictions or []
    review_path.write_text(
        render_review_markdown(predictions, str(payload.get("status", "")), str(payload.get("error_message", ""))),
        encoding="utf-8",
    )
    write_json(
        comments_path,
        {
            **payload,
            "comments": [prediction.to_dict() for prediction in predictions],
        },
    )
    write_json(result_path, payload)
    return payload


def run_mergemind_assist(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    started = time.perf_counter()
    diff_text = build_code_diff(args.before_code_dir, args.after_code_dir)
    if not diff_text.strip():
        return _write_result(
            output_dir,
            {
                "status": "skipped",
                "error_message": "No non-test code diff was produced by the programmer.",
                "comment_count": 0,
                "raw_comment_count": 0,
                "filtered_comment_count": 0,
                "apply_revision": False,
                "runtime": {"review_latency_seconds": time.perf_counter() - started},
            },
        )

    requirement_text = Path(args.requirement_path).read_text(encoding="utf-8", errors="ignore")
    failure_context_text = ""
    if "test_triage" in str(args.pipeline):
        failure_context_text = build_previous_failure_context(args.before_code_dir)
    config = apply_llm_provider(load_config(args.config), args.llm_provider)
    example = enrich_example(
        build_assist_example(
            task_id=args.task_id,
            repo_name=args.repo_name,
            repo_url=args.repo_url,
            current_sha=args.current_sha,
            image_sha=args.image_sha,
            requirement_text=requirement_text,
            diff_text=diff_text,
            failure_context_text=failure_context_text,
        )
    )
    write_json(output_dir / "mergemind_example.json", example.to_dict())

    generator, reranker, shared_client = build_pipeline_components(args.pipeline, config, Path(args.project_root))
    raw_predictions = run_inference(example, generator, reranker, top_n=args.top_n)
    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    predictions = [prediction for prediction in raw_predictions if prediction.reranker_score >= min_score]
    filtered_comment_count = len(raw_predictions) - len(predictions)
    max_revision_epochs = getattr(args, "max_revision_epochs", None)
    apply_revision = bool(predictions) and (
        max_revision_epochs is None or int(getattr(args, "epoch", 0) or 0) <= int(max_revision_epochs)
    )
    judge_result: dict[str, Any] = {}
    if pipeline_uses_llm_judge(args.pipeline):
        llm_config = dict(config.get("llm", {}))
        judge_client = shared_client or build_llm_client(config, Path(args.project_root))
        judge = OpenAICompatibleLLMJudge(
            judge_client,
            temperature=float(llm_config.get("temperature_judge", 0.0)),
            max_tokens=int(llm_config.get("max_tokens_judge", 400)),
        )
        judge_result = judge.evaluate(predictions, [], example)
        shared_client = judge_client

    return _write_result(
        output_dir,
        {
            "status": "success" if predictions else "skipped",
            "error_message": "" if predictions else "MergeMind returned no comments.",
            "task_id": args.task_id,
            "epoch": int(getattr(args, "epoch", 0) or 0),
            "repo_name": args.repo_name,
            "repo_url": args.repo_url,
            "current_sha": args.current_sha,
            "comment_count": len(predictions),
            "raw_comment_count": len(raw_predictions),
            "filtered_comment_count": filtered_comment_count,
            "min_score": min_score,
            "max_revision_epochs": max_revision_epochs,
            "apply_revision": apply_revision,
            "judge": judge_result,
            "runtime": {"review_latency_seconds": time.perf_counter() - started},
            "llm_stats": shared_client.stats() if shared_client is not None else {},
        },
        predictions,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MergeMind review guidance for one SWE-CI assisted epoch.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--llm-provider", default="")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-revision-epochs", type=int, default=None)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--current-sha", required=True)
    parser.add_argument("--image-sha", required=True)
    parser.add_argument("--before-code-dir", required=True)
    parser.add_argument("--after-code-dir", required=True)
    parser.add_argument("--requirement-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_mergemind_assist(args)
    except Exception as exc:  # noqa: BLE001 - helper must return structured failure to SWE-CI.
        output_dir = Path(args.output_dir).resolve()
        result = _write_result(
            output_dir,
            {
                "status": "error",
                "error_message": repr(exc),
                "task_id": args.task_id,
                "repo_name": args.repo_name,
                "repo_url": args.repo_url,
                "current_sha": args.current_sha,
                "comment_count": 0,
            },
        )
        print(json.dumps(result, ensure_ascii=True), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if result.get("status") in {"success", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
