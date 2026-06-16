"""Analyze requirement.xml drift and MergeMind injection points in SWE-CI runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


_bootstrap_path()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def _load_task_results(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "task_results.json"
    payload = _load_json(path)
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a result list or a 'results' list.")
    return [row for row in rows if isinstance(row, dict) and row.get("task_id")]


def _candidate_iteration_files(run_dir: Path, task_id: str) -> list[Path]:
    return sorted(
        path
        for path in run_dir.glob(f"workdirs/**/experiments/*/{task_id}/iteration.jsonl")
        if path.is_file()
    )


def _iteration_file_for_result(run_dir: Path, result: dict[str, Any]) -> Path | None:
    metrics = result.get("metrics", {})
    path_value = metrics.get("swe_ci_iteration_file") if isinstance(metrics, dict) else None
    if path_value:
        path = Path(str(path_value))
        if path.is_file():
            return path
        if not path.is_absolute():
            candidate = run_dir / path
            if candidate.is_file():
                return candidate
        parts = path.parts
        if "workdirs" in parts:
            suffix = Path(*parts[parts.index("workdirs") :])
            candidate = run_dir / suffix
            if candidate.is_file():
                return candidate
    candidates = _candidate_iteration_files(run_dir, str(result["task_id"]))
    return candidates[0] if candidates else None


def _requirement_paths(iteration_file: Path) -> list[Path]:
    task_dir = iteration_file.parent
    paths = [
        path
        for path in task_dir.glob("*/requirement.xml")
        if path.parent.name != "target" and path.is_file()
    ]
    return sorted(paths, key=lambda path: path.parent.name)


def _requirement_for_iteration(index: int, requirement_paths: list[Path], row_count: int) -> Path | None:
    if not requirement_paths:
        return None
    if len(requirement_paths) == row_count:
        return requirement_paths[index] if index < len(requirement_paths) else None
    adjusted = index - 1
    return requirement_paths[adjusted] if 0 <= adjusted < len(requirement_paths) else None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _excerpt(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _review_payload(row: dict[str, Any]) -> dict[str, Any]:
    review = row.get("mergemind_review")
    return review if isinstance(review, dict) else {}


def _revision_payload(row: dict[str, Any]) -> dict[str, Any]:
    revision = row.get("programmer_revision")
    return revision if isinstance(revision, dict) else {}


def analyze_task(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    task_id = str(result["task_id"])
    iteration_file = _iteration_file_for_result(run_dir, result)
    if iteration_file is None:
        return {
            "task_id": task_id,
            "status": "missing_iteration_file",
            "iteration_file": "",
            "iterations": [],
        }

    rows = _load_jsonl(iteration_file)
    requirements = _requirement_paths(iteration_file)
    previous_requirement_hash: str | None = None
    iterations: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        requirement_path = _requirement_for_iteration(index, requirements, len(rows))
        requirement_hash = None
        requirement_changed = None
        requirement_excerpt = ""
        if requirement_path is not None:
            requirement_text = requirement_path.read_text(encoding="utf-8", errors="replace")
            requirement_hash = _sha256_text(requirement_text)
            requirement_changed = (
                None if previous_requirement_hash is None else requirement_hash != previous_requirement_hash
            )
            previous_requirement_hash = requirement_hash
            requirement_excerpt = _excerpt(requirement_text)

        pytest_payload = row.get("pytest", {}) if isinstance(row.get("pytest"), dict) else {}
        review = _review_payload(row)
        revision = _revision_payload(row)
        iterations.append(
            {
                "iteration_index": index,
                "epoch": review.get("epoch") or (index if index > 0 else 0),
                "gap": _int_value(row.get("gap")),
                "passed": _int_value(pytest_payload.get("passed")),
                "requirement_path": str(requirement_path) if requirement_path is not None else "",
                "requirement_hash": requirement_hash,
                "requirement_changed_vs_previous": requirement_changed,
                "requirement_excerpt": requirement_excerpt,
                "mergemind_review_status": review.get("status", ""),
                "mergemind_comment_count": _int_value(review.get("comment_count")) or 0,
                "mergemind_apply_revision": review.get("apply_revision"),
                "target_sha_used_for_review": review.get("target_sha_used_for_review"),
                "programmer_revision_present": bool(revision),
                "programmer_revision_changed_files": revision.get("changed_files", []),
            }
        )

    return {
        "task_id": task_id,
        "status": "ok",
        "iteration_file": str(iteration_file),
        "requirement_file_count": len(requirements),
        "review_injection_point": (
            "MergeMind review is generated after the programmer patch, copied as "
            "/app/mergemind_review.md with /app/requirement.xml, and consumed by "
            "the programmer revision pass before pytest."
        ),
        "iterations": iterations,
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    tasks = [analyze_task(run_dir, result) for result in _load_task_results(run_dir)]
    return {
        "run_dir": str(run_dir),
        "task_count": len(tasks),
        "tasks": tasks,
    }


def _md(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SWE-CI Requirement And MergeMind Injection Analysis",
        "",
        f"- run: `{payload['run_dir']}`",
        f"- tasks: {payload['task_count']}",
        "",
        "MergeMind assisted injection point: after the programmer patch, before pytest.",
        "The revision container receives `/app/requirement.xml` and `/app/mergemind_review.md`.",
        "",
    ]
    for task in payload["tasks"]:
        lines.extend(
            [
                f"## {task['task_id']}",
                "",
                f"- status: `{task['status']}`",
                f"- iteration file: `{task['iteration_file']}`",
            ]
        )
        if task["status"] != "ok":
            lines.append("")
            continue
        lines.extend(
            [
                f"- requirement files: {task['requirement_file_count']}",
                f"- injection point: {task['review_injection_point']}",
                "",
                "| iter | epoch | gap | passed | requirement_changed | review_status | comments | apply_revision | revision_present | target_sha_used | requirement_excerpt |",
                "| ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for row in task["iterations"]:
            lines.append(
                "| {iteration_index} | {epoch} | {gap} | {passed} | {requirement_changed} | "
                "{review_status} | {comments} | {apply_revision} | {revision_present} | "
                "{target_sha_used} | {excerpt} |".format(
                    iteration_index=_md(row["iteration_index"]),
                    epoch=_md(row["epoch"]),
                    gap=_md(row["gap"]),
                    passed=_md(row["passed"]),
                    requirement_changed=_md(row["requirement_changed_vs_previous"]),
                    review_status=_md(row["mergemind_review_status"]),
                    comments=_md(row["mergemind_comment_count"]),
                    apply_revision=_md(row["mergemind_apply_revision"]),
                    revision_present=_md(row["programmer_revision_present"]),
                    target_sha_used=_md(row["target_sha_used_for_review"]),
                    excerpt=_md(row["requirement_excerpt"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze requirement.xml drift and MergeMind review injection in a SWE-CI run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="", help="Optional markdown output path.")
    parser.add_argument("--json-output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    payload = analyze_run(Path(args.run_dir))
    markdown = render_markdown(payload)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
