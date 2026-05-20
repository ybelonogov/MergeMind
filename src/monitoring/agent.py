"""Monitoring report generator for MergeMind project work."""

from __future__ import annotations

import html
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config
from src.monitoring.dashboard import collect_dashboard_status


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return [item for item in value["results"] if isinstance(item, dict)]
    return []


def _run_command(command: list[str], *, cwd: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "command": command, "stdout": "", "stderr": str(exc), "returncode": None}
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def collect_git_snapshot(project_root: Path) -> dict[str, Any]:
    status = _run_command(["git", "status", "--short", "--branch"], cwd=project_root)
    log = _run_command(["git", "log", "--oneline", "-8"], cwd=project_root)
    branch = status["stdout"].splitlines()[0] if status["stdout"] else ""
    dirty_files = [line for line in status["stdout"].splitlines()[1:] if line.strip()]
    return {
        "branch": branch,
        "dirty_file_count": len(dirty_files),
        "dirty_files": dirty_files,
        "latest_commits": log["stdout"].splitlines() if log["ok"] else [],
        "status_error": status["stderr"],
    }


def collect_swe_ci_runs(project_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    runs_dir = project_root / "artifacts" / "swe_ci_runs"
    if not runs_dir.exists():
        return []
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs[:limit]:
        metrics = _read_json(run_dir / "metrics.json")
        results = _read_json_list(run_dir / "task_results.json")
        latest_errors = []
        for result in results:
            message = str(result.get("error_message", "")).strip()
            if message:
                latest_errors.append({"task_id": result.get("task_id", ""), "error_message": message})
        runs.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "updated_at": run_dir.stat().st_mtime,
                "metrics": metrics,
                "task_count": len(results),
                "latest_errors": latest_errors[:5],
                "summary_path": str(run_dir / "summary.md") if (run_dir / "summary.md").exists() else "",
            }
        )
    return runs


def collect_latest_files(project_root: Path, limit: int = 12) -> list[dict[str, Any]]:
    roots = [project_root / "artifacts" / "runs", project_root / "artifacts" / "swe_ci_runs"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [
        {
            "path": str(path.relative_to(project_root)),
            "updated_at": path.stat().st_mtime,
            "size_bytes": path.stat().st_size,
        }
        for path in files[:limit]
    ]


def collect_monitoring_snapshot(
    *,
    project_root: Path,
    config_path: Path,
    run_tests: bool = False,
    limit: int = 8,
) -> dict[str, Any]:
    config = load_config(config_path)
    dashboard = collect_dashboard_status(config, project_root)
    snapshot: dict[str, Any] = {
        "snapshot_id": _utc_slug(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "git": collect_git_snapshot(project_root),
        "dashboard": dashboard,
        "swe_ci_runs": collect_swe_ci_runs(project_root, limit=limit),
        "latest_files": collect_latest_files(project_root, limit=limit),
    }
    if run_tests:
        snapshot["tests"] = _run_command(["python", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=project_root, timeout=240)
    else:
        snapshot["tests"] = {"ok": None, "stdout": "", "stderr": "", "returncode": None, "skipped": True}
    return snapshot


def _format_ts(timestamp: float) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.000"


def _latest_ab_modes(snapshot: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in snapshot.get("dashboard", {}).get("runs", []):
        for mode in run.get("modes", []):
            rows.append({"run_id": run.get("run_id", ""), **mode})
    return rows[:limit]


def render_chronicle(snapshot: dict[str, Any]) -> str:
    git = snapshot.get("git", {})
    lm = snapshot.get("dashboard", {}).get("lmstudio", {})
    tests = snapshot.get("tests", {})
    ab_modes = _latest_ab_modes(snapshot, limit=10)
    lines = [
        f"# MergeMind monitoring snapshot {snapshot.get('snapshot_id', '')}",
        "",
        f"- Created: `{snapshot.get('created_at', '')}`",
        f"- Branch: `{git.get('branch', '')}`",
        f"- Dirty files: `{git.get('dirty_file_count', 0)}`",
        f"- LM Studio: `{'ok' if lm.get('ok') else 'check'}` `{lm.get('configured_model', '')}` at `{lm.get('base_url', '')}`",
        f"- Tests: `{tests.get('returncode')}`" if not tests.get("skipped") else "- Tests: `not run`",
        "",
        "## Latest Commits",
        "",
    ]
    commits = git.get("latest_commits", [])
    lines.extend(f"- `{commit}`" for commit in commits[:8])
    if not commits:
        lines.append("- <none>")

    lines.extend(["", "## A/B Runs", ""])
    if ab_modes:
        lines.append("| run | mode | model | hit@k | best similarity | judge | latency | status |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
        for mode in ab_modes:
            lines.append(
                "| {run} | {mode} | {model} | {hit} | {sim} | {judge} | {latency}s | {status} |".format(
                    run=mode.get("run_id", ""),
                    mode=mode.get("mode", ""),
                    model=mode.get("model_id", ""),
                    hit=_fmt_float(mode.get("hit_rate_at_k")),
                    sim=_fmt_float(mode.get("best_similarity_at_k")),
                    judge=_fmt_float(mode.get("judge_score")),
                    latency=_fmt_float(mode.get("avg_total_wall_latency_sec"), 2),
                    status=mode.get("status", ""),
                )
            )
    else:
        lines.append("- No A/B run artifacts found.")

    lines.extend(["", "## SWE-CI Runs", ""])
    swe_runs = snapshot.get("swe_ci_runs", [])
    if swe_runs:
        lines.append("| run | tasks | success | failed | timeout | updated |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for run in swe_runs:
            metrics = run.get("metrics", {})
            lines.append(
                "| {run_id} | {tasks} | {success} | {failed} | {timeout} | {updated} |".format(
                    run_id=run.get("run_id", ""),
                    tasks=run.get("task_count", 0),
                    success=metrics.get("success", 0),
                    failed=metrics.get("failed", 0),
                    timeout=metrics.get("timeout", 0),
                    updated=_format_ts(float(run.get("updated_at", 0.0))),
                )
            )
        lines.extend(["", "### SWE-CI Errors", ""])
        for run in swe_runs:
            for error in run.get("latest_errors", []):
                lines.append(f"- `{run.get('run_id', '')}` `{error.get('task_id', '')}`: {error.get('error_message', '')}")
        if not any(run.get("latest_errors") for run in swe_runs):
            lines.append("- No SWE-CI task errors in collected runs.")
    else:
        lines.append("- No SWE-CI run artifacts found.")

    lines.extend(["", "## Latest Artifacts", ""])
    files = snapshot.get("latest_files", [])
    lines.extend(f"- `{item['path']}` ({item['size_bytes']} bytes, {_format_ts(item['updated_at'])})" for item in files)
    if not files:
        lines.append("- <none>")
    lines.append("")
    return "\n".join(lines)


def render_static_dashboard(snapshot: dict[str, Any]) -> str:
    lm = snapshot.get("dashboard", {}).get("lmstudio", {})
    gpu = (snapshot.get("dashboard", {}).get("gpu", {}).get("gpus") or [{}])[0]
    cards = [
        ("LM Studio", "OK" if lm.get("ok") else "CHECK", f"{lm.get('configured_model', '')} @ {lm.get('base_url', '')}"),
        ("GPU", f"{_fmt_float(gpu.get('utilization_gpu'), 0)}%", str(gpu.get("name", "n/a"))),
        ("GPU memory", f"{_fmt_float(gpu.get('memory_used_mb'), 0)} MB", f"of {_fmt_float(gpu.get('memory_total_mb'), 0)} MB"),
        ("SWE-CI runs", str(len(snapshot.get("swe_ci_runs", []))), "collected local artifacts"),
    ]
    rows = []
    for mode in _latest_ab_modes(snapshot, limit=12):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(mode.get('run_id', '')))}</td>"
            f"<td>{html.escape(str(mode.get('mode', '')))}</td>"
            f"<td>{html.escape(str(mode.get('model_id', '')))}</td>"
            f"<td>{_fmt_float(mode.get('hit_rate_at_k'))}</td>"
            f"<td>{_fmt_float(mode.get('best_similarity_at_k'))}</td>"
            f"<td>{_fmt_float(mode.get('judge_score'))}</td>"
            f"<td>{_fmt_float(mode.get('avg_total_wall_latency_sec'), 2)}s</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='7'>No A/B run artifacts found.</td></tr>")
    card_html = "".join(
        f"<section class='card'><div class='label'>{html.escape(title)}</div><div class='value'>{html.escape(value)}</div><div>{html.escape(detail)}</div></section>"
        for title, value, detail in cards
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>MergeMind Monitoring Snapshot</title>
  <style>
    body {{ margin: 0; padding: 28px; font-family: Arial, sans-serif; background: #f6f7f9; color: #151515; }}
    h1 {{ margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 18px 0; }}
    .card {{ background: white; border: 1px solid #d9dde5; border-radius: 14px; padding: 16px; }}
    .label {{ color: #667085; font-size: 13px; }}
    .value {{ font-size: 28px; font-weight: 700; margin: 6px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #d9dde5; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
  <h1>MergeMind Monitoring Snapshot</h1>
  <p>Snapshot: <code>{html.escape(str(snapshot.get('snapshot_id', '')))}</code></p>
  <div class="grid">{card_html}</div>
  <h2>A/B metrics</h2>
  <table>
    <tr><th>run</th><th>mode</th><th>model</th><th>hit@k</th><th>best similarity</th><th>judge</th><th>latency</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""


def render_presentation_markdown(snapshot: dict[str, Any]) -> str:
    lm = snapshot.get("dashboard", {}).get("lmstudio", {})
    swe_runs = snapshot.get("swe_ci_runs", [])
    ab_modes = _latest_ab_modes(snapshot, limit=4)
    best_line = "No completed A/B artifacts found."
    if ab_modes:
        best = max(ab_modes, key=lambda item: float(item.get("hit_rate_at_k") or 0.0))
        best_line = (
            f"Best latest mode: `{best.get('mode', '')}` in `{best.get('run_id', '')}`, "
            f"hit@k={_fmt_float(best.get('hit_rate_at_k'))}, judge={_fmt_float(best.get('judge_score'))}."
        )
    latest_swe = swe_runs[0]["run_id"] if swe_runs else "no SWE-CI run artifacts"
    return "\n".join(
        [
            "# MergeMind: отчет по текущему состоянию",
            "",
            "---",
            "",
            "## Что уже собрано",
            "",
            "- Локальный MR review pipeline: data -> context -> generator -> reranker -> rewriter -> validation.",
            "- Интеграция LM Studio / Qwen с cache, run artifacts и dashboard.",
            "- SWE-CI wrapper для baseline и MergeMind review-loop экспериментов.",
            "",
            "---",
            "",
            "## Текущий мониторинг",
            "",
            f"- Snapshot: `{snapshot.get('snapshot_id', '')}`.",
            f"- LM Studio: `{lm.get('configured_model', '')}` at `{lm.get('base_url', '')}`.",
            f"- Git: `{snapshot.get('git', {}).get('branch', '')}`.",
            f"- Latest SWE-CI run: `{latest_swe}`.",
            "",
            "---",
            "",
            "## Последние метрики",
            "",
            f"- {best_line}",
            "- Deterministic metrics используются как базовый сравнимый сигнал.",
            "- Judge/usefulness/groundedness используются как исследовательские сигналы.",
            "",
            "---",
            "",
            "## Следующие решения",
            "",
            "- Стабилизировать SWE-CI agent runtime и task downloads.",
            "- Решить, должен ли review-loop feedback стать реальной второй итерацией агента.",
            "- Добавить ручную разметку relevance/usefulness/groundedness.",
            "- Сравнивать размеры локальных моделей только после стабилизации оценки.",
            "",
        ]
    )


def write_monitoring_artifacts(
    snapshot: dict[str, Any],
    *,
    output_dir: Path,
    append_chronicle: bool = True,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / str(snapshot.get("snapshot_id", _utc_slug()))
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    chronicle = render_chronicle(snapshot)
    dashboard = render_static_dashboard(snapshot)
    presentation = render_presentation_markdown(snapshot)

    paths = {
        "snapshot": str(snapshot_dir / "snapshot.json"),
        "chronicle": str(snapshot_dir / "chronicle.md"),
        "dashboard": str(snapshot_dir / "dashboard.html"),
        "presentation": str(snapshot_dir / "presentation.md"),
        "latest_snapshot": str(output_dir / "latest_snapshot.json"),
        "latest_dashboard": str(output_dir / "latest_dashboard.html"),
        "latest_presentation": str(output_dir / "latest_presentation.md"),
    }

    Path(paths["snapshot"]).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["chronicle"]).write_text(chronicle, encoding="utf-8")
    Path(paths["dashboard"]).write_text(dashboard, encoding="utf-8")
    Path(paths["presentation"]).write_text(presentation, encoding="utf-8")
    Path(paths["latest_snapshot"]).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(paths["latest_dashboard"]).write_text(dashboard, encoding="utf-8")
    Path(paths["latest_presentation"]).write_text(presentation, encoding="utf-8")

    if append_chronicle:
        cumulative_path = output_dir / "chronicle.md"
        previous = cumulative_path.read_text(encoding="utf-8") if cumulative_path.exists() else "# MergeMind Chronicle\n\n"
        cumulative_path.write_text(previous.rstrip() + "\n\n---\n\n" + chronicle, encoding="utf-8")
        paths["cumulative_chronicle"] = str(cumulative_path)

    # Keep a tiny machine-readable heartbeat for external dashboards.
    heartbeat = {"snapshot_id": snapshot.get("snapshot_id"), "created_at": snapshot.get("created_at"), "paths": paths}
    (output_dir / "heartbeat.json").write_text(json.dumps(heartbeat, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["heartbeat"] = str(output_dir / "heartbeat.json")
    return paths


def sleep_until_next_interval(started_at: float, interval_seconds: int) -> None:
    elapsed = time.time() - started_at
    delay = max(0.0, interval_seconds - elapsed)
    time.sleep(delay)
