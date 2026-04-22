"""Local dashboard server for MergeMind experiment monitoring."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import resolve_path


GPU_FIELDS = [
    "name",
    "utilization_gpu",
    "memory_used_mb",
    "memory_total_mb",
    "temperature_c",
    "power_w",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return {}
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def _to_float(value: str) -> float:
    cleaned = value.strip().replace("%", "").replace("MiB", "").replace("W", "")
    if cleaned.lower() in {"n/a", "[n/a]", ""}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_nvidia_smi_csv(output: str) -> list[dict[str, Any]]:
    """Parse nvidia-smi CSV output into dashboard GPU records."""

    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(GPU_FIELDS):
            continue
        records.append(
            {
                "name": parts[0],
                "utilization_gpu": _to_float(parts[1]),
                "memory_used_mb": _to_float(parts[2]),
                "memory_total_mb": _to_float(parts[3]),
                "temperature_c": _to_float(parts[4]),
                "power_w": _to_float(parts[5]),
            }
        )
    return records


def collect_gpu_stats() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3)
    except Exception as exc:  # noqa: BLE001 - dashboard should report monitoring failures, not crash.
        return {"ok": False, "error": str(exc), "gpus": []}
    return {"ok": True, "error": "", "gpus": parse_nvidia_smi_csv(result.stdout)}


def collect_lmstudio_status(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = dict(config.get("llm", {}))
    base_url = str(llm_config.get("base_url", "http://localhost:1234/v1")).rstrip("/")
    configured_model = str(llm_config.get("model", ""))
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "base_url": base_url,
            "configured_model": configured_model,
            "available_models": [],
            "model_available": False,
            "error": str(exc),
        }

    models = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
    return {
        "ok": configured_model in models,
        "base_url": base_url,
        "configured_model": configured_model,
        "available_models": models,
        "model_available": configured_model in models,
        "error": "",
    }


def _mode_progress(summary: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    total = int(progress.get("total") or summary.get("example_count") or 0)
    completed = int(progress.get("completed") or (summary.get("example_count") if summary else 0) or 0)
    percent = (completed / total * 100.0) if total else 0.0
    llm_metrics = dict(progress.get("llm_metrics", {}))
    return {
        "completed": completed,
        "total": total,
        "percent": percent,
        "latest_example_id": progress.get("example_id", ""),
        "latest_latency_sec": float(progress.get("latency_sec") or 0.0),
        "latest_inference_latency_sec": float(progress.get("inference_latency_sec") or progress.get("latency_sec") or 0.0),
        "latest_judge_latency_sec": float(progress.get("judge_latency_sec") or 0.0),
        "latest_total_wall_latency_sec": float(progress.get("total_wall_latency_sec") or progress.get("latency_sec") or 0.0),
        "tokens_per_second": float(
            summary.get("uncached_tokens_per_second")
            or llm_metrics.get("uncached_tokens_per_second")
            or summary.get("tokens_per_second")
            or llm_metrics.get("tokens_per_second")
            or 0.0
        ),
        "uncached_tokens_per_second": float(
            summary.get("uncached_tokens_per_second") or llm_metrics.get("uncached_tokens_per_second") or 0.0
        ),
        "total_tokens": int(summary.get("total_tokens") or llm_metrics.get("total_tokens") or 0),
        "uncached_total_tokens": int(
            summary.get("uncached_total_tokens") or llm_metrics.get("uncached_total_tokens") or 0
        ),
        "cached_call_count": int(summary.get("cached_call_count") or llm_metrics.get("cached_call_count") or 0),
        "uncached_call_count": int(summary.get("uncached_call_count") or llm_metrics.get("uncached_call_count") or 0),
        "parse_error_rate": float(summary.get("parse_error_rate") or llm_metrics.get("parse_error_rate") or 0.0),
        "cache_hit_rate": float(summary.get("cache_hit_rate") or llm_metrics.get("cache_hit_rate") or 0.0),
    }


def collect_runs(runs_dir: Path, limit: int = 12) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    runs = []
    for run_dir in run_dirs[:limit]:
        metrics_table = _read_json(run_dir / "metrics_table.json")
        run_record = {
            "run_id": run_dir.name,
            "path": str(run_dir),
            "updated_at": run_dir.stat().st_mtime,
            "metrics_table": metrics_table.get("rows", []),
            "modes": [],
        }
        mode_dirs = [path for path in run_dir.iterdir() if path.is_dir()]
        mode_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for mode_dir in mode_dirs:
            summary = _read_json(mode_dir / "summary.json")
            manifest = _read_json(mode_dir / "run_manifest.json")
            progress = _read_last_jsonl(mode_dir / "progress.jsonl")
            status = "completed" if summary else ("running" if progress else "pending")
            progress_view = _mode_progress(summary, progress)
            run_record["modes"].append(
                {
                    "mode": mode_dir.name,
                    "path": str(mode_dir),
                    "status": status,
                    "profile": summary.get("profile") or manifest.get("profile") or progress.get("profile", ""),
                    "example_count": summary.get("example_count", progress_view["total"]),
                    "top1_similarity": summary.get("top1_similarity", 0.0),
                    "best_similarity_at_k": summary.get("best_similarity_at_k", 0.0),
                    "hit_rate_at_k": summary.get("hit_rate_at_k", 0.0),
                    "mrr_at_k": summary.get("mrr_at_k", 0.0),
                    "judge_score": summary.get("judge_score", 0.0),
                    "avg_latency_sec": summary.get("avg_latency_sec", 0.0),
                    "p95_latency_sec": summary.get("p95_latency_sec", 0.0),
                    "avg_inference_latency_sec": summary.get("avg_inference_latency_sec", summary.get("avg_latency_sec", 0.0)),
                    "p95_inference_latency_sec": summary.get("p95_inference_latency_sec", summary.get("p95_latency_sec", 0.0)),
                    "avg_judge_latency_sec": summary.get("avg_judge_latency_sec", 0.0),
                    "p95_judge_latency_sec": summary.get("p95_judge_latency_sec", 0.0),
                    "avg_total_wall_latency_sec": summary.get(
                        "avg_total_wall_latency_sec", summary.get("avg_latency_sec", 0.0)
                    ),
                    "p95_total_wall_latency_sec": summary.get(
                        "p95_total_wall_latency_sec", summary.get("p95_latency_sec", 0.0)
                    ),
                    "progress": progress_view,
                }
            )
        runs.append(run_record)
    return runs


def collect_dashboard_status(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    runs_dir = resolve_path(project_root, config.get("paths", {}).get("runs_dir", "artifacts/runs"))
    return {
        "timestamp": time.time(),
        "gpu": collect_gpu_stats(),
        "lmstudio": collect_lmstudio_status(config),
        "runs_dir": str(runs_dir),
        "runs": collect_runs(runs_dir),
    }


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MergeMind Dashboard</title>
  <style>
    :root {
      --bg: #101318;
      --panel: #181d24;
      --panel-2: #202734;
      --text: #eef2f8;
      --muted: #9aa8ba;
      --accent: #55d6be;
      --warn: #ffcc66;
      --bad: #ff7a7a;
      --line: #2e3745;
    }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, #213448, var(--bg) 45%);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { max-width: 1200px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: -0.04em; }
    h2 { margin: 0 0 14px; font-size: 20px; }
    .muted { color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 24px 0; }
    .card { background: color-mix(in srgb, var(--panel) 92%, transparent); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 16px 40px rgba(0,0,0,.24); }
    .metric { font-size: 30px; font-weight: 760; margin: 4px 0; }
    .pill { display: inline-flex; align-items: center; gap: 8px; padding: 5px 10px; border-radius: 999px; background: var(--panel-2); border: 1px solid var(--line); color: var(--muted); font-size: 13px; }
    .ok { color: var(--accent); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .progress { height: 10px; background: #0e1117; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; margin: 10px 0; }
    .bar { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent), #8ec5ff); transition: width .4s ease; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 14px; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; vertical-align: top; }
    th { color: var(--muted); font-weight: 650; background: rgba(255,255,255,.03); }
    code { color: #c4e3ff; word-break: break-all; }
    .mode { margin: 12px 0; padding: 14px; background: rgba(255,255,255,.03); border: 1px solid var(--line); border-radius: 14px; }
  </style>
</head>
<body>
<main>
  <h1>MergeMind Dashboard</h1>
  <div class="muted">Local Qwen / LM Studio monitoring. Auto-refreshes every <span id="refreshLabel"></span>s.</div>
  <section class="grid" id="topCards"></section>
  <section class="card">
    <h2>Recent Runs</h2>
    <div id="runs"></div>
  </section>
</main>
<script>
const refreshSeconds = Number(new URLSearchParams(location.search).get("refresh") || "__REFRESH__");
document.getElementById("refreshLabel").textContent = refreshSeconds;
function fmt(n, digits = 2) {
  const x = Number(n || 0);
  return x.toFixed(digits);
}
function card(title, value, detail, cls = "") {
  return `<div class="card"><div class="muted">${title}</div><div class="metric ${cls}">${value}</div><div class="muted">${detail || ""}</div></div>`;
}
function renderTop(data) {
  const gpu = (data.gpu.gpus || [])[0] || {};
  const lm = data.lmstudio || {};
  const mem = gpu.memory_total_mb ? `${fmt(gpu.memory_used_mb,0)} / ${fmt(gpu.memory_total_mb,0)} MB` : "n/a";
  const lmClass = lm.ok ? "ok" : "bad";
  document.getElementById("topCards").innerHTML = [
    card("GPU utilization", `${fmt(gpu.utilization_gpu,0)}%`, gpu.name || data.gpu.error || "n/a"),
    card("GPU memory", mem, `${fmt((gpu.memory_used_mb || 0) / Math.max(gpu.memory_total_mb || 1, 1) * 100,0)}% used`),
    card("GPU thermals", `${fmt(gpu.temperature_c,0)} C`, `${fmt(gpu.power_w,1)} W`),
    card("LM Studio", lm.ok ? "OK" : "CHECK", `${lm.configured_model || ""} @ ${lm.base_url || ""}`, lmClass),
  ].join("");
}
function renderRuns(data) {
  const runs = data.runs || [];
  if (!runs.length) {
    document.getElementById("runs").innerHTML = '<div class="muted">No runs found yet.</div>';
    return;
  }
  document.getElementById("runs").innerHTML = runs.map(run => {
    const modes = (run.modes || []).map(mode => {
      const p = mode.progress || {};
      const statusClass = mode.status === "completed" ? "ok" : (mode.status === "running" ? "warn" : "muted");
      return `<div class="mode">
        <div><span class="pill ${statusClass}">${mode.status}</span> <strong>${mode.mode}</strong> <span class="muted">${mode.profile || ""}</span></div>
        <div class="progress"><div class="bar" style="width:${Math.min(100, p.percent || 0)}%"></div></div>
        <table>
          <tr><th>progress</th><th>uncached tok/sec</th><th>tokens</th><th>calls cache/live</th><th>latency inf/judge/total</th><th>hit@k</th><th>best sim</th><th>parse/cache</th></tr>
          <tr>
            <td>${p.completed || 0}/${p.total || mode.example_count || 0} (${fmt(p.percent,1)}%)<br><span class="muted">${p.latest_example_id || ""}</span></td>
            <td>${fmt(p.uncached_tokens_per_second || p.tokens_per_second,2)}</td>
            <td>${p.uncached_total_tokens || 0} live<br><span class="muted">${p.total_tokens || 0} logical</span></td>
            <td>${p.cached_call_count || 0}/${p.uncached_call_count || 0}</td>
            <td>${fmt(mode.avg_inference_latency_sec,2)} / ${fmt(mode.avg_judge_latency_sec,2)} / ${fmt(mode.avg_total_wall_latency_sec,2)}s</td>
            <td>${fmt(mode.hit_rate_at_k,3)}</td>
            <td>${fmt(mode.best_similarity_at_k,3)}</td>
            <td>${fmt(p.parse_error_rate,3)} / ${fmt(p.cache_hit_rate,3)}</td>
          </tr>
        </table>
        <div class="muted"><code>${mode.path}</code></div>
      </div>`;
    }).join("");
    return `<div style="margin-bottom:22px">
      <h3>${run.run_id}</h3>
      <div class="muted"><code>${run.path}</code></div>
      ${modes || '<div class="muted">No mode artifacts yet.</div>'}
    </div>`;
  }).join("");
}
async function refresh() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    const data = await response.json();
    renderTop(data);
    renderRuns(data);
  } catch (error) {
    document.getElementById("topCards").innerHTML = card("Dashboard", "ERROR", String(error), "bad");
  }
}
refresh();
setInterval(refresh, refreshSeconds * 1000);
</script>
</body>
</html>
"""


def make_dashboard_handler(config: dict[str, Any], project_root: Path, refresh_seconds: int) -> type[BaseHTTPRequestHandler]:
    page = HTML_PAGE.replace("__REFRESH__", str(refresh_seconds))

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                payload = collect_dashboard_status(config, project_root)
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/", "/index.html"}:
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler
