"""Structured process monitoring helpers for SWE-CI runs."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def tail_lines(path: str | Path, max_lines: int = 10) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return {"gpu_available": False, "gpu_error": str(exc)}

    if completed.returncode != 0:
        return {"gpu_available": False, "gpu_error": completed.stderr.strip()}

    devices: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        name, utilization, memory_used, memory_total = parts
        devices.append(
            {
                "name": name,
                "utilization_gpu_percent": _safe_float(utilization),
                "memory_used_mb": _safe_float(memory_used),
                "memory_total_mb": _safe_float(memory_total),
            }
        )
    return {"gpu_available": bool(devices), "gpus": devices}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def process_resource_snapshot(pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {"cpu_percent": None, "memory_mb": None, "resource_error": "process has no pid"}

    try:
        import psutil  # type: ignore
    except ImportError:
        return {"cpu_percent": None, "memory_mb": None, "resource_error": "psutil is not installed"}

    try:
        process = psutil.Process(pid)
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return {"cpu_percent": process.cpu_percent(interval=None), "memory_mb": memory_mb}
    except psutil.Error as exc:
        return {"cpu_percent": None, "memory_mb": None, "resource_error": str(exc)}


def build_monitor_event(
    *,
    task_id: str,
    phase: str,
    started_monotonic: float,
    process_pid: int | None,
    stdout_path: str | Path,
    stderr_path: str | Path,
    tail_line_count: int = 10,
) -> dict[str, Any]:
    event = {
        "event": "monitor",
        "timestamp": utc_now_iso(),
        "task_id": task_id,
        "phase": phase,
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "process_pid": process_pid,
        "last_stdout_lines": tail_lines(stdout_path, max_lines=tail_line_count),
        "last_stderr_lines": tail_lines(stderr_path, max_lines=tail_line_count),
    }
    event.update(process_resource_snapshot(process_pid))
    event.update(_gpu_snapshot())
    return event
