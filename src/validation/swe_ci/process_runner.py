"""Safe subprocess runner used by the SWE-CI integration."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .monitor import append_jsonl, build_monitor_event, utc_now_iso
from .schemas import SweCiTaskRunResult

_SECRET_ARGS = {"--api_key", "--api-key", "--hf_token", "--hf-token"}


def redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue
        redacted.append(part)
        if part in _SECRET_ARGS:
            redact_next = True
    return redacted


def _popen_process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _running_docker_container_ids() -> set[str]:
    try:
        completed = subprocess.run(
            ["docker", "ps", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if completed.returncode != 0:
        return set()
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def _stop_new_docker_containers(previous_container_ids: set[str]) -> list[str]:
    current_container_ids = _running_docker_container_ids()
    new_container_ids = sorted(current_container_ids - previous_container_ids)
    stopped: list[str] = []
    for container_id in new_container_ids:
        completed = subprocess.run(
            ["docker", "stop", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if completed.returncode == 0:
            stopped.append(container_id)
    return stopped


def run_process(
    *,
    command: list[str],
    task_id: str,
    task_log_dir: str | Path,
    timeout_seconds: int,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    phase: str = "swe_ci.evaluate",
    monitor_interval_seconds: float = 5.0,
) -> SweCiTaskRunResult:
    if not command:
        raise ValueError("command must not be empty.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    log_dir = Path(task_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    events_path = log_dir / "events.jsonl"

    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    append_jsonl(
        events_path,
        {
            "event": "process_start",
            "timestamp": started_at,
            "task_id": task_id,
            "phase": phase,
            "command": redact_command(command),
            "cwd": str(cwd) if cwd is not None else None,
            "timeout_seconds": timeout_seconds,
        },
    )

    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                shell=False,
                **_popen_process_group_kwargs(),
            )
        except OSError as exc:
            finished_at = utc_now_iso()
            duration = time.monotonic() - started_monotonic
            append_jsonl(
                events_path,
                {
                    "event": "process_start_failed",
                    "timestamp": finished_at,
                    "task_id": task_id,
                    "phase": phase,
                    "error": str(exc),
                },
            )
            return SweCiTaskRunResult(
                task_id=task_id,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=round(duration, 3),
                exit_code=None,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                events_path=str(events_path),
                metrics={"phase": phase},
                error_message=f"Failed to start process: {exc}",
            )

        timed_out = False
        docker_containers_before = _running_docker_container_ids()
        stopped_docker_containers: list[str] = []
        last_monitor = 0.0
        while True:
            exit_code = process.poll()
            elapsed = time.monotonic() - started_monotonic
            if elapsed - last_monitor >= monitor_interval_seconds:
                stdout_handle.flush()
                stderr_handle.flush()
                append_jsonl(
                    events_path,
                    build_monitor_event(
                        task_id=task_id,
                        phase=phase,
                        started_monotonic=started_monotonic,
                        process_pid=process.pid,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    ),
                )
                last_monitor = elapsed

            if exit_code is not None:
                break

            if elapsed >= timeout_seconds:
                timed_out = True
                _kill_process_tree(process)
                stopped_docker_containers = _stop_new_docker_containers(docker_containers_before)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                    pass
                break

            time.sleep(min(0.2, monitor_interval_seconds))

        stdout_handle.flush()
        stderr_handle.flush()

    finished_at = utc_now_iso()
    duration = time.monotonic() - started_monotonic
    exit_code = process.returncode
    if timed_out:
        status = "timeout"
        error_message = f"Process timed out after {timeout_seconds} seconds."
    elif exit_code == 0:
        status = "success"
        error_message = ""
    else:
        status = "failed"
        error_message = f"Process exited with code {exit_code}."

    append_jsonl(
        events_path,
        {
            "event": "process_finish",
            "timestamp": finished_at,
            "task_id": task_id,
            "phase": phase,
            "status": status,
            "exit_code": exit_code,
            "duration_seconds": round(duration, 3),
            "timed_out": timed_out,
        },
    )

    return SweCiTaskRunResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(duration, 3),
        exit_code=exit_code,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        events_path=str(events_path),
        metrics={
            "phase": phase,
            "pid": getattr(process, "pid", None),
            "command": redact_command(command),
            "cwd": str(cwd) if cwd is not None else "",
            "docker_containers_stopped_on_timeout": stopped_docker_containers,
        },
        error_message=error_message,
    )


def merged_environment(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(os.environ)
    if extra_env:
        merged.update(extra_env)
    return merged
