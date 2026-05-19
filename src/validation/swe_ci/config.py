"""Preflight validation and command construction for SWE-CI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .schemas import SweCiRunConfig, SweCiTask

HONEST_SWE_CI_MODES = {"baseline", "mergemind_review_loop"}
ORACLE_SWE_CI_MODE = "oracle_comments"
EXPECTED_SWE_CI_FILES = (
    "src/swe_ci/evaluate.py",
    "src/swe_ci/summarize.py",
    "src/swe_ci/download.py",
    "config.toml",
)


class SweCiConfigError(ValueError):
    """Raised when SWE-CI setup is not ready."""


def validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise SweCiConfigError(f"{name} must be positive.")


def validate_swe_ci_repo_path(path: str | Path) -> list[str]:
    repo_path = Path(path)
    errors: list[str] = []
    if not repo_path.exists():
        return [f"SWE-CI repo path does not exist: {repo_path}"]
    if not repo_path.is_dir():
        return [f"SWE-CI repo path is not a directory: {repo_path}"]
    for relative in EXPECTED_SWE_CI_FILES:
        if not (repo_path / relative).exists():
            errors.append(f"SWE-CI expected file is missing: {repo_path / relative}")
    return errors


def check_command_available(name: str) -> str | None:
    executable = shutil.which(name)
    if executable is None:
        return f"Required command is not available on PATH: {name}"
    try:
        completed = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Could not run '{name} --version': {exc}"
    if completed.returncode != 0:
        return f"'{name} --version' failed with exit code {completed.returncode}: {completed.stderr.strip()}"
    return None


def validate_output_dir(path: str | Path) -> list[str]:
    output_dir = Path(path)
    errors: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        errors.append(f"Output directory is not writable: {output_dir} ({exc})")
    return errors


def validate_run_config(config: SweCiRunConfig, *, require_tools: bool = True) -> list[str]:
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ is required for MergeMind SWE-CI wrapper.")
    if require_tools:
        for command_name in ("git", "docker"):
            error = check_command_available(command_name)
            if error:
                errors.append(error)
    errors.extend(validate_swe_ci_repo_path(config.swe_ci_repo_path))
    if not config.tasks_path.exists():
        errors.append(f"SWE-CI tasks file does not exist: {config.tasks_path}")
    errors.extend(validate_output_dir(config.output_dir))
    try:
        validate_positive_int(config.max_iterations, "max_iterations")
    except SweCiConfigError as exc:
        errors.append(str(exc))
    try:
        validate_positive_int(config.timeout_seconds, "timeout_seconds")
    except SweCiConfigError as exc:
        errors.append(str(exc))
    if config.limit is not None and config.limit < 0:
        errors.append("limit must be >= 0 when provided.")
    if config.mode not in HONEST_SWE_CI_MODES:
        if config.mode == ORACLE_SWE_CI_MODE:
            errors.append("mode='oracle_comments' is reserved for leaky local debugging and is not benchmark-safe.")
        else:
            errors.append(f"Unsupported mode='{config.mode}'. Use one of: {', '.join(sorted(HONEST_SWE_CI_MODES))}.")
    if not config.run_id.strip():
        errors.append("run_id must not be empty.")
    if config.mode == "mergemind_review_loop":
        try:
            validate_positive_int(config.mergemind_top_n, "mergemind_top_n")
        except SweCiConfigError as exc:
            errors.append(str(exc))
        if config.mergemind_config_path is not None and not config.mergemind_config_path.exists():
            errors.append(f"MergeMind config file does not exist: {config.mergemind_config_path}")
    return errors


def ensure_run_config_ready(config: SweCiRunConfig, *, require_tools: bool = True) -> None:
    errors = validate_run_config(config, require_tools=require_tools)
    if errors:
        raise SweCiConfigError("\n".join(errors))


def task_output_dir(run_dir: str | Path, task: SweCiTask) -> Path:
    safe_task_id = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in task.task_id)
    return Path(run_dir) / "swe_ci_outputs" / safe_task_id


def build_swe_ci_command(config: SweCiRunConfig, task: SweCiTask, task_dir: str | Path) -> list[str]:
    metadata = dict(task.metadata)
    experiment_name = str(metadata.get("experiment_name") or f"{config.run_id}_{task.task_id}")
    splitting = str(config.splitting or metadata.get("splitting") or "default")
    command = [
        sys.executable,
        "-u",
        "-m",
        "swe_ci.evaluate",
        "--experiment_name",
        experiment_name,
        "--splitting",
        splitting,
        "--save_root_dir",
        str(task_dir),
        "--evolve.max_epoch",
        str(config.max_iterations),
    ]
    for config_key, cli_key in (
        ("api_key", "--api_key"),
        ("base_url", "--base_url"),
        ("model_name", "--model_name"),
        ("agent_name", "--agent_name"),
        ("config_file", "--config_file"),
        ("hf_token", "--hf_token"),
    ):
        value = getattr(config, config_key) or metadata.get(config_key)
        if value is not None and str(value) != "":
            command.extend([cli_key, str(value)])
    return command


def build_swe_ci_env(swe_ci_repo_path: str | Path) -> dict[str, str]:
    src_path = Path(swe_ci_repo_path) / "src"
    return {"PYTHONPATH": str(src_path)}


def describe_task_command(config: SweCiRunConfig, task: SweCiTask, run_dir: str | Path) -> dict[str, Any]:
    output_dir = task_output_dir(run_dir, task)
    return {
        "task_id": task.task_id,
        "command": build_swe_ci_command(config, task, output_dir),
        "cwd": str(config.swe_ci_repo_path),
        "output_dir": str(output_dir),
    }
