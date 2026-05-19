"""Dataclasses shared by the SWE-CI runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SweCiStatus = Literal["success", "failed", "timeout", "skipped"]


@dataclass(frozen=True)
class SweCiTask:
    task_id: str
    repo_name: str
    repo_url: str
    current_sha: str
    target_sha: str
    image_sha: str
    test_gap: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SweCiRunConfig:
    swe_ci_repo_path: Path
    tasks_path: Path
    output_dir: Path
    limit: int | None
    max_iterations: int
    timeout_seconds: int
    mode: str
    run_id: str
    splitting: str = "default"
    api_key: str | None = None
    base_url: str | None = None
    model_name: str | None = None
    agent_name: str | None = None
    config_file: str | None = None
    hf_token: str | None = None
    mergemind_config_path: Path | None = None
    mergemind_pipeline: str = "qwen35_rewriter"
    mergemind_llm_provider: str = ""
    mergemind_top_n: int = 3

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("swe_ci_repo_path", "tasks_path", "output_dir", "mergemind_config_path"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True)
class SweCiTaskRunResult:
    task_id: str
    status: SweCiStatus
    started_at: str
    finished_at: str
    duration_seconds: float
    exit_code: int | None
    stdout_path: str
    stderr_path: str
    events_path: str
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
