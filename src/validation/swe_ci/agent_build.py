"""Docker build helpers for SWE-CI agent images."""

from __future__ import annotations

import os
from pathlib import Path

AGENT_DEFAULTS = {
    "iflow": {
        "dockerfile": "Dockerfile.iflow",
        "node_version": "22.11.0",
        "npm_pkg": "@iflow-ai/iflow-cli",
        "npm_bin": "iflow",
    },
    "opencode": {
        "dockerfile": "Dockerfile.opencode",
        "node_version": "22.11.0",
        "npm_pkg": "opencode-ai",
        "npm_bin": "opencode",
    },
}


class SweCiAgentBuildError(ValueError):
    """Raised when an agent image build cannot be configured."""


def agent_defaults(agent_name: str) -> dict[str, str]:
    try:
        return dict(AGENT_DEFAULTS[agent_name])
    except KeyError as exc:
        supported = ", ".join(sorted(AGENT_DEFAULTS))
        raise SweCiAgentBuildError(f"Unsupported SWE-CI agent '{agent_name}'. Supported agents: {supported}.") from exc


def agent_dockerfile_path(swe_ci_repo_path: str | Path, agent_name: str) -> Path:
    defaults = agent_defaults(agent_name)
    return Path(swe_ci_repo_path) / "src" / "swe_ci" / "benchmark" / "agents" / defaults["dockerfile"]


def validate_agent_build_inputs(*, swe_ci_repo_path: str | Path, agent_name: str, base_image: str) -> list[str]:
    errors: list[str] = []
    repo_path = Path(swe_ci_repo_path)
    if not repo_path.exists():
        errors.append(f"SWE-CI repo path does not exist: {repo_path}")
    elif not repo_path.is_dir():
        errors.append(f"SWE-CI repo path is not a directory: {repo_path}")
    try:
        dockerfile = agent_dockerfile_path(repo_path, agent_name)
    except SweCiAgentBuildError as exc:
        errors.append(str(exc))
    else:
        if not dockerfile.exists():
            errors.append(f"SWE-CI agent Dockerfile is missing: {dockerfile}")
    if not base_image.strip():
        errors.append("base_image must not be empty.")
    return errors


def build_agent_image_command(
    *,
    swe_ci_repo_path: str | Path,
    agent_name: str,
    base_image: str,
    tag: str,
    node_version: str | None = None,
    npm_pkg: str | None = None,
    npm_bin: str | None = None,
    progress: str | None = None,
) -> list[str]:
    defaults = agent_defaults(agent_name)
    dockerfile = agent_dockerfile_path(swe_ci_repo_path, agent_name)
    command = [
        "docker",
        "build",
        "-t",
        tag,
        "-f",
        str(dockerfile),
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"NODE_VERSION={node_version or defaults['node_version']}",
        "--build-arg",
        f"AGENT_NPM_PKG={npm_pkg or defaults['npm_pkg']}",
        "--build-arg",
        f"AGENT_BIN={npm_bin or defaults['npm_bin']}",
    ]
    if progress:
        command.extend(["--progress", progress])
    command.append(str(Path(swe_ci_repo_path)))
    return command


def build_agent_image_env(*, builder: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    if builder == "legacy":
        env["DOCKER_BUILDKIT"] = "0"
    elif builder == "buildkit":
        env.pop("DOCKER_BUILDKIT", None)
    else:
        raise SweCiAgentBuildError("builder must be either 'legacy' or 'buildkit'.")
    return env
