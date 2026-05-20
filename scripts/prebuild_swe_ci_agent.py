"""Prebuild a SWE-CI coding-agent Docker layer before running the benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.validation.swe_ci.agent_build import (  # noqa: E402
    AGENT_DEFAULTS,
    build_agent_image_command,
    build_agent_image_env,
    validate_agent_build_inputs,
)
from src.validation.swe_ci.process_runner import redact_command, run_process  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebuild SWE-CI agent Docker image layers with visible logs.")
    parser.add_argument("--swe-ci-repo-path", required=True, help="Path to the cloned SWE-CI repository.")
    parser.add_argument("--base-image", required=True, help="Existing SWE-CI task image, e.g. image_owner__repo__sha:latest.")
    parser.add_argument("--agent-name", choices=sorted(AGENT_DEFAULTS), default="iflow", help="SWE-CI agent backend.")
    parser.add_argument("--tag", default="", help="Docker tag for the prebuilt image.")
    parser.add_argument("--node-version", default=None, help="Override Node version used by the SWE-CI Dockerfile.")
    parser.add_argument("--npm-pkg", default=None, help="Override agent npm package.")
    parser.add_argument("--npm-bin", default=None, help="Override agent binary name.")
    parser.add_argument("--builder", choices=["legacy", "buildkit"], default="legacy", help="Docker builder mode.")
    parser.add_argument("--timeout-seconds", type=int, default=3600, help="Build timeout.")
    parser.add_argument("--output-dir", default="artifacts/swe_ci_agent_builds", help="Directory for build logs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the docker build command without executing it.")
    return parser.parse_args()


def _default_tag(agent_name: str, base_image: str) -> str:
    safe_base = "".join(char if char.isalnum() else "-" for char in base_image).strip("-").lower()
    return f"mergemind-swe-ci-{agent_name}-{safe_base}"


def main() -> int:
    args = _parse_args()
    swe_ci_repo_path = Path(args.swe_ci_repo_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    tag = args.tag or _default_tag(args.agent_name, args.base_image)
    errors = validate_agent_build_inputs(
        swe_ci_repo_path=swe_ci_repo_path,
        agent_name=args.agent_name,
        base_image=args.base_image,
    )
    if args.timeout_seconds <= 0:
        errors.append("timeout_seconds must be positive.")
    if errors:
        print("[prebuild_swe_ci_agent] Build is not ready:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    command = build_agent_image_command(
        swe_ci_repo_path=swe_ci_repo_path,
        agent_name=args.agent_name,
        base_image=args.base_image,
        tag=tag,
        node_version=args.node_version,
        npm_pkg=args.npm_pkg,
        npm_bin=args.npm_bin,
        progress="plain" if args.builder == "buildkit" else None,
    )
    print("[prebuild_swe_ci_agent] command:")
    print(" ".join(redact_command(command)))
    print(f"[prebuild_swe_ci_agent] builder={args.builder}")
    print(f"[prebuild_swe_ci_agent] tag={tag}")
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_process(
        command=command,
        task_id=f"prebuild_{args.agent_name}",
        task_log_dir=output_dir / tag,
        timeout_seconds=args.timeout_seconds,
        cwd=swe_ci_repo_path,
        env=build_agent_image_env(builder=args.builder),
        phase="swe_ci.agent_prebuild",
        monitor_interval_seconds=10.0,
    )
    print(f"[prebuild_swe_ci_agent] status={result.status} exit_code={result.exit_code}")
    print(f"[prebuild_swe_ci_agent] stdout={result.stdout_path}")
    print(f"[prebuild_swe_ci_agent] stderr={result.stderr_path}")
    if result.status != "success":
        return 1

    inspect = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True, check=False)
    if inspect.returncode != 0:
        print(f"[prebuild_swe_ci_agent] Docker image was not created: {tag}", file=sys.stderr)
        print(inspect.stderr.strip(), file=sys.stderr)
        return 1
    print(f"[prebuild_swe_ci_agent] OK: image layer cache is ready for {args.agent_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
