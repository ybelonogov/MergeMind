"""Print Linux/LAN commands for running SWE-CI against a remote LM Studio server."""

from __future__ import annotations

import argparse
from urllib.parse import urlparse


def normalize_lmstudio_url(*, windows_host: str, lmstudio_url: str, port: int) -> str:
    raw = (lmstudio_url or "").strip()
    if not raw:
        host = windows_host.strip()
        if not host:
            raise ValueError("Provide --windows-host or --lmstudio-url.")
        raw = f"http://{host}:{port}/v1"
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if not parsed.netloc:
        raise ValueError(f"Invalid LM Studio URL: {raw}")
    normalized = raw.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare copy-paste commands for Linux SWE-CI + LAN LM Studio.")
    parser.add_argument("--windows-host", default="", help="Windows LAN IP or hostname that runs LM Studio.")
    parser.add_argument("--lmstudio-url", default="", help="Full LM Studio OpenAI-compatible URL, e.g. http://192.168.1.50:1234/v1.")
    parser.add_argument("--port", type=int, default=1234, help="LM Studio port when --windows-host is used.")
    parser.add_argument("--model-name", default="qwen3.6-27b@iq2_xxs", help="LM Studio model id.")
    parser.add_argument("--agent-name", choices=["iflow", "opencode"], default="iflow", help="SWE-CI agent backend.")
    parser.add_argument("--swe-ci-repo-path", default="../SWE-CI", help="Path to SWE-CI checkout on Linux.")
    parser.add_argument("--tasks-path", default="artifacts/swe_ci/tasks_smoke.jsonl", help="SWE-CI task manifest.")
    parser.add_argument("--output-dir", default="artifacts/swe_ci_runs", help="SWE-CI run output directory.")
    parser.add_argument("--base-image", default="image_pypa__build__ffe5ee__010b6c:latest", help="Task base Docker image for prebuild.")
    parser.add_argument("--run-id", default="sweci_linux_lan_smoke_001", help="Run id for smoke commands.")
    parser.add_argument("--max-iterations", type=int, default=1, help="SWE-CI max evolution iterations.")
    parser.add_argument("--timeout-seconds", type=int, default=14400, help="Per-task timeout.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        base_url = normalize_lmstudio_url(
            windows_host=args.windows_host,
            lmstudio_url=args.lmstudio_url,
            port=args.port,
        )
    except ValueError as exc:
        print(f"[prepare_linux_lan_swe_ci] {exc}")
        return 1

    print("# 1. Check LAN access to LM Studio")
    print(f"curl {base_url}/models")
    print()
    print("# 2. Activate MergeMind env and configure remote LM Studio")
    print("source .venv/bin/activate")
    print(f"export MERGEMIND_LLM_BASE_URL={base_url}")
    print(f"export MERGEMIND_LLM_MODEL={args.model_name}")
    print("export MERGEMIND_LLM_API_KEY=lm-studio")
    print()
    print("# 3. Check MergeMind LLM client")
    print("python scripts/check_llm.py --llm-provider local_qwen36_27b_iq2 --chat")
    print()
    print("# 4. Validate SWE-CI paths and task manifest")
    print("python scripts/setup_swe_ci.py \\")
    print(f"  --swe-ci-repo-path {args.swe_ci_repo_path} \\")
    print(f"  --tasks-path {args.tasks_path} \\")
    print(f"  --output-dir {args.output_dir} \\")
    print("  --limit 1")
    print()
    print("# 5. Prebuild SWE-CI agent Docker layer before benchmark")
    print("python scripts/prebuild_swe_ci_agent.py \\")
    print(f"  --swe-ci-repo-path {args.swe_ci_repo_path} \\")
    print(f"  --base-image {args.base_image} \\")
    print(f"  --agent-name {args.agent_name} \\")
    print("  --builder legacy \\")
    print("  --timeout-seconds 3600")
    print()
    print("# 6. Dry-run official SWE-CI command")
    print("python scripts/run_swe_ci.py \\")
    print(f"  --swe-ci-repo-path {args.swe_ci_repo_path} \\")
    print(f"  --tasks-path {args.tasks_path} \\")
    print(f"  --output-dir {args.output_dir} \\")
    print(f"  --run-id {args.run_id}_dry \\")
    print("  --limit 1 \\")
    print(f"  --max-iterations {args.max_iterations} \\")
    print(f"  --timeout-seconds {args.timeout_seconds} \\")
    print("  --mode baseline \\")
    print(f"  --agent-name {args.agent_name} \\")
    print(f"  --base-url {base_url} \\")
    print(f"  --model-name {args.model_name} \\")
    print("  --api-key lm-studio \\")
    print("  --dry-run")
    print()
    print("# 7. Real baseline smoke run")
    print("python scripts/run_swe_ci.py \\")
    print(f"  --swe-ci-repo-path {args.swe_ci_repo_path} \\")
    print(f"  --tasks-path {args.tasks_path} \\")
    print(f"  --output-dir {args.output_dir} \\")
    print(f"  --run-id {args.run_id} \\")
    print("  --limit 1 \\")
    print(f"  --max-iterations {args.max_iterations} \\")
    print(f"  --timeout-seconds {args.timeout_seconds} \\")
    print("  --mode baseline \\")
    print(f"  --agent-name {args.agent_name} \\")
    print(f"  --base-url {base_url} \\")
    print(f"  --model-name {args.model_name} \\")
    print("  --api-key lm-studio")
    print()
    print("# 8. Review-loop smoke run after baseline works")
    print("python scripts/run_swe_ci.py \\")
    print(f"  --swe-ci-repo-path {args.swe_ci_repo_path} \\")
    print(f"  --tasks-path {args.tasks_path} \\")
    print(f"  --output-dir {args.output_dir} \\")
    print(f"  --run-id {args.run_id}_review \\")
    print("  --limit 1 \\")
    print(f"  --max-iterations {args.max_iterations} \\")
    print(f"  --timeout-seconds {args.timeout_seconds} \\")
    print("  --mode mergemind_review_loop \\")
    print(f"  --agent-name {args.agent_name} \\")
    print(f"  --base-url {base_url} \\")
    print(f"  --model-name {args.model_name} \\")
    print("  --api-key lm-studio \\")
    print("  --mergemind-pipeline qwen35_rewriter \\")
    print("  --mergemind-llm-provider local_qwen36_27b_iq2 \\")
    print("  --mergemind-top-n 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
