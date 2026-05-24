"""Run monitored SWE-CI smoke runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.validation.swe_ci.assisted import (  # noqa: E402
    MERGEMIND_ASSISTED_MODE,
    build_assisted_environment,
    planned_execution_repo_path,
    prepare_swe_ci_execution_repo,
)
from src.validation.swe_ci.config import (
    build_swe_ci_command,
    build_swe_ci_env,
    describe_task_command,
    ensure_run_config_ready,
    experiment_name_for_task,
    official_experiment_task_dir,
    task_dataset_dir,
)
from src.validation.swe_ci.dataset import prepare_task_dataset_root
from src.validation.swe_ci.process_runner import merged_environment, redact_command, run_process
from src.validation.swe_ci.reporter import append_run_event, run_dir_for, write_report, write_run_inputs
from src.validation.swe_ci.result_parser import parse_swe_ci_result
from src.validation.swe_ci.review_loop import run_mergemind_patch_review
from src.validation.swe_ci.schemas import SweCiRunConfig, SweCiTaskRunResult
from src.validation.swe_ci.task_loader import load_swe_ci_tasks


def _safe_task_id(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in task_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real SWE-CI tasks with MergeMind monitoring.")
    parser.add_argument("--swe-ci-repo-path", required=True, help="Path to the cloned SWE-CI repository.")
    parser.add_argument("--tasks-path", required=True, help="Path to SWE-CI task manifest (.jsonl or .json).")
    parser.add_argument("--output-dir", required=True, help="Directory for MergeMind SWE-CI run artifacts.")
    parser.add_argument("--run-id", required=True, help="Unique run id.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks to run.")
    parser.add_argument("--max-iterations", type=int, default=3, help="SWE-CI max evolution iterations.")
    parser.add_argument("--timeout-seconds", type=int, default=7200, help="Per-task timeout.")
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "mergemind_review_loop", MERGEMIND_ASSISTED_MODE],
        help=(
            "baseline runs SWE-CI only; mergemind_review_loop reviews the patch after SWE-CI finishes; "
            "mergemind_assisted injects MergeMind comments before pytest inside each SWE-CI epoch."
        ),
    )
    parser.add_argument("--splitting", default="default", help="SWE-CI dataset split to pass to swe_ci.evaluate.")
    parser.add_argument("--api-key", default=None, help="Optional OpenAI-compatible API key for SWE-CI coding-agent.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL for SWE-CI coding-agent.")
    parser.add_argument("--model-name", default=None, help="Optional model name for SWE-CI coding-agent.")
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Optional SWE-CI agent backend, for example opencode, iflow, or direct_openai.",
    )
    parser.add_argument("--config-file", default=None, help="Optional SWE-CI config file path.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for SWE-CI.")
    parser.add_argument("--source-data-root", default=None, help="Optional SWE-CI data root containing task_id folders.")
    parser.add_argument("--assisted-work-dir", default=None, help="Optional workdir for instrumented SWE-CI checkout.")
    parser.add_argument("--docker-network", default=None, help="Optional Docker network mode for SWE-CI task containers, e.g. host.")
    parser.add_argument(
        "--opencode-no-think",
        action="store_true",
        help="Patch the copied SWE-CI checkout so OpenCode injects /no_think via instructions.",
    )
    parser.add_argument("--mergemind-config", default="configs/base.yaml", help="MergeMind config for review-loop mode.")
    parser.add_argument("--mergemind-pipeline", default="qwen35_rewriter", help="MergeMind pipeline for patch review.")
    parser.add_argument("--mergemind-llm-provider", default="", help="Optional MergeMind LLM provider override.")
    parser.add_argument("--mergemind-top-n", type=int, default=3, help="Number of MergeMind comments to keep.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without executing SWE-CI.")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> SweCiRunConfig:
    return SweCiRunConfig(
        swe_ci_repo_path=Path(args.swe_ci_repo_path).resolve(),
        tasks_path=Path(args.tasks_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        limit=args.limit,
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout_seconds,
        mode=args.mode,
        run_id=args.run_id,
        splitting=args.splitting,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        agent_name=args.agent_name,
        config_file=str(Path(args.config_file).resolve()) if args.config_file else None,
        hf_token=args.hf_token,
        source_data_root=Path(args.source_data_root).resolve() if args.source_data_root else None,
        assisted_work_dir=Path(args.assisted_work_dir).resolve() if args.assisted_work_dir else None,
        docker_network=args.docker_network,
        opencode_no_think=args.opencode_no_think,
        mergemind_config_path=PROJECT_ROOT / args.mergemind_config if not Path(args.mergemind_config).is_absolute() else Path(args.mergemind_config),
        mergemind_pipeline=args.mergemind_pipeline,
        mergemind_llm_provider=args.mergemind_llm_provider,
        mergemind_top_n=args.mergemind_top_n,
    )


def _print_dry_run(config: SweCiRunConfig) -> int:
    ensure_run_config_ready(config)
    tasks = load_swe_ci_tasks(config.tasks_path, limit=config.limit)
    run_dir = run_dir_for(config)
    execution_repo = planned_execution_repo_path(config, run_dir)
    print("[run_swe_ci] DRY RUN: no SWE-CI process will be executed.")
    print(f"[run_swe_ci] Tasks: {len(tasks)}")
    if execution_repo != config.swe_ci_repo_path:
        print(f"[run_swe_ci] Execution SWE-CI checkout: {execution_repo}")
    if config.docker_network:
        print(f"[run_swe_ci] Docker network: {config.docker_network}")
    if config.opencode_no_think:
        print("[run_swe_ci] OpenCode /no_think instruction: enabled")
    for task in tasks:
        command_info = describe_task_command(config, task, run_dir, execution_swe_ci_repo_path=execution_repo)
        print(f"\n[run_swe_ci] task_id={command_info['task_id']}")
        print(f"cwd: {command_info['cwd']}")
        print(f"dataset_root: {command_info['dataset_root']}")
        print(f"official_task_dir: {command_info['official_task_dir']}")
        print("command:")
        print(" ".join(redact_command(command_info["command"])))
        if config.mode == "mergemind_review_loop":
            print("post_step: MergeMind will review the coding-agent patch from SWE-CI outputs.")
        if config.mode == MERGEMIND_ASSISTED_MODE:
            print("in_loop: MergeMind will review programmer diffs before pytest in the instrumented SWE-CI checkout.")
    return 0


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    if args.dry_run:
        return _print_dry_run(config)

    ensure_run_config_ready(config)
    tasks = load_swe_ci_tasks(config.tasks_path, limit=config.limit)
    run_dir = run_dir_for(config)
    execution_repo = prepare_swe_ci_execution_repo(config, run_dir, PROJECT_ROOT)
    write_run_inputs(run_dir, config, tasks)
    append_run_event(run_dir, {"event": "run_start", "run_id": config.run_id, "task_count": len(tasks)})
    if execution_repo != config.swe_ci_repo_path:
        append_run_event(
            run_dir,
            {
                "event": "execution_repo_ready",
                "mode": config.mode,
                "source_repo": str(config.swe_ci_repo_path),
                "execution_repo": str(execution_repo),
                "docker_network": config.docker_network or "",
            },
        )

    env_updates = build_swe_ci_env(execution_repo, PROJECT_ROOT)
    if config.mode == MERGEMIND_ASSISTED_MODE:
        env_updates.update(build_assisted_environment(config, run_dir, PROJECT_ROOT))
    env = merged_environment(env_updates)
    results: list[SweCiTaskRunResult] = []
    for index, task in enumerate(tasks, start=1):
        append_run_event(run_dir, {"event": "task_start", "task_id": task.task_id, "index": index})
        dataset_root = prepare_task_dataset_root(config, task, run_dir)
        task_dir = official_experiment_task_dir(config, task, swe_ci_repo_path=execution_repo)
        task_log_dir = run_dir / "logs" / _safe_task_id(task.task_id)
        append_run_event(
            run_dir,
            {
                "event": "swe_ci_dataset_ready",
                "task_id": task.task_id,
                "dataset_root": str(dataset_root),
                "official_task_dir": str(task_dir),
            },
        )
        command = build_swe_ci_command(config, task, dataset_root)
        process_result = run_process(
            command=command,
            task_id=task.task_id,
            task_log_dir=task_log_dir,
            timeout_seconds=config.timeout_seconds,
            cwd=execution_repo,
            env=env,
            phase="swe_ci.evaluate",
        )
        parsed_result = parse_swe_ci_result(
            process_result,
            task_dir,
            swe_ci_repo_path=execution_repo,
            experiment_name=experiment_name_for_task(config, task),
        )
        if config.mode == "mergemind_review_loop":
            append_run_event(run_dir, {"event": "mergemind_review_start", "task_id": task.task_id})
            review_metrics = run_mergemind_patch_review(
                config=config,
                task=task,
                task_result=parsed_result,
                task_output_dir=task_dir,
                project_root=PROJECT_ROOT,
            )
            parsed_metrics = dict(parsed_result.metrics)
            parsed_metrics["mergemind_review"] = review_metrics
            parsed_result = SweCiTaskRunResult(
                task_id=parsed_result.task_id,
                status=parsed_result.status,
                started_at=parsed_result.started_at,
                finished_at=parsed_result.finished_at,
                duration_seconds=parsed_result.duration_seconds,
                exit_code=parsed_result.exit_code,
                stdout_path=parsed_result.stdout_path,
                stderr_path=parsed_result.stderr_path,
                events_path=parsed_result.events_path,
                metrics=parsed_metrics,
                error_message=parsed_result.error_message,
            )
            append_run_event(
                run_dir,
                {
                    "event": "mergemind_review_finish",
                    "task_id": task.task_id,
                    "status": review_metrics.get("status", ""),
                    "comment_count": review_metrics.get("comment_count", 0),
                    "comments_path": review_metrics.get("comments_path", ""),
                },
            )
        results.append(parsed_result)
        append_run_event(
            run_dir,
            {
                "event": "task_finish",
                "task_id": task.task_id,
                "status": parsed_result.status,
                "exit_code": parsed_result.exit_code,
                "error_message": parsed_result.error_message,
            },
        )
        write_report(run_dir, config.run_id, results)

    metrics = write_report(run_dir, config.run_id, results)
    append_run_event(run_dir, {"event": "run_finish", "run_id": config.run_id, "metrics": metrics})
    print(f"[run_swe_ci] Wrote run artifacts to {run_dir}")
    print(f"[run_swe_ci] Summary: {run_dir / 'summary.md'}")
    return 0 if metrics.get("failed", 0) == 0 and metrics.get("timeout", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
