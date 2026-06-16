"""Run chunked SWE-CI paired baseline vs MergeMind-assisted experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.validation.swe_ci.process_runner import redact_command  # noqa: E402


def _chunk_run_id(index: int) -> str:
    return f"chunk_{index:02d}"


def _run(command: list[str], *, dry_run: bool, cwd: Path) -> int:
    print(" ".join(redact_command(command)), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def _append_command_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _run_swe_ci_command(
    *,
    mode: str,
    chunk_path: Path,
    output_dir: Path,
    chunk_run_id: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_swe_ci.py"),
        "--swe-ci-repo-path",
        str(Path(args.swe_ci_repo_path).resolve()),
        "--tasks-path",
        str(chunk_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--run-id",
        chunk_run_id,
        "--mode",
        mode,
        "--agent-name",
        args.agent_name,
        "--base-url",
        args.base_url,
        "--model-name",
        args.model_name,
        "--api-key",
        args.api_key,
        "--docker-network",
        args.docker_network,
        "--max-iterations",
        str(args.max_iterations),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.source_data_root:
        command.extend(["--source-data-root", str(Path(args.source_data_root).resolve())])
    if mode == "mergemind_assisted":
        command.extend(
            [
                "--mergemind-pipeline",
                args.mergemind_pipeline,
                "--mergemind-llm-provider",
                args.mergemind_llm_provider,
                "--mergemind-top-n",
                str(args.mergemind_top_n),
                "--mergemind-min-score",
                str(args.mergemind_min_score),
                "--mergemind-max-revision-epochs",
                str(args.mergemind_max_revision_epochs),
            ]
        )
    if args.dry_run:
        command.append("--dry-run")
    return command


def _combine_command(parent: Path, output_dir: Path, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "combine_swe_ci_run_chunks.py"),
        "--chunks-parent",
        str(parent.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--run-id",
        run_id,
    ]


def _compare_command(baseline_dir: Path, assisted_dir: Path, output_root: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare_swe_ci_runs.py"),
        "--baseline-run-dir",
        str(baseline_dir.resolve()),
        "--assisted-run-dir",
        str(assisted_dir.resolve()),
        "--output",
        str((output_root / "paired_summary.md").resolve()),
        "--json-output",
        str((output_root / "paired_summary.json").resolve()),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-CI Pair30 chunks.")
    parser.add_argument("--swe-ci-repo-path", required=True)
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--chunk-glob", default="*_chunk_*.jsonl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-data-root", default="")
    parser.add_argument("--agent-name", default="direct_openai")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model-name", default="qwen3.6-27b@iq2_xxs")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--docker-network", default="host")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--mergemind-pipeline", default="qwen35_rewriter_sweci_triage")
    parser.add_argument("--mergemind-llm-provider", default="local_qwen36_27b_iq2")
    parser.add_argument("--mergemind-top-n", type=int, default=1)
    parser.add_argument("--mergemind-min-score", type=float, default=0.75)
    parser.add_argument("--mergemind-max-revision-epochs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    chunks = sorted(Path(args.chunks_dir).resolve().glob(args.chunk_glob))
    if not chunks:
        raise ValueError(f"No chunks matched {args.chunk_glob!r} in {args.chunks_dir}.")

    output_root = Path(args.output_root).resolve() / args.run_id
    baseline_parent = output_root / "baseline"
    assisted_parent = output_root / "assisted"
    command_log = output_root / "commands.jsonl"
    for index, chunk_path in enumerate(chunks, start=1):
        chunk_run_id = _chunk_run_id(index)
        for mode, parent in (("baseline", baseline_parent), ("mergemind_assisted", assisted_parent)):
            command = _run_swe_ci_command(
                mode=mode,
                chunk_path=chunk_path,
                output_dir=parent,
                chunk_run_id=chunk_run_id,
                args=args,
            )
            _append_command_log(
                command_log,
                {
                    "mode": mode,
                    "chunk": str(chunk_path),
                    "run_id": chunk_run_id,
                    "command": redact_command(command),
                },
            )
            exit_code = _run(command, dry_run=args.dry_run, cwd=PROJECT_ROOT)
            if exit_code != 0:
                return exit_code

    baseline_all = baseline_parent / "all"
    assisted_all = assisted_parent / "all"
    for parent, output_dir, run_id in (
        (baseline_parent, baseline_all, "baseline_all"),
        (assisted_parent, assisted_all, "assisted_all"),
    ):
        command = _combine_command(parent, output_dir, run_id)
        _append_command_log(command_log, {"mode": "combine", "command": redact_command(command)})
        exit_code = _run(command, dry_run=args.dry_run, cwd=PROJECT_ROOT)
        if exit_code != 0:
            return exit_code

    command = _compare_command(baseline_all, assisted_all, output_root)
    _append_command_log(command_log, {"mode": "compare", "command": redact_command(command)})
    return _run(command, dry_run=args.dry_run, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
