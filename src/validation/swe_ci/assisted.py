"""Prepare instrumented SWE-CI workdirs for MergeMind-assisted runs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .reporter import run_dir_for
from .schemas import SweCiRunConfig, SweCiTask

MERGEMIND_ASSISTED_MODE = "mergemind_assisted"
DIRECT_OPENAI_AGENT = "direct_openai"

_COPY_IGNORE = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "data",
    "experiments",
    "mergemind_runs",
}

_INSTRUMENTED_HELPER = r'''"""Generated MergeMind bridge for instrumented SWE-CI runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MERGEMIND_PROGRAMMER_REVISION_PROMPT = """
You are revising your current code change before pytest is executed.

Inputs:
- /app/code/ contains your current implementation.
- /app/requirement.xml contains the architect requirement.
- /app/mergemind_review.md contains review comments generated from your current patch.
- /app/mergemind_allowed_files.txt lists the only files you may edit.
- /app/mergemind_before_files.md contains before-patch snapshots for those files.

Workflow:
1. Read /app/mergemind_review.md and /app/requirement.xml.
2. Read /app/mergemind_allowed_files.txt and inspect only those files under /app/code/.
3. Use /app/mergemind_before_files.md only to restore or reconcile code that your previous patch deleted or replaced by mistake.
4. Apply the smallest code revision needed to address actionable review comments.
5. If a comment points outside the files you changed, is already addressed, is uncertain,
   or would require broad rewrites, leave that area unchanged.

Constraints:
- Do not edit tests.
- Do not edit /app/requirement.xml or /app/mergemind_review.md.
- Do not edit /app/mergemind_allowed_files.txt.
- Do not edit /app/mergemind_before_files.md.
- Do not run pytest, unittest, or any test command.
- Do not create new files.
- Do not modify files that are absent from /app/mergemind_allowed_files.txt.
- Keep changes minimal and aligned with the existing requirement.
- Prefer no edit over an ungrounded edit.
""".strip()


def _safe_task_id(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in task_id)


def _metadata_value(task_metadata: dict[str, Any], *names: str) -> str:
    for name in names:
        value = task_metadata.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def run_mergemind_assist(
    *,
    task_metadata: dict[str, Any],
    task_dir: str | Path,
    epoch: int,
    before_code_dir: str | Path,
    after_code_dir: str | Path,
    requirement_path: str | Path,
) -> dict[str, Any]:
    project_root = os.environ.get("MERGEMIND_PROJECT_ROOT", "")
    if not project_root:
        return {"status": "skipped", "error_message": "MERGEMIND_PROJECT_ROOT is not set.", "comment_count": 0}

    task_id = _metadata_value(task_metadata, "task_id")
    output_root = Path(os.environ.get("MERGEMIND_ASSIST_OUTPUT_ROOT", "")) if os.environ.get("MERGEMIND_ASSIST_OUTPUT_ROOT") else Path(task_dir) / "mergemind"
    output_dir = output_root / _safe_task_id(task_id) / f"epoch_{epoch:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        os.environ.get("MERGEMIND_ASSIST_PYTHON", sys.executable),
        "-u",
        str(Path(project_root) / "scripts" / "swe_ci_mergemind_assist.py"),
        "--project-root",
        project_root,
        "--config",
        os.environ.get("MERGEMIND_CONFIG_PATH", str(Path(project_root) / "configs" / "base.yaml")),
        "--pipeline",
        os.environ.get("MERGEMIND_PIPELINE", "qwen35_rewriter"),
        "--llm-provider",
        os.environ.get("MERGEMIND_LLM_PROVIDER", ""),
        "--top-n",
        os.environ.get("MERGEMIND_TOP_N", "3"),
        "--min-score",
        os.environ.get("MERGEMIND_MIN_SCORE", "0.0"),
        "--epoch",
        str(epoch),
        "--task-id",
        task_id,
        "--repo-name",
        _metadata_value(task_metadata, "repo_name"),
        "--repo-url",
        _metadata_value(task_metadata, "repo_url", "url"),
        "--current-sha",
        _metadata_value(task_metadata, "current_sha"),
        "--image-sha",
        _metadata_value(task_metadata, "image_sha"),
        "--before-code-dir",
        str(before_code_dir),
        "--after-code-dir",
        str(after_code_dir),
        "--requirement-path",
        str(requirement_path),
        "--output-dir",
        str(output_dir),
    ]
    max_revision_epochs = os.environ.get("MERGEMIND_MAX_REVISION_EPOCHS", "")
    if max_revision_epochs:
        command.extend(["--max-revision-epochs", max_revision_epochs])
    timeout = int(os.environ.get("MERGEMIND_ASSIST_TIMEOUT_SECONDS", "1800"))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    (output_dir / "helper_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "helper_stderr.log").write_text(completed.stderr, encoding="utf-8")

    result_path = output_dir / "assist_result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
    else:
        result = {}
    if not isinstance(result, dict):
        result = {}
    if completed.returncode != 0 and not result:
        return {
            "status": "error",
            "error_message": completed.stderr.strip() or f"MergeMind helper exited with {completed.returncode}.",
            "comment_count": 0,
            "stdout_path": str(output_dir / "helper_stdout.log"),
            "stderr_path": str(output_dir / "helper_stderr.log"),
            "target_sha_used_for_review": False,
        }
    result.setdefault("status", "success" if completed.returncode == 0 else "error")
    result.setdefault("comment_count", 0)
    result["stdout_path"] = str(output_dir / "helper_stdout.log")
    result["stderr_path"] = str(output_dir / "helper_stderr.log")
    result["target_sha_used_for_review"] = False
    return result
'''


class SweCiAssistedError(RuntimeError):
    """Raised when an instrumented SWE-CI workdir cannot be prepared."""


def needs_instrumented_workdir(config: SweCiRunConfig) -> bool:
    return (
        config.mode == MERGEMIND_ASSISTED_MODE
        or bool(config.docker_network)
        or config.opencode_no_think
        or config.agent_name == DIRECT_OPENAI_AGENT
    )


def planned_execution_repo_path(config: SweCiRunConfig, run_dir: str | Path | None = None) -> Path:
    if not needs_instrumented_workdir(config):
        return config.swe_ci_repo_path
    if config.assisted_work_dir is not None:
        return config.assisted_work_dir
    directory = Path(run_dir) if run_dir is not None else run_dir_for(config)
    suffix = "assisted_swe_ci" if config.mode == MERGEMIND_ASSISTED_MODE else "network_swe_ci"
    return directory / "workdirs" / suffix


def _ignore_copy(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _COPY_IGNORE}


def _git_commit(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _copy_swe_ci_repo(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if destination_resolved == source_resolved:
        raise SweCiAssistedError("Instrumented SWE-CI workdir must differ from the source checkout.")
    try:
        if destination_resolved.is_relative_to(source_resolved):
            raise SweCiAssistedError(
                "Instrumented SWE-CI workdir must not be inside the source checkout; "
                "pass --assisted-work-dir outside --swe-ci-repo-path."
            )
    except AttributeError:
        pass
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=_ignore_copy)
    source_commit = _git_commit(source)
    if source_commit:
        (destination / ".mergemind_source_commit").write_text(source_commit + "\n", encoding="utf-8")


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SweCiAssistedError(f"Could not find expected patch target in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _ensure_docker_network_config(repo_path: Path) -> None:
    config_path = repo_path / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    if "network =" not in text:
        marker = "[docker]\n"
        if marker not in text:
            raise SweCiAssistedError("Could not find [docker] section in SWE-CI config.toml.")
        text = text.replace(marker, marker + 'network = ""                               # Optional Docker network mode, e.g. "host".\n', 1)
        config_path.write_text(text, encoding="utf-8")


def _patch_tools_for_network(repo_path: Path) -> None:
    tools_path = repo_path / "src" / "swe_ci" / "benchmark" / "tools.py"
    old = "def container_extra_args() -> list[str]:\n    extra_args = []\n"
    new = (
        "def container_extra_args() -> list[str]:\n"
        "    extra_args = []\n"
        "    docker_network = getattr(CONFIG.docker, \"network\", \"\")\n"
        "    if docker_network != \"\":\n"
        "        extra_args.extend([\"--network\", str(docker_network)])\n"
    )
    text = tools_path.read_text(encoding="utf-8")
    if "docker_network = getattr(CONFIG.docker" not in text:
        _replace_once(tools_path, old, new)


def _patch_opencode_no_think(repo_path: Path) -> None:
    config_path = repo_path / "config.toml"
    text = config_path.read_text(encoding="utf-8")
    if "opencode_no_think" not in text:
        lines = text.splitlines()
        patched_lines: list[str] = []
        inserted = False
        for line in lines:
            patched_lines.append(line)
            if not inserted and line.strip().startswith("model_name"):
                patched_lines.append(
                    'opencode_no_think = true                  # Add /no_think instruction for local Qwen/OpenCode runs.'
                )
                inserted = True
        if not inserted:
            raise SweCiAssistedError("Could not find model_name in SWE-CI config.toml.")
        text = "\n".join(patched_lines) + "\n"
    else:
        text = text.replace("opencode_no_think = false", "opencode_no_think = true")
    config_path.write_text(text, encoding="utf-8")

    opencode_path = repo_path / "src" / "swe_ci" / "benchmark" / "agents" / "opencode.py"
    text = opencode_path.read_text(encoding="utf-8")
    if "NO_THINK_INSTRUCTIONS_FILE" in text:
        return
    _replace_once(
        opencode_path,
        'CFG_FILE = "opencode.json"\n',
        'CFG_FILE = "opencode.json"\nNO_THINK_INSTRUCTIONS_DIR = f"{HOME_DIR}/mergemind"\nNO_THINK_INSTRUCTIONS_FILE = "NO_THINK.md"\n',
    )
    _replace_once(
        opencode_path,
        '    auth_payload = json.dumps(auth, indent=4, ensure_ascii=False) + "\\n"\n',
        '''    if getattr(CONFIG, "opencode_no_think", False):
        cfg["instructions"] = [f"{NO_THINK_INSTRUCTIONS_DIR}/{NO_THINK_INSTRUCTIONS_FILE}"]

    auth_payload = json.dumps(auth, indent=4, ensure_ascii=False) + "\\n"
''',
    )
    _replace_once(
        opencode_path,
        '''    subprocess.run([
        "docker", "exec", "-i", "-u", "root", container_name, "sh", "-c", 
        f"mkdir -p {AUTH_DIR} && cat > {AUTH_DIR}/{AUTH_FILE}"
        ], input=auth_payload, text=True, check=True)

''',
        '''    subprocess.run([
        "docker", "exec", "-i", "-u", "root", container_name, "sh", "-c", 
        f"mkdir -p {AUTH_DIR} && cat > {AUTH_DIR}/{AUTH_FILE}"
        ], input=auth_payload, text=True, check=True)

    if getattr(CONFIG, "opencode_no_think", False):
        subprocess.run([
            "docker", "exec", "-i", "-u", "root", container_name, "sh", "-c",
            f"mkdir -p {NO_THINK_INSTRUCTIONS_DIR} && cat > {NO_THINK_INSTRUCTIONS_DIR}/{NO_THINK_INSTRUCTIONS_FILE}"
            ], input="/no_think\\n", text=True, check=True)

''',
    )
    _replace_once(
        opencode_path,
        '''        "opencode", "run", "--model", f"custom/{CONFIG.model_name}", prompt,
''',
        '''        "opencode", "run",
        "--dangerously-skip-permissions", "--print-logs", "--log-level", "DEBUG",
        "--model", f"custom/{CONFIG.model_name}", prompt,
''',
    )


def _patch_run_for_mergemind(repo_path: Path) -> None:
    benchmark_dir = repo_path / "src" / "swe_ci" / "benchmark"
    (benchmark_dir / "mergemind_assist.py").write_text(_INSTRUMENTED_HELPER, encoding="utf-8")
    run_path = benchmark_dir / "run.py"
    text = run_path.read_text(encoding="utf-8")
    if "run_mergemind_assist" in text:
        return

    _replace_once(
        run_path,
        "from swe_ci.config import CONFIG\n",
        "from swe_ci.config import CONFIG\nfrom .mergemind_assist import MERGEMIND_PROGRAMMER_REVISION_PROMPT, run_mergemind_assist\n",
    )
    _replace_once(
        run_path,
        "            # Step 4.3: Call programmer agent with retry mechanism\n",
        "            mergemind_review = {}\n            programmer_revision_result = {}\n\n            # Step 4.3: Call programmer agent with retry mechanism\n",
    )
    _replace_once(
        run_path,
        '                    copy_dir_from_container(container_name, "/app/code", tmp_dir, mkdir=True)\n                    logger.info(prefix + "✅ The programmer agent has modified the code.")\n                    break\n',
        '''                    copy_dir_from_container(container_name, "/app/code", tmp_dir, mkdir=True)
                    logger.info(prefix + "✅ The programmer agent has modified the code.")
                    mergemind_review = run_mergemind_assist(
                        task_metadata=task_metadata,
                        task_dir=task_dir,
                        epoch=current_epoch + 1,
                        before_code_dir=current_dir / "code",
                        after_code_dir=tmp_dir / "code",
                        requirement_path=current_dir / "requirement.xml",
                    )
                    if mergemind_review.get("status") == "success" and int(mergemind_review.get("comment_count") or 0) > 0 and bool(mergemind_review.get("apply_revision", True)):
                        logger.info(prefix + f"✅ MergeMind generated {mergemind_review.get('comment_count')} review comment(s).")
                        for review_attempt in range(1, CONFIG.evolve.programmer.max_try+1):
                            review_prefix = f"(3b/7) (Review attempt {review_attempt}/{CONFIG.evolve.programmer.max_try}) "
                            try:
                                if has_container(container_name):
                                    remove_container(container_name)
                                run_container(image_tag, container_name, extra_args=CONTAINER_EXTRA_ARGS)
                                copy_dir_to_container(container_name, tmp_dir/"code", "/app")
                                copy_file_to_container(container_name, current_dir/"requirement.xml", "/app")
                                copy_file_to_container(container_name, mergemind_review["review_path"], "/app", rename="mergemind_review.md")
                                original_changed_files = sorted(set(programmer_result.get("changed_files") or []))
                                allowed_files_path = task_dir / "mergemind_allowed_files.txt"
                                allowed_files_path.write_text("\\n".join(original_changed_files) + "\\n", encoding="utf-8")
                                copy_file_to_container(container_name, allowed_files_path, "/app", rename="mergemind_allowed_files.txt")
                                before_files_path = task_dir / "mergemind_before_files.md"
                                before_sections = ["# Before-patch snapshots for MergeMind revision"]
                                for original_file in original_changed_files:
                                    before_source_path = current_dir / "code" / original_file
                                    before_sections.append("## " + original_file)
                                    if before_source_path.is_file():
                                        before_content = before_source_path.read_text(encoding="utf-8", errors="replace")[:20000]
                                        before_sections.append("```python\\n" + before_content + "\\n```")
                                    else:
                                        before_sections.append("<file did not exist before programmer patch>")
                                before_files_path.write_text("\\n\\n".join(before_sections) + "\\n", encoding="utf-8")
                                copy_file_to_container(container_name, before_files_path, "/app", rename="mergemind_before_files.md")
                                programmer_revision_result = call_cli_agent(
                                    container_name, MERGEMIND_PROGRAMMER_REVISION_PROMPT,
                                    timeout=CONFIG.evolve.programmer.timeout
                                    )
                                original_changed_files = set(original_changed_files)
                                revision_changed_files = set(programmer_revision_result.get("changed_files") or [])
                                unexpected_revision_files = sorted(revision_changed_files - original_changed_files)
                                if unexpected_revision_files:
                                    raise ValueError(
                                        "MergeMind revision changed files outside the programmer patch: "
                                        + ", ".join(unexpected_revision_files)
                                    )
                                shutil.rmtree(tmp_dir/"code", ignore_errors=True)
                                copy_dir_from_container(container_name, "/app/code", tmp_dir, mkdir=True)
                                logger.info(review_prefix + "✅ The programmer revised the code with MergeMind review.")
                                break
                            except Exception as e:
                                info = f"⚠️ Error occurred when applying MergeMind review: {repr(e)}"
                                logger.exception(review_prefix + info)
                                mergemind_review["revision_error"] = repr(e)
                            finally:
                                if has_container(container_name):
                                    remove_container(container_name)
                    elif mergemind_review.get("status") == "success" and int(mergemind_review.get("comment_count") or 0) > 0:
                        logger.info(prefix + f"ℹ️ MergeMind generated {mergemind_review.get('comment_count')} review comment(s), but revision pass was skipped by guard policy.")
                    break
''',
    )
    addition_block = '''                addition = {"architect": architect_result, "programmer": programmer_result}
                if mergemind_review:
                    addition["mergemind_review"] = mergemind_review
                if programmer_revision_result:
                    addition["programmer_revision"] = programmer_revision_result
'''
    _replace_once(
        run_path,
        '''                update_iteration(
                    gap, task_dir / "iteration.jsonl", current_dir / "test_report.json",
                    addition = {"architect": architect_result, "programmer": programmer_result}
                    )
''',
        addition_block
        + '''                update_iteration(
                    gap, task_dir / "iteration.jsonl", current_dir / "test_report.json",
                    addition = addition
                    )
''',
    )
    _replace_once(
        run_path,
        '''                update_iteration(
                    -1, task_dir / "iteration.jsonl", None,
                    addition = {"architect": architect_result, "programmer": programmer_result}
                    )
''',
        addition_block
        + '''                update_iteration(
                    -1, task_dir / "iteration.jsonl", None,
                    addition = addition
                    )
''',
    )


def _patch_direct_openai_agent(repo_path: Path) -> None:
    agents_dir = repo_path / "src" / "swe_ci" / "benchmark" / "agents"
    shutil.copyfile(Path(__file__).with_name("direct_openai_agent_template.py"), agents_dir / "direct_openai.py")
    (agents_dir / "Dockerfile.direct_openai").write_text(
        "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nWORKDIR /app\n",
        encoding="utf-8",
    )

    init_path = agents_dir / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    if "call_direct_openai" not in text:
        init_path.write_text(text + "\nfrom .direct_openai import call_direct_openai\n", encoding="utf-8")

    config_path = repo_path / "src" / "swe_ci" / "config.py"
    _replace_once(
        config_path,
        '''    elif cfg.agent_name == "opencode":
        if not hasattr(cfg, "agent"): cfg.agent = SimpleNamespace()
        cfg.agent.node_version = getattr(cfg.agent, "node_version", None) or "22.18.0"
        cfg.agent.npm_pkg = getattr(cfg.agent, "npm_pkg", None) or "opencode-ai"
        cfg.agent.npm_bin = getattr(cfg.agent, "npm_bin", None) or "opencode"
        cfg.agent.dockerfile = str(agent_dir / "Dockerfile.opencode")
    else:
        print(f"Unsupported agent: {cfg.agent_name}", flush=True)
        sys.exit(1)
''',
        '''    elif cfg.agent_name == "opencode":
        if not hasattr(cfg, "agent"): cfg.agent = SimpleNamespace()
        cfg.agent.node_version = getattr(cfg.agent, "node_version", None) or "22.18.0"
        cfg.agent.npm_pkg = getattr(cfg.agent, "npm_pkg", None) or "opencode-ai"
        cfg.agent.npm_bin = getattr(cfg.agent, "npm_bin", None) or "opencode"
        cfg.agent.dockerfile = str(agent_dir / "Dockerfile.opencode")
    elif cfg.agent_name == "direct_openai":
        if not hasattr(cfg, "agent"): cfg.agent = SimpleNamespace()
        cfg.agent.node_version = ""
        cfg.agent.npm_pkg = ""
        cfg.agent.npm_bin = ""
        cfg.agent.dockerfile = str(agent_dir / "Dockerfile.direct_openai")
    else:
        print(f"Unsupported agent: {cfg.agent_name}", flush=True)
        sys.exit(1)
''',
    )

    tools_path = repo_path / "src" / "swe_ci" / "benchmark" / "tools.py"
    _replace_once(
        tools_path,
        '''    func_map = {
        "iflow": call_iflow,
        "opencode": call_opencode,
    }
''',
        '''    func_map = {
        "iflow": call_iflow,
        "opencode": call_opencode,
        "direct_openai": call_direct_openai,
    }
''',
    )


def prepare_swe_ci_execution_repo(config: SweCiRunConfig, run_dir: str | Path, project_root: str | Path) -> Path:
    """Return the SWE-CI checkout to execute, copying and patching when needed."""

    execution_repo = planned_execution_repo_path(config, run_dir)
    if not needs_instrumented_workdir(config):
        return execution_repo
    _copy_swe_ci_repo(config.swe_ci_repo_path, execution_repo)
    if config.docker_network:
        _ensure_docker_network_config(execution_repo)
        _patch_tools_for_network(execution_repo)
    if config.opencode_no_think:
        _patch_opencode_no_think(execution_repo)
    if config.agent_name == DIRECT_OPENAI_AGENT:
        _patch_direct_openai_agent(execution_repo)
    if config.mode == MERGEMIND_ASSISTED_MODE:
        _patch_run_for_mergemind(execution_repo)
    (execution_repo / ".mergemind_project_root").write_text(str(Path(project_root).resolve()) + "\n", encoding="utf-8")
    return execution_repo


def build_assisted_environment(config: SweCiRunConfig, run_dir: str | Path, project_root: str | Path) -> dict[str, str]:
    """Environment consumed by the generated SWE-CI MergeMind bridge."""

    output_root = Path(run_dir) / "mergemind_assist"
    env = {
        "MERGEMIND_PROJECT_ROOT": str(Path(project_root).resolve()),
        "MERGEMIND_CONFIG_PATH": str((config.mergemind_config_path or (Path(project_root) / "configs" / "base.yaml")).resolve()),
        "MERGEMIND_PIPELINE": config.mergemind_pipeline,
        "MERGEMIND_LLM_PROVIDER": config.mergemind_llm_provider,
        "MERGEMIND_TOP_N": str(config.mergemind_top_n),
        "MERGEMIND_MIN_SCORE": str(config.mergemind_min_score),
        "MERGEMIND_ASSIST_OUTPUT_ROOT": str(output_root.resolve()),
    }
    if config.mergemind_max_revision_epochs is not None:
        env["MERGEMIND_MAX_REVISION_EPOCHS"] = str(config.mergemind_max_revision_epochs)
    return env


def execution_task_dir(swe_ci_repo_path: str | Path, config: SweCiRunConfig, task: SweCiTask) -> Path:
    from .config import experiment_name_for_task

    return Path(swe_ci_repo_path) / "experiments" / experiment_name_for_task(config, task) / task.task_id
