from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.assisted import prepare_swe_ci_execution_repo
from src.validation.swe_ci.schemas import SweCiRunConfig


def _fake_swe_ci_repo(path: Path) -> None:
    (path / "src" / "swe_ci" / "benchmark").mkdir(parents=True)
    (path / "src" / "swe_ci").mkdir(parents=True, exist_ok=True)
    (path / "src" / "swe_ci" / "evaluate.py").write_text("# fake\n", encoding="utf-8")
    (path / "src" / "swe_ci" / "download.py").write_text("# fake\n", encoding="utf-8")
    (path / "src" / "swe_ci" / "summarize.py").write_text("# fake\n", encoding="utf-8")
    (path / "src" / "swe_ci" / "config.py").write_text(
        "\n".join(
            [
                "from types import SimpleNamespace",
                "def load_config():",
                "    cfg = SimpleNamespace()",
                '    if cfg.agent_name == "iflow":',
                "        pass",
                '    elif cfg.agent_name == "opencode":',
                '        if not hasattr(cfg, "agent"): cfg.agent = SimpleNamespace()',
                '        cfg.agent.node_version = getattr(cfg.agent, "node_version", None) or "22.18.0"',
                '        cfg.agent.npm_pkg = getattr(cfg.agent, "npm_pkg", None) or "opencode-ai"',
                '        cfg.agent.npm_bin = getattr(cfg.agent, "npm_bin", None) or "opencode"',
                '        cfg.agent.dockerfile = str(agent_dir / "Dockerfile.opencode")',
                "    else:",
                '        print(f"Unsupported agent: {cfg.agent_name}", flush=True)',
                "        sys.exit(1)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "src" / "swe_ci" / "benchmark" / "tools.py").write_text(
        "\n".join(
            [
                "def container_extra_args() -> list[str]:",
                "    extra_args = []",
                "    return extra_args",
                "",
                "def call_cli_agent():",
                "    func_map = {",
                '        "iflow": call_iflow,',
                '        "opencode": call_opencode,',
                "    }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "src" / "swe_ci" / "benchmark" / "agents").mkdir(parents=True)
    (path / "src" / "swe_ci" / "benchmark" / "agents" / "__init__.py").write_text(
        "from .iflow import call_iflow\nfrom .opencode import call_opencode\n",
        encoding="utf-8",
    )
    (path / "src" / "swe_ci" / "benchmark" / "agents" / "iflow.py").write_text(
        "def call_iflow(*args, **kwargs):\n    return {}\n",
        encoding="utf-8",
    )
    (path / "src" / "swe_ci" / "benchmark" / "agents" / "opencode.py").write_text(
        "\n".join(
            [
                "import json",
                "import subprocess",
                "from swe_ci.config import CONFIG",
                'HOME_DIR = "/opt/agent/home"',
                'AUTH_DIR = f"{HOME_DIR}/.local/share/opencode"',
                'CFG_DIR = f"{HOME_DIR}/.config/opencode"',
                'AUTH_FILE = "auth.json"',
                'CFG_FILE = "opencode.json"',
                "def setup_opencode(container_name: str) -> None:",
                "    auth = {}",
                "    cfg = {}",
                '    auth_payload = json.dumps(auth, indent=4, ensure_ascii=False) + "\\n"',
                '    cfg_payload = json.dumps(cfg, indent=4, ensure_ascii=False) + "\\n"',
                "    subprocess.run([",
                '        "docker", "exec", "-i", "-u", "root", container_name, "sh", "-c", ',
                '        f"mkdir -p {AUTH_DIR} && cat > {AUTH_DIR}/{AUTH_FILE}"',
                "        ], input=auth_payload, text=True, check=True)",
                "",
                "    subprocess.run([",
                '        "docker", "exec", "-i", "-u", "root", container_name, "sh", "-c", ',
                '        f"mkdir -p {CFG_DIR} && cat > {CFG_DIR}/{CFG_FILE}"',
                "        ], input=cfg_payload, text=True, check=True)",
                "def call_opencode(container_name: str, prompt: str) -> None:",
                "    subprocess.run([",
                '        "docker", "exec", "-w", "/app",',
                '        "-e", f"HOME={HOME_DIR}", "-e", "DISABLE_SEND_PV=1",',
                "        container_name,",
                '        "opencode", "run", "--model", f"custom/{CONFIG.model_name}", prompt,',
                "        ])",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "src" / "swe_ci" / "benchmark" / "run.py").write_text(
        "\n".join(
            [
                "import shutil",
                "from pathlib import Path",
                "from swe_ci.config import CONFIG",
                "def f():",
                "            # Step 4.3: Call programmer agent with retry mechanism",
                "                    copy_dir_from_container(container_name, \"/app/code\", tmp_dir, mkdir=True)",
                "                    logger.info(prefix + \"✅ The programmer agent has modified the code.\")",
                "                    break",
                "                update_iteration(",
                "                    gap, task_dir / \"iteration.jsonl\", current_dir / \"test_report.json\",",
                "                    addition = {\"architect\": architect_result, \"programmer\": programmer_result}",
                "                    )",
                "                update_iteration(",
                "                    -1, task_dir / \"iteration.jsonl\", None,",
                "                    addition = {\"architect\": architect_result, \"programmer\": programmer_result}",
                "                    )",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "config.toml").write_text('model_name = ""\n[docker]\nstorage_disk = ""\n', encoding="utf-8")
    (path / "data").mkdir()
    (path / "data" / "large.bin").write_text("do not copy", encoding="utf-8")


class SweCiAssistedWorkdirTests(unittest.TestCase):
    def test_prepares_assisted_copy_without_mutating_source(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "SWE-CI"
            _fake_swe_ci_repo(source)
            config = SweCiRunConfig(
                swe_ci_repo_path=source,
                tasks_path=base / "tasks.jsonl",
                output_dir=base / "runs",
                limit=1,
                max_iterations=1,
                timeout_seconds=60,
                mode="mergemind_assisted",
                run_id="run-1",
                docker_network="host",
                opencode_no_think=True,
            )

            execution_repo = prepare_swe_ci_execution_repo(config, base / "runs" / "run-1", base / "project")

            source_run = (source / "src" / "swe_ci" / "benchmark" / "run.py").read_text(encoding="utf-8")
            copied_run = (execution_repo / "src" / "swe_ci" / "benchmark" / "run.py").read_text(encoding="utf-8")
            copied_helper = (execution_repo / "src" / "swe_ci" / "benchmark" / "mergemind_assist.py").read_text(
                encoding="utf-8"
            )
            copied_tools = (execution_repo / "src" / "swe_ci" / "benchmark" / "tools.py").read_text(encoding="utf-8")
            copied_opencode = (execution_repo / "src" / "swe_ci" / "benchmark" / "agents" / "opencode.py").read_text(encoding="utf-8")
            copied_config = (execution_repo / "config.toml").read_text(encoding="utf-8")

        self.assertNotEqual(execution_repo, source)
        self.assertNotIn("run_mergemind_assist", source_run)
        self.assertIn("run_mergemind_assist", copied_run)
        self.assertIn("apply_revision", copied_run)
        self.assertIn("unexpected_revision_files", copied_run)
        self.assertIn("changed files outside the programmer patch", copied_run)
        self.assertIn("Non-retryable MergeMind revision guard", copied_run)
        self.assertIn("revision_skipped_files", copied_run)
        self.assertIn('file.endswith(".py")', copied_run)
        self.assertIn("no eligible source-code files", copied_run)
        self.assertIn("MERGEMIND_MIN_SCORE", copied_helper)
        self.assertIn("MERGEMIND_MAX_REVISION_EPOCHS", copied_helper)
        self.assertIn("Do not create new files.", copied_helper)
        self.assertIn("mergemind_allowed_files.txt", copied_helper)
        self.assertIn("mergemind_before_files.md", copied_helper)
        self.assertIn("Do not modify files that are absent", copied_helper)
        self.assertIn('copy_dir_to_container(container_name, tmp_dir/"code", "/app")', copied_run)
        self.assertIn('copy_file_to_container(container_name, allowed_files_path, "/app", rename="mergemind_allowed_files.txt")', copied_run)
        self.assertIn('allowed_files_path.write_text("\\n".join(original_changed_files) + "\\n", encoding="utf-8")', copied_run)
        self.assertIn('copy_file_to_container(container_name, before_files_path, "/app", rename="mergemind_before_files.md")', copied_run)
        self.assertIn("Before-patch snapshots for MergeMind revision", copied_run)
        self.assertIn("--network", copied_tools)
        self.assertIn("NO_THINK_INSTRUCTIONS_FILE", copied_opencode)
        self.assertIn('input="/no_think\\n"', copied_opencode)
        self.assertIn("--dangerously-skip-permissions", copied_opencode)
        self.assertIn("--print-logs", copied_opencode)
        self.assertIn("opencode_no_think = true", copied_config)
        self.assertFalse((execution_repo / "data" / "large.bin").exists())

    def test_direct_openai_agent_is_added_to_copied_checkout(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "SWE-CI"
            _fake_swe_ci_repo(source)
            config = SweCiRunConfig(
                swe_ci_repo_path=source,
                tasks_path=base / "tasks.jsonl",
                output_dir=base / "runs",
                limit=1,
                max_iterations=1,
                timeout_seconds=60,
                mode="baseline",
                run_id="run-1",
                agent_name="direct_openai",
            )

            execution_repo = prepare_swe_ci_execution_repo(config, base / "runs" / "run-1", base / "project")

            copied_init = (execution_repo / "src" / "swe_ci" / "benchmark" / "agents" / "__init__.py").read_text(
                encoding="utf-8"
            )
            copied_tools = (execution_repo / "src" / "swe_ci" / "benchmark" / "tools.py").read_text(encoding="utf-8")
            copied_config = (execution_repo / "src" / "swe_ci" / "config.py").read_text(encoding="utf-8")
            copied_direct_agent = (
                execution_repo / "src" / "swe_ci" / "benchmark" / "agents" / "direct_openai.py"
            ).read_text(encoding="utf-8")
            has_direct_agent = (
                execution_repo / "src" / "swe_ci" / "benchmark" / "agents" / "direct_openai.py"
            ).exists()
            has_direct_dockerfile = (
                execution_repo / "src" / "swe_ci" / "benchmark" / "agents" / "Dockerfile.direct_openai"
            ).exists()

        self.assertNotEqual(execution_repo, source)
        self.assertTrue(has_direct_agent)
        self.assertTrue(has_direct_dockerfile)
        self.assertIn("call_direct_openai", copied_init)
        self.assertIn('"direct_openai": call_direct_openai', copied_tools)
        self.assertIn('cfg.agent.dockerfile = str(agent_dir / "Dockerfile.direct_openai")', copied_config)
        self.assertIn("MergeMind review guidance", copied_direct_agent)
        self.assertIn("outside allowed revision set", copied_direct_agent)
        self.assertIn("If no safe edit exists in those files", copied_direct_agent)
        self.assertIn("/app/mergemind_review.md", copied_direct_agent)
        self.assertIn("Allowed revision files", copied_direct_agent)
        self.assertIn("/app/mergemind_allowed_files.txt", copied_direct_agent)
        self.assertIn("Before-patch source snapshots", copied_direct_agent)
        self.assertIn("/app/mergemind_before_files.md", copied_direct_agent)


if __name__ == "__main__":
    unittest.main()
