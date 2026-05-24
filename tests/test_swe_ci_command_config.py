from __future__ import annotations

import os
import unittest
from pathlib import Path

from src.validation.swe_ci.config import build_swe_ci_command, build_swe_ci_env
from src.validation.swe_ci.reporter import run_dir_for
from src.validation.swe_ci.process_runner import redact_command
from src.validation.swe_ci.schemas import SweCiRunConfig, SweCiTask


class SweCiCommandConfigTests(unittest.TestCase):
    def test_global_model_arguments_are_added_to_command(self) -> None:
        config = SweCiRunConfig(
            swe_ci_repo_path=Path("SWE-CI"),
            tasks_path=Path("tasks.jsonl"),
            output_dir=Path("runs"),
            limit=1,
            max_iterations=3,
            timeout_seconds=60,
            mode="baseline",
            run_id="run-1",
            splitting="default",
            api_key="secret-key",
            base_url="http://host.docker.internal:1234/v1",
            model_name="qwen3.6-27b@iq2_xxs",
            agent_name="opencode",
        )
        task = SweCiTask(
            task_id="task-1",
            repo_name="owner/repo",
            repo_url="https://github.com/owner/repo",
            current_sha="abc",
            target_sha="def",
            image_sha="sha256:image",
            test_gap={},
        )

        command = build_swe_ci_command(config, task, Path("out"))

        self.assertIn("--base_url", command)
        self.assertIn("http://host.docker.internal:1234/v1", command)
        self.assertIn("--model_name", command)
        self.assertIn("qwen3.6-27b@iq2_xxs", command)
        self.assertIn("--api_key", command)
        self.assertIn("secret-key", command)
        self.assertIn("--agent_name", command)
        self.assertIn("opencode", command)
        self.assertNotIn("secret-key", " ".join(redact_command(command)))

    def test_docker_network_is_added_to_command(self) -> None:
        config = SweCiRunConfig(
            swe_ci_repo_path=Path("SWE-CI"),
            tasks_path=Path("tasks.jsonl"),
            output_dir=Path("runs"),
            limit=1,
            max_iterations=3,
            timeout_seconds=60,
            mode="mergemind_assisted",
            run_id="run-1",
            docker_network="host",
        )
        task = SweCiTask(
            task_id="task-1",
            repo_name="owner/repo",
            repo_url="https://github.com/owner/repo",
            current_sha="abc",
            target_sha="def",
            image_sha="sha256:image",
            test_gap={},
        )

        command = build_swe_ci_command(config, task, Path("out"))

        self.assertIn("--docker.network", command)
        self.assertIn("host", command)

    def test_absolute_task_dir_is_passed_to_swe_ci(self) -> None:
        config = SweCiRunConfig(
            swe_ci_repo_path=Path("SWE-CI"),
            tasks_path=Path("tasks.jsonl"),
            output_dir=Path.cwd() / "tmp" / "mergemind-swe-ci-runs",
            limit=1,
            max_iterations=1,
            timeout_seconds=60,
            mode="baseline",
            run_id="run-1",
        )
        task = SweCiTask(
            task_id="task-1",
            repo_name="owner/repo",
            repo_url="https://github.com/owner/repo",
            current_sha="abc",
            target_sha="def",
            image_sha="sha256:image",
            test_gap={},
        )
        run_dir = run_dir_for(config)
        command = build_swe_ci_command(config, task, run_dir / "swe_ci_outputs" / task.task_id)
        save_root_dir = command[command.index("--save_root_dir") + 1]

        self.assertTrue(Path(save_root_dir).is_absolute())

    def test_build_swe_ci_env_uses_platform_path_separator(self) -> None:
        swe_ci_repo_path = Path("/repo/SWE-CI")
        project_root = Path("/repo/MergeMind")

        env = build_swe_ci_env(swe_ci_repo_path, project_root)

        self.assertEqual(env["PYTHONPATH"], os.pathsep.join([str(swe_ci_repo_path / "src"), str(project_root)]))


if __name__ == "__main__":
    unittest.main()
