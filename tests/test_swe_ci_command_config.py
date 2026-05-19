from __future__ import annotations

import unittest
from pathlib import Path

from src.validation.swe_ci.config import build_swe_ci_command
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


if __name__ == "__main__":
    unittest.main()
