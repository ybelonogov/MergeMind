from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.validation.swe_ci.config import EXPECTED_SWE_CI_FILES, validate_run_config
from src.validation.swe_ci.schemas import SweCiRunConfig


def _create_fake_swe_ci_repo(root: Path) -> None:
    for relative in EXPECTED_SWE_CI_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake file for setup validation tests\n", encoding="utf-8")


class SweCiSetupValidationTests(unittest.TestCase):
    def test_valid_minimal_setup(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "SWE-CI"
            tasks = base / "tasks.jsonl"
            output = base / "runs"
            _create_fake_swe_ci_repo(repo)
            tasks.write_text("", encoding="utf-8")
            config = SweCiRunConfig(repo, tasks, output, None, 3, 60, "baseline", "run-1")

            with patch("src.validation.swe_ci.config.check_command_available", return_value=None):
                errors = validate_run_config(config)

        self.assertEqual(errors, [])

    def test_reports_missing_repo_and_tasks(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = SweCiRunConfig(
                base / "missing-swe-ci",
                base / "missing-tasks.jsonl",
                base / "runs",
                None,
                3,
                60,
                "baseline",
                "run-1",
            )

            with patch("src.validation.swe_ci.config.check_command_available", return_value=None):
                errors = validate_run_config(config)

        joined = "\n".join(errors)
        self.assertIn("SWE-CI repo path does not exist", joined)
        self.assertIn("tasks file does not exist", joined)

    def test_rejects_invalid_timeout_and_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "SWE-CI"
            tasks = base / "tasks.jsonl"
            _create_fake_swe_ci_repo(repo)
            tasks.write_text("", encoding="utf-8")
            config = SweCiRunConfig(repo, tasks, base / "runs", None, 0, 0, "unknown-mode", "")

            with patch("src.validation.swe_ci.config.check_command_available", return_value=None):
                errors = validate_run_config(config)

        joined = "\n".join(errors)
        self.assertIn("max_iterations must be positive", joined)
        self.assertIn("timeout_seconds must be positive", joined)
        self.assertIn("Unsupported mode='unknown-mode'", joined)
        self.assertIn("run_id must not be empty", joined)

    def test_accepts_mergemind_review_loop_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "SWE-CI"
            tasks = base / "tasks.jsonl"
            config_path = base / "base.yaml"
            _create_fake_swe_ci_repo(repo)
            tasks.write_text("", encoding="utf-8")
            config_path.write_text("paths:\n  data_dir: artifacts/data\n", encoding="utf-8")
            config = SweCiRunConfig(
                repo,
                tasks,
                base / "runs",
                None,
                3,
                60,
                "mergemind_review_loop",
                "run-1",
                mergemind_config_path=config_path,
            )

            with patch("src.validation.swe_ci.config.check_command_available", return_value=None):
                errors = validate_run_config(config)

        self.assertEqual(errors, [])

    def test_accepts_mergemind_assisted_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "SWE-CI"
            tasks = base / "tasks.jsonl"
            config_path = base / "base.yaml"
            _create_fake_swe_ci_repo(repo)
            tasks.write_text("", encoding="utf-8")
            config_path.write_text("paths:\n  data_dir: artifacts/data\n", encoding="utf-8")
            config = SweCiRunConfig(
                repo,
                tasks,
                base / "runs",
                None,
                3,
                60,
                "mergemind_assisted",
                "run-1",
                mergemind_config_path=config_path,
                assisted_work_dir=base / "assisted-swe-ci",
                docker_network="host",
            )

            with patch("src.validation.swe_ci.config.check_command_available", return_value=None):
                errors = validate_run_config(config)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
