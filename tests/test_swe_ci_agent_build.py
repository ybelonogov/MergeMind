from __future__ import annotations

import unittest
from pathlib import Path

from src.validation.swe_ci.agent_build import (
    agent_dockerfile_path,
    build_agent_image_command,
    build_agent_image_env,
    validate_agent_build_inputs,
)


class SweCiAgentBuildTests(unittest.TestCase):
    def test_iflow_build_command_uses_expected_defaults(self) -> None:
        command = build_agent_image_command(
            swe_ci_repo_path=Path("/repo/SWE-CI"),
            agent_name="iflow",
            base_image="image_owner__repo:latest",
            tag="mergemind-iflow",
        )

        self.assertEqual(command[:4], ["docker", "build", "-t", "mergemind-iflow"])
        self.assertIn("BASE_IMAGE=image_owner__repo:latest", command)
        self.assertIn("NODE_VERSION=22.11.0", command)
        self.assertIn("AGENT_NPM_PKG=@iflow-ai/iflow-cli", command)
        self.assertIn("AGENT_BIN=iflow", command)
        self.assertEqual(command[-1], str(Path("/repo/SWE-CI")))

    def test_opencode_dockerfile_path(self) -> None:
        dockerfile = agent_dockerfile_path("/repo/SWE-CI", "opencode")

        self.assertEqual(
            dockerfile,
            Path("/repo/SWE-CI") / "src" / "swe_ci" / "benchmark" / "agents" / "Dockerfile.opencode",
        )

    def test_legacy_builder_sets_docker_buildkit_zero(self) -> None:
        env = build_agent_image_env(builder="legacy", extra_env={"EXAMPLE": "1"})

        self.assertEqual(env["DOCKER_BUILDKIT"], "0")
        self.assertEqual(env["EXAMPLE"], "1")

    def test_validate_agent_build_inputs_reports_missing_repo(self) -> None:
        errors = validate_agent_build_inputs(
            swe_ci_repo_path=Path("missing-SWE-CI"),
            agent_name="iflow",
            base_image="image:latest",
        )

        self.assertTrue(any("does not exist" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
