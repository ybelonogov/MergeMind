from __future__ import annotations

import subprocess
import sys
import unittest

from scripts.prepare_linux_lan_swe_ci import normalize_lmstudio_url


class PrepareLinuxLanSweCiTests(unittest.TestCase):
    def test_normalize_lmstudio_url_from_host(self) -> None:
        self.assertEqual(
            normalize_lmstudio_url(windows_host="192.168.1.50", lmstudio_url="", port=1234),
            "http://192.168.1.50:1234/v1",
        )

    def test_normalize_lmstudio_url_appends_v1(self) -> None:
        self.assertEqual(
            normalize_lmstudio_url(windows_host="", lmstudio_url="http://llm-box:1234", port=1234),
            "http://llm-box:1234/v1",
        )

    def test_script_prints_lan_url_and_swe_ci_commands(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/prepare_linux_lan_swe_ci.py",
                "--windows-host",
                "192.168.1.50",
                "--model-name",
                "qwen3.6-27b@iq2_xxs",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("curl http://192.168.1.50:1234/v1/models", completed.stdout)
        self.assertIn("MERGEMIND_LLM_BASE_URL=http://192.168.1.50:1234/v1", completed.stdout)
        self.assertIn("scripts/prebuild_swe_ci_agent.py", completed.stdout)
        self.assertIn("--mode mergemind_review_loop", completed.stdout)


if __name__ == "__main__":
    unittest.main()
