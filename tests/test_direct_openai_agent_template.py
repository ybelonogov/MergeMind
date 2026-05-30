from __future__ import annotations

import unittest
import sys
import types

config_module = types.ModuleType("swe_ci.config")
config_module.CONFIG = types.SimpleNamespace()
sys.modules.setdefault("swe_ci", types.ModuleType("swe_ci"))
sys.modules.setdefault("swe_ci.config", config_module)
from src.validation.swe_ci.direct_openai_agent_template import _normalize_response_path


class DirectOpenAIAgentTemplateTests(unittest.TestCase):
    def test_normalize_response_path_accepts_container_code_paths(self) -> None:
        self.assertEqual(_normalize_response_path("/app/code/pkg/module.py"), "pkg/module.py")
        self.assertEqual(_normalize_response_path("code/pkg/module.py"), "pkg/module.py")

    def test_normalize_response_path_accepts_container_app_paths(self) -> None:
        self.assertEqual(_normalize_response_path("/app/pkg/module.py"), "pkg/module.py")
        self.assertEqual(_normalize_response_path("app/pkg/module.py"), "pkg/module.py")

    def test_normalize_response_path_preserves_unsafe_absolute_paths(self) -> None:
        self.assertEqual(_normalize_response_path("/etc/passwd"), "/etc/passwd")
        self.assertEqual(_normalize_response_path("/app/../secret.py"), "../secret.py")


if __name__ == "__main__":
    unittest.main()
