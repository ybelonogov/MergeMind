from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys
import types

config_module = types.ModuleType("swe_ci.config")
config_module.CONFIG = types.SimpleNamespace()
sys.modules.setdefault("swe_ci", types.ModuleType("swe_ci"))
sys.modules.setdefault("swe_ci.config", config_module)
from src.validation.swe_ci.direct_openai_agent_template import _normalize_response_path
from src.validation.swe_ci import direct_openai_agent_template as direct_agent


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

    def test_apply_surgical_edits_replaces_one_allowed_snippet(self) -> None:
        files = {"pkg/module.py": "def f():\n    return 1\n"}

        def fake_read(_: str, path: str, max_chars: int = 12000) -> str:
            del max_chars
            return files[path.removeprefix("/app/code/")]

        def fake_write(_: str, path: str, content: str) -> None:
            files[path.removeprefix("/app/code/")] = content

        original_read = direct_agent._read_container_file
        original_write = direct_agent._write_container_file
        try:
            direct_agent._read_container_file = fake_read
            direct_agent._write_container_file = fake_write

            changed = direct_agent._apply_surgical_edits(
                "container",
                '{"edits":[{"path":"/app/code/pkg/module.py","old":"return 1","new":"return 2"}]}',
                allowed_paths={"pkg/module.py"},
            )
        finally:
            direct_agent._read_container_file = original_read
            direct_agent._write_container_file = original_write

        self.assertEqual(changed, ["pkg/module.py"])
        self.assertEqual(files["pkg/module.py"], "def f():\n    return 2\n")

    def test_apply_surgical_edits_rejects_ambiguous_old_text(self) -> None:
        files = {"pkg/module.py": "value = 1\nvalue = 1\n"}

        def fake_read(_: str, path: str, max_chars: int = 12000) -> str:
            del max_chars
            return files[path.removeprefix("/app/code/")]

        original_read = direct_agent._read_container_file
        try:
            direct_agent._read_container_file = fake_read
            with self.assertRaisesRegex(RuntimeError, "expected one old-text match"):
                direct_agent._apply_surgical_edits(
                    "container",
                    '{"edits":[{"path":"pkg/module.py","old":"value = 1","new":"value = 2"}]}',
                    allowed_paths={"pkg/module.py"},
                )
        finally:
            direct_agent._read_container_file = original_read

    def test_apply_surgical_edits_rejects_outside_allowed_files(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside allowed revision set"):
            direct_agent._apply_surgical_edits(
                "container",
                '{"edits":[{"path":"pkg/other.py","old":"x","new":"y"}]}',
                allowed_paths={"pkg/module.py"},
            )

    def test_apply_file_replacements_can_reject_broad_revision(self) -> None:
        files = {"pkg/module.py": "a = 1\nb = 2\nc = 3\n"}
        written: list[str] = []

        def fake_read(_: str, path: str, max_chars: int = 12000) -> str:
            del max_chars
            return files[path.removeprefix("/app/code/")]

        def fake_write(_: str, path: str, content: str) -> None:
            written.append(path)
            files[path.removeprefix("/app/code/")] = content

        original_read = direct_agent._read_container_file
        original_write = direct_agent._write_container_file
        try:
            direct_agent._read_container_file = fake_read
            direct_agent._write_container_file = fake_write
            with self.assertRaisesRegex(RuntimeError, "changed 4 lines"):
                direct_agent._apply_file_replacements(
                    "container",
                    '{"files":[{"path":"pkg/module.py","content":"a = 10\\nb = 20\\nc = 3\\n"}]}',
                    allowed_paths={"pkg/module.py"},
                    max_changed_lines=2,
                )
        finally:
            direct_agent._read_container_file = original_read
            direct_agent._write_container_file = original_write

        self.assertEqual(written, [])

    def test_chat_writes_prompt_log_without_api_key(self) -> None:
        def fake_request(payload: dict, timeout: int) -> dict:
            del payload, timeout
            return {
                "choices": [{"message": {"content": '{"files":[]}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }

        with TemporaryDirectory() as tmp:
            previous_log_dir = os.environ.get("SWE_CI_DIRECT_PROMPT_LOG_DIR")
            previous_task_id = os.environ.get("SWE_CI_DIRECT_TASK_ID")
            os.environ["SWE_CI_DIRECT_PROMPT_LOG_DIR"] = str(Path(tmp) / "logs")
            os.environ["SWE_CI_DIRECT_TASK_ID"] = "task-1"
            original_request = direct_agent._request_chat
            original_base_url = getattr(direct_agent.CONFIG, "base_url", None)
            original_model = getattr(direct_agent.CONFIG, "model_name", None)
            original_api_key = getattr(direct_agent.CONFIG, "api_key", None)
            try:
                direct_agent._request_chat = fake_request
                direct_agent.CONFIG.base_url = "http://127.0.0.1:1234/v1"
                direct_agent.CONFIG.model_name = "qwen"
                direct_agent.CONFIG.api_key = "secret"
                content, usage, _elapsed = direct_agent._chat(
                    [{"role": "user", "content": "fix it"}],
                    timeout=10,
                    max_tokens=20,
                    log_role="programmer",
                    log_stage="programmer",
                )
            finally:
                direct_agent._request_chat = original_request
                direct_agent.CONFIG.base_url = original_base_url
                direct_agent.CONFIG.model_name = original_model
                direct_agent.CONFIG.api_key = original_api_key
                if previous_log_dir is None:
                    os.environ.pop("SWE_CI_DIRECT_PROMPT_LOG_DIR", None)
                else:
                    os.environ["SWE_CI_DIRECT_PROMPT_LOG_DIR"] = previous_log_dir
                if previous_task_id is None:
                    os.environ.pop("SWE_CI_DIRECT_TASK_ID", None)
                else:
                    os.environ["SWE_CI_DIRECT_TASK_ID"] = previous_task_id

            log_text = (Path(tmp) / "logs" / "direct_openai.jsonl").read_text(encoding="utf-8")
            row = json.loads(log_text.splitlines()[0])

        self.assertEqual(content, '{"files":[]}')
        self.assertEqual(usage["total_tokens"], 6)
        self.assertEqual(row["task_id"], "task-1")
        self.assertEqual(row["stage"], "programmer")
        self.assertEqual(row["messages"][0]["content"], "fix it")
        self.assertNotIn("secret", log_text)


if __name__ == "__main__":
    unittest.main()
