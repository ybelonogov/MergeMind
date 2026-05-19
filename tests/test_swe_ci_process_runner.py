from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.process_runner import run_process


class SweCiProcessRunnerTests(unittest.TestCase):
    def test_saves_stdout_stderr_and_success_status(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_process(
                command=[
                    sys.executable,
                    "-c",
                    "import sys; print('ok'); print('warn', file=sys.stderr)",
                ],
                task_id="task-1",
                task_log_dir=Path(tmp) / "logs" / "task-1",
                timeout_seconds=10,
                monitor_interval_seconds=0.1,
            )

            stdout = Path(result.stdout_path).read_text(encoding="utf-8")
            stderr = Path(result.stderr_path).read_text(encoding="utf-8")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ok", stdout)
        self.assertIn("warn", stderr)

    def test_timeout_kills_process(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_process(
                command=[sys.executable, "-c", "import time; time.sleep(5)"],
                task_id="slow-task",
                task_log_dir=Path(tmp) / "logs" / "slow-task",
                timeout_seconds=1,
                monitor_interval_seconds=0.1,
            )

        self.assertEqual(result.status, "timeout")
        self.assertIn("timed out", result.error_message)

    def test_event_log_redacts_secret_arguments(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_process(
                command=[sys.executable, "-c", "print('ok')", "--api_key", "secret-value"],
                task_id="secret-task",
                task_log_dir=Path(tmp) / "logs" / "secret-task",
                timeout_seconds=10,
                monitor_interval_seconds=0.1,
            )
            events = [
                json.loads(line)
                for line in Path(result.events_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            event_text = Path(result.events_path).read_text(encoding="utf-8")

            self.assertEqual(events[0]["command"][-1], "***")
            self.assertNotIn("secret-value", event_text)


if __name__ == "__main__":
    unittest.main()
