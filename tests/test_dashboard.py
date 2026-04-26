"""Dashboard monitoring helper tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.monitoring.dashboard import collect_lmstudio_status, collect_runs, parse_nvidia_smi_csv


class DashboardMonitoringTests(unittest.TestCase):
    def test_collect_lmstudio_status_uses_model_env_override(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"data": [{"id": "qwen3.6-27b@iq2_xxs"}]}).encode("utf-8")

        config = {"llm": {"base_url": "http://localhost:1234/v1", "model": "qwen/qwen3.5-9b"}}
        with patch.dict("os.environ", {"MERGEMIND_LLM_MODEL": "qwen3.6-27b@iq2_xxs"}), patch(
            "src.monitoring.dashboard.urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            status = collect_lmstudio_status(config)

        self.assertTrue(status["ok"])
        self.assertEqual(status["configured_model"], "qwen3.6-27b@iq2_xxs")
        self.assertEqual(status["config_model"], "qwen/qwen3.5-9b")
        self.assertTrue(status["env_model_override"])
        self.assertEqual(status["available_model_count"], 1)

    def test_parse_nvidia_smi_csv(self) -> None:
        output = "NVIDIA GeForce RTX 5070, 42, 6144, 12227, 63, 155.50\n"

        records = parse_nvidia_smi_csv(output)

        self.assertEqual(records[0]["name"], "NVIDIA GeForce RTX 5070")
        self.assertEqual(records[0]["utilization_gpu"], 42.0)
        self.assertEqual(records[0]["memory_used_mb"], 6144.0)
        self.assertEqual(records[0]["power_w"], 155.5)

    def test_collect_runs_reads_summary_and_progress(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            mode_dir = runs_dir / "demo_run" / "qwen35_generator_qwen35_reranker"
            mode_dir.mkdir(parents=True)
            (mode_dir / "progress.jsonl").write_text(
                json.dumps(
                    {
                        "completed": 3,
                        "total": 10,
                        "example_id": "mr-3",
                        "latency_sec": 2.0,
                        "llm_metrics": {"tokens_per_second": 15.5, "total_tokens": 120},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (mode_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "example_count": 10,
                        "best_similarity_at_k": 0.25,
                        "tokens_per_second": 20.0,
                        "total_tokens": 500,
                    }
                ),
                encoding="utf-8",
            )
            (mode_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "model_id": "qwen3.6-27b@iq2_xxs",
                        "base_url": "http://localhost:1234/v1",
                        "llm_provider": "local_qwen36_27b_iq2",
                    }
                ),
                encoding="utf-8",
            )

            runs = collect_runs(runs_dir)

        self.assertEqual(runs[0]["run_id"], "demo_run")
        mode = runs[0]["modes"][0]
        self.assertEqual(mode["status"], "completed")
        self.assertEqual(mode["progress"]["completed"], 3)
        self.assertEqual(mode["progress"]["total_tokens"], 500)
        self.assertEqual(mode["progress"]["tokens_per_second"], 20.0)
        self.assertEqual(mode["model_id"], "qwen3.6-27b@iq2_xxs")
        self.assertEqual(runs[0]["model_ids"], ["qwen3.6-27b@iq2_xxs"])


if __name__ == "__main__":
    unittest.main()
