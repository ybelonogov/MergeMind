"""End-to-end smoke tests for the CLI scripts."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_temp_config(base_dir: Path) -> Path:
    config_text = "\n".join(
        [
            "project_name: MergeMind",
            "paths:",
            "  raw:",
            f"    codereviewer: {PROJECT_ROOT / 'sample_data/raw/codereviewer.jsonl'}",
            f"    codereviewqa: {PROJECT_ROOT / 'sample_data/raw/codereviewqa.jsonl'}",
            f"    codocbench: {PROJECT_ROOT / 'sample_data/raw/codocbench.jsonl'}",
            f"  prepared_dir: {base_dir / 'data'}",
            f"  model_dir: {base_dir / 'models'}",
            f"  evaluation_dir: {base_dir / 'evaluation'}",
            "pipeline:",
            "  - datasets",
            "  - context_processing",
            "  - models",
            "  - validation",
            "model:",
            "  retrieval_top_examples: 4",
            "  max_candidates: 5",
            "  reranker:",
            "    retrieval_weight: 0.65",
            "    identifier_weight: 0.2",
            "    file_weight: 0.1",
            "    action_weight: 0.08",
            "    hedge_penalty: 0.08",
            "    verbosity_penalty: 0.05",
            "    short_penalty: 0.04",
            "validation:",
            "  use_llm_judge: false",
            "  use_tests_ci: true",
            "  similarity_threshold: 0.35",
            "  top_n: 3",
            "demo:",
            "  top_n: 3",
        ]
    )
    config_path = base_dir / "base.yaml"
    config_path.write_text(config_text + "\n", encoding="utf-8")
    return config_path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


class EndToEndScriptTests(unittest.TestCase):
    def test_end_to_end_scripts_run_successfully(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            config_path = _write_temp_config(base_dir)

            prepare_result = _run(["python", "scripts/prepare_data.py", "--config", str(config_path)])
            train_result = _run(["python", "scripts/train_baseline.py", "--config", str(config_path)])
            evaluate_result = _run(["python", "scripts/evaluate.py", "--config", str(config_path)])
            demo_result = _run(["python", "scripts/demo_mr.py", "--config", str(config_path)])

            summary_path = base_dir / "evaluation" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertIn("[prepare_data]", prepare_result.stdout)
        self.assertIn("[train_baseline]", train_result.stdout)
        self.assertIn("[evaluate]", evaluate_result.stdout)
        self.assertIn("[demo_mr]", demo_result.stdout)
        self.assertEqual(summary["example_count"], 4)
        self.assertIn("Stock is decremented", demo_result.stdout)


if __name__ == "__main__":
    unittest.main()
