from __future__ import annotations

import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.assist_helper import (
    build_assist_example,
    build_code_diff,
    build_previous_failure_context,
    pipeline_uses_failure_context,
    pipeline_uses_python_only_diff,
    run_mergemind_assist,
)


class SweCiAssistHelperTests(unittest.TestCase):
    def test_build_code_diff_excludes_tests(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            before = base / "before"
            after = base / "after"
            (before / "pkg").mkdir(parents=True)
            (after / "pkg").mkdir(parents=True)
            (before / "tests").mkdir()
            (after / "tests").mkdir()
            (before / "pkg" / "app.py").write_text("value = 1\n", encoding="utf-8")
            (after / "pkg" / "app.py").write_text("value = 2\n", encoding="utf-8")
            (before / "tests" / "test_app.py").write_text("assert 1\n", encoding="utf-8")
            (after / "tests" / "test_app.py").write_text("assert 2\n", encoding="utf-8")

            diff = build_code_diff(before, after)

        self.assertIn("pkg/app.py", diff)
        self.assertNotIn("tests/test_app.py", diff)

    def test_build_code_diff_empty_when_only_tests_change(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            before = base / "before"
            after = base / "after"
            (before / "tests").mkdir(parents=True)
            (after / "tests").mkdir(parents=True)
            (before / "tests" / "test_app.py").write_text("assert 1\n", encoding="utf-8")
            (after / "tests" / "test_app.py").write_text("assert 2\n", encoding="utf-8")

            diff = build_code_diff(before, after)

        self.assertEqual(diff, "")

    def test_build_code_diff_can_limit_to_python_source(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            before = base / "before"
            after = base / "after"
            (before / "pkg").mkdir(parents=True)
            (after / "pkg").mkdir(parents=True)
            (before / "pkg" / "app.py").write_text("value = 1\n", encoding="utf-8")
            (after / "pkg" / "app.py").write_text("value = 2\n", encoding="utf-8")
            (before / "pkg" / "data.md").write_text("old\n", encoding="utf-8")
            (after / "pkg" / "data.md").write_text("new\n", encoding="utf-8")

            diff = build_code_diff(before, after, allowed_suffixes={".py"})

        self.assertIn("pkg/app.py", diff)
        self.assertNotIn("pkg/data.md", diff)

    def test_pipeline_feature_flags_for_test_guard(self) -> None:
        self.assertTrue(pipeline_uses_failure_context("qwen35_rewriter_sweci_test_guard"))
        self.assertTrue(pipeline_uses_python_only_diff("qwen35_rewriter_sweci_test_guard"))
        self.assertTrue(pipeline_uses_failure_context("qwen35_caveman_test_triage"))
        self.assertFalse(pipeline_uses_python_only_diff("qwen35_caveman_test_triage"))

    def test_assist_example_does_not_include_target_sha(self) -> None:
        example = build_assist_example(
            task_id="task-1",
            repo_name="owner/repo",
            repo_url="https://github.com/owner/repo",
            current_sha="BASE_SECRET",
            image_sha="sha256:image",
            requirement_text="Do the thing.",
            diff_text="diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-a\n+b\n",
        )
        payload = json.dumps(example.to_dict())

        self.assertIn("BASE_SECRET", payload)
        self.assertNotIn("TARGET_SECRET", payload)
        self.assertFalse(example.metadata["target_sha_used_for_review"])

    def test_previous_failure_context_reads_visible_test_report(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            code_dir = base / "code"
            code_dir.mkdir()
            (base / "test_report.json").write_text(
                json.dumps(
                    {
                        "summary": {"failed": 1, "error": 1, "passed": 3, "total": 5},
                        "tests": [
                            {"nodeid": "tests/test_a.py::test_a", "outcome": "passed"},
                            {"nodeid": "tests/test_b.py::test_b", "outcome": "failed", "call": {"outcome": "failed"}},
                            {"nodeid": "tests/test_c.py::test_c", "outcome": "error", "setup": {"outcome": "failed"}},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            context = build_previous_failure_context(code_dir)

        self.assertIn("failed=1", context)
        self.assertIn("error=1", context)
        self.assertIn("tests/test_b.py::test_b", context)
        self.assertIn("tests/test_c.py::test_c", context)

    def test_helper_skips_no_diff_and_writes_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            before = base / "before"
            after = base / "after"
            before.mkdir()
            after.mkdir()
            (base / "requirement.xml").write_text("<requirements />", encoding="utf-8")
            args = Namespace(
                output_dir=str(base / "out"),
                before_code_dir=str(before),
                after_code_dir=str(after),
                requirement_path=str(base / "requirement.xml"),
                config="configs/base.yaml",
                llm_provider="",
                pipeline="qwen35_rewriter",
                project_root=str(Path.cwd()),
                task_id="task-1",
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo",
                current_sha="abc",
                image_sha="sha256:image",
                top_n=3,
            )

            result = run_mergemind_assist(args)
            review_exists = Path(result["review_path"]).exists()

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["comment_count"], 0)
            self.assertFalse(result["apply_revision"])
            self.assertTrue(review_exists)


if __name__ == "__main__":
    unittest.main()
