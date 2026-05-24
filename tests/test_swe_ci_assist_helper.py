from __future__ import annotations

import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from src.validation.swe_ci.assist_helper import build_assist_example, build_code_diff, run_mergemind_assist


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
        self.assertTrue(review_exists)


if __name__ == "__main__":
    unittest.main()
