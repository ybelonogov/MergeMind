from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.schema import CandidateComment
from src.validation.swe_ci.review_loop import (
    AgentPatch,
    build_review_example,
    find_agent_patch,
    write_mergemind_review_artifacts,
)
from src.validation.swe_ci.schemas import SweCiTask


def _task() -> SweCiTask:
    return SweCiTask(
        task_id="task-1",
        repo_name="owner/repo",
        repo_url="https://github.com/owner/repo",
        current_sha="BASE_SECRET_SHA",
        target_sha="TARGET_SECRET_SHA",
        image_sha="sha256:image",
        test_gap={},
    )


class SweCiReviewLoopTests(unittest.TestCase):
    def test_finds_agent_patch_file(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "task"
            output_dir.mkdir()
            patch_path = output_dir / "agent.patch"
            patch_path.write_text(
                "\n".join(
                    [
                        "diff --git a/app.py b/app.py",
                        "--- a/app.py",
                        "+++ b/app.py",
                        "@@ -1 +1 @@",
                        "-old",
                        "+new",
                    ]
                ),
                encoding="utf-8",
            )

            patch = find_agent_patch(output_dir)

        self.assertIsNotNone(patch)
        self.assertEqual(patch.source_type, "file")
        self.assertIn("diff --git", patch.text)

    def test_review_example_does_not_include_target_sha(self) -> None:
        patch = AgentPatch(
            text="diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
            source_path="agent.patch",
            source_type="file",
        )

        example = build_review_example(_task(), patch)
        payload = json.dumps(example.to_dict())

        self.assertIn("BASE_SECRET_SHA", payload)
        self.assertNotIn("TARGET_SECRET_SHA", payload)
        self.assertFalse(example.metadata["target_sha_used_for_review"])

    def test_writes_mergemind_review_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "task"
            patch = AgentPatch(
                text="diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
                source_path="agent.patch",
                source_type="file",
            )
            example = build_review_example(_task(), patch)
            prediction = CandidateComment(text="Consider validating this new behavior.", reranker_score=0.8)

            metrics = write_mergemind_review_artifacts(
                task_output_dir=output_dir,
                task=_task(),
                patch=patch,
                example=example,
                predictions=[prediction],
            )
            comments = json.loads((output_dir / "mergemind_comments.json").read_text(encoding="utf-8"))
            review = (output_dir / "mergemind_review.md").read_text(encoding="utf-8")

        self.assertEqual(metrics["comment_count"], 1)
        self.assertFalse(comments["target_sha_used_for_review"])
        self.assertIn("Consider validating", review)


if __name__ == "__main__":
    unittest.main()
