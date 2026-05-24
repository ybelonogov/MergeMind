"""GitHub PR experiment runner tests."""

from __future__ import annotations

import unittest

from scripts.run_github_pr_experiment import _manifest_pipelines, _manifest_prs, _render_summary


class GitHubPRExperimentTests(unittest.TestCase):
    def test_manifest_normalizes_urls_and_pipeline_aliases(self) -> None:
        manifest = {
            "pipelines": ["qwen35_rewriter", "qwen35_review_contract"],
            "prs": [
                "https://github.com/acme/widgets/pull/1",
                {"url": "https://github.com/acme/widgets/pull/2", "notes": "has review"},
            ],
        }

        self.assertEqual(len(_manifest_prs(manifest)), 2)
        self.assertEqual(
            _manifest_pipelines(manifest),
            ["qwen35_full_with_rewriter", "qwen35_rewriter_sweci_contract"],
        )

    def test_summary_includes_metrics_and_artifact_context(self) -> None:
        rows = [
            {
                "url": "https://github.com/acme/widgets/pull/1",
                "title": "Fix widget state",
                "pipeline": "qwen35_rewriter_sweci_contract",
                "gold_comment_count": 2,
                "comment_count": 3,
                "hit_at_k": 1,
                "best_similarity_at_k": 0.6,
                "judge_enabled": True,
                "judge_score": 0.8,
                "groundedness": 0.9,
                "usefulness": 0.7,
                "total_tokens": 100,
                "total_wall_latency_sec": 12.5,
                "fallback_count": 0,
                "notes": "useful",
            }
        ]

        summary = _render_summary("demo", rows, ["python", "scripts/run_github_pr_experiment.py"])

        self.assertIn("GitHub PR Experiment: demo", summary)
        self.assertIn("qwen35_rewriter_sweci_contract", summary)
        self.assertIn("Fix widget state", summary)
        self.assertIn("100", summary)


if __name__ == "__main__":
    unittest.main()
