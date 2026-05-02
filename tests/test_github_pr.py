"""Tests for GitHub Pull Request ingestion."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.data.github import GitHubClient, GitHubClientError, fetch_github_pr_example, parse_github_pr_url


class FakeGitHubClient:
    def get_pull_request(self, ref):
        return {
            "title": "Fix inventory update",
            "body": "",
            "user": {"login": "octocat"},
            "base": {"ref": "main"},
            "head": {"ref": "bugfix", "sha": "abc123"},
            "changed_files": 2,
            "additions": 4,
            "deletions": 1,
        }

    def get_pull_request_diff(self, ref):
        return ""

    def list_pull_request_files(self, ref):
        return [
            {
                "filename": "src/inventory.py",
                "status": "modified",
                "additions": 4,
                "deletions": 1,
                "changes": 5,
                "patch": "@@ -1,3 +1,4 @@\n def reserve(stock):\n-    return stock\n+    if stock <= 0:\n+        raise ValueError('empty')\n+    return stock - 1",
            },
            {
                "filename": "assets/logo.png",
                "status": "modified",
                "additions": 0,
                "deletions": 0,
                "changes": 0,
            },
        ]

    def list_pull_request_review_comments(self, ref):
        return [
            {
                "body": "Please add a regression test for empty stock.",
                "user": {"login": "reviewer", "type": "User"},
            },
            {
                "body": "Automated formatting note.",
                "user": {"login": "ci-bot[bot]", "type": "Bot"},
            },
        ]

    def list_pull_request_reviews(self, ref):
        return [
            {
                "body": "The guard looks useful, but the behavior should be covered by tests.",
                "state": "COMMENTED",
                "user": {"login": "maintainer", "type": "User"},
            }
        ]

    def get_file_text(self, ref, path, git_ref):
        if path.endswith(".png"):
            return ""
        return "def reserve(stock):\n    return stock - 1\n"


class GitHubPullRequestTests(unittest.TestCase):
    def test_parse_github_pr_url(self) -> None:
        ref = parse_github_pr_url("https://github.com/acme/widgets/pull/42?tab=files")

        self.assertEqual(ref.owner, "acme")
        self.assertEqual(ref.repo, "widgets")
        self.assertEqual(ref.number, 42)
        self.assertEqual(ref.example_id, "acme/widgets#42")
        self.assertEqual(ref.safe_id, "acme_widgets_pull_42")

    def test_parse_github_pr_url_rejects_non_pr_url(self) -> None:
        with self.assertRaises(ValueError):
            parse_github_pr_url("https://github.com/acme/widgets/issues/42")

    def test_fetch_github_pr_example_normalizes_to_mr_example(self) -> None:
        example, ref = fetch_github_pr_example(
            "https://github.com/acme/widgets/pull/42",
            client=FakeGitHubClient(),
        )

        self.assertEqual(ref.example_id, "acme/widgets#42")
        self.assertEqual(example.source_dataset, "github_pr")
        self.assertEqual(example.example_id, "acme/widgets#42")
        self.assertEqual(example.split, "live")
        self.assertEqual(example.repo, "acme/widgets")
        self.assertEqual(example.title, "Fix inventory update")
        self.assertEqual(len(example.gold_comments), 2)
        self.assertEqual(example.gold_comments[0].source, "github_review_comment")
        self.assertIn("regression test", example.gold_comments[0].text)
        self.assertIn("src/inventory.py", example.repository_files)
        self.assertEqual(len(example.changed_files), 1)
        self.assertEqual(example.changed_files[0].path, "src/inventory.py")
        self.assertIn("reserve", example.repository_context)
        self.assertEqual(example.metadata["author"], "octocat")
        self.assertEqual(example.metadata["gold_comment_count"], 2)
        self.assertEqual(example.metadata["review_comment_count"], 2)
        self.assertEqual(example.metadata["review_body_count"], 1)
        self.assertEqual(example.metadata["files"][1]["has_patch"], False)

    def test_github_client_reports_access_errors_clearly(self) -> None:
        response = Mock()
        response.status_code = 404
        response.text = "Not Found"

        with patch("src.data.github.requests.get", return_value=response):
            client = GitHubClient(token="", api_url="https://api.github.test")
            with self.assertRaises(GitHubClientError) as error:
                client.get_pull_request(parse_github_pr_url("https://github.com/acme/widgets/pull/42"))

        self.assertIn("GITHUB_TOKEN", str(error.exception))


if __name__ == "__main__":
    unittest.main()
