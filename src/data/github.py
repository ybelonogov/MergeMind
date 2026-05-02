"""GitHub Pull Request ingestion for live MergeMind demos."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from src.context.processing import enrich_example
from src.data.schema import MRExample, ReviewComment

GITHUB_API_URL = "https://api.github.com"
PR_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)(?:[/?#].*)?$"
)


class GitHubClientError(RuntimeError):
    """Raised when GitHub PR ingestion cannot continue."""


@dataclass(frozen=True)
class GitHubPullRequestRef:
    owner: str
    repo: str
    number: int

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def example_id(self) -> str:
        return f"{self.full_name}#{self.number}"

    @property
    def safe_id(self) -> str:
        return f"{self.owner}_{self.repo}_pull_{self.number}"


def parse_github_pr_url(url: str) -> GitHubPullRequestRef:
    match = PR_URL_RE.match(url.strip())
    if not match:
        raise ValueError("Expected GitHub PR URL like https://github.com/OWNER/REPO/pull/123")
    return GitHubPullRequestRef(
        owner=match.group("owner"),
        repo=match.group("repo"),
        number=int(match.group("number")),
    )


class GitHubClient:
    """Small read-only GitHub API client for PR demo inference."""

    def __init__(
        self,
        token: str | None = None,
        api_url: str = GITHUB_API_URL,
        timeout_seconds: int = 30,
    ) -> None:
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
        self.api_url = api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "MergeMind",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, *, accept: str = "application/vnd.github+json", params: dict[str, Any] | None = None) -> requests.Response:
        url = f"{self.api_url}{path}"
        response = requests.get(url, headers=self._headers(accept), params=params, timeout=self.timeout_seconds)
        if response.status_code in {401, 403, 404}:
            hint = "Check GITHUB_TOKEN permissions for private repositories." if not self.token else response.text[:300]
            raise GitHubClientError(f"GitHub request failed ({response.status_code}) for {path}. {hint}")
        if response.status_code >= 400:
            raise GitHubClientError(f"GitHub request failed ({response.status_code}) for {path}: {response.text[:300]}")
        return response

    def get_pull_request(self, ref: GitHubPullRequestRef) -> dict[str, Any]:
        return self._get(f"/repos/{ref.full_name}/pulls/{ref.number}").json()

    def get_pull_request_diff(self, ref: GitHubPullRequestRef) -> str:
        return self._get(f"/repos/{ref.full_name}/pulls/{ref.number}", accept="application/vnd.github.v3.diff").text

    def list_pull_request_files(self, ref: GitHubPullRequestRef) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        page = 1
        while True:
            response = self._get(
                f"/repos/{ref.full_name}/pulls/{ref.number}/files",
                params={"per_page": 100, "page": page},
            )
            batch = response.json()
            if not batch:
                break
            files.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return files

    def list_pull_request_review_comments(self, ref: GitHubPullRequestRef) -> list[dict[str, Any]]:
        comments: list[dict[str, Any]] = []
        page = 1
        while True:
            response = self._get(
                f"/repos/{ref.full_name}/pulls/{ref.number}/comments",
                params={"per_page": 100, "page": page},
            )
            batch = response.json()
            if not batch:
                break
            comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return comments

    def list_pull_request_reviews(self, ref: GitHubPullRequestRef) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        page = 1
        while True:
            response = self._get(
                f"/repos/{ref.full_name}/pulls/{ref.number}/reviews",
                params={"per_page": 100, "page": page},
            )
            batch = response.json()
            if not batch:
                break
            reviews.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return reviews

    def get_file_text(self, ref: GitHubPullRequestRef, path: str, git_ref: str) -> str:
        response = self._get(
            f"/repos/{ref.full_name}/contents/{path}",
            params={"ref": git_ref},
        )
        payload = response.json()
        if isinstance(payload, list) or payload.get("type") != "file":
            return ""
        if payload.get("encoding") != "base64":
            return ""
        content = str(payload.get("content", ""))
        try:
            raw = base64.b64decode(content, validate=False)
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""


def _build_files_diff(files: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for item in files:
        filename = item.get("filename", "unknown")
        patch = item.get("patch", "")
        if not patch:
            continue
        sections.append(f"diff --git a/{filename} b/{filename}\n--- a/{filename}\n+++ b/{filename}\n{patch}")
    return "\n".join(sections)


def _is_human_comment(payload: dict[str, Any]) -> bool:
    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    login = str(user.get("login", "")).lower()
    user_type = str(user.get("type", "")).lower()
    if user_type == "bot" or login.endswith("[bot]"):
        return False
    if payload.get("minimized") is True:
        return False
    return True


def _review_comments_from_payloads(
    review_comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[ReviewComment]:
    comments: list[ReviewComment] = []
    seen: set[str] = set()

    def append_comment(text: str, source: str) -> None:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        comments.append(ReviewComment(text=normalized, is_useful=True, source=source))

    for payload in review_comments:
        if not _is_human_comment(payload):
            continue
        append_comment(str(payload.get("body") or ""), "github_review_comment")

    for payload in reviews:
        if not _is_human_comment(payload):
            continue
        state = str(payload.get("state", "")).lower()
        if state in {"commented", "changes_requested"}:
            append_comment(str(payload.get("body") or ""), "github_review_body")

    return comments


def fetch_github_pr_example(
    url: str,
    client: GitHubClient | None = None,
    max_repository_files: int = 20,
) -> tuple[MRExample, GitHubPullRequestRef]:
    """Fetch a GitHub Pull Request and normalize it into MRExample."""

    ref = parse_github_pr_url(url)
    github = client or GitHubClient()
    pr_payload = github.get_pull_request(ref)
    files_payload = github.list_pull_request_files(ref)
    review_comments_payload = github.list_pull_request_review_comments(ref)
    reviews_payload = github.list_pull_request_reviews(ref)
    diff = github.get_pull_request_diff(ref).strip() or _build_files_diff(files_payload)

    head_sha = str(pr_payload.get("head", {}).get("sha", ""))
    repository_files: dict[str, str] = {}
    for item in files_payload[:max_repository_files]:
        filename = str(item.get("filename", ""))
        status = str(item.get("status", ""))
        if not filename or status == "removed":
            continue
        try:
            repository_files[filename] = github.get_file_text(ref, filename, head_sha)
        except GitHubClientError:
            repository_files[filename] = ""

    example = MRExample(
        source_dataset="github_pr",
        example_id=ref.example_id,
        split="live",
        repo=ref.full_name,
        title=str(pr_payload.get("title", "")),
        description=str(pr_payload.get("body") or ""),
        diff=diff,
        repository_files=repository_files,
        gold_comments=_review_comments_from_payloads(review_comments_payload, reviews_payload),
        metadata={
            "url": url,
            "number": ref.number,
            "owner": ref.owner,
            "repo": ref.repo,
            "author": pr_payload.get("user", {}).get("login", ""),
            "base_ref": pr_payload.get("base", {}).get("ref", ""),
            "head_ref": pr_payload.get("head", {}).get("ref", ""),
            "head_sha": head_sha,
            "changed_files": pr_payload.get("changed_files", len(files_payload)),
            "additions": pr_payload.get("additions", 0),
            "deletions": pr_payload.get("deletions", 0),
            "review_comment_count": len(review_comments_payload),
            "review_body_count": len(reviews_payload),
            "gold_comment_count": len(_review_comments_from_payloads(review_comments_payload, reviews_payload)),
            "files": [
                {
                    "filename": item.get("filename", ""),
                    "status": item.get("status", ""),
                    "additions": item.get("additions", 0),
                    "deletions": item.get("deletions", 0),
                    "changes": item.get("changes", 0),
                    "has_patch": bool(item.get("patch", "")),
                }
                for item in files_payload
            ],
        },
    )
    return enrich_example(example), ref
