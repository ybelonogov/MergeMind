"""Shared data types for MergeMind."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReviewComment:
    text: str
    is_useful: bool = True
    outcome: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReviewComment":
        return cls(
            text=payload.get("text", ""),
            is_useful=bool(payload.get("is_useful", True)),
            outcome=payload.get("outcome", ""),
            source=payload.get("source", ""),
        )


@dataclass
class DiffHunk:
    header: str
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    context_lines: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiffHunk":
        return cls(
            header=payload.get("header", ""),
            added_lines=list(payload.get("added_lines", [])),
            removed_lines=list(payload.get("removed_lines", [])),
            context_lines=list(payload.get("context_lines", [])),
        )


@dataclass
class ChangedFile:
    path: str
    language: str = "text"
    hunks: list[DiffHunk] = field(default_factory=list)
    changed_identifiers: list[str] = field(default_factory=list)
    structural_symbols: list[str] = field(default_factory=list)
    repository_snippet: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChangedFile":
        return cls(
            path=payload.get("path", ""),
            language=payload.get("language", "text"),
            hunks=[DiffHunk.from_dict(item) for item in payload.get("hunks", [])],
            changed_identifiers=list(payload.get("changed_identifiers", [])),
            structural_symbols=list(payload.get("structural_symbols", [])),
            repository_snippet=payload.get("repository_snippet", ""),
        )


@dataclass
class CandidateComment:
    text: str
    generator_score: float = 0.0
    reranker_score: float = 0.0
    source_example_id: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateComment":
        return cls(
            text=payload.get("text", ""),
            generator_score=float(payload.get("generator_score", 0.0)),
            reranker_score=float(payload.get("reranker_score", 0.0)),
            source_example_id=payload.get("source_example_id", ""),
            evidence=list(payload.get("evidence", [])),
        )


@dataclass
class MRExample:
    source_dataset: str
    example_id: str
    split: str
    repo: str
    title: str
    description: str
    diff: str
    changed_files: list[ChangedFile] = field(default_factory=list)
    repository_files: dict[str, str] = field(default_factory=dict)
    repository_context: str = ""
    gold_comments: list[ReviewComment] = field(default_factory=list)
    follow_up: str = ""
    ci_signals: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MRExample":
        return cls(
            source_dataset=payload.get("source_dataset", ""),
            example_id=payload.get("example_id", ""),
            split=payload.get("split", "train"),
            repo=payload.get("repo", ""),
            title=payload.get("title", ""),
            description=payload.get("description", ""),
            diff=payload.get("diff", ""),
            changed_files=[ChangedFile.from_dict(item) for item in payload.get("changed_files", [])],
            repository_files=dict(payload.get("repository_files", {})),
            repository_context=payload.get("repository_context", ""),
            gold_comments=[ReviewComment.from_dict(item) for item in payload.get("gold_comments", [])],
            follow_up=payload.get("follow_up", ""),
            ci_signals=dict(payload.get("ci_signals", {})),
            metadata=dict(payload.get("metadata", {})),
        )
