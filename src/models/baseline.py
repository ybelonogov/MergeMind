"""Retrieval generator and learned reranker for MergeMind."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.data.schema import CandidateComment, MRExample

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ACTION_TERMS = {"guard", "check", "handle", "avoid", "prevent", "ensure", "validate", "return", "raise", "fallback"}
HEDGE_TERMS = {"maybe", "might", "probably", "possibly"}
FEATURE_NAMES = [
    "generator_score",
    "identifier_overlap",
    "file_overlap",
    "diff_overlap",
    "action_terms",
    "hedge_terms",
    "token_count",
    "char_count",
    "mentions_exception",
    "mentions_test_signal",
]


def _normalize_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def build_feature_text(example: MRExample) -> str:
    file_parts = []
    for changed_file in example.changed_files:
        file_parts.append(changed_file.path)
        file_parts.extend(changed_file.changed_identifiers[:10])
        file_parts.extend(changed_file.structural_symbols[:10])
    return "\n".join(
        [
            example.title,
            example.description,
            example.diff[:4000],
            example.repository_context[:2500],
            " ".join(file_parts),
        ]
    )


def _example_token_sets(example: MRExample) -> tuple[set[str], set[str], set[str]]:
    file_tokens = {
        Path(changed_file.path).stem.lower()
        for changed_file in example.changed_files
        if changed_file.path
    }
    identifier_tokens = {
        identifier.lower()
        for changed_file in example.changed_files
        for identifier in changed_file.changed_identifiers + changed_file.structural_symbols
    }
    diff_tokens = _normalize_tokens(example.diff)
    return file_tokens, identifier_tokens, diff_tokens


def _candidate_feature_map(example: MRExample, candidate: CandidateComment) -> dict[str, float]:
    file_tokens, identifier_tokens, diff_tokens = _example_token_sets(example)
    tokens = _normalize_tokens(candidate.text)
    lower_text = candidate.text.lower()

    return {
        "generator_score": float(candidate.generator_score),
        "identifier_overlap": float(len(tokens & identifier_tokens)),
        "file_overlap": float(len(tokens & file_tokens)),
        "diff_overlap": float(len(tokens & diff_tokens)),
        "action_terms": float(len(tokens & ACTION_TERMS)),
        "hedge_terms": float(len(tokens & HEDGE_TERMS)),
        "token_count": float(max(len(tokens), 1)),
        "char_count": float(len(candidate.text)),
        "mentions_exception": float(any(word in lower_text for word in ("error", "exception", "fail", "bug"))),
        "mentions_test_signal": float(any(word in lower_text for word in ("test", "ci", "assert", "case"))),
    }


def _feature_vector(feature_map: dict[str, float]) -> list[float]:
    return [feature_map[name] for name in FEATURE_NAMES]


@dataclass
class RetrievalRecord:
    example_id: str
    comments: list[str]
    feature_text: str


class RetrievalGenerator:
    """Retrieval-based generator over normalized CodeReviewer examples."""

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix: Any,
        records: list[RetrievalRecord],
        top_examples: int = 4,
        max_candidates: int = 5,
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.records = records
        self.top_examples = top_examples
        self.max_candidates = max_candidates

    @classmethod
    def fit(cls, train_examples: list[MRExample], config: dict[str, Any] | None = None) -> "RetrievalGenerator":
        config = config or {}
        filtered_examples = [
            example
            for example in train_examples
            if example.source_dataset == "CodeReviewer" and example.gold_comments
        ]
        max_train_examples = int(config.get("max_train_examples", 12000))
        if max_train_examples > 0:
            filtered_examples = filtered_examples[:max_train_examples]

        records = [
            RetrievalRecord(
                example_id=example.example_id,
                comments=[comment.text for comment in example.gold_comments if comment.text],
                feature_text=build_feature_text(example),
            )
            for example in filtered_examples
        ]
        if not records:
            raise ValueError("Training requires at least one CodeReviewer example with a gold comment.")

        min_df = 1 if len(records) < 50 else int(config.get("vectorizer_min_df", 2))
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            max_features=int(config.get("vectorizer_max_features", 50000)),
            min_df=min_df,
        )
        matrix = vectorizer.fit_transform([record.feature_text for record in records])
        return cls(
            vectorizer=vectorizer,
            matrix=matrix,
            records=records,
            top_examples=int(config.get("retrieval_top_examples", 4)),
            max_candidates=int(config.get("max_candidates", 5)),
        )

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalGenerator":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError("Invalid retrieval generator artifact.")
        return loaded

    def sample_negative_comments(self, exclude_example_id: str, limit: int) -> list[CandidateComment]:
        negatives: list[CandidateComment] = []
        seen: set[str] = set()
        for record in self.records:
            if record.example_id == exclude_example_id:
                continue
            for comment in record.comments:
                normalized = comment.strip().lower()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                negatives.append(
                    CandidateComment(
                        text=comment,
                        generator_score=0.0,
                        source_example_id=record.example_id,
                        evidence=[f"sampled_negative_from={record.example_id}"],
                    )
                )
                if len(negatives) >= limit:
                    return negatives
        return negatives

    def generate(
        self,
        example: MRExample,
        top_examples: int | None = None,
        max_candidates: int | None = None,
    ) -> list[CandidateComment]:
        top_examples = self.top_examples if top_examples is None else top_examples
        max_candidates = self.max_candidates if max_candidates is None else max_candidates

        query_vector = self.vectorizer.transform([build_feature_text(example)])
        similarities = cosine_similarity(query_vector, self.matrix)[0]
        sorted_indices = similarities.argsort()[::-1]

        candidates: list[CandidateComment] = []
        seen_texts: set[str] = set()

        for index in sorted_indices[:top_examples]:
            record = self.records[index]
            similarity = float(similarities[index])
            for comment in record.comments:
                normalized = comment.strip().lower()
                if not normalized or normalized in seen_texts:
                    continue
                seen_texts.add(normalized)
                candidates.append(
                    CandidateComment(
                        text=comment,
                        generator_score=similarity,
                        source_example_id=record.example_id,
                        evidence=[f"retrieved_from={record.example_id}", f"similarity={similarity:.3f}"],
                    )
                )
                if len(candidates) >= max_candidates:
                    return candidates

        return candidates


class Reranker:
    """Feature-based reranker with logistic and heuristic scoring modes."""

    def __init__(
        self,
        weights: dict[str, float],
        mode: str = "heuristic",
        classifier: LogisticRegression | None = None,
    ) -> None:
        self.weights = weights
        self.mode = mode
        self.classifier = classifier

    @classmethod
    def fit(
        cls,
        train_examples: list[MRExample],
        generator: RetrievalGenerator,
        config: dict[str, Any] | None = None,
    ) -> "Reranker":
        config = config or {}
        mode = str(config.get("mode", "logistic"))
        weights = {
            key: float(value)
            for key, value in config.items()
            if key
            not in {
                "mode",
                "negative_candidates_per_example",
                "training_examples",
            }
        }

        if mode != "logistic":
            return cls(weights=weights, mode="heuristic")

        examples = [
            example
            for example in train_examples
            if example.source_dataset == "CodeReviewer" and example.gold_comments
        ]
        max_examples = int(config.get("training_examples", 2500))
        if max_examples > 0:
            examples = examples[:max_examples]

        negative_candidates_per_example = int(config.get("negative_candidates_per_example", 3))
        feature_rows: list[list[float]] = []
        labels: list[int] = []

        for example in examples:
            gold_texts = {comment.text.strip().lower() for comment in example.gold_comments if comment.text.strip()}
            for comment in example.gold_comments[:1]:
                positive_candidate = CandidateComment(
                    text=comment.text,
                    generator_score=1.0,
                    source_example_id=example.example_id,
                    evidence=["gold_comment"],
                )
                feature_rows.append(_feature_vector(_candidate_feature_map(example, positive_candidate)))
                labels.append(1)

            negatives = [
                candidate
                for candidate in generator.generate(
                    example,
                    top_examples=generator.top_examples + 2,
                    max_candidates=generator.max_candidates + negative_candidates_per_example + 4,
                )
                if candidate.text.strip().lower() not in gold_texts and candidate.source_example_id != example.example_id
            ]
            if len(negatives) < negative_candidates_per_example:
                negatives.extend(
                    generator.sample_negative_comments(
                        exclude_example_id=example.example_id,
                        limit=negative_candidates_per_example - len(negatives),
                    )
                )

            for candidate in negatives[:negative_candidates_per_example]:
                feature_rows.append(_feature_vector(_candidate_feature_map(example, candidate)))
                labels.append(0)

        if len(set(labels)) < 2 or len(feature_rows) < 10:
            return cls(weights=weights, mode="heuristic")

        classifier = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
        classifier.fit(feature_rows, labels)
        return cls(weights=weights, mode="logistic", classifier=classifier)

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "Reranker":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError("Invalid reranker artifact.")
        return loaded

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Reranker":
        mode = str(config.get("mode", "heuristic"))
        weights = {
            key: float(value)
            for key, value in config.items()
            if key
            not in {
                "mode",
                "negative_candidates_per_example",
                "training_examples",
            }
        }
        return cls(weights=weights, mode=mode)

    def _heuristic_score(self, example: MRExample, candidate: CandidateComment) -> tuple[float, dict[str, float]]:
        feature_map = _candidate_feature_map(example, candidate)
        verbosity_penalty = 0.0
        if feature_map["token_count"] > 35:
            verbosity_penalty += self.weights.get("verbosity_penalty", 0.05)
        if feature_map["token_count"] < 6:
            verbosity_penalty += self.weights.get("short_penalty", 0.04)

        score = (
            self.weights.get("retrieval_weight", 0.6) * feature_map["generator_score"]
            + self.weights.get("identifier_weight", 0.2) * min(feature_map["identifier_overlap"], 3.0)
            + self.weights.get("file_weight", 0.1) * min(feature_map["file_overlap"], 2.0)
            + self.weights.get("action_weight", 0.1) * min(feature_map["action_terms"], 2.0)
            - self.weights.get("hedge_penalty", 0.08) * feature_map["hedge_terms"]
            - verbosity_penalty
        )
        return score, feature_map

    def rerank(self, example: MRExample, candidates: list[CandidateComment], top_n: int = 3) -> list[CandidateComment]:
        scored: list[CandidateComment] = []
        for candidate in candidates:
            heuristic_score, feature_map = self._heuristic_score(example, candidate)

            if self.mode == "logistic" and self.classifier is not None:
                probability = float(self.classifier.predict_proba([_feature_vector(feature_map)])[0][1])
                final_score = 0.7 * probability + 0.3 * heuristic_score
            else:
                probability = 0.0
                final_score = heuristic_score

            evidence = list(candidate.evidence)
            for key in ("identifier_overlap", "file_overlap", "action_terms", "hedge_terms"):
                value = feature_map[key]
                if value:
                    evidence.append(f"{key}={int(value)}")
            if probability:
                evidence.append(f"logistic_prob={probability:.3f}")
            evidence.append(f"mode={self.mode}")

            scored.append(
                CandidateComment(
                    text=candidate.text,
                    generator_score=candidate.generator_score,
                    reranker_score=final_score,
                    source_example_id=candidate.source_example_id,
                    evidence=evidence,
                )
            )

        scored.sort(key=lambda item: item.reranker_score, reverse=True)
        return scored[:top_n]
