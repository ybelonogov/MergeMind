"""Offline evaluation metrics for MergeMind."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean
from typing import Any

from src.data.schema import CandidateComment, MRExample
from src.inference.pipeline import run_inference
from src.models.baseline import RetrievalGenerator, Reranker


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()


@dataclass
class LocalHeuristicJudge:
    threshold: float = 0.35

    def score(self, predictions: list[CandidateComment], gold_comments: list[str]) -> float:
        if not predictions or not gold_comments:
            return 0.0
        return max(
            _text_similarity(prediction.text, gold_comment)
            for prediction in predictions
            for gold_comment in gold_comments
        )


def evaluate_examples(
    examples: list[MRExample],
    generator: RetrievalGenerator,
    reranker: Reranker,
    top_n: int,
    similarity_threshold: float,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    example_summaries: list[dict[str, Any]] = []
    top1_scores: list[float] = []
    best_scores: list[float] = []
    hits_at_k: list[int] = []
    reciprocal_ranks: list[float] = []
    judge_scores: list[float] = []
    judge = LocalHeuristicJudge(threshold=similarity_threshold) if use_llm_judge else None

    for example in examples:
        predictions = run_inference(example, generator, reranker, top_n=top_n)
        gold_comments = [comment.text for comment in example.gold_comments]

        top1_similarity = 0.0
        best_similarity = 0.0
        first_hit_rank = 0

        for rank, prediction in enumerate(predictions, start=1):
            current_similarity = max((_text_similarity(prediction.text, gold) for gold in gold_comments), default=0.0)
            best_similarity = max(best_similarity, current_similarity)
            if rank == 1:
                top1_similarity = current_similarity
            if not first_hit_rank and current_similarity >= similarity_threshold:
                first_hit_rank = rank

        hit_at_k = int(best_similarity >= similarity_threshold)
        reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank else 0.0

        top1_scores.append(top1_similarity)
        best_scores.append(best_similarity)
        hits_at_k.append(hit_at_k)
        reciprocal_ranks.append(reciprocal_rank)

        record = {
            "example_id": example.example_id,
            "source_dataset": example.source_dataset,
            "top1_similarity": top1_similarity,
            "best_similarity_at_k": best_similarity,
            "hit_at_k": hit_at_k,
            "predictions": [candidate.to_dict() for candidate in predictions],
            "gold_comments": gold_comments,
        }

        if judge is not None:
            judge_score = judge.score(predictions, gold_comments)
            judge_scores.append(judge_score)
            record["judge_score"] = judge_score

        example_summaries.append(record)

    summary = {
        "example_count": len(examples),
        "top1_similarity": mean(top1_scores) if top1_scores else 0.0,
        "best_similarity_at_k": mean(best_scores) if best_scores else 0.0,
        "hit_rate_at_k": mean(hits_at_k) if hits_at_k else 0.0,
        "mrr_at_k": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "examples": example_summaries,
    }
    if judge_scores:
        summary["judge_score"] = mean(judge_scores)
    return summary
