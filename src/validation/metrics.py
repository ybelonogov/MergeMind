"""Offline evaluation metrics for MergeMind."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean
from typing import Any

from openai import OpenAI

from src.data.schema import CandidateComment, MRExample
from src.inference.pipeline import run_inference
from src.models.baseline import RetrievalGenerator, Reranker


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()


@dataclass
class LocalHeuristicJudge:
    threshold: float = 0.35

    def score(self, predictions: list[CandidateComment], gold_comments: list[str], example: MRExample | None = None) -> float:
        if not predictions or not gold_comments:
            return 0.0
        return max(
            _text_similarity(prediction.text, gold_comment)
            for prediction in predictions
            for gold_comment in gold_comments
        )


class OpenAILLMJudge:
    """Optional G-Eval style judge backed by OpenAI Responses API."""

    def __init__(self, model: str) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the LLM judge.")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model

    def score(self, predictions: list[CandidateComment], gold_comments: list[str], example: MRExample | None = None) -> float:
        if not predictions or not gold_comments:
            return 0.0

        top_predictions = predictions[:3]
        prompt = "\n".join(
            [
                "You are grading code review comments for usefulness and alignment.",
                "Return only JSON with a single numeric field: {\"score\": <float between 0 and 1>}.",
                "Score based on whether the predicted comments identify the same issue as the gold comments,",
                "are actionable, and are grounded in the diff.",
                "",
                f"Title: {example.title if example else ''}",
                f"Diff:\n{(example.diff if example else '')[:3000]}",
                "",
                "Gold comments:",
                *[f"- {comment}" for comment in gold_comments[:3]],
                "",
                "Predicted comments:",
                *[f"- {candidate.text}" for candidate in top_predictions],
            ]
        )

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0,
        )
        output_text = getattr(response, "output_text", "") or ""
        try:
            payload = json.loads(output_text)
            return max(0.0, min(1.0, float(payload["score"])))
        except Exception:
            return 0.0


def _build_judge(
    use_llm_judge: bool,
    similarity_threshold: float,
    llm_judge_model: str = "",
) -> tuple[Any, str]:
    if use_llm_judge and llm_judge_model and os.getenv("OPENAI_API_KEY"):
        return OpenAILLMJudge(model=llm_judge_model), "openai"
    return LocalHeuristicJudge(threshold=similarity_threshold), "heuristic"


def evaluate_examples(
    examples: list[MRExample],
    generator: RetrievalGenerator,
    reranker: Reranker,
    top_n: int,
    similarity_threshold: float,
    use_llm_judge: bool = False,
    llm_judge_model: str = "",
    llm_judge_max_examples: int = 25,
) -> dict[str, Any]:
    example_summaries: list[dict[str, Any]] = []
    top1_scores: list[float] = []
    best_scores: list[float] = []
    hits_at_k: list[int] = []
    reciprocal_ranks: list[float] = []
    judge_scores: list[float] = []
    judge, judge_backend = _build_judge(use_llm_judge, similarity_threshold, llm_judge_model=llm_judge_model)

    for index, example in enumerate(examples):
        predictions = run_inference(example, generator, reranker, top_n=top_n)
        gold_comments = [comment.text for comment in example.gold_comments if comment.text]

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

        if index < llm_judge_max_examples:
            judge_score = judge.score(predictions, gold_comments, example)
            judge_scores.append(judge_score)
            record["judge_score"] = judge_score

        example_summaries.append(record)

    summary = {
        "example_count": len(examples),
        "top1_similarity": mean(top1_scores) if top1_scores else 0.0,
        "best_similarity_at_k": mean(best_scores) if best_scores else 0.0,
        "hit_rate_at_k": mean(hits_at_k) if hits_at_k else 0.0,
        "mrr_at_k": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "judge_backend": judge_backend,
        "examples": example_summaries,
    }
    if judge_scores:
        summary["judge_score"] = mean(judge_scores)
    return summary
