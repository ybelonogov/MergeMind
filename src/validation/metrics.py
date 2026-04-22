"""Offline evaluation metrics for MergeMind."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean
from typing import Any, Callable

from openai import OpenAI

from src.data.schema import CandidateComment, MRExample
from src.inference.pipeline import run_inference
from src.models.llm import JUDGE_SCHEMA, OpenAICompatibleLLMClient


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


class OpenAICompatibleLLMJudge:
    """Local OpenAI-compatible judge for LM Studio and other local servers."""

    def __init__(
        self,
        client: OpenAICompatibleLLMClient,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_count = 0

    def score(self, predictions: list[CandidateComment], gold_comments: list[str], example: MRExample | None = None) -> float:
        if not predictions or not gold_comments:
            return 0.0

        prompt = "\n".join(
            [
                "Grade predicted code review comments against the gold comments.",
                "Return JSON only. Score 1.0 when predictions identify the same concrete issue",
                "and are actionable and grounded in the diff. Score 0.0 when they miss the issue.",
                "",
                f"Title: {example.title if example else ''}",
                f"Diff:\n{(example.diff if example else '')[:3000]}",
                "",
                "Gold comments:",
                *[f"- {comment}" for comment in gold_comments[:3]],
                "",
                "Predicted comments:",
                *[f"- {candidate.text}" for candidate in predictions[:3]],
            ]
        )
        result = self.client.chat_json(
            role="judge",
            messages=[
                {"role": "system", "content": "You are a strict evaluator for automated code review quality."},
                {"role": "user", "content": prompt},
            ],
            response_schema=JUDGE_SCHEMA,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if result.parse_error or "score" not in result.payload:
            self.fallback_count += 1
            return 0.0
        try:
            return max(0.0, min(1.0, float(result.payload.get("score", 0.0))))
        except (TypeError, ValueError):
            self.fallback_count += 1
            return 0.0


def _build_judge(
    use_llm_judge: bool,
    similarity_threshold: float,
    llm_judge_model: str = "",
) -> tuple[Any, str]:
    if use_llm_judge and llm_judge_model and os.getenv("OPENAI_API_KEY"):
        return OpenAILLMJudge(model=llm_judge_model), "openai"
    return LocalHeuristicJudge(threshold=similarity_threshold), "heuristic"


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(0.95 * (len(sorted_values) - 1))
    return sorted_values[index]


def _collect_llm_metrics(*components: Any, example_count: int) -> dict[str, Any]:
    clients: dict[int, OpenAICompatibleLLMClient] = {}
    fallback_count = 0
    for component in components:
        if component is None:
            continue
        client = getattr(component, "client", None)
        if isinstance(client, OpenAICompatibleLLMClient):
            clients[id(client)] = client
        fallback_count += int(getattr(component, "fallback_count", 0))

    merged = {
        "llm_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_hit_rate": 0.0,
        "parse_error_rate": 0.0,
        "llm_total_latency_sec": 0.0,
        "llm_avg_latency_sec": 0.0,
        "llm_p95_latency_sec": 0.0,
        "tokens_per_second": 0.0,
        "fallback_rate": fallback_count / example_count if example_count else 0.0,
    }
    if not clients:
        return merged

    stats = [client.stats() for client in clients.values()]
    total_calls = sum(int(item["llm_call_count"]) for item in stats)
    merged["llm_call_count"] = total_calls
    merged["prompt_tokens"] = sum(int(item["prompt_tokens"]) for item in stats)
    merged["completion_tokens"] = sum(int(item["completion_tokens"]) for item in stats)
    merged["total_tokens"] = sum(int(item["total_tokens"]) for item in stats)
    merged["llm_total_latency_sec"] = sum(float(item["llm_total_latency_sec"]) for item in stats)
    if total_calls:
        merged["cache_hit_rate"] = sum(item["cache_hit_rate"] * item["llm_call_count"] for item in stats) / total_calls
        merged["parse_error_rate"] = sum(item["parse_error_rate"] * item["llm_call_count"] for item in stats) / total_calls
        merged["llm_avg_latency_sec"] = sum(item["llm_avg_latency_sec"] * item["llm_call_count"] for item in stats) / total_calls
        merged["llm_p95_latency_sec"] = max(float(item["llm_p95_latency_sec"]) for item in stats)
    if merged["llm_total_latency_sec"]:
        merged["tokens_per_second"] = merged["total_tokens"] / merged["llm_total_latency_sec"]
    return merged


def evaluate_examples(
    examples: list[MRExample],
    generator: Any,
    reranker: Any,
    top_n: int,
    similarity_threshold: float,
    use_llm_judge: bool = False,
    llm_judge_model: str = "",
    llm_judge_max_examples: int = 25,
    judge_override: Any | None = None,
    judge_backend_override: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    example_summaries: list[dict[str, Any]] = []
    top1_scores: list[float] = []
    best_scores: list[float] = []
    hits_at_k: list[int] = []
    reciprocal_ranks: list[float] = []
    judge_scores: list[float] = []
    inference_latencies: list[float] = []
    candidate_counts: list[int] = []
    if judge_override is not None:
        judge, judge_backend = judge_override, judge_backend_override or "local_llm"
    else:
        judge, judge_backend = _build_judge(use_llm_judge, similarity_threshold, llm_judge_model=llm_judge_model)
    total_examples = len(examples)

    for index, example in enumerate(examples):
        started_at = time.perf_counter()
        predictions = run_inference(example, generator, reranker, top_n=top_n)
        latency = time.perf_counter() - started_at
        inference_latencies.append(latency)
        candidate_counts.append(len(predictions))
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
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": index + 1,
                    "total": total_examples,
                    "example_id": example.example_id,
                    "source_dataset": example.source_dataset,
                    "latency_sec": latency,
                    "prediction_count": len(predictions),
                    "top1_similarity": top1_similarity,
                    "best_similarity_at_k": best_similarity,
                    "hit_at_k": hit_at_k,
                    "judge_score": record.get("judge_score"),
                    "llm_metrics": _collect_llm_metrics(generator, reranker, judge, example_count=index + 1),
                    "timestamp": time.time(),
                }
            )

    summary = {
        "example_count": len(examples),
        "top1_similarity": mean(top1_scores) if top1_scores else 0.0,
        "best_similarity_at_k": mean(best_scores) if best_scores else 0.0,
        "hit_rate_at_k": mean(hits_at_k) if hits_at_k else 0.0,
        "mrr_at_k": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "judge_backend": judge_backend,
        "avg_latency_sec": mean(inference_latencies) if inference_latencies else 0.0,
        "p95_latency_sec": _p95(inference_latencies),
        "avg_candidates_per_mr": mean(candidate_counts) if candidate_counts else 0.0,
        "examples": example_summaries,
    }
    summary.update(_collect_llm_metrics(generator, reranker, judge, example_count=len(examples)))
    if judge_scores:
        summary["judge_score"] = mean(judge_scores)
    return summary
