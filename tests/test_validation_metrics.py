"""Validation metric aggregation tests."""

from __future__ import annotations

import unittest

from src.data.schema import CandidateComment, MRExample, ReviewComment
from src.validation.metrics import evaluate_examples


def _example() -> MRExample:
    return MRExample(
        source_dataset="CodeReviewer",
        example_id="mr-1",
        split="validation",
        repo="demo/repo",
        title="Guard cart",
        description="Handle empty carts.",
        diff="diff --git a/cart.py b/cart.py\n+if not items:\n+    return 0\n",
        repository_context="File: cart.py",
        gold_comments=[ReviewComment(text="Guard empty carts.")],
    )


class FakeGenerator:
    def __init__(self) -> None:
        self.last_raw_generated_count = 0
        self.last_deduped_candidate_count = 0

    def generate(self, example: MRExample) -> list[CandidateComment]:
        self.last_raw_generated_count = 4
        self.last_deduped_candidate_count = 3
        return [
            CandidateComment(text="Guard empty carts before checkout.", generator_score=0.9),
            CandidateComment(text="Add a negative test.", generator_score=0.6),
        ]


class FakeReranker:
    def rerank(
        self,
        example: MRExample,
        candidates: list[CandidateComment],
        top_n: int = 3,
    ) -> list[CandidateComment]:
        return candidates[:1]


class FakeJudge:
    def evaluate(
        self,
        predictions: list[CandidateComment],
        gold_comments: list[str],
        example: MRExample,
    ) -> dict:
        return {
            "gold_alignment_score": 0.4,
            "valid_alternative_score": 0.7,
            "groundedness": 0.8,
            "usefulness": 0.9,
            "reason": "Useful alternative.",
        }


class ValidationMetricsTests(unittest.TestCase):
    def test_evaluate_examples_records_latency_candidate_and_judge_details(self) -> None:
        summary = evaluate_examples(
            examples=[_example()],
            generator=FakeGenerator(),
            reranker=FakeReranker(),
            top_n=1,
            similarity_threshold=0.35,
            judge_override=FakeJudge(),
            judge_backend_override="fake",
        )

        record = summary["examples"][0]

        self.assertEqual(summary["judge_backend"], "fake")
        self.assertEqual(summary["judge_score"], 0.7)
        self.assertEqual(summary["gold_alignment_score"], 0.4)
        self.assertEqual(summary["valid_alternative_score"], 0.7)
        self.assertEqual(summary["avg_raw_generated_count"], 4)
        self.assertEqual(summary["avg_deduped_candidate_count"], 3)
        self.assertIn("avg_inference_latency_sec", summary)
        self.assertIn("avg_judge_latency_sec", summary)
        self.assertIn("avg_total_wall_latency_sec", summary)
        self.assertEqual(record["raw_generated_count"], 4)
        self.assertEqual(record["deduped_candidate_count"], 3)
        self.assertEqual(record["judge"]["valid_alternative_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
