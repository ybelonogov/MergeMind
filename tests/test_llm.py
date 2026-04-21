"""Local LLM component tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.schema import CandidateComment, MRExample
from src.models.llm import (
    GENERATOR_SCHEMA,
    JUDGE_SCHEMA,
    LLMGenerator,
    LLMReranker,
    OpenAICompatibleLLMClient,
    SQLiteLLMCache,
    parse_json_payload,
)
from src.validation.metrics import OpenAICompatibleLLMJudge


def _completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _example() -> MRExample:
    return MRExample(
        source_dataset="CodeReviewer",
        example_id="demo",
        split="test",
        repo="repo/demo",
        title="Handle empty cart",
        description="Avoid failing on empty cart checkout.",
        diff=(
            "diff --git a/cart.py b/cart.py\n"
            "--- a/cart.py\n"
            "+++ b/cart.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+if not items:\n"
            "+    return 0\n"
        ),
        repository_context="File: cart.py\nfunction checkout(items)",
    )


class LocalLLMComponentTests(unittest.TestCase):
    def test_parse_json_payload_tolerates_code_fences(self) -> None:
        payload = parse_json_payload('```json\n{"score": 0.7, "reason": "grounded"}\n```')
        self.assertEqual(payload["score"], 0.7)

    def test_sqlite_cache_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache = SQLiteLLMCache(Path(temp_dir) / "cache.sqlite")
            cache.set("key", {"payload": {"ok": True}})

            self.assertEqual(cache.get("key"), {"payload": {"ok": True}})

    def test_client_cache_avoids_duplicate_completion_calls(self) -> None:
        calls = {"count": 0}

        def completion_fn(**_: object) -> dict:
            calls["count"] += 1
            return _completion('{"comments": []}')

        with TemporaryDirectory() as temp_dir:
            client = OpenAICompatibleLLMClient(
                model="qwen/qwen3.5-9b",
                cache_path=Path(temp_dir) / "cache.sqlite",
                completion_fn=completion_fn,
            )
            messages = [{"role": "user", "content": "review this"}]
            first = client.chat_json("generator", messages, GENERATOR_SCHEMA)
            second = client.chat_json("generator", messages, GENERATOR_SCHEMA)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(calls["count"], 1)

    def test_client_retries_malformed_json(self) -> None:
        calls = {"count": 0}

        def completion_fn(**_: object) -> dict:
            calls["count"] += 1
            return _completion("not json")

        client = OpenAICompatibleLLMClient(completion_fn=completion_fn, retries=1)
        response = client.chat_json("judge", [{"role": "user", "content": "x"}], JUDGE_SCHEMA)

        self.assertTrue(response.parse_error)
        self.assertEqual(calls["count"], 2)

    def test_llm_generator_builds_candidates_from_json(self) -> None:
        payload = {
            "comments": [
                {
                    "text": "Guard empty items before reading the first cart entry.",
                    "confidence": 0.8,
                    "reason": "The diff adds an empty path.",
                }
            ]
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: _completion(json.dumps(payload)))
        generator = LLMGenerator(client, max_candidates=3)

        candidates = generator.generate(_example())

        self.assertEqual(len(candidates), 1)
        self.assertIn("empty items", candidates[0].text)
        self.assertAlmostEqual(candidates[0].generator_score, 0.8)

    def test_llm_reranker_preserves_candidate_indices(self) -> None:
        payload = {
            "ranked_comments": [
                {
                    "candidate_id": 1,
                    "score": 0.91,
                    "reason": "More specific.",
                    "usefulness": 0.9,
                    "groundedness": 0.9,
                    "actionability": 0.9,
                    "specificity": 0.9,
                },
                {
                    "candidate_id": 0,
                    "score": 0.2,
                    "reason": "Too generic.",
                    "usefulness": 0.2,
                    "groundedness": 0.2,
                    "actionability": 0.2,
                    "specificity": 0.2,
                },
            ]
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: _completion(json.dumps(payload)))
        reranker = LLMReranker(client)
        candidates = [
            CandidateComment(text="Generic comment", generator_score=0.7),
            CandidateComment(text="Guard empty items in checkout.", generator_score=0.4),
        ]

        ranked = reranker.rerank(_example(), candidates, top_n=2)

        self.assertEqual(ranked[0].text, "Guard empty items in checkout.")
        self.assertAlmostEqual(ranked[0].reranker_score, 0.91)

    def test_llm_judge_returns_bounded_score(self) -> None:
        client = OpenAICompatibleLLMClient(
            completion_fn=lambda **_: _completion('{"score": 1.2, "reason": "same issue"}')
        )
        judge = OpenAICompatibleLLMJudge(client)

        score = judge.score(
            [CandidateComment(text="Guard empty items before checkout.")],
            ["Check for empty items before reading from the cart."],
            _example(),
        )

        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
