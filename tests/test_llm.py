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
    LLMRewriter,
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

    def test_client_stats_separate_cached_and_uncached_calls(self) -> None:
        calls = {"count": 0}

        def completion_fn(**_: object) -> dict:
            calls["count"] += 1
            return _completion('{"comments": []}', prompt_tokens=20, completion_tokens=10)

        with TemporaryDirectory() as temp_dir:
            client = OpenAICompatibleLLMClient(
                model="qwen/qwen3.5-9b",
                cache_path=Path(temp_dir) / "cache.sqlite",
                completion_fn=completion_fn,
            )
            messages = [{"role": "user", "content": "review this"}]
            client.chat_json("generator", messages, GENERATOR_SCHEMA)
            client.chat_json("generator", messages, GENERATOR_SCHEMA)

        stats = client.stats()

        self.assertEqual(stats["llm_call_count"], 2)
        self.assertEqual(stats["cached_call_count"], 1)
        self.assertEqual(stats["uncached_call_count"], 1)
        self.assertEqual(stats["total_tokens"], 60)
        self.assertEqual(stats["uncached_total_tokens"], 30)
        self.assertGreaterEqual(stats["uncached_tokens_per_second"], 0.0)

    def test_client_can_use_json_object_response_format(self) -> None:
        formats = []

        def completion_fn(**kwargs: object) -> dict:
            formats.append(kwargs["response_format"])
            return _completion('{"comments": []}')

        client = OpenAICompatibleLLMClient(
            response_format_mode="json_object",
            completion_fn=completion_fn,
        )

        response = client.chat_json("generator", [{"role": "user", "content": "json"}], GENERATOR_SCHEMA)

        self.assertFalse(response.parse_error)
        self.assertEqual(formats[0], {"type": "json_object"})

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
        self.assertEqual(generator.last_raw_generated_count, 1)
        self.assertEqual(generator.last_deduped_candidate_count, 1)
        self.assertIn("raw_generated_count=1", candidates[0].evidence)

    def test_llm_generator_prompt_requests_minimum_diverse_candidates(self) -> None:
        seen_prompts: list[str] = []

        def completion_fn(**kwargs: object) -> dict:
            messages = kwargs["messages"]
            assert isinstance(messages, list)
            seen_prompts.append(messages[-1]["content"])
            return _completion('{"comments": []}')

        client = OpenAICompatibleLLMClient(completion_fn=completion_fn)
        generator = LLMGenerator(client, max_candidates=5, min_candidates=3)

        generator.generate(_example())

        self.assertIn("between 3 and 5", seen_prompts[0])
        self.assertIn("correctness", seen_prompts[0])
        self.assertIn("missing tests", seen_prompts[0])

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
        self.assertGreater(ranked[0].reranker_score, ranked[1].reranker_score)

    def test_llm_reranker_calibrates_saturated_and_nice_to_have_scores(self) -> None:
        payload = {
            "ranked_comments": [
                {
                    "candidate_id": 0,
                    "score": 1.0,
                    "reason": "Concrete bug risk.",
                    "usefulness": 1.0,
                    "groundedness": 1.0,
                    "actionability": 1.0,
                    "specificity": 1.0,
                },
                {
                    "candidate_id": 1,
                    "score": 1.0,
                    "reason": "Documentation nice-to-have.",
                    "usefulness": 1.0,
                    "groundedness": 1.0,
                    "actionability": 1.0,
                    "specificity": 1.0,
                },
                {
                    "candidate_id": 2,
                    "score": 1.0,
                    "reason": "Useful but ranked later.",
                    "usefulness": 1.0,
                    "groundedness": 1.0,
                    "actionability": 1.0,
                    "specificity": 1.0,
                },
            ]
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: _completion(json.dumps(payload)))
        reranker = LLMReranker(client)
        candidates = [
            CandidateComment(text="Guard empty items before indexing into the cart.", generator_score=0.5),
            CandidateComment(text="Consider adding documentation for the cart helper.", generator_score=0.5),
            CandidateComment(text="Handle empty items consistently in checkout.", generator_score=0.5),
        ]

        ranked = reranker.rerank(_example(), candidates, top_n=3)
        doc_candidate = next(candidate for candidate in ranked if "documentation" in candidate.text)
        later_candidate = next(candidate for candidate in ranked if "consistently" in candidate.text)

        self.assertAlmostEqual(ranked[0].reranker_score, 1.0)
        self.assertLess(doc_candidate.reranker_score, 0.7)
        self.assertLess(later_candidate.reranker_score, 0.9)
        self.assertTrue(any("calibrated_score=" in item for item in ranked[0].evidence))

    def test_llm_reranker_fills_when_model_returns_too_few_ranked_candidates(self) -> None:
        payload = {
            "ranked_comments": [
                {
                    "candidate_id": 0,
                    "score": 0.9,
                    "reason": "Best issue.",
                    "usefulness": 0.9,
                    "groundedness": 0.9,
                    "actionability": 0.9,
                    "specificity": 0.9,
                }
            ]
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: _completion(json.dumps(payload)))
        reranker = LLMReranker(client)
        candidates = [
            CandidateComment(text="Guard empty items before indexing into the cart.", generator_score=0.9),
            CandidateComment(text="Add a negative test for empty carts.", generator_score=0.8),
            CandidateComment(text="Clarify checkout behavior in docs.", generator_score=0.4),
        ]

        ranked = reranker.rerank(_example(), candidates, top_n=3)

        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0].text, "Guard empty items before indexing into the cart.")
        self.assertTrue(any("llm_reranker_unranked_fill=true" in item for item in ranked[1].evidence))

    def test_llm_judge_returns_bounded_score(self) -> None:
        client = OpenAICompatibleLLMClient(
            completion_fn=lambda **_: _completion(
                json.dumps(
                    {
                        "gold_alignment_score": 1.2,
                        "valid_alternative_score": 0.3,
                        "groundedness": 0.9,
                        "usefulness": 0.8,
                        "reason": "same issue",
                    }
                )
            )
        )
        judge = OpenAICompatibleLLMJudge(client)

        score = judge.score(
            [CandidateComment(text="Guard empty items before checkout.")],
            ["Check for empty items before reading from the cart."],
            _example(),
        )

        self.assertEqual(score, 1.0)

    def test_llm_judge_returns_detailed_scores(self) -> None:
        client = OpenAICompatibleLLMClient(
            completion_fn=lambda **_: _completion(
                json.dumps(
                    {
                        "gold_alignment_score": 0.2,
                        "valid_alternative_score": 0.7,
                        "groundedness": 0.8,
                        "usefulness": 0.9,
                        "reason": "Useful alternative issue.",
                    }
                )
            )
        )
        judge = OpenAICompatibleLLMJudge(client)

        result = judge.evaluate(
            [CandidateComment(text="Add a negative test case.")],
            ["Rename this helper."],
            _example(),
        )

        self.assertEqual(result["judge_score"], 0.7)
        self.assertEqual(result["gold_alignment_score"], 0.2)
        self.assertEqual(result["valid_alternative_score"], 0.7)
        self.assertIn("Useful alternative", result["reason"])

    def test_llm_judge_scores_live_pr_without_gold_comments(self) -> None:
        seen_prompt = []

        def completion_fn(**kwargs: object) -> dict:
            messages = kwargs["messages"]
            assert isinstance(messages, list)
            seen_prompt.append(messages[-1]["content"])
            return _completion(
                json.dumps(
                    {
                        "gold_alignment_score": 0.0,
                        "valid_alternative_score": 0.8,
                        "groundedness": 0.9,
                        "usefulness": 0.7,
                        "reason": "Grounded live PR comment without benchmark gold.",
                    }
                )
            )

        client = OpenAICompatibleLLMClient(completion_fn=completion_fn)
        judge = OpenAICompatibleLLMJudge(client)

        result = judge.evaluate(
            [CandidateComment(text="Add a regression test case.")],
            [],
            _example(),
        )

        self.assertIn("no gold comments", seen_prompt[0])
        self.assertEqual(result["gold_alignment_score"], 0.0)
        self.assertEqual(result["judge_score"], 0.8)
        self.assertEqual(result["groundedness"], 0.9)
        self.assertIn("Grounded live PR", result["reason"])

    def test_llm_rewriter_preserves_original_and_adds_essence(self) -> None:
        payload = {
            "rewritten_comments": [
                {
                    "candidate_id": 0,
                    "rewritten_comment": "Guard empty carts before reading the first item.",
                    "essence": "Empty cart guard",
                    "severity": "medium",
                    "confidence": 0.92,
                    "reason": "Shortened without changing the issue.",
                }
            ]
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: _completion(json.dumps(payload)))
        rewriter = LLMRewriter(client)
        candidates = [
            CandidateComment(
                text="You should add a guard for empty carts before the checkout code reads the first item.",
                generator_score=0.8,
                reranker_score=0.9,
                evidence=["llm_reranker"],
            )
        ]

        rewritten = rewriter.rewrite(_example(), candidates)

        self.assertEqual(rewritten[0].text, "Guard empty carts before reading the first item.")
        self.assertIn("You should add a guard", rewritten[0].original_text)
        self.assertEqual(rewritten[0].essence, "Empty cart guard")
        self.assertEqual(rewritten[0].severity, "medium")
        self.assertAlmostEqual(rewritten[0].rewrite_confidence, 0.92)
        self.assertTrue(any(item == "llm_rewriter" for item in rewritten[0].evidence))

    def test_llm_rewriter_falls_back_on_invalid_payload(self) -> None:
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: _completion('{"not_comments": []}'))
        rewriter = LLMRewriter(client)
        candidates = [CandidateComment(text="Keep the original comment.", reranker_score=0.7)]

        rewritten = rewriter.rewrite(_example(), candidates)

        self.assertEqual(rewritten[0].text, "Keep the original comment.")
        self.assertEqual(rewriter.fallback_count, 1)
        self.assertTrue(any("llm_rewriter_fallback=true" == item for item in rewritten[0].evidence))


if __name__ == "__main__":
    unittest.main()
