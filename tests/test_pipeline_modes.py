"""Pipeline mode factory tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.inference.factory import (
    QWEN_FULL_REWRITER_JUDGE_MODE,
    QWEN_FULL_REWRITER_MODE,
    QWEN_REWRITER_SWECI_CONTRACT_MODE,
    QWEN_REWRITER_SWECI_TRIAGE_MODE,
    build_pipeline_components,
    canonical_pipeline_mode,
    pipeline_uses_llm,
    pipeline_uses_llm_judge,
)
from src.models.llm import OpenAICompatibleLLMClient


class PipelineModeTests(unittest.TestCase):
    def test_rewriter_alias_and_judge_detection(self) -> None:
        self.assertEqual(canonical_pipeline_mode("qwen35_rewriter"), QWEN_FULL_REWRITER_MODE)
        self.assertEqual(canonical_pipeline_mode("qwen35_rewriter_judge"), QWEN_FULL_REWRITER_JUDGE_MODE)
        self.assertEqual(canonical_pipeline_mode("qwen35_review_contract"), QWEN_REWRITER_SWECI_CONTRACT_MODE)
        self.assertEqual(canonical_pipeline_mode("qwen35_review_triage"), QWEN_REWRITER_SWECI_TRIAGE_MODE)
        self.assertTrue(pipeline_uses_llm("qwen35_rewriter"))
        self.assertTrue(pipeline_uses_llm("qwen35_review_contract"))
        self.assertTrue(pipeline_uses_llm("qwen35_review_triage"))
        self.assertFalse(pipeline_uses_llm_judge("qwen35_rewriter"))
        self.assertTrue(pipeline_uses_llm_judge("qwen35_rewriter_judge"))

    def test_rewriter_mode_wraps_reranker_without_loading_baseline(self) -> None:
        config = {
            "llm": {
                "max_candidates": 2,
                "temperature_generator": 0.0,
                "temperature_reranker": 0.0,
                "temperature_rewriter": 0.0,
                "max_tokens_generator": 100,
                "max_tokens_reranker": 100,
                "max_tokens_rewriter": 100,
            },
            "model": {"max_candidates": 2},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, shared_client = build_pipeline_components(
            QWEN_FULL_REWRITER_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertIs(shared_client, client)
        self.assertEqual(generator.client, client)
        self.assertEqual(reranker.client, client)
        self.assertTrue(hasattr(reranker, "rewriter"))

    def test_contract_mode_uses_contract_agents_and_token_limits(self) -> None:
        config = {
            "llm": {
                "max_candidates": 2,
                "min_candidates": 1,
                "max_tokens_contract_generator": 1200,
                "max_tokens_contract_reranker": 900,
                "max_tokens_contract_rewriter": 1200,
            },
            "model": {"max_candidates": 2},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, shared_client = build_pipeline_components(
            QWEN_REWRITER_SWECI_CONTRACT_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertIs(shared_client, client)
        self.assertEqual(generator.__class__.__name__, "SWEContractLLMGenerator")
        self.assertEqual(reranker.__class__.__name__, "RewritingReranker")
        self.assertEqual(reranker.reranker.__class__.__name__, "SWEContractLLMReranker")
        self.assertEqual(reranker.rewriter.__class__.__name__, "SWEContractLLMRewriter")
        self.assertEqual(generator.max_tokens, 1200)
        self.assertEqual(reranker.reranker.max_tokens, 900)
        self.assertEqual(reranker.rewriter.max_tokens, 1200)

    def test_triage_mode_uses_triage_agents_and_caps_candidates(self) -> None:
        config = {
            "llm": {
                "max_candidates": 5,
                "min_candidates": 3,
                "max_tokens_contract_generator": 1200,
                "max_tokens_contract_reranker": 900,
                "max_tokens_contract_rewriter": 1200,
            },
            "model": {"max_candidates": 5},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, shared_client = build_pipeline_components(
            QWEN_REWRITER_SWECI_TRIAGE_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertIs(shared_client, client)
        self.assertEqual(generator.__class__.__name__, "SWETriageLLMGenerator")
        self.assertEqual(reranker.reranker.__class__.__name__, "SWETriageLLMReranker")
        self.assertEqual(reranker.rewriter.__class__.__name__, "SWETriageLLMRewriter")
        self.assertEqual(generator.max_candidates, 3)
        self.assertEqual(generator.min_candidates, 1)


if __name__ == "__main__":
    unittest.main()
