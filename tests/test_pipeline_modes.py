"""Pipeline mode factory tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.inference.factory import (
    QWEN_CAVEMAN_DIRECT_TOP1_MODE,
    QWEN_CAVEMAN_TEST_TRIAGE_MODE,
    QWEN_CAVEMAN_TOP1_MODE,
    QWEN_FULL_REWRITER_JUDGE_MODE,
    QWEN_FULL_REWRITER_MODE,
    QWEN_REWRITER_SWECI_CONTRACT_MODE,
    QWEN_REWRITER_SWECI_SAFE_TRIAGE_MODE,
    QWEN_REWRITER_SWECI_TEST_GUARD_MODE,
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
        self.assertEqual(canonical_pipeline_mode("qwen35_review_safe_triage"), QWEN_REWRITER_SWECI_SAFE_TRIAGE_MODE)
        self.assertEqual(canonical_pipeline_mode("qwen35_review_test_guard"), QWEN_REWRITER_SWECI_TEST_GUARD_MODE)
        self.assertTrue(pipeline_uses_llm("qwen35_rewriter"))
        self.assertTrue(pipeline_uses_llm("qwen35_review_contract"))
        self.assertTrue(pipeline_uses_llm("qwen35_review_triage"))
        self.assertTrue(pipeline_uses_llm("qwen35_review_safe_triage"))
        self.assertTrue(pipeline_uses_llm("qwen35_review_test_guard"))
        self.assertTrue(pipeline_uses_llm(QWEN_CAVEMAN_TOP1_MODE))
        self.assertTrue(pipeline_uses_llm(QWEN_CAVEMAN_DIRECT_TOP1_MODE))
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

    def test_caveman_pipeline_overrides_apply_only_to_selected_mode(self) -> None:
        config = {
            "llm": {
                "max_candidates": 8,
                "min_candidates": 3,
                "max_tokens_contract_generator": 1200,
                "max_tokens_contract_reranker": 900,
                "max_tokens_contract_rewriter": 1200,
            },
            "llm_pipeline_overrides": {
                QWEN_CAVEMAN_TOP1_MODE: {
                    "max_candidates": 3,
                    "min_candidates": 1,
                    "max_tokens_contract_generator": 900,
                    "max_tokens_contract_reranker": 600,
                    "max_tokens_contract_rewriter": 800,
                }
            },
            "model": {"max_candidates": 8},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, _ = build_pipeline_components(
            QWEN_CAVEMAN_TOP1_MODE,
            config,
            Path("."),
            llm_client=client,
        )
        control_generator, control_reranker, _ = build_pipeline_components(
            QWEN_REWRITER_SWECI_TRIAGE_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertEqual(generator.__class__.__name__, "CavemanLLMGenerator")
        self.assertEqual(reranker.reranker.__class__.__name__, "CavemanLLMReranker")
        self.assertEqual(reranker.rewriter.__class__.__name__, "CavemanLLMRewriter")
        self.assertEqual(generator.max_candidates, 3)
        self.assertEqual(generator.min_candidates, 1)
        self.assertEqual(generator.max_tokens, 900)
        self.assertEqual(reranker.reranker.max_tokens, 600)
        self.assertEqual(reranker.rewriter.max_tokens, 800)
        self.assertEqual(control_generator.max_tokens, 1200)
        self.assertEqual(control_reranker.reranker.max_tokens, 900)

    def test_safe_triage_mode_uses_safe_agents_and_token_overrides(self) -> None:
        config = {
            "llm": {
                "max_candidates": 8,
                "min_candidates": 3,
                "max_tokens_contract_generator": 1200,
                "max_tokens_contract_reranker": 900,
                "max_tokens_contract_rewriter": 1200,
            },
            "llm_pipeline_overrides": {
                QWEN_REWRITER_SWECI_SAFE_TRIAGE_MODE: {
                    "max_candidates": 2,
                    "min_candidates": 1,
                    "max_tokens_contract_generator": 800,
                    "max_tokens_contract_reranker": 500,
                    "max_tokens_contract_rewriter": 700,
                }
            },
            "model": {"max_candidates": 8},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, _ = build_pipeline_components(
            QWEN_REWRITER_SWECI_SAFE_TRIAGE_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertEqual(generator.__class__.__name__, "SWESafeTriageLLMGenerator")
        self.assertEqual(reranker.reranker.__class__.__name__, "SWESafeTriageLLMReranker")
        self.assertEqual(reranker.rewriter.__class__.__name__, "SWESafeTriageLLMRewriter")
        self.assertEqual(generator.max_candidates, 2)
        self.assertEqual(generator.min_candidates, 1)
        self.assertEqual(generator.max_tokens, 800)
        self.assertEqual(reranker.reranker.max_tokens, 500)
        self.assertEqual(reranker.rewriter.max_tokens, 700)

    def test_test_guard_mode_uses_test_guard_agents_and_token_overrides(self) -> None:
        config = {
            "llm": {
                "max_candidates": 8,
                "min_candidates": 3,
                "max_tokens_contract_generator": 1200,
                "max_tokens_contract_reranker": 900,
                "max_tokens_contract_rewriter": 1200,
            },
            "llm_pipeline_overrides": {
                QWEN_REWRITER_SWECI_TEST_GUARD_MODE: {
                    "max_candidates": 2,
                    "min_candidates": 1,
                    "max_tokens_contract_generator": 750,
                    "max_tokens_contract_reranker": 500,
                    "max_tokens_contract_rewriter": 650,
                }
            },
            "model": {"max_candidates": 8},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, _ = build_pipeline_components(
            QWEN_REWRITER_SWECI_TEST_GUARD_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertEqual(generator.__class__.__name__, "SWETestGuardLLMGenerator")
        self.assertEqual(reranker.reranker.__class__.__name__, "SWETestGuardLLMReranker")
        self.assertEqual(reranker.rewriter.__class__.__name__, "SWETestGuardLLMRewriter")
        self.assertEqual(generator.max_candidates, 2)
        self.assertEqual(generator.min_candidates, 1)
        self.assertEqual(generator.max_tokens, 750)
        self.assertEqual(reranker.reranker.max_tokens, 500)
        self.assertEqual(reranker.rewriter.max_tokens, 650)

    def test_caveman_direct_top1_skips_rewriter(self) -> None:
        config = {
            "llm": {
                "max_candidates": 3,
                "min_candidates": 1,
                "max_tokens_contract_generator": 1100,
                "max_tokens_contract_reranker": 600,
            },
            "model": {"max_candidates": 3},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, _ = build_pipeline_components(
            QWEN_CAVEMAN_DIRECT_TOP1_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertEqual(generator.__class__.__name__, "CavemanDirectLLMGenerator")
        self.assertEqual(reranker.__class__.__name__, "CavemanLLMReranker")
        self.assertFalse(hasattr(reranker, "rewriter"))

    def test_caveman_test_triage_uses_test_triage_generator(self) -> None:
        config = {
            "llm": {
                "max_candidates": 3,
                "min_candidates": 1,
                "max_tokens_contract_generator": 1000,
                "max_tokens_contract_reranker": 700,
                "max_tokens_contract_rewriter": 800,
            },
            "model": {"max_candidates": 3},
        }
        client = OpenAICompatibleLLMClient(completion_fn=lambda **_: {"choices": [{"message": {"content": "{}"}}]})

        generator, reranker, _ = build_pipeline_components(
            QWEN_CAVEMAN_TEST_TRIAGE_MODE,
            config,
            Path("."),
            llm_client=client,
        )

        self.assertEqual(generator.__class__.__name__, "CavemanTestTriageLLMGenerator")
        self.assertEqual(reranker.reranker.__class__.__name__, "CavemanLLMReranker")


if __name__ == "__main__":
    unittest.main()
