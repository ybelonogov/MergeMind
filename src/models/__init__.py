"""Model components for MergeMind.

Imports are intentionally lazy so lightweight LLM tooling does not load the
scikit-learn baseline stack just to run a model health check.
"""

from __future__ import annotations

__all__ = [
    "RetrievalGenerator",
    "Reranker",
    "LLMGenerator",
    "LLMReranker",
    "LLMRewriter",
    "OpenAICompatibleLLMClient",
    "CavemanDirectLLMGenerator",
    "CavemanLLMGenerator",
    "CavemanLLMReranker",
    "CavemanLLMRewriter",
    "CavemanTestTriageLLMGenerator",
    "SWEContractLLMGenerator",
    "SWEContractLLMReranker",
    "SWEContractLLMRewriter",
    "SWETriageLLMGenerator",
    "SWETriageLLMReranker",
    "SWETriageLLMRewriter",
]


def __getattr__(name: str) -> object:
    if name in {"RetrievalGenerator", "Reranker"}:
        from .baseline import RetrievalGenerator, Reranker

        return {"RetrievalGenerator": RetrievalGenerator, "Reranker": Reranker}[name]
    if name in {
        "LLMGenerator",
        "LLMReranker",
        "LLMRewriter",
        "OpenAICompatibleLLMClient",
        "CavemanDirectLLMGenerator",
        "CavemanLLMGenerator",
        "CavemanLLMReranker",
        "CavemanLLMRewriter",
        "CavemanTestTriageLLMGenerator",
        "SWEContractLLMGenerator",
        "SWEContractLLMReranker",
        "SWEContractLLMRewriter",
        "SWETriageLLMGenerator",
        "SWETriageLLMReranker",
        "SWETriageLLMRewriter",
    }:
        from .llm import (
            CavemanDirectLLMGenerator,
            CavemanLLMGenerator,
            CavemanLLMReranker,
            CavemanLLMRewriter,
            CavemanTestTriageLLMGenerator,
            LLMGenerator,
            LLMReranker,
            LLMRewriter,
            OpenAICompatibleLLMClient,
            SWEContractLLMGenerator,
            SWEContractLLMReranker,
            SWEContractLLMRewriter,
            SWETriageLLMGenerator,
            SWETriageLLMReranker,
            SWETriageLLMRewriter,
        )

        return {
            "LLMGenerator": LLMGenerator,
            "LLMReranker": LLMReranker,
            "LLMRewriter": LLMRewriter,
            "OpenAICompatibleLLMClient": OpenAICompatibleLLMClient,
            "CavemanDirectLLMGenerator": CavemanDirectLLMGenerator,
            "CavemanLLMGenerator": CavemanLLMGenerator,
            "CavemanLLMReranker": CavemanLLMReranker,
            "CavemanLLMRewriter": CavemanLLMRewriter,
            "CavemanTestTriageLLMGenerator": CavemanTestTriageLLMGenerator,
            "SWEContractLLMGenerator": SWEContractLLMGenerator,
            "SWEContractLLMReranker": SWEContractLLMReranker,
            "SWEContractLLMRewriter": SWEContractLLMRewriter,
            "SWETriageLLMGenerator": SWETriageLLMGenerator,
            "SWETriageLLMReranker": SWETriageLLMReranker,
            "SWETriageLLMRewriter": SWETriageLLMRewriter,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
