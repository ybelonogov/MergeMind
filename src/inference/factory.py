"""Pipeline component factory for baseline and local LLM experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import resolve_path
from src.models.baseline import RetrievalGenerator, Reranker
from src.models.llm import LLMGenerator, LLMReranker, OpenAICompatibleLLMClient, build_llm_client

BASELINE_MODE = "baseline_retrieval_logistic"
QWEN_GENERATOR_LOGISTIC_MODE = "qwen35_generator_logistic_reranker"
RETRIEVAL_QWEN_RERANKER_MODE = "retrieval_generator_qwen35_reranker"
QWEN_FULL_MODE = "qwen35_generator_qwen35_reranker"
QWEN_FULL_ALIAS = "qwen35_full"
QWEN_FULL_JUDGE_MODE = "qwen35_full_with_qwen35_judge"

PIPELINE_ALIASES = {
    QWEN_FULL_ALIAS: QWEN_FULL_MODE,
}

PIPELINE_MODES = {
    BASELINE_MODE,
    QWEN_GENERATOR_LOGISTIC_MODE,
    RETRIEVAL_QWEN_RERANKER_MODE,
    QWEN_FULL_MODE,
    QWEN_FULL_ALIAS,
    QWEN_FULL_JUDGE_MODE,
}


def canonical_pipeline_mode(mode: str) -> str:
    normalized = mode.strip()
    return PIPELINE_ALIASES.get(normalized, normalized)


def pipeline_uses_llm_judge(mode: str) -> bool:
    return mode == QWEN_FULL_JUDGE_MODE


def pipeline_uses_llm(mode: str) -> bool:
    canonical_mode = canonical_pipeline_mode(mode)
    return canonical_mode in {
        QWEN_GENERATOR_LOGISTIC_MODE,
        RETRIEVAL_QWEN_RERANKER_MODE,
        QWEN_FULL_MODE,
        QWEN_FULL_JUDGE_MODE,
    }


def _load_retrieval_generator(config: dict[str, Any], project_root: Path) -> RetrievalGenerator:
    model_dir = resolve_path(project_root, config["paths"]["model_dir"])
    return RetrievalGenerator.load(model_dir / "generator.pkl")


def _load_baseline_reranker(config: dict[str, Any], project_root: Path) -> Reranker:
    model_dir = resolve_path(project_root, config["paths"]["model_dir"])
    return Reranker.load(model_dir / "reranker.pkl")


def _llm_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = dict(config.get("llm", {}))
    return {
        "max_candidates": int(llm_config.get("max_candidates", config.get("model", {}).get("max_candidates", 5))),
        "temperature": float(llm_config.get("temperature_generator", 0.2)),
        "max_tokens": int(llm_config.get("max_tokens_generator", 700)),
    }


def _llm_reranker_config(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = dict(config.get("llm", {}))
    return {
        "temperature": float(llm_config.get("temperature_reranker", 0.0)),
        "max_tokens": int(llm_config.get("max_tokens_reranker", 900)),
    }


def build_pipeline_components(
    mode: str,
    config: dict[str, Any],
    project_root: Path,
    llm_client: OpenAICompatibleLLMClient | None = None,
) -> tuple[Any, Any, OpenAICompatibleLLMClient | None]:
    """Build generator/reranker pair for an evaluation mode."""

    canonical_mode = canonical_pipeline_mode(mode)
    if canonical_mode not in PIPELINE_MODES:
        raise ValueError(f"Unknown pipeline mode: {mode}")

    shared_client = llm_client
    if canonical_mode in {
        QWEN_GENERATOR_LOGISTIC_MODE,
        RETRIEVAL_QWEN_RERANKER_MODE,
        QWEN_FULL_MODE,
        QWEN_FULL_JUDGE_MODE,
    }:
        shared_client = shared_client or build_llm_client(config, project_root)

    if canonical_mode == BASELINE_MODE:
        return _load_retrieval_generator(config, project_root), _load_baseline_reranker(config, project_root), None

    if canonical_mode == QWEN_GENERATOR_LOGISTIC_MODE:
        assert shared_client is not None
        return (
            LLMGenerator(shared_client, **_llm_generation_config(config)),
            _load_baseline_reranker(config, project_root),
            shared_client,
        )

    if canonical_mode == RETRIEVAL_QWEN_RERANKER_MODE:
        assert shared_client is not None
        return (
            _load_retrieval_generator(config, project_root),
            LLMReranker(shared_client, **_llm_reranker_config(config)),
            shared_client,
        )

    if canonical_mode in {QWEN_FULL_MODE, QWEN_FULL_JUDGE_MODE}:
        assert shared_client is not None
        return (
            LLMGenerator(shared_client, **_llm_generation_config(config)),
            LLMReranker(shared_client, **_llm_reranker_config(config)),
            shared_client,
        )

    raise ValueError(f"Unsupported pipeline mode: {mode}")


def resolve_profile_limit(config: dict[str, Any], profile: str = "", explicit_limit: int | None = None) -> int:
    if explicit_limit is not None:
        return explicit_limit
    if not profile:
        return 0
    profiles = dict(config.get("evaluation_profiles", {}))
    if profile not in profiles:
        raise ValueError(f"Unknown evaluation profile: {profile}")
    return int(profiles[profile])
