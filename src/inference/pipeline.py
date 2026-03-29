"""End-to-end inference pipeline."""

from __future__ import annotations

from src.context.processing import enrich_example
from src.data.schema import CandidateComment, MRExample
from src.models.baseline import RetrievalGenerator, Reranker


def _rewrite_candidates(candidates: list[CandidateComment]) -> list[CandidateComment]:
    return candidates


def run_inference(
    example: MRExample,
    generator: RetrievalGenerator,
    reranker: Reranker,
    top_n: int = 3,
) -> list[CandidateComment]:
    if not example.changed_files or not example.repository_context:
        example = enrich_example(example)
    generated = generator.generate(example)
    reranked = reranker.rerank(example, generated, top_n=top_n)
    return _rewrite_candidates(reranked)
