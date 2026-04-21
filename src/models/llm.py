"""Local OpenAI-compatible LLM components for MergeMind."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from openai import OpenAI

from src.data.schema import CandidateComment, MRExample


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

GENERATOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "review_comment_candidates",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["text", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["comments"],
            "additionalProperties": False,
        },
    },
}

RERANKER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "reranked_review_comments",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ranked_comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "integer"},
                            "score": {"type": "number"},
                            "reason": {"type": "string"},
                            "usefulness": {"type": "number"},
                            "groundedness": {"type": "number"},
                            "actionability": {"type": "number"},
                            "specificity": {"type": "number"},
                        },
                        "required": [
                            "candidate_id",
                            "score",
                            "reason",
                            "usefulness",
                            "groundedness",
                            "actionability",
                            "specificity",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ranked_comments"],
            "additionalProperties": False,
        },
    },
}

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "review_quality_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _strip_code_fence(text: str) -> str:
    match = JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse an LLM JSON response, tolerating code fences and extra prose."""

    cleaned = _strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def _stable_cache_key(
    model: str,
    role: str,
    messages: list[dict[str, str]],
    response_schema: dict[str, Any],
    params: dict[str, Any],
) -> str:
    payload = {
        "model": model,
        "role": role,
        "messages": messages,
        "response_schema": response_schema,
        "params": params,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SQLiteLLMCache:
    """Tiny SQLite cache for deterministic local LLM experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_table(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    value_json TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, cache_key: str, value: dict[str, Any]) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_cache(cache_key, created_at, value_json)
                VALUES (?, ?, ?)
                """,
                (cache_key, time.time(), json.dumps(value, ensure_ascii=True)),
            )
            connection.commit()
        finally:
            connection.close()


@dataclass
class LLMJSONResponse:
    payload: dict[str, Any]
    raw_text: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0
    cache_hit: bool = False
    parse_error: bool = False
    error: str = ""


class OpenAICompatibleLLMClient:
    """OpenAI-compatible client for LM Studio and similar local servers."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "qwen/qwen3.5-9b",
        cache_path: str | Path | None = None,
        timeout_seconds: int = 180,
        retries: int = 2,
        completion_fn: Callable[..., Any] | None = None,
        list_models_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.cache = SQLiteLLMCache(cache_path) if cache_path else None
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_seconds)
        self._completion_fn = completion_fn
        self._list_models_fn = list_models_fn
        self.calls: list[LLMJSONResponse] = []

    def list_models(self) -> list[str]:
        if self._list_models_fn is not None:
            return self._list_models_fn()

        response = self._client.models.list()
        model_ids = []
        for item in getattr(response, "data", []):
            model_id = getattr(item, "id", "")
            if model_id:
                model_ids.append(model_id)
        return model_ids

    def health_check(self) -> dict[str, Any]:
        models = self.list_models()
        return {
            "base_url": self.base_url,
            "configured_model": self.model,
            "available_models": models,
            "model_available": self.model in models,
        }

    def _send_chat_completion(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        if self._completion_fn is not None:
            return self._completion_fn(
                model=self.model,
                messages=messages,
                response_format=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            timeout=self.timeout_seconds,
        )

    def chat_json(
        self,
        role: str,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMJSONResponse:
        params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_seconds": self.timeout_seconds,
        }
        cache_key = _stable_cache_key(self.model, role, messages, response_schema, params)
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            result = LLMJSONResponse(
                payload=dict(cached.get("payload", {})),
                raw_text=str(cached.get("raw_text", "")),
                usage=dict(cached.get("usage", {})),
                latency_seconds=0.0,
                cache_hit=True,
            )
            self.calls.append(result)
            return result

        last_error = ""
        for _attempt in range(self.retries + 1):
            started_at = time.perf_counter()
            try:
                response = self._send_chat_completion(
                    messages=messages,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = time.perf_counter() - started_at
                raw_text = self._extract_text(response)
                usage = self._extract_usage(response)
                payload = parse_json_payload(raw_text)
                result = LLMJSONResponse(
                    payload=payload,
                    raw_text=raw_text,
                    usage=usage,
                    latency_seconds=latency,
                )
                if self.cache:
                    self.cache.set(
                        cache_key,
                        {
                            "payload": payload,
                            "raw_text": raw_text,
                            "usage": usage,
                        },
                    )
                self.calls.append(result)
                return result
            except Exception as exc:  # noqa: BLE001 - retries should catch client and parse failures.
                last_error = str(exc)

        result = LLMJSONResponse(
            payload={},
            raw_text="",
            latency_seconds=0.0,
            parse_error=True,
            error=last_error,
        )
        self.calls.append(result)
        return result

    def _extract_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", "")
            if content:
                return content
            reasoning_content = getattr(message, "reasoning_content", "")
            return reasoning_content or ""
        if isinstance(response, dict):
            message = response.get("choices", [{}])[0].get("message", {})
            return message.get("content") or message.get("reasoning_content") or ""
        return ""

    def _extract_usage(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage", {})
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) if usage is not None else 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) if usage is not None else 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) if usage is not None else 0),
        }

    def stats(self) -> dict[str, float]:
        if not self.calls:
            return {
                "llm_call_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_hit_rate": 0.0,
                "parse_error_rate": 0.0,
                "llm_avg_latency_sec": 0.0,
                "llm_p95_latency_sec": 0.0,
            }

        latencies = [call.latency_seconds for call in self.calls if not call.cache_hit]
        sorted_latencies = sorted(latencies)
        p95_index = int(0.95 * (len(sorted_latencies) - 1)) if sorted_latencies else 0
        return {
            "llm_call_count": len(self.calls),
            "prompt_tokens": sum(call.usage.get("prompt_tokens", 0) for call in self.calls),
            "completion_tokens": sum(call.usage.get("completion_tokens", 0) for call in self.calls),
            "total_tokens": sum(call.usage.get("total_tokens", 0) for call in self.calls),
            "cache_hit_rate": mean([float(call.cache_hit) for call in self.calls]),
            "parse_error_rate": mean([float(call.parse_error) for call in self.calls]),
            "llm_avg_latency_sec": mean(latencies) if latencies else 0.0,
            "llm_p95_latency_sec": sorted_latencies[p95_index] if sorted_latencies else 0.0,
        }


def build_llm_client(config: dict[str, Any], project_root: Path) -> OpenAICompatibleLLMClient:
    llm_config = dict(config.get("llm", {}))
    raw_cache_path = Path(str(llm_config.get("cache_path", "artifacts/llm_cache.sqlite")))
    cache_path = raw_cache_path if raw_cache_path.is_absolute() else project_root / raw_cache_path
    return OpenAICompatibleLLMClient(
        base_url=str(llm_config.get("base_url", "http://localhost:1234/v1")),
        api_key=str(llm_config.get("api_key", "lm-studio")),
        model=str(llm_config.get("model", "qwen/qwen3.5-9b")),
        cache_path=cache_path,
        timeout_seconds=int(llm_config.get("timeout_seconds", 180)),
        retries=int(llm_config.get("retries", 2)),
    )


def _format_changed_files(example: MRExample) -> str:
    parts = []
    for changed_file in example.changed_files[:8]:
        identifiers = ", ".join(changed_file.changed_identifiers[:12])
        symbols = ", ".join(changed_file.structural_symbols[:12])
        parts.append(
            "\n".join(
                [
                    f"File: {changed_file.path}",
                    f"Language: {changed_file.language}",
                    f"Identifiers: {identifiers}",
                    f"Symbols: {symbols}",
                ]
            )
        )
    return "\n\n".join(parts)


def _example_prompt(example: MRExample) -> str:
    return "\n".join(
        [
            f"MR title: {example.title}",
            f"MR description: {example.description}",
            "",
            "Changed files:",
            _format_changed_files(example),
            "",
            "Repository context:",
            example.repository_context[:3500],
            "",
            "Unified diff:",
            example.diff[:6000],
        ]
    )


class LLMGenerator:
    """Generate review comment candidates with a local LLM."""

    def __init__(
        self,
        client: OpenAICompatibleLLMClient,
        max_candidates: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> None:
        self.client = client
        self.max_candidates = max_candidates
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_count = 0

    def generate(
        self,
        example: MRExample,
        top_examples: int | None = None,
        max_candidates: int | None = None,
    ) -> list[CandidateComment]:
        del top_examples
        limit = self.max_candidates if max_candidates is None else max_candidates
        messages = [
            {
                "role": "system",
                "content": (
                    "You are MergeMind, a code review assistant. Generate only useful, specific, "
                    "actionable review comments grounded in the provided diff. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate up to {limit} concise review comments. Avoid duplicates, style nits, "
                    "and generic advice. Each comment should point to a concrete risk or improvement.\n\n"
                    f"{_example_prompt(example)}"
                ),
            },
        ]
        result = self.client.chat_json(
            role="generator",
            messages=messages,
            response_schema=GENERATOR_SCHEMA,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        comments = result.payload.get("comments", [])
        if not isinstance(comments, list):
            self.fallback_count += 1
            return []

        candidates: list[CandidateComment] = []
        seen: set[str] = set()
        for item in comments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            normalized = text.lower()
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            reason = str(item.get("reason", "")).strip()
            evidence = ["llm_generator", f"model={self.client.model}"]
            if reason:
                evidence.append(f"reason={reason}")
            if result.cache_hit:
                evidence.append("cache_hit=true")
            candidates.append(
                CandidateComment(
                    text=text,
                    generator_score=_bounded(item.get("confidence"), default=0.5),
                    source_example_id=example.example_id,
                    evidence=evidence,
                )
            )
            if len(candidates) >= limit:
                break

        if not candidates:
            self.fallback_count += 1
        return candidates


class LLMReranker:
    """Rank review candidates with a local LLM."""

    def __init__(
        self,
        client: OpenAICompatibleLLMClient,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_count = 0
        self.mode = "llm"

    def _fallback(self, candidates: list[CandidateComment], top_n: int) -> list[CandidateComment]:
        self.fallback_count += 1
        sorted_candidates = sorted(candidates, key=lambda item: item.generator_score, reverse=True)
        output: list[CandidateComment] = []
        for candidate in sorted_candidates[:top_n]:
            evidence = list(candidate.evidence) + ["llm_reranker_fallback=true"]
            output.append(
                CandidateComment(
                    text=candidate.text,
                    generator_score=candidate.generator_score,
                    reranker_score=candidate.generator_score,
                    source_example_id=candidate.source_example_id,
                    evidence=evidence,
                )
            )
        return output

    def rerank(self, example: MRExample, candidates: list[CandidateComment], top_n: int = 3) -> list[CandidateComment]:
        if not candidates:
            return []

        candidate_lines = [
            f"{index}. {candidate.text}\n   generator_score={candidate.generator_score:.3f}"
            for index, candidate in enumerate(candidates)
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict code review reranker. Prefer comments that are useful, grounded "
                    "in the diff, actionable, and specific. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Rank the best {top_n} candidates. Use the zero-based candidate_id values exactly.\n\n"
                    f"{_example_prompt(example)}\n\nCandidate comments:\n"
                    + "\n".join(candidate_lines)
                ),
            },
        ]
        result = self.client.chat_json(
            role="reranker",
            messages=messages,
            response_schema=RERANKER_SCHEMA,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        ranked = result.payload.get("ranked_comments", [])
        if not isinstance(ranked, list):
            return self._fallback(candidates, top_n)

        by_id = {index: candidate for index, candidate in enumerate(candidates)}
        scored: list[CandidateComment] = []
        seen_ids: set[int] = set()
        for item in ranked:
            if not isinstance(item, dict):
                continue
            try:
                candidate_id = int(item.get("candidate_id"))
            except (TypeError, ValueError):
                continue
            if candidate_id in seen_ids or candidate_id not in by_id:
                continue
            seen_ids.add(candidate_id)
            candidate = by_id[candidate_id]
            score = _bounded(item.get("score"), default=candidate.generator_score)
            evidence = list(candidate.evidence)
            evidence.extend(
                [
                    "llm_reranker",
                    f"model={self.client.model}",
                    f"usefulness={_bounded(item.get('usefulness')):.3f}",
                    f"groundedness={_bounded(item.get('groundedness')):.3f}",
                    f"actionability={_bounded(item.get('actionability')):.3f}",
                    f"specificity={_bounded(item.get('specificity')):.3f}",
                ]
            )
            reason = str(item.get("reason", "")).strip()
            if reason:
                evidence.append(f"reason={reason}")
            scored.append(
                CandidateComment(
                    text=candidate.text,
                    generator_score=candidate.generator_score,
                    reranker_score=score,
                    source_example_id=candidate.source_example_id,
                    evidence=evidence,
                )
            )

        if not scored:
            return self._fallback(candidates, top_n)
        scored.sort(key=lambda candidate: candidate.reranker_score, reverse=True)
        return scored[:top_n]
