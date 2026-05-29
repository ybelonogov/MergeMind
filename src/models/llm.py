"""Local OpenAI-compatible LLM components for MergeMind."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from openai import OpenAI

from src.config import load_dotenv
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

REWRITER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rewritten_review_comments",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "integer"},
                            "rewritten_comment": {"type": "string"},
                            "essence": {"type": "string"},
                            "severity": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "candidate_id",
                            "rewritten_comment",
                            "essence",
                            "severity",
                            "confidence",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["rewritten_comments"],
            "additionalProperties": False,
        },
    },
}

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "review_quality_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "gold_alignment_score": {"type": "number"},
                "valid_alternative_score": {"type": "number"},
                "groundedness": {"type": "number"},
                "usefulness": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": [
                "gold_alignment_score",
                "valid_alternative_score",
                "groundedness",
                "usefulness",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}

NICE_TO_HAVE_TERMS = {
    "documentation",
    "docstring",
    "style",
    "formatting",
    "naming convention",
    "cleanup",
    "refactor",
    "consistency",
}

BUG_RISK_TERMS = {
    "bug",
    "break",
    "crash",
    "error",
    "exception",
    "fail",
    "incorrect",
    "regression",
    "security",
    "leak",
    "race",
    "null",
    "none",
}


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _contains_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


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
        response_format_mode: str = "json_schema",
        completion_fn: Callable[..., Any] | None = None,
        list_models_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.response_format_mode = response_format_mode
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

    def _response_format(self, response_schema: dict[str, Any]) -> dict[str, Any]:
        if self.response_format_mode == "json_object":
            return {"type": "json_object"}
        return response_schema

    def _send_chat_completion(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        if self._completion_fn is not None:
            return self._completion_fn(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
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
            "response_format_mode": self.response_format_mode,
        }
        response_format = self._response_format(response_schema)
        cache_key = _stable_cache_key(self.model, role, messages, response_format, params)
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
                    response_format=response_format,
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
                "cached_call_count": 0,
                "uncached_call_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "uncached_prompt_tokens": 0,
                "uncached_completion_tokens": 0,
                "uncached_total_tokens": 0,
                "cache_hit_rate": 0.0,
                "parse_error_rate": 0.0,
                "llm_total_latency_sec": 0.0,
                "llm_avg_latency_sec": 0.0,
                "llm_p95_latency_sec": 0.0,
                "tokens_per_second": 0.0,
                "uncached_tokens_per_second": 0.0,
            }

        cached_calls = [call for call in self.calls if call.cache_hit]
        uncached_calls = [call for call in self.calls if not call.cache_hit]
        latencies = [call.latency_seconds for call in uncached_calls]
        sorted_latencies = sorted(latencies)
        p95_index = math.ceil(0.95 * len(sorted_latencies)) - 1 if sorted_latencies else 0
        total_tokens = sum(call.usage.get("total_tokens", 0) for call in self.calls)
        uncached_total_tokens = sum(call.usage.get("total_tokens", 0) for call in uncached_calls)
        total_latency = sum(latencies)
        uncached_tokens_per_second = uncached_total_tokens / total_latency if total_latency else 0.0
        return {
            "llm_call_count": len(self.calls),
            "cached_call_count": len(cached_calls),
            "uncached_call_count": len(uncached_calls),
            "prompt_tokens": sum(call.usage.get("prompt_tokens", 0) for call in self.calls),
            "completion_tokens": sum(call.usage.get("completion_tokens", 0) for call in self.calls),
            "total_tokens": total_tokens,
            "uncached_prompt_tokens": sum(call.usage.get("prompt_tokens", 0) for call in uncached_calls),
            "uncached_completion_tokens": sum(call.usage.get("completion_tokens", 0) for call in uncached_calls),
            "uncached_total_tokens": uncached_total_tokens,
            "cache_hit_rate": mean([float(call.cache_hit) for call in self.calls]),
            "parse_error_rate": mean([float(call.parse_error) for call in self.calls]),
            "llm_total_latency_sec": total_latency,
            "llm_avg_latency_sec": mean(latencies) if latencies else 0.0,
            "llm_p95_latency_sec": sorted_latencies[p95_index] if sorted_latencies else 0.0,
            "tokens_per_second": uncached_tokens_per_second,
            "uncached_tokens_per_second": uncached_tokens_per_second,
        }


def build_llm_client(config: dict[str, Any], project_root: Path) -> OpenAICompatibleLLMClient:
    load_dotenv(project_root / ".env")
    llm_config = dict(config.get("llm", {}))
    raw_cache_path = Path(str(llm_config.get("cache_path", "artifacts/llm_cache.sqlite")))
    cache_path = raw_cache_path if raw_cache_path.is_absolute() else project_root / raw_cache_path
    api_key_env = str(llm_config.get("api_key_env", "")).strip()
    api_key = os.getenv(api_key_env, "") if api_key_env else str(llm_config.get("api_key", "lm-studio"))
    api_key = os.getenv("MERGEMIND_LLM_API_KEY", api_key)
    base_url = os.getenv("MERGEMIND_LLM_BASE_URL", str(llm_config.get("base_url", "http://localhost:1234/v1")))
    model = os.getenv("MERGEMIND_LLM_MODEL", str(llm_config.get("model", "qwen/qwen3.5-9b")))
    return OpenAICompatibleLLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        cache_path=cache_path,
        timeout_seconds=int(llm_config.get("timeout_seconds", 180)),
        retries=int(llm_config.get("retries", 2)),
        response_format_mode=str(llm_config.get("response_format", "json_schema")),
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
        min_candidates: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> None:
        self.client = client
        self.max_candidates = max_candidates
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_count = 0
        self.min_candidates = max(1, min(min_candidates, self.max_candidates))
        self.last_raw_generated_count = 0
        self.last_deduped_candidate_count = 0

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind, a code review system. Generate useful, specific, "
            "actionable review comments grounded in the provided diff. Prefer recall first: "
            "surface several plausible review angles before reranking removes noise. "
            "Return JSON only."
        )

    def _user_prompt(self, example: MRExample, minimum: int, limit: int) -> str:
        return (
            f"Generate between {minimum} and {limit} concise review comments when possible. "
            "Cover diverse grounded angles such as correctness, API/behavior compatibility, "
            "missing tests, edge cases, maintainability, and style only when it affects review value. "
            "Avoid duplicates and generic advice. Do not invent issues: if fewer than "
            f"{minimum} grounded comments exist, return only the grounded ones.\n\n"
            f"{_example_prompt(example)}"
        )

    def generate(
        self,
        example: MRExample,
        top_examples: int | None = None,
        max_candidates: int | None = None,
    ) -> list[CandidateComment]:
        del top_examples
        limit = self.max_candidates if max_candidates is None else max_candidates
        minimum = min(self.min_candidates, limit)
        self.last_raw_generated_count = 0
        self.last_deduped_candidate_count = 0
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._user_prompt(example, minimum, limit),
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
        self.last_raw_generated_count = len(comments)

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
            evidence = ["llm_generator", f"model={self.client.model}", f"raw_generated_count={len(comments)}"]
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

        self.last_deduped_candidate_count = len(candidates)
        if not candidates:
            self.fallback_count += 1
        return candidates


class SWEContractLLMGenerator(LLMGenerator):
    """Generate SWE-CI-oriented candidate comments focused on patch risk."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI patch-risk reviewer. Generate only correctness, "
            "behavior, API compatibility, data-loss, or regression-risk review comments "
            "grounded in the provided requirement and diff. Ignore style-only, naming-only, "
            "or documentation-only feedback unless it can break behavior. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, minimum: int, limit: int) -> str:
        return (
            f"Generate between {minimum} and {limit} compact review findings for the programmer's patch. "
            "Assume the patch may be wrong and look for concrete ways it could increase failing tests "
            "or fail the stated requirement. Use only the provided current diff, repository context, "
            "and requirement text. Do not infer a hidden solution or rely on unavailable reference code. "
            "Every comment must identify an actionable source-code risk; if fewer grounded risks exist, "
            "return only those grounded risks.\n\n"
            f"{_example_prompt(example)}"
        )


class SWETriageLLMGenerator(SWEContractLLMGenerator):
    """Generate a small set of root-cause triage comments for SWE-CI repair."""

    def __init__(
        self,
        client: OpenAICompatibleLLMClient,
        max_candidates: int = 3,
        min_candidates: int = 1,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> None:
        super().__init__(
            client,
            max_candidates=max(1, min(max_candidates, 3)),
            min_candidates=max(1, min(min_candidates, 1)),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI test-failure triage reviewer. Generate only the "
            "smallest set of source-code repair blockers that a programmer can act on before "
            "pytest. Prefer one root-cause finding over multiple broad review notes. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, minimum: int, limit: int) -> str:
        return (
            f"Generate up to {limit} triage findings; one strong root-cause finding is better than "
            "several weak comments. Focus only on source-code changes that are likely to reduce the "
            "failing-test gap in the next revision. A valid finding must name the broken invariant, "
            "cite diff or requirement evidence, and describe the smallest safe source-code repair. "
            "Reject style, docs, naming, test-editing, speculative rewrites, and broad redesign advice. "
            "Use only the provided current diff, repository context, and requirement text. Do not rely "
            "on unavailable reference code. If there is no grounded root-cause risk, return no comments.\n\n"
            f"{_example_prompt(example)}"
        )


class SWESafeTriageLLMGenerator(SWETriageLLMGenerator):
    """Generate only regression-guarded SWE-CI comments."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI regression-guard reviewer. Your default answer is "
            "no comment. Generate a comment only when the programmer patch introduces a "
            "clear local source-code regression that can be repaired without broad feature "
            "implementation. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, minimum: int, limit: int) -> str:
        return (
            f"Generate 0 or 1 comments; never exceed {limit}. Prefer 0 comments unless the "
            "risk is grounded in the current programmer diff and the requirement.\n\n"
            "Safe-triage rules:\n"
            "1. Comment only on changed source files, not generated data, fixtures, docs, tests, or unrelated files.\n"
            "2. Do not ask for a new feature-shaped repair unless the programmer patch already changed that exact behavior.\n"
            "3. Prefer preserving existing passing behavior over chasing a plausible requirement interpretation.\n"
            "4. The expected revision must be the smallest correction to the current patch, not a rewrite.\n"
            "5. If the comment could plausibly increase the failing-test gap, return no comments.\n"
            "6. If evidence is only a broad requirement with no diff-level bug, return no comments.\n"
            "7. Do not mention hidden reference solutions, unavailable code, style, docs, naming, or tests.\n\n"
            "Required fields inside each comment:\n"
            "Finding: <one concrete regression risk>\n"
            "Evidence: <specific diff and requirement evidence>\n"
            "Expected revision: <smallest behavior-preserving source-code correction>\n"
            "Do not change: <tests, unrelated files, public APIs, and currently passing behavior>\n"
            "Confidence: <0.0-1.0>\n\n"
            f"{_example_prompt(example)}"
        )


class SWETestGuardLLMGenerator(SWESafeTriageLLMGenerator):
    """Generate comments only when visible pytest failures support the finding."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI test-failure guard reviewer. Your default answer is "
            "no comment. Generate a comment only when a visible previous pytest failure, the "
            "current programmer diff, and the architect requirement all support the same small "
            "Python source-code repair. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, minimum: int, limit: int) -> str:
        return (
            f"Generate 0 or 1 comments; never exceed {limit}. Prefer 0 comments unless there is "
            "a direct chain from visible failed nodeid or traceback to changed Python source code.\n\n"
            "Test-guard rules:\n"
            "1. A high-confidence comment must cite a visible pytest failure and a changed Python source line.\n"
            "2. If no visible failure context is present, return no comments.\n"
            "3. If the diff only changes docs, generated data, fixtures, tests, or unrelated files, return no comments.\n"
            "4. Do not propose broad feature implementation, rewrites, or behavior outside the programmer patch.\n"
            "5. Do not ask the programmer to edit tests or generated artifacts.\n"
            "6. If the fix could plausibly add new failing tests, return no comments.\n"
            "7. Do not mention hidden reference solutions, unavailable code, style, docs, naming, or preferences.\n\n"
            "Required fields inside each comment:\n"
            "Failing test: <visible nodeid or traceback symptom>\n"
            "Finding: <one concrete root cause in changed Python source>\n"
            "Evidence: <specific failure context plus specific diff evidence>\n"
            "Expected revision: <smallest safe Python source-code correction>\n"
            "Do not change: <tests, unrelated files, public APIs, generated data, and currently passing behavior>\n"
            "Confidence: <0.0-1.0>\n\n"
            f"{_example_prompt(example)}"
        )


class CavemanLLMGenerator(SWEContractLLMGenerator):
    """Generate blunt, low-noise SWE-CI repair comments."""

    def _system_prompt(self) -> str:
        return (
            "You are Caveman Reviewer for SWE-CI. Your only job is to reduce failing tests "
            "and patch risk. You are not a stylist, architect, or teacher. Generate comments "
            "only for concrete source-code bug risks grounded in the requirement, diff, or "
            "visible test-failure context. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, minimum: int, limit: int) -> str:
        return (
            f"Generate between 0 and {limit} comments. Prefer zero comments over weak comments.\n\n"
            "Caveman rules:\n"
            "1. Comment only on changed source code or directly affected source behavior.\n"
            "2. One comment must describe one concrete bug risk.\n"
            "3. Every comment must cite specific evidence from diff, requirement, or visible test failure context.\n"
            "4. If you cannot name a specific failing behavior, return no comment.\n"
            "5. Prefer comments likely to reduce the current failing-test gap.\n"
            "6. Prefer local fixes over rewrites.\n"
            "7. Do not suggest editing tests.\n"
            "8. Do not mention style, docs, naming, formatting, or preferences.\n"
            "9. Do not summarize or praise.\n"
            "10. If confidence is low, say nothing.\n\n"
            "Each comment text must already include these fields:\n"
            "Finding: <one sentence bug/risk>\n"
            "Evidence: <specific diff, requirement, or visible test evidence>\n"
            "Expected revision: <small local source-code change>\n"
            "Do not change: <tests, unrelated files, or behavior to preserve>\n"
            "Confidence: <0.0-1.0>\n\n"
            f"{_example_prompt(example)}"
        )


class CavemanDirectLLMGenerator(CavemanLLMGenerator):
    """Generate final repair-contract comments without a rewriter step."""

    def _system_prompt(self) -> str:
        return (
            "You are Caveman Direct Reviewer for SWE-CI. Generate final agent-facing repair "
            "contracts directly. Be blunt, local, and conservative. Return JSON only."
        )


class CavemanTestTriageLLMGenerator(CavemanLLMGenerator):
    """Generate comments using visible previous pytest failures as first-class evidence."""

    def _system_prompt(self) -> str:
        return (
            "You are Caveman Test Triage Reviewer for SWE-CI. Use only the requirement, diff, "
            "and visible previous pytest failures to find one local source-code repair risk. "
            "Do not infer unseen tests or hidden solutions. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, minimum: int, limit: int) -> str:
        return (
            f"Generate between 0 and {limit} comments. Prefer exactly one root-cause comment "
            "when the visible pytest failures and diff point to a local source bug; otherwise "
            "return no comments.\n\n"
            "Priority:\n"
            "1. Import/runtime errors introduced by the patch.\n"
            "2. Visible failed nodeids explained by changed source behavior.\n"
            "3. Wrong branch, wrong default, wrong type, or lost state in changed code.\n"
            "4. Anything else only if it is concrete and local.\n\n"
            "Required comment fields:\n"
            "Finding: <one likely root cause>\n"
            "Evidence: <specific diff/requirement/test-failure evidence>\n"
            "Expected revision: <small source-code edit>\n"
            "Do not change: <tests, unrelated files, public APIs, or behavior to preserve>\n"
            "Confidence: <0.0-1.0>\n\n"
            "Forbidden: style-only, docs-only, naming, test edits, broad refactors, speculation.\n\n"
            f"{_example_prompt(example)}"
        )


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

    def _system_prompt(self) -> str:
        return (
            "You are a skeptical senior code reviewer reranking candidate review comments. "
            "Only score a comment highly if you would personally leave it on the merge request. "
            "Prefer comments that identify concrete correctness, safety, API, data-loss, "
            "performance, or maintainability risks grounded in the diff. Return JSON only."
        )

    def _user_prompt(
        self,
        example: MRExample,
        candidate_lines: list[str],
        top_n: int,
    ) -> str:
        return (
            "Use this strict scoring rubric:\n"
            "- 0.90-1.00: clear high-value issue, concrete risk, directly grounded, actionable.\n"
            "- 0.70-0.89: useful and grounded, but lower severity or missing some context.\n"
            "- 0.40-0.69: minor, nice-to-have, documentation/style, or somewhat speculative.\n"
            "- 0.10-0.39: generic, weak, mostly restates the diff, or low confidence.\n"
            "- 0.00: unrelated, duplicate, hallucinated, or not worth leaving.\n"
            "Do not copy generator_score. Do not give multiple candidates the same score unless "
            "they are truly equally useful. At most one candidate should exceed 0.90 unless there "
            "are multiple independent defects. Documentation, style, and naming-only comments "
            "should usually stay below 0.70 unless they prevent a real bug.\n\n"
            f"Rank the best {top_n} candidates. Use the zero-based candidate_id values exactly.\n\n"
            f"{_example_prompt(example)}\n\nCandidate comments:\n"
            + "\n".join(candidate_lines)
        )

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

    def _fill_missing_ranked(
        self,
        scored: list[CandidateComment],
        candidates: list[CandidateComment],
        seen_ids: set[int],
        top_n: int,
    ) -> list[CandidateComment]:
        if len(scored) >= top_n:
            return scored[:top_n]

        remaining = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if index not in seen_ids
        ]
        remaining.sort(key=lambda item: item[1].generator_score, reverse=True)
        filled = list(scored)
        last_score = filled[-1].reranker_score if filled else 0.55
        for index, candidate in remaining:
            if len(filled) >= top_n:
                break
            fill_score = _bounded(min(candidate.generator_score, max(0.05, last_score - 0.05)))
            last_score = fill_score
            evidence = list(candidate.evidence)
            evidence.extend(
                [
                    "llm_reranker_unranked_fill=true",
                    f"filled_candidate_id={index}",
                    "reason=LLM reranker returned fewer than requested candidates; filled by generator score.",
                ]
            )
            filled.append(
                CandidateComment(
                    text=candidate.text,
                    generator_score=candidate.generator_score,
                    reranker_score=fill_score,
                    source_example_id=candidate.source_example_id,
                    evidence=evidence,
                )
            )
        return filled[:top_n]

    def _calibrated_score(self, candidate: CandidateComment, item: dict[str, Any], output_position: int) -> float:
        usefulness = _bounded(item.get("usefulness"), default=0.5)
        groundedness = _bounded(item.get("groundedness"), default=0.5)
        actionability = _bounded(item.get("actionability"), default=0.5)
        specificity = _bounded(item.get("specificity"), default=0.5)
        component_score = (
            0.35 * usefulness
            + 0.30 * groundedness
            + 0.20 * actionability
            + 0.15 * specificity
        )
        raw_score = _bounded(item.get("score"), default=component_score)
        score = 0.55 * raw_score + 0.45 * component_score

        candidate_context = " ".join([candidate.text, str(item.get("reason", ""))])
        if _contains_any(candidate_context, NICE_TO_HAVE_TERMS) and not _contains_any(candidate_context, BUG_RISK_TERMS):
            score = min(score, 0.68)
        if groundedness < 0.7:
            score *= 0.80
        if usefulness < 0.7:
            score *= 0.85
        if actionability < 0.6:
            score -= 0.08

        # Keep saturated local-model scores honest: later ranked candidates need
        # stronger evidence to stay near the top.
        score = min(score, 1.0 - 0.08 * output_position)
        return _bounded(score)

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
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._user_prompt(example, candidate_lines, top_n),
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
        for output_position, item in enumerate(ranked):
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
            raw_score = _bounded(item.get("score"), default=candidate.generator_score)
            score = self._calibrated_score(candidate, item, output_position)
            evidence = list(candidate.evidence)
            evidence.extend(
                [
                    "llm_reranker",
                    f"model={self.client.model}",
                    f"llm_raw_score={raw_score:.3f}",
                    f"calibrated_score={score:.3f}",
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
        return self._fill_missing_ranked(scored, candidates, seen_ids, top_n)


class SWEContractLLMReranker(LLMReranker):
    """Rerank comments for likely SWE-CI repair impact."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI repair-impact critic. Rank comments by how likely "
            "they are to help the programmer reduce failing tests in the next revision. "
            "Favor concrete, grounded, minimal-fix guidance. Return JSON only."
        )

    def _user_prompt(
        self,
        example: MRExample,
        candidate_lines: list[str],
        top_n: int,
    ) -> str:
        return (
            "Use this strict scoring rubric:\n"
            "- 0.90-1.00: likely to prevent a regression or directly reduce failing tests.\n"
            "- 0.70-0.89: grounded and actionable, but lower confidence or indirect test impact.\n"
            "- 0.40-0.69: plausible but speculative, broad, or mostly maintainability-focused.\n"
            "- 0.10-0.39: style-only, weak, generic, or not useful for the next revision.\n"
            "- 0.00: unrelated, duplicate, hallucinated, or based on hidden solution assumptions.\n"
            "Prefer findings tied to changed source behavior, edge cases, API compatibility, state handling, "
            "or expected requirement behavior. Do not reward comments that require editing tests. "
            "Use only the zero-based candidate_id values exactly.\n\n"
            f"Rank the best {top_n} comments for the next programmer revision.\n\n"
            f"{_example_prompt(example)}\n\nCandidate comments:\n"
            + "\n".join(candidate_lines)
        )


class SWETriageLLMReranker(SWEContractLLMReranker):
    """Rerank SWE-CI comments as root-cause triage blockers."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI root-cause triage critic. Select only comments that "
            "are likely to produce a minimal, behavior-preserving repair in the next programmer "
            "revision. Penalize noise aggressively. Return JSON only."
        )

    def _user_prompt(
        self,
        example: MRExample,
        candidate_lines: list[str],
        top_n: int,
    ) -> str:
        return (
            "Use this strict scoring rubric:\n"
            "- 0.90-1.00: grounded root cause, clear invariant, minimal repair likely reduces failing tests.\n"
            "- 0.75-0.89: actionable and grounded but lower confidence about test-gap impact.\n"
            "- 0.40-0.74: plausible review note, but too broad, indirect, or not a revision blocker.\n"
            "- 0.10-0.39: style, docs, naming, test-editing, generic, or likely to distract the programmer.\n"
            "- 0.00: unrelated, duplicate, hallucinated, or based on unavailable information.\n"
            "For automated SWE-CI repair, false positives are expensive: if a comment could cause "
            "a broad rewrite or preserve the wrong behavior, keep it below 0.75. Rank only comments "
            "that should be shown before the next revision. Use the zero-based candidate_id values exactly.\n\n"
            f"Rank the best {top_n} comments for root-cause repair triage.\n\n"
            f"{_example_prompt(example)}\n\nCandidate comments:\n"
            + "\n".join(candidate_lines)
        )


class SWESafeTriageLLMReranker(SWETriageLLMReranker):
    """Reject SWE-CI comments likely to trigger broad or regressive revisions."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI safe-triage critic. False positives are more costly "
            "than missed comments. Select only comments that should be shown to an automated "
            "programmer before pytest. Return JSON only."
        )

    def _user_prompt(
        self,
        example: MRExample,
        candidate_lines: list[str],
        top_n: int,
    ) -> str:
        return (
            "Use this stricter scoring rubric:\n"
            "- 0.90-1.00: changed-source regression, exact diff evidence, tiny behavior-preserving repair.\n"
            "- 0.75-0.89: grounded local risk, but modest uncertainty about test-gap impact.\n"
            "- 0.40-0.74: plausible but broad, feature-shaped, or weakly tied to the programmer diff.\n"
            "- 0.10-0.39: generated/data/docs/tests advice, style, naming, generic, or likely to distract.\n"
            "- 0.00: unrelated, duplicate, hidden-reference based, or unsafe.\n"
            "Keep comments below 0.75 if they suggest implementing new behavior not already attempted "
            "by the programmer patch, could touch unrelated files, or could increase regressions. "
            "High-scoring comments must include Finding, Evidence, Expected revision, Do not change, "
            "and Confidence. Use the zero-based candidate_id values exactly.\n\n"
            f"Rank at most {top_n} comments for a safe next revision.\n\n"
            f"{_example_prompt(example)}\n\nCandidate comments:\n"
            + "\n".join(candidate_lines)
        )


class SWETestGuardLLMReranker(SWESafeTriageLLMReranker):
    """Rank only comments tied to visible test failures and changed Python source."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI test-failure guard critic. False positives are very "
            "expensive. Select only comments that connect a visible pytest failure to changed "
            "Python source and a tiny safe revision. Return JSON only."
        )

    def _user_prompt(
        self,
        example: MRExample,
        candidate_lines: list[str],
        top_n: int,
    ) -> str:
        return (
            "Use this strict scoring rubric:\n"
            "- 0.90-1.00: explicit visible failed nodeid/traceback, exact changed Python evidence, tiny safe fix.\n"
            "- 0.75-0.89: visible failure and changed source are connected, but some uncertainty remains.\n"
            "- 0.40-0.74: plausible code review note, but failure link or source evidence is weak.\n"
            "- 0.10-0.39: no visible failure link, broad rewrite, non-code artifact advice, style, docs, naming, or tests.\n"
            "- 0.00: unrelated, duplicate, hidden-reference based, unsafe, or likely to increase regressions.\n"
            "Keep any comment below 0.75 unless it includes Failing test, Finding, Evidence, "
            "Expected revision, Do not change, and Confidence. Use the zero-based candidate_id values exactly.\n\n"
            f"Rank at most {top_n} comments for a test-aware safe revision.\n\n"
            f"{_example_prompt(example)}\n\nCandidate comments:\n"
            + "\n".join(candidate_lines)
        )


class CavemanLLMReranker(SWEContractLLMReranker):
    """Aggressively filter SWE-CI comments for local repair impact."""

    def _system_prompt(self) -> str:
        return (
            "You are Caveman Critic for SWE-CI. False positives are expensive. Select only "
            "comments that a coding agent should follow before pytest. Return JSON only."
        )

    def _user_prompt(
        self,
        example: MRExample,
        candidate_lines: list[str],
        top_n: int,
    ) -> str:
        return (
            "Use this strict scoring rubric:\n"
            "- 0.90-1.00: concrete local bug risk, strong evidence, small safe revision.\n"
            "- 0.75-0.89: grounded and actionable, but weaker test-gap confidence.\n"
            "- 0.40-0.74: plausible but too broad, indirect, or missing evidence.\n"
            "- 0.10-0.39: generic, style/docs/naming, test-edit advice, or distracting.\n"
            "- 0.00: unrelated, duplicate, hallucinated, or unsafe.\n"
            "A high-scoring comment must contain Finding, Evidence, Expected revision, "
            "Do not change, and Confidence. Penalize comments that could cause broad rewrites "
            "or edits outside the programmer patch. Use the zero-based candidate_id values exactly.\n\n"
            f"Rank the best {top_n} comments for the next programmer revision.\n\n"
            f"{_example_prompt(example)}\n\nCandidate comments:\n"
            + "\n".join(candidate_lines)
        )


class LLMRewriter:
    """Rewrite final review comments into concise human-facing feedback."""

    def __init__(
        self,
        client: OpenAICompatibleLLMClient,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_count = 0
        self.mode = "llm_rewriter"

    def _system_prompt(self) -> str:
        return (
            "You are a final code-review editor. Rewrite already-selected comments so they are "
            "clear, concise, and useful to a developer. Do not add new facts, new risks, or "
            "claims that are not present in the original comment and diff. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, candidate_lines: list[str]) -> str:
        return (
            "For each candidate, produce:\n"
            "- rewritten_comment: one or two short sentences in a practical code-review style.\n"
            "- essence: at most 12 words describing the core issue.\n"
            "- severity: one of low, medium, high.\n"
            "- confidence: 0..1 for whether the rewrite preserved the original meaning.\n\n"
            "Rules:\n"
            "- Preserve the original technical meaning.\n"
            "- Do not invent missing code paths, tests, bugs, or fixes.\n"
            "- If the original comment is already clear, keep it close to the original.\n"
            "- Prefer direct, actionable wording over long explanations.\n\n"
            f"{_example_prompt(example)}\n\nSelected comments:\n"
            + "\n".join(candidate_lines)
        )

    def _fallback(self, candidates: list[CandidateComment]) -> list[CandidateComment]:
        self.fallback_count += 1
        output: list[CandidateComment] = []
        for candidate in candidates:
            evidence = list(candidate.evidence) + ["llm_rewriter_fallback=true"]
            output.append(
                CandidateComment(
                    text=candidate.text,
                    generator_score=candidate.generator_score,
                    reranker_score=candidate.reranker_score,
                    source_example_id=candidate.source_example_id,
                    evidence=evidence,
                    original_text=candidate.original_text,
                    essence=candidate.essence,
                    severity=candidate.severity,
                    rewrite_confidence=candidate.rewrite_confidence,
                )
            )
        return output

    def rewrite(self, example: MRExample, candidates: list[CandidateComment]) -> list[CandidateComment]:
        if not candidates:
            return []

        candidate_lines = [
            "\n".join(
                [
                    f"{index}. {candidate.text}",
                    f"   reranker_score={candidate.reranker_score:.3f}",
                ]
            )
            for index, candidate in enumerate(candidates)
        ]
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._user_prompt(example, candidate_lines),
            },
        ]
        result = self.client.chat_json(
            role="rewriter",
            messages=messages,
            response_schema=REWRITER_SCHEMA,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        rewritten = result.payload.get("rewritten_comments", [])
        if not isinstance(rewritten, list):
            return self._fallback(candidates)

        by_id = {index: candidate for index, candidate in enumerate(candidates)}
        used_ids: set[int] = set()
        output_by_id: dict[int, CandidateComment] = {}
        for item in rewritten:
            if not isinstance(item, dict):
                continue
            try:
                candidate_id = int(item.get("candidate_id"))
            except (TypeError, ValueError):
                continue
            if candidate_id in used_ids or candidate_id not in by_id:
                continue
            used_ids.add(candidate_id)
            candidate = by_id[candidate_id]
            rewritten_text = str(item.get("rewritten_comment", "")).strip() or candidate.text
            original_text = candidate.original_text or candidate.text
            essence = str(item.get("essence", "")).strip()
            severity = str(item.get("severity", "")).strip().lower()
            if severity not in {"low", "medium", "high"}:
                severity = "medium"
            confidence = _bounded(item.get("confidence"), default=0.5)
            evidence = list(candidate.evidence)
            evidence.extend(
                [
                    "llm_rewriter",
                    f"model={self.client.model}",
                    f"rewrite_confidence={confidence:.3f}",
                    f"rewrite_severity={severity}",
                ]
            )
            if essence:
                evidence.append(f"essence={essence}")
            reason = str(item.get("reason", "")).strip()
            if reason:
                evidence.append(f"rewrite_reason={reason}")
            if result.cache_hit:
                evidence.append("rewrite_cache_hit=true")
            output_by_id[candidate_id] = CandidateComment(
                text=rewritten_text,
                generator_score=candidate.generator_score,
                reranker_score=candidate.reranker_score,
                source_example_id=candidate.source_example_id,
                evidence=evidence,
                original_text=original_text,
                essence=essence,
                severity=severity,
                rewrite_confidence=confidence,
            )

        if not output_by_id:
            return self._fallback(candidates)

        output: list[CandidateComment] = []
        for index, candidate in enumerate(candidates):
            output.append(output_by_id.get(index, candidate))
        return output


class SWEContractLLMRewriter(LLMRewriter):
    """Rewrite selected comments into an agent-facing revision contract."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's final repair-guidance editor. Rewrite selected comments "
            "as compact instructions for an automated programmer revision. Preserve only "
            "facts grounded in the original comment and diff. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, candidate_lines: list[str]) -> str:
        return (
            "For each candidate, set rewritten_comment to this exact Markdown contract:\n"
            "Finding: <one concrete risk>\n"
            "Evidence: <specific diff or requirement evidence>\n"
            "Expected revision: <minimal source-code change the programmer should make>\n"
            "Do not change: <tests, unrelated files, or behavior that should remain stable>\n\n"
            "Rules:\n"
            "- Keep each field to one concise sentence.\n"
            "- Do not invent code paths, tests, failing output, or hidden solution details.\n"
            "- Do not ask the programmer to edit tests.\n"
            "- Prefer minimal source edits that preserve existing public behavior.\n"
            "- If the candidate lacks enough evidence, keep the original issue but lower confidence.\n\n"
            f"{_example_prompt(example)}\n\nSelected comments:\n"
            + "\n".join(candidate_lines)
        )


class SWETriageLLMRewriter(SWEContractLLMRewriter):
    """Rewrite comments into one minimal root-cause repair contract."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI root-cause repair editor. Convert selected triage comments "
            "into compact instructions for an automated programmer. Preserve only grounded facts, "
            "avoid broad rewrites, and prefer no-op guidance over speculative fixes. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, candidate_lines: list[str]) -> str:
        return (
            "For each candidate, set rewritten_comment to this exact Markdown contract:\n"
            "Finding: <one likely root cause or invariant violation>\n"
            "Evidence: <specific requirement or diff evidence; no test-output guesses>\n"
            "Expected revision: <smallest source-code edit that should repair the invariant>\n"
            "Do not change: <tests, unrelated files, public APIs, or behavior that should stay stable>\n\n"
            "Rules:\n"
            "- Keep each field to one concise sentence.\n"
            "- Do not invent failing tests, hidden implementation details, or unavailable solution code.\n"
            "- Do not ask the programmer to edit tests.\n"
            "- Do not preserve newly introduced behavior unless the requirement explicitly demands it.\n"
            "- If the candidate is weak, rewrite it as a narrow verification/check instead of a broad rewrite.\n\n"
            f"{_example_prompt(example)}\n\nSelected comments:\n"
            + "\n".join(candidate_lines)
        )


class SWESafeTriageLLMRewriter(SWETriageLLMRewriter):
    """Rewrite comments into a conservative regression-guard contract."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI safe repair editor. Convert selected comments into "
            "conservative instructions for an automated programmer. Preserve currently "
            "passing behavior and avoid broad repairs. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, candidate_lines: list[str]) -> str:
        return (
            "For each candidate, set rewritten_comment to this exact Markdown contract:\n"
            "Finding: <one concrete regression risk in the current patch>\n"
            "Evidence: <specific diff and requirement evidence>\n"
            "Expected revision: <smallest behavior-preserving correction to the current patch>\n"
            "Do not change: <tests, unrelated files, public APIs, generated data, and currently passing behavior>\n\n"
            "Rules:\n"
            "- Keep each field to one concise sentence.\n"
            "- Do not add new requirements, hidden solution details, or failing-test guesses.\n"
            "- Do not ask for tests, docs, generated-data edits, broad feature implementation, or unrelated files.\n"
            "- If the selected comment is broad, rewrite it as a narrow guard/check or preserve-current-behavior note.\n"
            "- The revision should be safe even if the comment is only partially correct.\n\n"
            f"{_example_prompt(example)}\n\nSelected comments:\n"
            + "\n".join(candidate_lines)
        )


class SWETestGuardLLMRewriter(SWESafeTriageLLMRewriter):
    """Rewrite comments into a test-aware safe repair contract."""

    def _system_prompt(self) -> str:
        return (
            "You are MergeMind's SWE-CI test-aware safe repair editor. Convert selected "
            "comments into conservative instructions for an automated programmer. Preserve "
            "only facts grounded in visible pytest failures and changed Python source. "
            "Return JSON only."
        )

    def _user_prompt(self, example: MRExample, candidate_lines: list[str]) -> str:
        return (
            "For each candidate, set rewritten_comment to this exact Markdown contract:\n"
            "Failing test: <visible nodeid or traceback symptom>\n"
            "Finding: <one concrete root cause in changed Python source>\n"
            "Evidence: <specific failure context plus specific diff evidence>\n"
            "Expected revision: <smallest safe Python source-code correction>\n"
            "Do not change: <tests, unrelated files, public APIs, generated data, and currently passing behavior>\n"
            "Confidence: <0.0-1.0>\n\n"
            "Rules:\n"
            "- Keep each field to one concise sentence.\n"
            "- Do not add hidden solution details, unseen tests, or new requirements.\n"
            "- Do not ask for tests, docs, generated-data edits, broad feature implementation, or unrelated files.\n"
            "- If the selected comment lacks visible failure evidence, rewrite it as a no-op verification note with low confidence.\n"
            "- The revision must be safe even if the comment is only partially correct.\n\n"
            f"{_example_prompt(example)}\n\nSelected comments:\n"
            + "\n".join(candidate_lines)
        )


class CavemanLLMRewriter(SWEContractLLMRewriter):
    """Rewrite selected comments into a strict caveman repair contract."""

    def _system_prompt(self) -> str:
        return (
            "You are Caveman Repair Editor. Rewrite selected comments as terse instructions "
            "for an automated SWE-CI programmer revision. Do not add facts. Return JSON only."
        )

    def _user_prompt(self, example: MRExample, candidate_lines: list[str]) -> str:
        return (
            "For each candidate, set rewritten_comment to this exact contract:\n"
            "Finding: <one concrete bug/risk>\n"
            "Evidence: <specific diff, requirement, or visible test-failure evidence>\n"
            "Expected revision: <smallest local source-code change>\n"
            "Do not change: <tests, unrelated files, public APIs, or behavior to preserve>\n"
            "Confidence: <0.0-1.0>\n\n"
            "Rules:\n"
            "- Keep each field to one short sentence.\n"
            "- Do not ask for tests, docs, naming changes, broad refactors, or style edits.\n"
            "- Preserve only facts present in the selected comment and provided context.\n"
            "- Prefer no new claim over a guess.\n\n"
            f"{_example_prompt(example)}\n\nSelected comments:\n"
            + "\n".join(candidate_lines)
        )
