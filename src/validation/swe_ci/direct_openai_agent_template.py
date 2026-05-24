import json
import os
import posixpath
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from swe_ci.config import CONFIG


HOME_DIR = "/opt/agent/home"
DEFAULT_CONTEXT_CHARS = 90000


def _run_bytes(args: list[str], *, input_bytes: bytes | None = None, timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, input=input_bytes, capture_output=True, timeout=timeout, check=False)


def _read_container_file(container_name: str, path: str, max_chars: int = 12000) -> str:
    result = _run_bytes(["docker", "exec", container_name, "sh", "-c", f"cat {shlex.quote(path)} 2>/dev/null"], timeout=30)
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", "replace")[:max_chars]


def _write_container_file(container_name: str, path: str, content: str) -> None:
    directory = posixpath.dirname(path)
    command = f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}"
    result = _run_bytes(["docker", "exec", "-i", container_name, "sh", "-c", command], input_bytes=content.encode("utf-8"), timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))


def _list_container_files(container_name: str, root: str, pattern: str = "*.py", max_files: int = 80) -> list[str]:
    command = f"find {shlex.quote(root)} -type f -name {shlex.quote(pattern)} | sort | head -n {max_files}"
    result = _run_bytes(["docker", "exec", container_name, "sh", "-c", command], timeout=30)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.decode("utf-8", "replace").splitlines() if line.strip()]


def _list_tree(container_name: str, root: str = "/app/code") -> str:
    command = f"find {shlex.quote(root)} -maxdepth 4 -type f | sed 's#^/app/code/##' | sort | head -n 220"
    result = _run_bytes(["docker", "exec", container_name, "sh", "-c", command], timeout=30)
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", "replace")[:12000]


def _json_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _test_paths_from_summary(summary: str) -> list[str]:
    paths: list[str] = []
    for row in _json_lines(summary):
        test_id = str(row.get("test", ""))
        if "::" in test_id:
            path = test_id.split("::", 1)[0]
            if path and path not in paths:
                paths.append(path)
    return paths


def _paths_from_text(text: str) -> list[str]:
    matches = re.findall(r"(?:(?:/app/code/)?(?:src|tests)/[A-Za-z0-9_./-]+\.py)", text)
    seen: list[str] = []
    for match in matches:
        normalized = match
        if normalized.startswith("/app/code/"):
            normalized = normalized.removeprefix("/app/code/")
        if normalized not in seen:
            seen.append(normalized)
    return seen


def _append_section(parts: list[str], title: str, body: str, remaining: list[int]) -> None:
    if not body or remaining[0] <= 0:
        return
    chunk = body[: remaining[0]]
    parts.append(f"\n\n## {title}\n{chunk}")
    remaining[0] -= len(chunk)


def _collect_context(container_name: str, role: str, prompt: str) -> str:
    max_chars = int(os.environ.get("SWE_CI_DIRECT_CONTEXT_CHARS", str(DEFAULT_CONTEXT_CHARS)))
    remaining = [max_chars]
    parts: list[str] = []
    _append_section(parts, "File tree", _list_tree(container_name), remaining)

    summary = _read_container_file(container_name, "/app/non-passed/summary.jsonl", 18000)
    if summary:
        _append_section(parts, "Non-passed summary", summary, remaining)
        detail_files = _list_container_files(container_name, "/app/non-passed", "*", max_files=24)
        for detail_file in detail_files:
            if detail_file.endswith("summary.jsonl"):
                continue
            _append_section(parts, f"Failure detail {detail_file}", _read_container_file(container_name, detail_file, 3500), remaining)
        for test_path in _test_paths_from_summary(summary):
            _append_section(parts, f"Test file {test_path}", _read_container_file(container_name, f"/app/code/{test_path}", 9000), remaining)

    requirement = _read_container_file(container_name, "/app/requirement.xml", 16000)
    if requirement:
        _append_section(parts, "Requirement XML", requirement, remaining)

    candidate_paths = _paths_from_text(summary + "\n" + requirement + "\n" + prompt)
    source_paths = [path for path in candidate_paths if path.startswith("src/")]
    if not source_paths:
        source_paths = [path.removeprefix("/app/code/") for path in _list_container_files(container_name, "/app/code/src", "*.py", max_files=80)]
    for source_path in source_paths:
        if remaining[0] <= 0:
            break
        _append_section(parts, f"Source file {source_path}", _read_container_file(container_name, f"/app/code/{source_path}", 10000), remaining)

    if role == "programmer":
        for test_path in [path for path in candidate_paths if path.startswith("tests/")]:
            if remaining[0] <= 0:
                break
            _append_section(parts, f"Referenced test {test_path}", _read_container_file(container_name, f"/app/code/{test_path}", 7000), remaining)

    return "".join(parts)


def _message_text(message: dict[str, Any], choice: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        content = "\n".join(parts)
    if isinstance(content, str) and content.strip():
        return content
    for key in ("reasoning_content", "reasoning", "text"):
        value = message.get(key, choice.get(key))
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _request_chat(payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if getattr(CONFIG, "api_key", ""):
        headers["Authorization"] = f"Bearer {CONFIG.api_key}"
    request = urllib.request.Request(CONFIG.base_url.rstrip("/") + "/chat/completions", data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def _chat(
    messages: list[dict[str, str]],
    timeout: int,
    max_tokens: int,
    *,
    response_format: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], float]:
    if not CONFIG.base_url:
        raise RuntimeError("CONFIG.base_url is required for direct_openai.")
    if not CONFIG.model_name:
        raise RuntimeError("CONFIG.model_name is required for direct_openai.")
    payload = {
        "model": CONFIG.model_name,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format
    started = time.monotonic()
    try:
        try:
            response_payload = _request_chat(payload, timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if response_format and exc.code in {400, 422}:
                payload.pop("response_format", None)
                response_payload = _request_chat(payload, timeout)
            else:
                raise RuntimeError(f"direct_openai HTTP {exc.code}: {body[:1000]}") from exc
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"direct_openai HTTP {exc.code}: {body[:1000]}") from exc
    choice = response_payload.get("choices", [{}])[0]
    message = choice.get("message") or {}
    content = _message_text(message, choice)
    if not content:
        finish_reason = choice.get("finish_reason")
        keys = sorted(message.keys())
        raise RuntimeError(f"direct_openai empty response content; finish_reason={finish_reason!r}; message_keys={keys}")
    return content, response_payload.get("usage") or {}, time.monotonic() - started


def _extract_xml(text: str) -> str:
    fence = re.search(r"```(?:xml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("<")
    end = text.rfind(">")
    if start >= 0 and end > start:
        return text[start : end + 1].strip() + "\n"
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<requirements><requirement><description>{escaped}</description></requirement></requirements>\n"


def _extract_json(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"direct_openai response did not contain JSON: {text[:500]}")
    return json.loads(text[start : end + 1])


def _apply_file_replacements(container_name: str, response_text: str) -> list[str]:
    payload = _extract_json(response_text)
    files = payload.get("files") or payload.get("changes") or []
    if not isinstance(files, list):
        raise RuntimeError("direct_openai JSON must contain a files list.")
    changed: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        content = item.get("content")
        if not path or not isinstance(content, str):
            continue
        path = path.removeprefix("/app/code/").removeprefix("code/")
        normalized = posixpath.normpath(path)
        if normalized.startswith("../") or normalized == ".." or normalized.startswith("/"):
            raise RuntimeError(f"Unsafe path from direct_openai response: {path}")
        if normalized.startswith("tests/") or "/tests/" in normalized:
            raise RuntimeError(f"direct_openai attempted to edit tests: {normalized}")
        _write_container_file(container_name, f"/app/code/{normalized}", content)
        changed.append(normalized)
    if not changed:
        raise RuntimeError("direct_openai did not return any file replacements.")
    return changed


def _usage(input_tokens: int | None, output_tokens: int | None, elapsed: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "execution_time": elapsed,
    }
    if extra:
        result.update(extra)
    return result


def call_direct_openai(
    container_name: str,
    prompt: str,
    *,
    work_dir: str = "/app",
    timeout: int,
) -> dict[str, Any]:
    del work_dir
    role = "architect" if "/app/requirement.xml" in prompt and "/app/non-passed" in prompt else "programmer"
    context = _collect_context(container_name, role, prompt)
    if role == "architect":
        user_prompt = f"""/no_think
Create only the XML requirement document requested by the SWE-CI architect prompt.
Do not modify files in your answer. Use this extracted repository context instead of browsing tools.

Original prompt:
{prompt}

Repository context:
{context}
"""
        content, usage, elapsed = _chat(
            [
                {"role": "system", "content": "You write concise SWE-CI XML requirements. Output XML only."},
                {"role": "user", "content": user_prompt},
            ],
            timeout=timeout,
            max_tokens=6000,
        )
        _write_container_file(container_name, "/app/requirement.xml", _extract_xml(content))
        return _usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), elapsed, {"agent": "direct_openai", "role": role})

    file_replacements_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "swe_ci_file_replacements",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    }
                },
                "required": ["files"],
            },
        },
    }
    programmer_max_tokens = int(os.environ.get("SWE_CI_DIRECT_PROGRAMMER_MAX_TOKENS", "10000"))
    user_prompt = f"""/no_think
You are the SWE-CI programmer. Modify source code only. Do not edit tests.
Return JSON only in this exact shape:
{{"files":[{{"path":"src/package/module.py","content":"full replacement file content"}}]}}
Include full replacement content for every changed file. Keep the patch minimal.

Original prompt:
{prompt}

Repository context:
{context}
"""
    content, usage, elapsed = _chat(
        [
            {"role": "system", "content": "You fix Python code and return strict JSON file replacements."},
            {"role": "user", "content": user_prompt},
        ],
        timeout=timeout,
        max_tokens=programmer_max_tokens,
        response_format=file_replacements_schema,
    )
    changed_files = _apply_file_replacements(container_name, content)
    return _usage(
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        elapsed,
        {"agent": "direct_openai", "role": role, "changed_files": changed_files},
    )
