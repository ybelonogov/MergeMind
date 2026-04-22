"""Configuration helpers for MergeMind."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value == "":
        return ""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _clean_lines(text: str) -> list[tuple[int, str]]:
    cleaned: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        cleaned.append((indent, line.strip()))
    return cleaned


def _parse_block(lines: list[tuple[int, str]], start_index: int, indent: int) -> tuple[Any, int]:
    if start_index >= len(lines):
        return {}, start_index

    current_indent, current_content = lines[start_index]
    if current_indent < indent:
        return {}, start_index

    if current_content.startswith("- "):
        items: list[Any] = []
        index = start_index
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent != indent or not content.startswith("- "):
                break
            item_text = content[2:].strip()
            index += 1
            if item_text:
                items.append(_parse_scalar(item_text))
            elif index < len(lines) and lines[index][0] > indent:
                nested_item, index = _parse_block(lines, index, lines[index][0])
                items.append(nested_item)
            else:
                items.append(None)
        return items, index

    mapping: dict[str, Any] = {}
    index = start_index
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {content}")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise ValueError(f"Expected ':' in config line: {content}")

        key, raw_value = content.split(":", 1)
        key = key.strip()
        value_text = raw_value.strip()
        index += 1

        if value_text:
            mapping[key] = _parse_scalar(value_text)
            continue

        if index < len(lines) and lines[index][0] > indent:
            nested_value, index = _parse_block(lines, index, lines[index][0])
            mapping[key] = nested_value
        else:
            mapping[key] = {}

    return mapping, index


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    lines = _clean_lines(config_path.read_text(encoding="utf-8"))
    if not lines:
        return {}
    parsed, _ = _parse_block(lines, 0, lines[0][0])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level config must be a mapping.")
    return parsed


def load_dotenv(path: str | Path, override: bool = False) -> None:
    """Load simple KEY=VALUE secrets from a local .env file."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def apply_llm_provider(config: dict[str, Any], provider_name: str) -> dict[str, Any]:
    """Return a config copy with a named LLM provider merged into llm settings."""

    if not provider_name:
        return config
    providers = dict(config.get("llm_providers", {}))
    if provider_name not in providers:
        available = ", ".join(sorted(providers)) or "<none>"
        raise ValueError(f"Unknown LLM provider '{provider_name}'. Available providers: {available}")

    provider = dict(providers[provider_name])
    merged = copy.deepcopy(config)
    merged.setdefault("llm", {})
    for key, value in provider.items():
        merged["llm"][key] = value
    merged["llm"]["provider"] = provider_name
    return merged


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path
