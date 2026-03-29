"""JSON and JSONL helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return

    with file_path.open("r", encoding="utf-8") as handle:
        count = 0
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)
            count += 1
            if limit is not None and count >= limit:
                break


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, limit=limit))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
