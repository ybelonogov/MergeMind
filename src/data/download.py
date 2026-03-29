"""Download helpers for real MergeMind datasets."""

from __future__ import annotations
import os
import zipfile
from pathlib import Path
from typing import Any

import requests

from src.data.io import write_json

CHUNK_SIZE = 4 * 1024 * 1024
CODE_REVIEWER_ARCHIVE = "Comment_Generation.zip"
CODE_REVIEWER_FILES = [
    "Comment_Generation/msg-train.jsonl",
    "Comment_Generation/msg-valid.jsonl",
    "Comment_Generation/msg-test.jsonl",
]
CODOCBENCH_FILES = ["train.jsonl", "test.jsonl", "codocbench.jsonl"]


def _download_file(url: str, destination: Path, headers: dict[str, str] | None = None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers=headers, timeout=120) as response:
        if response.status_code >= 400:
            return {
                "status": "error",
                "status_code": response.status_code,
                "url": url,
                "message": response.text[:300],
            }

        total_bytes = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                total_bytes += len(chunk)

    return {
        "status": "downloaded",
        "path": str(destination),
        "bytes": total_bytes,
        "url": url,
    }


def _ensure_codereviewer(config: dict[str, Any], project_root: Path, force: bool = False) -> dict[str, Any]:
    download_dir = project_root / str(config["paths"]["download_dir"])
    raw_dir = project_root / str(config["paths"]["raw"]["codereviewer"])
    archive_path = download_dir / CODE_REVIEWER_ARCHIVE
    extracted_dir = raw_dir / "Comment_Generation"

    if force or not archive_path.exists():
        result = _download_file(config["download"]["codereviewer_comment_generation_url"], archive_path)
        if result["status"] != "downloaded":
            return result

    raw_dir.mkdir(parents=True, exist_ok=True)
    missing_files = [name for name in CODE_REVIEWER_FILES if not (raw_dir / name).exists()]
    if force or missing_files:
        with zipfile.ZipFile(archive_path) as archive:
            for member in CODE_REVIEWER_FILES:
                archive.extract(member, raw_dir)

    return {
        "status": "ready",
        "archive": str(archive_path),
        "extracted_dir": str(extracted_dir),
        "files": CODE_REVIEWER_FILES,
    }


def _ensure_codocbench(config: dict[str, Any], project_root: Path, force: bool = False) -> dict[str, Any]:
    raw_dir = project_root / str(config["paths"]["raw"]["codocbench"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_url = str(config["download"]["codocbench_base_url"]).rstrip("/")
    results: list[dict[str, Any]] = []

    for file_name in CODOCBENCH_FILES:
        destination = raw_dir / file_name
        if destination.exists() and not force:
            results.append({"status": "ready", "path": str(destination), "url": f"{base_url}/{file_name}"})
            continue
        results.append(_download_file(f"{base_url}/{file_name}", destination))

    return {"status": "ready", "files": results}


def _ensure_codereviewqa(config: dict[str, Any], project_root: Path, force: bool = False) -> dict[str, Any]:
    raw_file = project_root / str(config["paths"]["raw"]["codereviewqa"])
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if raw_file.exists() and not force:
        return {"status": "ready", "path": str(raw_file), "source": "existing_file"}

    if not token:
        note = (
            "CodeReviewQA is a gated Hugging Face dataset. "
            "Set HF_TOKEN or HUGGINGFACE_TOKEN after accepting the dataset conditions, "
            "then rerun scripts/download_datasets.py."
        )
        (raw_file.parent / "DOWNLOAD_BLOCKED.txt").write_text(note + "\n", encoding="utf-8")
        return {"status": "blocked", "reason": "missing_hf_token", "message": note}

    result = _download_file(
        str(config["download"]["codereviewqa_url"]),
        raw_file,
        headers={"Authorization": f"Bearer {token}"},
    )
    if result["status"] == "error":
        note = (
            "Failed to download CodeReviewQA. "
            "Make sure the Hugging Face account tied to HF_TOKEN has accepted the dataset terms."
        )
        (raw_file.parent / "DOWNLOAD_BLOCKED.txt").write_text(note + "\n", encoding="utf-8")
        result["message"] = note
        result["status"] = "blocked"
    return result


def download_datasets(config: dict[str, Any], project_root: Path, force: bool = False) -> dict[str, Any]:
    manifest = {
        "codereviewer": _ensure_codereviewer(config, project_root, force=force),
        "codocbench": _ensure_codocbench(config, project_root, force=force),
        "codereviewqa": _ensure_codereviewqa(config, project_root, force=force),
    }
    manifest["all_ready"] = all(entry.get("status") in {"ready"} for entry in manifest.values())
    write_json(project_root / str(config["paths"]["raw_manifest"]), manifest)
    return manifest
