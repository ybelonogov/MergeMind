"""Dataset adapters for MergeMind MVP."""

from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from src.context.processing import enrich_example
from src.data.io import iter_jsonl, write_json, write_jsonl
from src.data.schema import MRExample, ReviewComment

CODE_REVIEWER_SPLITS = {
    "msg-train.jsonl": "train",
    "msg-valid.jsonl": "validation",
    "msg-test.jsonl": "test",
}


def _build_unified_diff(old_text: str, new_text: str, path: str) -> str:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    return "\n".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )


def _trim_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _compress_example(example: MRExample, prepare_config: dict[str, Any]) -> MRExample:
    diff_limit = int(prepare_config.get("max_patch_chars", 6000))
    context_limit = int(prepare_config.get("max_context_chars", 4000))
    file_limit = int(prepare_config.get("max_repository_file_chars", 12000))

    example = enrich_example(example)
    example.diff = _trim_text(example.diff, diff_limit)
    example.repository_context = _trim_text(example.repository_context, context_limit)
    example.repository_files = {
        path: _trim_text(content, file_limit)
        for path, content in example.repository_files.items()
    }
    for changed_file in example.changed_files:
        changed_file.repository_snippet = _trim_text(changed_file.repository_snippet, 1200)
    return example


def _normalize_codereviewer_sample(row: dict[str, Any]) -> MRExample:
    return MRExample(
        source_dataset="CodeReviewer",
        example_id=row["id"],
        split=row.get("split", "train"),
        repo=row.get("repo", ""),
        title=row.get("title", ""),
        description=row.get("description", ""),
        diff=row.get("diff", ""),
        repository_files=dict(row.get("repository_files", {})),
        gold_comments=[
            ReviewComment(
                text=text,
                is_useful=True,
                outcome=row.get("outcome", ""),
                source="CodeReviewer",
            )
            for text in row.get("review_comments", [])
        ],
        follow_up=row.get("follow_up", ""),
        ci_signals=dict(row.get("ci", {})),
        metadata={"dataset_role": "primary", "source_format": "sample_fixture"},
    )


def _normalize_codereviewer_real(row: dict[str, Any], split: str) -> MRExample | None:
    if int(row.get("y", 0)) != 1:
        return None

    synthetic_path = f"codereviewer/{row['id']}.txt"
    patch = row.get("patch", "").strip()
    if not patch:
        return None

    diff_text = "\n".join(
        [
            f"diff --git a/{synthetic_path} b/{synthetic_path}",
            f"--- a/{synthetic_path}",
            f"+++ b/{synthetic_path}",
            patch,
        ]
    )
    comment = row.get("msg", "").strip()
    if not comment:
        return None

    return MRExample(
        source_dataset="CodeReviewer",
        example_id=str(row["id"]),
        split=split,
        repo="CodeReviewer",
        title=comment.split(".")[0][:120],
        description="Real CodeReviewer comment-generation example.",
        diff=diff_text,
        repository_files={synthetic_path: row.get("oldf", "")},
        gold_comments=[ReviewComment(text=comment, is_useful=True, outcome="observed_review", source="CodeReviewer")],
        metadata={"dataset_role": "primary", "source_format": "zenodo_comment_generation"},
    )


def _normalize_codereviewqa_sample(row: dict[str, Any]) -> MRExample:
    return MRExample(
        source_dataset="CodeReviewQA",
        example_id=row["question_id"],
        split=row.get("split", "validation"),
        repo=row.get("repo", ""),
        title=row.get("change_title", ""),
        description=row.get("change_description", ""),
        diff=row.get("diff", ""),
        repository_files=dict(row.get("repository_files", {})),
        gold_comments=[
            ReviewComment(
                text=row.get("reference_comment", ""),
                is_useful=True,
                outcome="reasoning_check",
                source="CodeReviewQA",
            )
        ],
        metadata={
            "dataset_role": "validation_reasoning",
            "qa_answer": row.get("answer", ""),
            "source_format": "sample_fixture",
        },
    )


def _normalize_codereviewqa_real(row: dict[str, Any]) -> MRExample:
    language = str(row.get("lang", "text")).lower()
    extension = {
        "python": "py",
        "javascript": "js",
        "java": "java",
        "go": "go",
        "php": "php",
        "ruby": "rb",
        "c": "c",
        "c++": "cpp",
        "csharp": "cs",
    }.get(language, "txt")
    synthetic_path = f"codereviewqa/{row.get('lang', 'text').lower()}_{hash(row.get('review', '')) & 0xfffffff}.{extension}"
    old_text = row.get("old", "")
    new_text = row.get("new", "")
    diff_text = _build_unified_diff(old_text, new_text, synthetic_path)

    return MRExample(
        source_dataset="CodeReviewQA",
        example_id=f"CodeReviewQA-{hash((old_text, row.get('review', ''), new_text)) & 0xfffffff}",
        split="validation",
        repo="CodeReviewQA",
        title=row.get("type_correct", "Code review reasoning benchmark"),
        description="Real CodeReviewQA comprehension example.",
        diff=diff_text,
        repository_files={synthetic_path: old_text},
        gold_comments=[ReviewComment(text=row.get("review", ""), is_useful=True, outcome="reasoning_check", source="CodeReviewQA")],
        follow_up=new_text,
        metadata={
            "dataset_role": "validation_reasoning",
            "lang": row.get("lang", ""),
            "type_correct": row.get("type_correct", ""),
            "solution_correct": row.get("solution_correct", ""),
        },
    )


def _normalize_codocbench_sample(row: dict[str, Any]) -> MRExample:
    return MRExample(
        source_dataset="CoDocBench",
        example_id=row["sample_id"],
        split=row.get("split", "validation"),
        repo=row.get("repo", ""),
        title=row.get("change_summary", ""),
        description=row.get("change_notes", ""),
        diff=row.get("diff", ""),
        repository_files=dict(row.get("repository_files", {})),
        gold_comments=[
            ReviewComment(
                text=row.get("linked_note", ""),
                is_useful=True,
                outcome="alignment_signal",
                source="CoDocBench",
            )
        ],
        metadata={"dataset_role": "aux_alignment", "source_format": "sample_fixture"},
    )


def _normalize_codocbench_real(row: dict[str, Any], split: str) -> MRExample:
    version_data = row.get("version_data", [])
    if not version_data:
        return MRExample(
            source_dataset="CoDocBench",
            example_id=f"{row.get('owner', '')}/{row.get('project', '')}:{row.get('function', '')}",
            split=split,
            repo=f"{row.get('owner', '')}/{row.get('project', '')}",
            title=row.get("function", ""),
            description="Real CoDocBench alignment example.",
            diff=row.get("diff_code", ""),
            gold_comments=[],
            metadata={"dataset_role": "aux_alignment", "source_format": "github_jsonl"},
        )

    first_version = version_data[0]
    last_version = version_data[-1]
    file_path = row.get("file_path", row.get("file", "codocbench.py"))
    old_code = first_version.get("code", "")
    new_code = last_version.get("code", old_code)
    diff_text = row.get("diff_code", "") or _build_unified_diff(old_code, new_code, file_path)
    natural_language_target = last_version.get("docstring", "").strip() or last_version.get("commit_message", "").strip()

    return MRExample(
        source_dataset="CoDocBench",
        example_id=f"{row.get('owner', '')}/{row.get('project', '')}:{row.get('function', '')}:{last_version.get('commit_sha', '')}",
        split=split,
        repo=f"{row.get('owner', '')}/{row.get('project', '')}",
        title=row.get("function", ""),
        description="Real CoDocBench alignment example.",
        diff=diff_text,
        repository_files={file_path: old_code},
        gold_comments=[
            ReviewComment(
                text=natural_language_target,
                is_useful=True,
                outcome="alignment_signal",
                source="CoDocBench",
            )
        ]
        if natural_language_target
        else [],
        follow_up=new_code,
        metadata={
            "dataset_role": "aux_alignment",
            "source_format": "github_jsonl",
            "diff_docstring": row.get("diff_docstring", ""),
            "function": row.get("function", ""),
        },
    )


def _iter_codereviewer(path: Path, prepare_config: dict[str, Any]) -> Iterator[MRExample]:
    if path.is_file():
        for row in iter_jsonl(path):
            yield _compress_example(_normalize_codereviewer_sample(row), prepare_config)
        return

    if (path / "Comment_Generation").is_dir():
        path = path / "Comment_Generation"

    for file_name, split in CODE_REVIEWER_SPLITS.items():
        file_path = path / file_name
        if not file_path.exists():
            continue
        limit = int(prepare_config.get(f"{split}_limit", 0)) or None
        emitted = 0
        for row in iter_jsonl(file_path):
            example = _normalize_codereviewer_real(row, split)
            if example is None:
                continue
            yield _compress_example(example, prepare_config)
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def _iter_codereviewqa(path: Path, prepare_config: dict[str, Any]) -> Iterator[MRExample]:
    if not path.exists():
        return

    if path.is_file():
        first_row = next(iter_jsonl(path, limit=1), None)
        if first_row is None:
            return
        if "question_id" in first_row:
            for row in iter_jsonl(path):
                yield _compress_example(_normalize_codereviewqa_sample(row), prepare_config)
            return

        limit = int(prepare_config.get("validation_limit", 0)) or None
        count = 0
        for row in iter_jsonl(path):
            yield _compress_example(_normalize_codereviewqa_real(row), prepare_config)
            count += 1
            if limit is not None and count >= limit:
                break


def _iter_codocbench(path: Path, prepare_config: dict[str, Any]) -> Iterator[MRExample]:
    if path.is_file():
        for row in iter_jsonl(path):
            yield _compress_example(_normalize_codocbench_sample(row), prepare_config)
        return

    split_files = {
        "train.jsonl": "validation",
        "test.jsonl": "test",
    }
    for file_name, split in split_files.items():
        file_path = path / file_name
        if not file_path.exists():
            continue
        limit = int(prepare_config.get(f"{split}_limit", 0)) or None
        count = 0
        for row in iter_jsonl(file_path):
            yield _compress_example(_normalize_codocbench_real(row, split), prepare_config)
            count += 1
            if limit is not None and count >= limit:
                break


ITERATORS = {
    "codereviewer": _iter_codereviewer,
    "codereviewqa": _iter_codereviewqa,
    "codocbench": _iter_codocbench,
}


def prepare_datasets(
    raw_paths: dict[str, str],
    prepared_dir: Path,
    prepare_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepare_config = prepare_config or {}
    grouped_examples: dict[str, list[MRExample]] = defaultdict(list)
    warnings: list[str] = []
    all_sources: set[str] = set()

    for dataset_name, raw_path in raw_paths.items():
        path = Path(raw_path)
        iterator = ITERATORS[dataset_name]
        if not path.exists():
            warnings.append(f"{dataset_name}: source path not found at {path}")
            continue

        count = 0
        for example in iterator(path, prepare_config):
            grouped_examples[example.split].append(example)
            all_sources.add(example.source_dataset)
            count += 1

        if count == 0:
            warnings.append(f"{dataset_name}: no usable examples were loaded from {path}")

    prepared_dir.mkdir(parents=True, exist_ok=True)

    demo_examples = grouped_examples.pop("demo", [])
    if not demo_examples:
        fallback_demo = None
        for split_name in ("validation", "test", "train"):
            if grouped_examples.get(split_name):
                fallback_demo = grouped_examples[split_name][0]
                break
        if fallback_demo is None:
            raise ValueError("No usable examples were prepared from the configured datasets.")
        demo_examples = [fallback_demo]

    summary = {
        "counts": {split: len(items) for split, items in grouped_examples.items()},
        "sources": sorted(all_sources),
        "demo_example_id": demo_examples[0].example_id,
        "warnings": warnings,
    }

    for split in ("train", "validation", "test"):
        write_jsonl(prepared_dir / f"{split}.jsonl", (example.to_dict() for example in grouped_examples.get(split, [])))

    write_json(prepared_dir / "demo.json", demo_examples[0].to_dict())
    write_json(prepared_dir / "manifest.json", summary)
    return summary
