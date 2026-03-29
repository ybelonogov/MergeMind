"""Diff parsing and lightweight structural context extraction."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src.data.schema import ChangedFile, DiffHunk, MRExample

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PYTHON_KEYWORDS = {
    "and",
    "as",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
}


def _infer_language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
    }.get(suffix, "text")


def _extract_identifiers(lines: list[str]) -> list[str]:
    seen: list[str] = []
    for line in lines:
        for identifier in IDENTIFIER_RE.findall(line):
            if identifier in PYTHON_KEYWORDS:
                continue
            if identifier not in seen:
                seen.append(identifier)
    return seen


def _extract_python_symbols(source: str) -> list[str]:
    try:
        parsed = ast.parse(source)
    except SyntaxError:
        return []

    symbols: list[str] = []
    for node in ast.walk(parsed):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
    return symbols


def _extract_tree_sitter_symbols(source: str) -> list[str]:
    try:
        import tree_sitter  # type: ignore # noqa: F401
    except ImportError:
        return []
    return []


def _extract_structural_symbols(path: str, source: str) -> list[str]:
    tree_sitter_symbols = _extract_tree_sitter_symbols(source)
    if tree_sitter_symbols:
        return tree_sitter_symbols

    language = _infer_language(path)
    if language == "python":
        return _extract_python_symbols(source)

    symbols: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "function ")):
            parts = stripped.split("(", 1)[0].split()
            if parts:
                symbol = parts[-1].rstrip(":")
                if symbol not in symbols:
                    symbols.append(symbol)
    return symbols


def _select_repository_snippet(source: str, identifiers: list[str], symbols: list[str]) -> str:
    if not source:
        return ""

    interesting_terms = identifiers[:5] + symbols[:5]
    lines = source.splitlines()
    selected: list[str] = []

    if interesting_terms:
        for line in lines:
            if any(term in line for term in interesting_terms):
                selected.append(line)
            if len(selected) >= 8:
                break

    if not selected:
        selected = lines[:8]

    return "\n".join(selected)


def parse_diff(diff_text: str, repository_files: dict[str, str] | None = None) -> list[ChangedFile]:
    repository_files = repository_files or {}
    changed_files: list[ChangedFile] = []
    current_file: ChangedFile | None = None
    current_hunk: DiffHunk | None = None

    def finalize_hunk() -> None:
        nonlocal current_hunk, current_file
        if current_file is not None and current_hunk is not None:
            current_file.hunks.append(current_hunk)
            current_hunk = None

    def finalize_file() -> None:
        nonlocal current_file
        if current_file is None:
            return
        source = repository_files.get(current_file.path, "")
        current_file.language = _infer_language(current_file.path)
        identifiers = []
        for hunk in current_file.hunks:
            identifiers.extend(_extract_identifiers(hunk.added_lines + hunk.removed_lines))
        current_file.changed_identifiers = list(dict.fromkeys(identifiers))
        current_file.structural_symbols = _extract_structural_symbols(current_file.path, source)
        current_file.repository_snippet = _select_repository_snippet(
            source=source,
            identifiers=current_file.changed_identifiers,
            symbols=current_file.structural_symbols,
        )
        changed_files.append(current_file)
        current_file = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            finalize_hunk()
            finalize_file()
            current_file = ChangedFile(path="unknown")
            continue

        if current_file is None:
            current_file = ChangedFile(path="unknown")

        if line.startswith("+++ b/"):
            current_file.path = line[6:].strip()
            continue
        if line.startswith("@@"):
            finalize_hunk()
            current_hunk = DiffHunk(header=line)
            continue
        if current_hunk is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current_hunk.added_lines.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            current_hunk.removed_lines.append(line[1:])
        else:
            current_hunk.context_lines.append(line[1:] if line.startswith(" ") else line)

    finalize_hunk()
    finalize_file()
    return changed_files


def _build_repository_context(changed_files: list[ChangedFile]) -> str:
    sections: list[str] = []
    for changed_file in changed_files:
        identifiers = ", ".join(changed_file.changed_identifiers[:6]) or "n/a"
        symbols = ", ".join(changed_file.structural_symbols[:6]) or "n/a"
        snippet = changed_file.repository_snippet or "No repository snippet available."
        sections.append(
            "\n".join(
                [
                    f"File: {changed_file.path}",
                    f"Identifiers: {identifiers}",
                    f"Symbols: {symbols}",
                    "Snippet:",
                    snippet,
                ]
            )
        )
    return "\n\n".join(sections)


def enrich_example(example: MRExample) -> MRExample:
    changed_files = parse_diff(example.diff, example.repository_files)
    example.changed_files = changed_files
    example.repository_context = _build_repository_context(changed_files)
    return example
