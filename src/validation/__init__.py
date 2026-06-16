"""Validation helpers."""

try:
    from .metrics import evaluate_examples
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in minimal SWE-CI runner environments.
    if exc.name != "openai":
        raise

    def evaluate_examples(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("evaluate_examples requires the optional 'openai' package.") from exc

__all__ = ["evaluate_examples"]
