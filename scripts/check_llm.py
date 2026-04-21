"""Check local LM Studio / OpenAI-compatible model availability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.config import load_config
from src.models.llm import build_llm_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local OpenAI-compatible LLM server.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    client = build_llm_client(config, PROJECT_ROOT)
    try:
        status = client.health_check()
    except Exception as exc:  # noqa: BLE001 - CLI should explain local server issues clearly.
        print(
            json.dumps(
                {
                    "ok": False,
                    "base_url": dict(config.get("llm", {})).get("base_url", ""),
                    "error": str(exc),
                    "hint": "Start LM Studio local server and load the configured Qwen model.",
                },
                indent=2,
            )
        )
        raise SystemExit(1)

    status["ok"] = bool(status["model_available"])
    print(json.dumps(status, indent=2))
    if not status["model_available"]:
        print(
            f"Configured model '{status['configured_model']}' was not found. "
            "Use one of the available model ids above in configs/base.yaml.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
