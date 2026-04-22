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

from src.config import apply_llm_provider, load_config, load_dotenv
from src.models.llm import GENERATOR_SCHEMA
from src.models.llm import build_llm_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local OpenAI-compatible LLM server.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--llm-provider", default="", help="Optional provider from llm_providers, e.g. qwen_cloud.")
    parser.add_argument("--chat", action="store_true", help="Run a tiny JSON chat completion smoke check.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config(PROJECT_ROOT / args.config)
    config = apply_llm_provider(config, args.llm_provider)
    client = build_llm_client(config, PROJECT_ROOT)
    try:
        status = client.health_check()
    except Exception as exc:  # noqa: BLE001 - CLI should explain local server issues clearly.
        print(
            json.dumps(
                {
                    "ok": False,
                    "base_url": dict(config.get("llm", {})).get("base_url", ""),
                    "provider": dict(config.get("llm", {})).get("provider", ""),
                    "error": str(exc),
                    "hint": "Start the configured OpenAI-compatible server or check provider secrets in .env.",
                },
                indent=2,
            )
        )
        raise SystemExit(1)

    status["ok"] = bool(status["model_available"])
    if args.chat and status["model_available"]:
        response = client.chat_json(
            role="health_check",
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {
                    "role": "user",
                    "content": 'Return {"comments": []} as JSON.',
                },
            ],
            response_schema=GENERATOR_SCHEMA,
            temperature=0.0,
            max_tokens=60,
        )
        status["chat_ok"] = not response.parse_error
        status["chat_error"] = response.error
        status["chat_cache_hit"] = response.cache_hit
        status["chat_usage"] = response.usage
        status["ok"] = status["ok"] and status["chat_ok"]
    print(json.dumps(status, indent=2))
    if not status["model_available"]:
        print(
            f"Configured model '{status['configured_model']}' was not found. "
            "Use one of the available model ids above in configs/base.yaml or the selected provider.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.chat and not status.get("chat_ok", False):
        print(
            "Model is listed, but chat completion failed. Check provider quota, billing/free-token activation, "
            "model access, or response_format support.",
            file=sys.stderr,
        )
        raise SystemExit(3)


if __name__ == "__main__":
    main()
