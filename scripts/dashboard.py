"""Serve the local MergeMind dashboard."""

from __future__ import annotations

import argparse
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path


def _bootstrap_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _bootstrap_path()

from src.config import apply_llm_provider, load_config, load_dotenv
from src.monitoring.dashboard import make_dashboard_handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the MergeMind local monitoring dashboard.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host.")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port.")
    parser.add_argument("--refresh-seconds", type=int, default=3, help="Browser auto-refresh interval.")
    parser.add_argument("--llm-provider", default="", help="Optional provider from llm_providers.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config(PROJECT_ROOT / args.config)
    config = apply_llm_provider(config, args.llm_provider)
    handler = make_dashboard_handler(config, PROJECT_ROOT, refresh_seconds=max(args.refresh_seconds, 1))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[dashboard] serving {url}")
    print("[dashboard] press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
