"""
CLI Entry Point for Compressor Reflex MCP.
Usage:
  compressor-reflex-mcp serve
  compressor-reflex-mcp proxy -- <command> [args...]
  compressor-reflex-mcp download
  compressor-reflex-mcp info
  compressor-reflex-mcp compress <file_path> [--intent "search intent"]
"""

import sys
import argparse
from pathlib import Path

from compressor_reflex_mcp.model_manager import ensure_model_files, get_model_info
from compressor_reflex_mcp.server import run_stdio_server
from compressor_reflex_mcp.proxy import run_proxy
from compressor_reflex_mcp.engine import get_default_engine

MAX_CLI_READ_BYTES = 10 * 1024 * 1024 # 10 MB limit

def main():
    parser = argparse.ArgumentParser(
        prog="compressor-reflex-mcp",
        description="Compressor Reflex MCP — Model Context Protocol Server & Proxy for High-Fidelity Token Compression"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # serve command
    subparsers.add_parser("serve", help="Run the stdio MCP server for Cursor, Antigravity, and Claude Desktop")

    # proxy command
    proxy_parser = subparsers.add_parser("proxy", help="Run as transparent MCP proxy wrapping another MCP command")
    proxy_parser.add_argument("proxy_args", nargs=argparse.REMAINDER, help="Target MCP command preceded by --")

    # download command
    subparsers.add_parser("download", help="Pre-download and cache model weights from Hugging Face Hub")

    # info command
    subparsers.add_parser("info", help="Display model repository info, local cache location, and threshold parameters")

    # compress command
    comp_parser = subparsers.add_parser("compress", help="Compress a file or text from command line")
    comp_parser.add_argument("file", type=str, help="Path to file to compress (or '-' for stdin)")
    comp_parser.add_argument("--intent", type=str, default="", help="Optional extraction intent")
    comp_parser.add_argument("--threshold", type=float, default=0.50, help="Extraction threshold tau (default: 0.50)")

    args = parser.parse_args()

    cmd = args.command or "serve"

    if cmd == "serve":
        run_stdio_server()

    elif cmd == "proxy":
        proxy_cmd = args.proxy_args
        if proxy_cmd and proxy_cmd[0] == "--":
            proxy_cmd = proxy_cmd[1:]
        if not proxy_cmd:
            print("Error: Please specify the child MCP command to wrap, e.g.:", file=sys.stderr)
            print("  compressor-reflex-mcp proxy -- npx -y @modelcontextprotocol/server-filesystem /path", file=sys.stderr)
            sys.exit(1)
        run_proxy(proxy_cmd)

    elif cmd == "download":
        print("Ensuring Compressor Reflex model weights are cached and verified locally...")
        path = ensure_model_files(force_download=True)
        print(f"Model successfully verified and cached at: {path}")

    elif cmd == "info":
        info = get_model_info()
        print("\n--- Compressor Reflex MCP Info ---")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("  calibrated_threshold: 0.50")
        print("  fail_open_policy: <= 5 lines or <= 64 tokens\n")

    elif cmd == "compress":
        if args.file == "-":
            text = sys.stdin.read(MAX_CLI_READ_BYTES)
        else:
            p = Path(args.file)
            if not p.exists() or not p.is_file():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            if p.stat().st_size > MAX_CLI_READ_BYTES:
                print(f"Error: File size ({p.stat().st_size:,} bytes) exceeds safety limit ({MAX_CLI_READ_BYTES:,} bytes)", file=sys.stderr)
                sys.exit(1)
            text = p.read_text(encoding="utf-8", errors="replace")

        eng = get_default_engine()
        res = eng.compress(text=text, intent=args.intent, threshold=args.threshold)
        print(res["compressed_text"])
        token_stats = (
            "token counts unavailable (resource-limit bypass)"
            if res["raw_tokens"] is None
            else f"{res['raw_tokens']} -> {res['kept_tokens']} tokens"
        )
        print(
            f"\n# Stats: {token_stats} "
            f"({round(res['compression_ratio'] * 100, 1)}% saved) | "
            f"latency: {res['compressor_latency_ms']}ms"
            f"{' | BYPASS' if res['bypass_applied'] else ''}",
            file=sys.stderr
        )

if __name__ == "__main__":
    main()
