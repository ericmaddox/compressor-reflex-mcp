"""
Transparent MCP Proxy Middleware for Compressor Reflex.
Wraps any existing stdio MCP server command (e.g. filesystem, git, bash, fetch)
and transparently compresses tool output text before returning it to the IDE.
Includes resilient process lifecycle management and unicode decode safety.
"""

import sys
import json
import subprocess
import threading
from typing import Dict, Any, List

from compressor_reflex_mcp.engine import get_default_engine

def run_proxy(child_command: List[str]):
    """
    Runs child_command as an MCP subprocess, intercepting tools/call responses
    and applying Compressor Reflex compression to text results.
    """
    if not child_command or not isinstance(child_command, list):
        print("[CompressorProxy Security] Error: child_command must be a non-empty list of command arguments.", file=sys.stderr)
        sys.exit(1)

    # Sanitize and ensure all elements are strings
    safe_command = [str(arg) for arg in child_command]

    print(f"[CompressorProxy] Launching wrapped MCP server: {' '.join(safe_command)}", file=sys.stderr)

    try:
        proc = subprocess.Popen(
            safe_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding="utf-8",
            errors="replace", # Security: prevent crashes on binary or invalid unicode output
            shell=False,      # Security: strictly disallow shell execution
            bufsize=1
        )
    except Exception as e:
        print(f"[CompressorProxy] Failed to launch child process: {e}", file=sys.stderr)
        sys.exit(1)

    pending_tool_calls: Dict[Any, str] = {}
    lock = threading.Lock()
    engine = None

    def get_engine_lazy():
        nonlocal engine
        if engine is None:
            engine = get_default_engine()
        return engine

    def read_child_stdout():
        try:
            for line in iter(proc.stdout.readline, ""):
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                except Exception:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue

                req_id = data.get("id")
                with lock:
                    tool_name = pending_tool_calls.pop(req_id, None)

                # If this is a successful response to tools/call, compress text content blocks
                if tool_name and "result" in data and isinstance(data["result"], dict):
                    res = data["result"]
                    # Do not compress if tool returned an error (preserve error fidelity)
                    if not res.get("isError"):
                        contents = res.get("content", [])
                        if isinstance(contents, list):
                            try:
                                eng = get_engine_lazy()
                                for item in contents:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        raw_text = item.get("text", "")
                                        comp_res = eng.compress(
                                            text=raw_text,
                                            intent=f"Tool call: {tool_name}"
                                        )
                                        item["text"] = comp_res["compressed_text"]
                            except Exception as comp_err:
                                print(f"[CompressorProxy] Warning: compression failed: {comp_err}", file=sys.stderr)

                sys.stdout.write(json.dumps(data) + "\n")
                sys.stdout.flush()
        except Exception as e:
            print(f"[CompressorProxy] Stream handler closed: {e}", file=sys.stderr)

    t = threading.Thread(target=read_child_stdout, daemon=True)
    t.start()

    # Read from client (IDE) and forward to child
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
                if isinstance(data, dict) and data.get("method") == "tools/call":
                    req_id = data.get("id")
                    tool_name = data.get("params", {}).get("name", "tool")
                    with lock:
                        pending_tool_calls[req_id] = tool_name
            except Exception:
                pass

            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.write(line)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        t.join(timeout=1.0)