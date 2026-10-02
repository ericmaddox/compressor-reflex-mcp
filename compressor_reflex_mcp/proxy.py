"""
Transparent MCP Proxy Middleware for Compressor Reflex.
Wraps any existing stdio MCP server command (e.g. filesystem, git, bash, fetch)
and transparently compresses tool output text before returning it to the IDE.
"""

import sys
import json
import subprocess
import threading
from typing import Dict, Any, Optional

from compressor_reflex_mcp.engine import get_default_engine

def run_proxy(child_command: list):
    """
    Runs child_command as an MCP subprocess, intercepting tools/call responses
    and applying Compressor Reflex compression to text results.
    """
    print(f"[CompressorProxy] Launching wrapped MCP server: {' '.join(child_command)}", file=sys.stderr)

    proc = subprocess.Popen(
        child_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        encoding="utf-8",
        bufsize=1
    )

    pending_tool_calls: Dict[Any, str] = {} # id -> tool_name
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

                # If this is a response to tools/call, compress text content blocks
                if tool_name and "result" in data and isinstance(data["result"], dict):
                    res = data["result"]
                    contents = res.get("content", [])
                    if isinstance(contents, list):
                        eng = get_engine_lazy()
                        for item in contents:
                            if isinstance(item, dict) and item.get("type") == "text":
                                raw_text = item.get("text", "")
                                # Compress output
                                comp_res = eng.compress(
                                    text=raw_text,
                                    intent=f"Tool call: {tool_name}"
                                )
                                item["text"] = comp_res["compressed_text"]

                sys.stdout.write(json.dumps(data) + "\n")
                sys.stdout.flush()
        except Exception as e:
            print(f"[CompressorProxy] Error reading child stdout: {e}", file=sys.stderr)

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
                if data.get("method") == "tools/call":
                    req_id = data.get("id")
                    tool_name = data.get("params", {}).get("name", "tool")
                    with lock:
                        pending_tool_calls[req_id] = tool_name
            except Exception:
                pass

            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write(line)
                proc.stdin.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
        t.join(timeout=1.0)
