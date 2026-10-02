"""
MCP Server for Compressor Reflex.
Implements Model Context Protocol (JSON-RPC 2.0 over stdio) to expose tool output compression
to Cursor, Antigravity IDE, Claude Desktop, and other MCP clients.
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from compressor_reflex_mcp.engine import get_default_engine, CALIBRATED_THRESHOLD
from compressor_reflex_mcp.model_manager import get_model_info

SERVER_NAME = "compressor-reflex-mcp"
SERVER_VERSION = "0.2.0"

TOOLS = [
    {
        "name": "compress_tool_output",
        "description": (
            "Compress long tool outputs (file views, terminal logs, test failures, git diffs, directory trees) "
            "using the fine-tuned Compressor Reflex model. Preserves 100% of critical anchors, error traces, and targets "
            "while reducing tokens by up to 90%. Outputs <= 5 lines or <= 64 tokens are passed through verbatim."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The raw tool output text or terminal output to compress."
                },
                "intent": {
                    "type": "string",
                    "description": "Optional context or user intent explaining what the agent is searching for (e.g. 'check test failures in auth')."
                },
                "threshold": {
                    "type": "number",
                    "description": "Extraction decision threshold tau (default: 0.50 calibrated). Higher = more aggressive compression.",
                    "default": CALIBRATED_THRESHOLD
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "compress_file",
        "description": "Reads a file from disk and compresses its contents using Compressor Reflex, keeping only relevant lines.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the local file to read and compress."
                },
                "intent": {
                    "type": "string",
                    "description": "Optional query or task intent (e.g. 'find database connection settings')."
                },
                "threshold": {
                    "type": "number",
                    "description": "Extraction threshold tau (default: 0.50).",
                    "default": CALIBRATED_THRESHOLD
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "get_model_status",
        "description": "Returns status of the Compressor Reflex ONNX model, Hugging Face Hub source, and cache path.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

def handle_compress_tool_output(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text", "")
    intent = args.get("intent", "")
    threshold = float(args.get("threshold", CALIBRATED_THRESHOLD))

    engine = get_default_engine()
    result = engine.compress(text=text, intent=intent, threshold=threshold)

    summary_header = (
        f"[Compressor Reflex: {result['raw_tokens']} -> {result['kept_tokens']} tokens "
        f"({round(result['compression_ratio'] * 100, 1)}% saved) | "
        f"latency: {result['compressor_latency_ms']}ms"
        f"{' | BYPASS' if result['bypass_applied'] else ''}]\n"
    )

    return {
        "content": [
            {
                "type": "text",
                "text": result["compressed_text"]
            }
        ],
        "metadata": {
            "raw_tokens": result["raw_tokens"],
            "kept_tokens": result["kept_tokens"],
            "compression_ratio": result["compression_ratio"],
            "bypass_applied": result["bypass_applied"],
            "latency_ms": result["compressor_latency_ms"]
        }
    }

def handle_compress_file(args: Dict[str, Any]) -> Dict[str, Any]:
    file_path = args.get("file_path", "")
    intent = args.get("intent", "")
    threshold = float(args.get("threshold", CALIBRATED_THRESHOLD))

    p = Path(file_path).resolve()
    if not p.exists() or not p.is_file():
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Error: File not found: {file_path}"}]
        }

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Error reading {file_path}: {e}"}]
        }

    return handle_compress_tool_output({"text": content, "intent": intent or f"viewing {p.name}", "threshold": threshold})

def handle_get_model_status(_args: Dict[str, Any]) -> Dict[str, Any]:
    info = get_model_info()
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(info, indent=2)
            }
        ]
    }

def process_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = request.get("method")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }

    elif method in ["notifications/initialized", "initialized"]:
        return None

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS
            }
        }

    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "compress_tool_output":
            res = handle_compress_tool_output(args)
        elif tool_name == "compress_file":
            res = handle_compress_file(args)
        elif tool_name == "get_model_status":
            res = handle_get_model_status(args)
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": res
        }

    else:
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        return None

def run_stdio_server():
    """Main stdio loop for MCP server."""
    # Ensure stdout is in unbuffered utf-8 mode
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    print(f"[{SERVER_NAME}] Starting stdio MCP server v{SERVER_VERSION}...", file=sys.stderr)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            try:
                req = json.loads(line_str)
            except Exception as e:
                print(f"[{SERVER_NAME}] Failed to parse JSON: {e}", file=sys.stderr)
                continue

            resp = process_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{SERVER_NAME}] Server error: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_stdio_server()
