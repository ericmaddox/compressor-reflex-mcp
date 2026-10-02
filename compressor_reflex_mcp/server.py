"""
MCP Server for Compressor Reflex.
Implements Model Context Protocol (JSON-RPC 2.0 over stdio) to expose tool output compression
to Cursor, Antigravity IDE, Claude Desktop, and other MCP clients with enterprise security controls.
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from compressor_reflex_mcp.engine import get_default_engine, CALIBRATED_THRESHOLD, validate_threshold
from compressor_reflex_mcp.model_manager import get_model_info

SERVER_NAME = "compressor-reflex-mcp"
SERVER_VERSION = "0.2.1"

# Security constraints
MAX_FILE_READ_BYTES = 10 * 1024 * 1024  # 10 MB maximum file read
MAX_STDIO_LINE_BYTES = 10 * 1024 * 1024 # 10 MB line limit to prevent stdin memory exhaustion

# Blacklist of sensitive file names, extensions, and directory segments
BLOCKED_PATTERNS = {
    ".env", ".git", ".ssh", ".aws", ".gnupg",
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "credentials", "secrets", "passwd", "shadow", "sam", "system32",
    ".pem", ".key", ".pfx", ".p12", ".kdbx"
}

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
                    "description": "Extraction decision threshold tau (default: 0.50 calibrated). Range [0.0, 1.0].",
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
                    "description": "Path to the local file to read and compress."
                },
                "intent": {
                    "type": "string",
                    "description": "Optional query or task intent (e.g. 'find database connection settings')."
                },
                "threshold": {
                    "type": "number",
                    "description": "Extraction threshold tau (default: 0.50). Range [0.0, 1.0].",
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

def is_path_safe(file_path: Path) -> Tuple[bool, str]:
    """
    Validates file path against directory traversal, sensitive credential stores,
    and configured workspace root boundaries.
    """
    try:
        resolved = file_path.resolve()
    except Exception as e:
        return False, f"Invalid path resolution: {e}"

    # Check for blocked sensitive files or directories
    parts_lower = [part.lower() for part in resolved.parts]
    name_lower = resolved.name.lower()
    suffix_lower = resolved.suffix.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern.startswith("."):
            if suffix_lower == pattern or name_lower == pattern or any(p == pattern for p in parts_lower):
                return False, f"Access denied: reading sensitive credential/config files ({pattern}) is prohibited"
        else:
            if pattern in name_lower or any(pattern in p for p in parts_lower):
                return False, f"Access denied: reading protected path ({pattern}) is prohibited"

    # Workspace containment validation (if COMPRESSOR_WORKSPACE_ROOT is set)
    workspace_root = os.environ.get("COMPRESSOR_WORKSPACE_ROOT")
    if workspace_root:
        root_path = Path(workspace_root).resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError:
            return False, f"Access denied: path is outside allowed workspace root ({root_path})"

    return True, ""

def handle_compress_tool_output(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text", "")
    intent = args.get("intent", "")
    raw_threshold = args.get("threshold", CALIBRATED_THRESHOLD)
    threshold = validate_threshold(raw_threshold)

    engine = get_default_engine()
    result = engine.compress(text=text, intent=intent, threshold=threshold)

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
            "latency_ms": result["compressor_latency_ms"],
            "truncated": result.get("input_truncated", False)
        }
    }

def handle_compress_file(args: Dict[str, Any]) -> Dict[str, Any]:
    file_path = args.get("file_path", "")
    intent = args.get("intent", "")
    raw_threshold = args.get("threshold", CALIBRATED_THRESHOLD)
    threshold = validate_threshold(raw_threshold)

    if not file_path or not isinstance(file_path, str):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Error: Missing or invalid file_path argument"}]
        }

    p = Path(file_path)
    safe, reason = is_path_safe(p)
    if not safe:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Security Error: {reason}"}]
        }

    resolved = p.resolve()
    if not resolved.exists() or not resolved.is_file():
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Error: File not found: {file_path}"}]
        }

    # Size limit guardrail against DoS
    try:
        file_size = resolved.stat().st_size
        if file_size > MAX_FILE_READ_BYTES:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"Security Error: File size ({file_size:,} bytes) exceeds safety limit ({MAX_FILE_READ_BYTES:,} bytes)"
                }]
            }
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Error reading file: {e}"}]
        }

    return handle_compress_tool_output({
        "text": content,
        "intent": intent or f"file contents for {resolved.name}",
        "threshold": threshold
    })

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
    if not isinstance(request, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request: must be a JSON object"}
        }

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
        if not isinstance(params, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params"}
            }

        tool_name = params.get("name")
        args = params.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

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
    """Main stdio loop for MCP server with line-length protection."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    print(f"[{SERVER_NAME}] Starting secure stdio MCP server v{SERVER_VERSION}...", file=sys.stderr)

    while True:
        try:
            line = sys.stdin.readline(MAX_STDIO_LINE_BYTES)
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue

            try:
                req = json.loads(line_str)
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
                continue

            resp = process_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{SERVER_NAME}] Internal server error: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_stdio_server()
