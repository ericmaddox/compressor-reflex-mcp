"""
Security Verification Test Suite for Compressor Reflex MCP.
Validates path traversal prevention, sensitive file blacklists, payload DoS limits,
intent sanitization, threshold bounds, and cryptographic integrity.
"""

import os
import math
import tempfile
from pathlib import Path

from compressor_reflex_mcp.server import is_path_safe, handle_compress_file, process_request
from compressor_reflex_mcp.serializer import sanitize_intent, chunk_tool_output
from compressor_reflex_mcp.engine import validate_threshold, CALIBRATED_THRESHOLD, CompressorEngine
from compressor_reflex_mcp.model_manager import check_local_model_dir, VERIFIED_SHA256

def test_sensitive_file_blocking():
    """Verify that path traversal to sensitive credential stores is denied."""
    blocked_test_cases = [
        Path(".env"),
        Path("subfolder/.env"),
        Path("id_rsa"),
        Path("~/.ssh/id_rsa"),
        Path("/etc/passwd"),
        Path("credentials.json"),
        Path("server.key"),
        Path("cert.pem"),
        Path("C:/Windows/System32/config/SAM"),
    ]

    for p in blocked_test_cases:
        safe, reason = is_path_safe(p)
        assert not safe, f"Path should have been blocked: {p}"
        assert "Access denied" in reason

def test_workspace_boundary_enforcement():
    """Verify that paths outside the workspace root are denied by default."""
    # Test file inside cwd
    inside_file = Path.cwd() / "allowed_test.txt"
    inside_file.write_text("safe content")
    try:
        safe, _ = is_path_safe(inside_file)
        assert safe, "Files within workspace should be allowed"
    finally:
        if inside_file.exists():
            inside_file.unlink()

    # Test file outside cwd (e.g. system temp dir)
    outside_file = Path(tempfile.gettempdir()) / "outside_root.txt"
    outside_file.write_text("unsafe content")
    try:
        safe, reason = is_path_safe(outside_file)
        assert not safe, "Files outside workspace must be blocked by default"
        assert "outside allowed workspace" in reason
    finally:
        if outside_file.exists():
            outside_file.unlink()

def test_threshold_validation():
    """Verify threshold parameter cannot be corrupted with NaN, Inf, or out-of-bounds numbers."""
    assert validate_threshold(0.50) == 0.50
    assert validate_threshold("0.75") == 0.75
    assert validate_threshold(-0.1) == 0.0
    assert validate_threshold(1.5) == 1.0
    assert validate_threshold(float("nan")) == CALIBRATED_THRESHOLD
    assert validate_threshold(float("inf")) == CALIBRATED_THRESHOLD
    assert validate_threshold(float("-inf")) == CALIBRATED_THRESHOLD
    assert validate_threshold("malicious_string") == CALIBRATED_THRESHOLD
    assert validate_threshold(None) == CALIBRATED_THRESHOLD

def test_intent_sanitization():
    """Verify intent is cleansed of null bytes, control chars, and capped at max length."""
    dirty_intent = "find error\x00\x07\x1b[31m and secret" + ("A" * 1000)
    cleaned = sanitize_intent(dirty_intent)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert len(cleaned) <= 512
    assert cleaned.startswith("find error")

def test_file_size_limit():
    """Verify reading oversized files returns a security error rather than exhausting memory."""
    dummy_file = Path.cwd() / "oversized_test_dummy.bin"
    with open(dummy_file, "wb") as f:
        f.seek(11 * 1024 * 1024)
        f.write(b"\0")

    try:
        res = handle_compress_file({"file_path": str(dummy_file)})
        assert res.get("isError") is True
        assert "Security Error: File size" in res["content"][0]["text"]
    finally:
        if dummy_file.exists():
            dummy_file.unlink()

def test_mcp_invalid_request_handling():
    """Verify server returns clean JSON-RPC errors on malformed payloads."""
    resp = process_request("string_not_json")
    assert resp["error"]["code"] == -32600

    resp2 = process_request({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/call",
        "params": {"name": "non_existent_tool"}
    })
    assert resp2["error"]["code"] == -32601