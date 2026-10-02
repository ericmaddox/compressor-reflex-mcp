"""
Tests for MCP JSON-RPC Server protocol handling.
"""

from compressor_reflex_mcp.server import process_request, TOOLS

def test_initialize():
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }
    resp = process_request(req)
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "compressor-reflex-mcp"
    assert "tools" in resp["result"]["capabilities"]

def test_tools_list():
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    resp = process_request(req)
    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "compress_tool_output" in tool_names
    assert "compress_file" in tool_names
    assert "get_model_status" in tool_names

def test_tool_call_bypass():
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "compress_tool_output",
            "arguments": {
                "text": "short output line 1\nshort output line 2"
            }
        }
    }
    resp = process_request(req)
    assert resp["id"] == 3
    assert resp["result"]["metadata"]["bypass_applied"] is True
