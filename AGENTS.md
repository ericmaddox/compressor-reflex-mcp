# Compressor Reflex — AI Agent & IDE Integration Guide (`AGENTS.md`)

This guide is designed for **AI coding agents** (Antigravity IDE, Cursor, Claude Code, Windsurf, Roo Code) and developer configuration. It documents how to configure, call, and benefit from the **Compressor Reflex MCP Server**.

---

## 1. What is Compressor Reflex?

**Compressor Reflex** is a specialized, neural line-level extraction model (based on ModernBERT-151M) designed to solve the context explosion problem in multi-turn coding sessions.

### Key Capabilities
- **89.7% Tool Output Compression:** Compresses terminal logs, git diffs, directory trees, and file contents down to their essential lines.
- **100.0% Critical Anchor Retention:** Guaranteed retention on compiler errors, pytest failure lines, tracebacks, and target locations (verified at calibrated threshold $\tau^* = 0.50$).
- **Fail-Open Bypass Policy:** Any output with **$\le 5$ physical lines** OR **$\le 64$ tokens** automatically bypasses compression and passes through verbatim. Zero overhead on short outputs.
- **Weights on Hugging Face:** Pre-trained weights hosted at [`aialchemist-dev/compressor-reflex`](https://huggingface.co/aialchemist-dev/compressor-reflex) and cached automatically on first run.

---

## 2. Setting Up in Your IDE

### A. Antigravity IDE Setup

Add to your workspace or global MCP configuration:

**Option 1: Global Config (`~/.gemini/config/mcp_config.json`) or Workspace Root (`.gemini/mcp_config.json`):**
```json
{
  "mcpServers": {
    "compressor-reflex": {
      "command": "python",
      "args": ["-m", "compressor_reflex_mcp.server"],
      "env": {
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

**Option 2: Transparent Proxy for Other MCP Tools:**
Wrap filesystem, git, or terminal tools so their outputs are automatically compressed before hitting context:
```json
{
  "mcpServers": {
    "filesystem-compressed": {
      "command": "compressor-reflex-mcp",
      "args": [
        "proxy",
        "--",
        "npx",
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "."
      ]
    }
  }
}
```

---

### B. Cursor Setup

Add to your project's `.cursor/mcp.json` or Cursor Global Settings (`Cursor Settings -> Features -> MCP`):

```json
{
  "mcpServers": {
    "compressor-reflex": {
      "command": "python",
      "args": ["-m", "compressor_reflex_mcp.server"]
    }
  }
}
```

If using a dedicated virtual environment or `uv`:
```json
{
  "mcpServers": {
    "compressor-reflex": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/ericmaddox/compressor-reflex-mcp", "compressor-reflex-mcp"]
    }
  }
}
```

---

### C. Claude Desktop Setup

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "compressor-reflex": {
      "command": "python",
      "args": ["-m", "compressor_reflex_mcp.server"]
    }
  }
}
```

---

## 3. MCP Tools Reference

The server exposes 3 standard tools:

### `compress_tool_output`
Compresses arbitrary text or terminal logs with optional search intent.

**Arguments:**
- `text` *(string, required)*: The raw multi-line tool output or terminal stdout.
- `intent` *(string, optional)*: High-level goal or search context (e.g. `"check pytest failures in auth_flow"`).
- `threshold` *(number, optional)*: Extraction decision threshold $\tau$ (default: `0.50`).

**Example Call:**
```json
{
  "name": "compress_tool_output",
  "arguments": {
    "text": "PASSED test_1\nPASSED test_2\nFAILED test_auth.py::test_login - Token expired\nTraceback...\n",
    "intent": "locate auth failure"
  }
}
```

### `compress_file`
Reads a file from the local workspace and compresses its contents according to intent.

**Arguments:**
- `file_path` *(string, required)*: Path to file on disk.
- `intent` *(string, optional)*: What information to extract.
- `threshold` *(number, optional)*: Extraction threshold (default: `0.50`).

### `get_model_status`
Returns information regarding the Hugging Face repository source, local model cache status, and calibrated threshold.

---

## 4. Agent Guidelines & Behavioral Instructions

When operating as an autonomous coding agent, adopt these best practices:

1. **Long File / Output Filtering:** When viewing large files (> 150 lines) or terminal logs where you only need specific logic or failure messages, invoke `compress_tool_output` or `compress_file`.
2. **Provide Search Intent:** Supplying an `intent` argument (e.g. `"find database port config"` or `"check missing import error"`) improves cross-attention scoring on targeted lines.
3. **Respect Fail-Open Bypass:** Do not attempt to pre-filter small outputs manually. The model automatically bypasses any output with $\le 5$ physical lines or $\le 64$ tokens.
4. **Never Alter Calibrated $\tau^*$:** The threshold $\tau^* = 0.50$ is mathematically calibrated for 100% retention on held-out evaluations. Avoid lowering or raising unless specifically testing compression trade-offs.
