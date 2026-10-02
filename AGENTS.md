# Compressor Reflex - AI Agent & IDE Integration Guide (AGENTS.md)

This document provides system prompt instructions and execution guidelines for AI coding agents (Antigravity IDE, Cursor, Claude Code, Windsurf, Roo Code) utilizing the Compressor Reflex MCP Server.

---

## 1. Overview and Core Invariants

Compressor Reflex is a neural line-level extraction model (based on ModernBERT-151M) calibrated to reduce dynamic context bloat from tool returns.

- **Tool Compression Ratio:** ~89.7% reduction on typical code, diff, and log outputs.
- **Anchor Retention Guarantee:** 100.0% retention of compiler diagnostics, test failure traces, target locations, and syntax errors at calibrated threshold tau* = 0.50.
- **Fail-Open Bypass:** Outputs containing <= 5 physical lines or <= 64 tokens automatically bypass neural extraction and return verbatim.
- **Model Distribution:** Weights are hosted on Hugging Face at `aialchemist-dev/compressor-reflex`.

---

## 2. Configuration for Supported Environments

### Antigravity IDE

Add to your workspace or global MCP configuration (`~/.gemini/config/mcp_config.json` or `.gemini/mcp_config.json`):

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

#### Transparent Proxy Mode
To automatically compress outputs from other tools (e.g. filesystem or terminal execution):

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

### Cursor

Add to `.cursor/mcp.json`:

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

Or using `uvx`:

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

### Claude Desktop

Add to `claude_desktop_config.json`:

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

## 3. Tool Calling Conventions

### `compress_tool_output`

Compresses arbitrary multi-line text (such as terminal stdout, test suites, or git diffs).

- **`text` (string, required):** The full raw tool output text.
- **`intent` (string, optional):** Clarifying context describing what the agent is investigating (e.g. "locate authentication failure in unit tests").
- **`threshold` (number, optional):** Decision boundary tau (default: 0.50). Do not alter unless evaluating compression trade-offs.

### `compress_file`

Reads a file from the workspace filesystem and extracts lines matching the intent.

- **`file_path` (string, required):** Path to target file on disk.
- **`intent` (string, optional):** Query describing target information (e.g. "find database connection pool configuration").
- **`threshold` (number, optional):** Extraction threshold (default: 0.50).

### `get_model_status`

Returns runtime metadata including model weight cache status, Hugging Face Hub source URL, and active threshold parameters.

---

## 4. Agent Operational Rules

1. **Selective Invocation:** Use `compress_tool_output` on verbose outputs (> 15 lines) when diagnosing failures or synthesizing broad logs.
2. **Intent Specificity:** When supplying the `intent` parameter, state the target error or concept concisely. Cross-attention routing utilizes intent to bias line-level extraction probabilities.
3. **Respect Built-in Bypass:** The model automatically preserves short snippets (<= 5 lines or <= 64 tokens) without calling the neural head. Manual pre-filtering is unnecessary.
4. **Preserve Calibrated Threshold:** Always use the default threshold of 0.50 for production coding workflows. It has been empirically validated to maintain 100% critical anchor retention.