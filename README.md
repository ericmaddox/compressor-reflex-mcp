# Compressor Reflex MCP ⚡

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-aialchemist--dev%2Fcompressor--reflex-yellow)](https://huggingface.co/aialchemist-dev/compressor-reflex)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Protocol%20Compatible-green.svg)](https://modelcontextprotocol.io/)

**Compressor Reflex MCP** is a Model Context Protocol (MCP) server and transparent proxy that brings ultra-high-fidelity tool output compression directly into **Cursor**, **Antigravity IDE**, **Claude Desktop**, and other MCP-enabled AI coding environments.

Powered by [`aialchemist-dev/compressor-reflex`](https://huggingface.co/aialchemist-dev/compressor-reflex) (fine-tuned ModernBERT-151M), it reduces tool-output tokens by up to **90%** while preserving **100%** of compiler errors, test failures, target locations, and decisive anchors.

---

## 🌟 Key Features

- **⚡ 89.7% Empirical Tool Token Reduction:** Cuts massive test logs, git diffs, directory trees, and file dumps down to the lines that actually matter.
- **🎯 100% Critical Info Retention:** Validated at calibrated threshold $\tau^* = 0.50$ across held-out evaluation sets. Zero loss of error tracebacks or must-keep anchors.
- **🛡️ Fail-Open Deployment Bypass:** Outputs with **$\le 5$ physical lines** or **$\le 64$ tokens** are passed through verbatim. Zero overhead on short commands or simple edits.
- **🔌 Dual-Mode Integration:**
  1. **Direct MCP Tools:** Exposes `compress_tool_output` and `compress_file` for deliberate agent use.
  2. **Transparent Proxy:** Wraps any existing MCP server (like filesystem, git, bash) and automatically compresses text responses before they reach the model.
- **📦 Zero-Friction Weight Management:** Automatically downloads and caches pre-quantized INT8 ONNX weights from Hugging Face on first run.

---

## 🚀 Quickstart

### 1. Installation

Install via `pip`:
```bash
pip install git+https://github.com/ericmaddox/compressor-reflex-mcp.git
```

Or clone and install locally:
```bash
git clone https://github.com/ericmaddox/compressor-reflex-mcp.git
cd compressor-reflex-mcp
pip install -e .
```

*(Optional)* Pre-download the model weights from Hugging Face:
```bash
compressor-reflex-mcp download
```

---

## 🛠️ IDE Configuration

### Cursor Setup

Add the server to `.cursor/mcp.json` in your workspace or global settings:

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

### Antigravity IDE Setup

Add to `~/.gemini/config/mcp_config.json` or your project's `.gemini/mcp_config.json`:

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

#### Transparent Proxy Mode (Antigravity & Cursor)
To automatically compress all outputs from standard tools (such as filesystem or git):
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

### Claude Desktop Setup

Edit `claude_desktop_config.json`:
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

## 🧰 Available MCP Tools

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `compress_tool_output` | `text` *(req)*, `intent` *(opt)*, `threshold` *(opt)* | Compresses terminal output, logs, or git diffs with optional intent guidance. |
| `compress_file` | `file_path` *(req)*, `intent` *(opt)*, `threshold` *(opt)* | Reads and extracts decisive lines from a workspace file. |
| `get_model_status` | *(none)* | Inspects model cache, Hugging Face Hub link, and calibrated threshold. |

---

## 💻 CLI Usage

```bash
# Start the stdio MCP server
compressor-reflex-mcp serve

# Transparent proxy wrapping another MCP server
compressor-reflex-mcp proxy -- npx -y @modelcontextprotocol/server-filesystem /path/to/project

# Compress a file directly from terminal
compressor-reflex-mcp compress path/to/pytest_output.log --intent "find failures"

# Check model cache and configuration
compressor-reflex-mcp info
```

---

## 📊 Measured Performance

Empirical results from real multi-turn coding sessions:

| Metric | Measured Value | Note |
| :--- | :---: | :--- |
| **Tool Compression Ratio** | **89.69%** | Measured on 181 tool outputs (129k $\rightarrow$ 13k tokens) |
| **Anchor Retention** | **100.0%** | 611/611 critical lines preserved at $\tau^* = 0.50$ |
| **Bypass Rate** | **9.39%** | Bypasses $\le 5$ physical lines or $\le 64$ tokens |
| **Peak Session Savings** | **60.54%** | Measured in deep exploratory coding session |
| **Model Size** | **143 MB** | INT8 ONNX (ModernBERT-151M) |

---

## 🤖 Instructions for AI Agents

For detailed system prompt instructions and agent workflows, see [`AGENTS.md`](./AGENTS.md).

---

## 📄 License

MIT License. See [LICENSE](./LICENSE) for details.
Model weights hosted on [Hugging Face](https://huggingface.co/aialchemist-dev/compressor-reflex).
