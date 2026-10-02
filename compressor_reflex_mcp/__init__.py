"""
Compressor Reflex MCP — Model Context Protocol Server & Proxy for High-Fidelity Tool Output Compression.
Reduces tool-output token consumption by up to 90% in Cursor, Antigravity IDE, and Claude Desktop.
Model weights: https://huggingface.co/aialchemist-dev/compressor-reflex
"""

__version__ = "0.2.1"
__author__ = "Eric Maddox"

from compressor_reflex_mcp.engine import CompressorEngine, get_default_engine
from compressor_reflex_mcp.model_manager import ensure_model_files

__all__ = ["CompressorEngine", "get_default_engine", "ensure_model_files", "__version__"]
