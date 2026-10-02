"""
Tests for Compressor Reflex Engine and Fail-Open Bypass Policy.
"""

# no external test framework required
from compressor_reflex_mcp.engine import CompressorEngine, CALIBRATED_THRESHOLD

def test_bypass_policy():
    """Verify <= 5 lines or <= 64 tokens triggers bypass."""
    # Test short 3-line output
    short_text = "line 1\nline 2\nline 3"
    engine = CompressorEngine()
    is_bypass, lines, tokens = engine.check_bypass(short_text)
    assert is_bypass is True
    assert lines == 3

    res = engine.compress(short_text)
    assert res["bypass_applied"] is True
    assert res["compressed_text"] == short_text
    assert res["compression_ratio"] == 0.0

def test_long_output_compression():
    """Verify long output triggers neural compression."""
    long_output = "\n".join([
        "PASSED tests/test_core.py::test_init",
        "PASSED tests/test_core.py::test_load",
        "FAILED tests/test_auth.py::test_login - ValueError: Invalid password",
        "Traceback (most recent call last):",
        "  File '/app/auth.py', line 45, in test_login",
        "    raise ValueError('Invalid password')",
        "ValueError: Invalid password",
    ] + [f"INFO log line {i}: routine health check normal status" for i in range(40)])

    engine = CompressorEngine()
    res = engine.compress(long_output, intent="check test failure in auth")
    assert res["bypass_applied"] is False
    assert res["compression_ratio"] > 0.50
    # Must preserve the failure anchor
    assert "FAILED tests/test_auth.py::test_login" in res["compressed_text"]
