"""Wrapper regressions: no model downloads or inference runtime required."""

import io
import json
import queue
import subprocess
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from compressor_reflex_mcp import cli, engine as engine_module, server
from compressor_reflex_mcp.engine import CompressorEngine
from compressor_reflex_mcp.serializer import (
    build_chunk_model_inputs,
    chunk_tool_output,
)


class CharacterTokenizer:
    """Deterministic tokenizer with explicit special tokens."""

    sep_token_id = 2
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        tokens = [ord(c) + 10 for c in text]
        return [1] + tokens + [2] if add_special_tokens else tokens

    def decode(self, tokens):
        return "".join(chr(t - 10) for t in tokens)


class KeepAllSession:
    def run(self, names, inputs):
        return [np.ones(inputs["line_mask"].shape[0], dtype=np.float32)]


def make_engine():
    engine = CompressorEngine.__new__(CompressorEngine)
    engine.tokenizer = CharacterTokenizer()
    engine.session = KeepAllSession()
    return engine


class OutputPreservationTests(unittest.TestCase):
    def test_selected_long_line_preserves_original_tail(self):
        text = "\n".join(["x" * 3000 + " CRITICAL_ANCHOR"] + ["routine data"] * 6)
        result = make_engine().compress(text)
        self.assertEqual(result["compressed_text"], text)
        self.assertFalse(result["input_truncated"])

    def test_long_line_fallback_preserves_original_tail(self):
        engine = make_engine()
        engine.session = Mock()
        engine.session.run.side_effect = lambda _, inputs: [
            np.zeros(inputs["line_mask"].shape[0], dtype=np.float32)
        ]
        result = engine.compress(
            "\n".join(["x" * 3000 + " ERROR_TAIL"] + ["routine"] * 6)
        )
        self.assertIn("ERROR_TAIL", result["compressed_text"])

    def test_character_limit_bypasses_without_tokenizing_or_truncating(self):
        engine = make_engine()
        engine.tokenizer = Mock()
        engine.tokenizer.encode.side_effect = AssertionError(
            "oversized input was tokenized"
        )
        engine.session = Mock()
        text = "x" * engine_module.MAX_INPUT_CHARS + " ERROR_TAIL"
        result = engine.compress(text)
        self.assertEqual(result["compressed_text"], text)
        self.assertTrue(result["bypass_applied"])
        self.assertFalse(result["input_truncated"])
        self.assertIsNone(result["raw_tokens"])
        self.assertIsNone(result["kept_tokens"])
        self.assertEqual(result["compression_ratio"], 0.0)
        engine.tokenizer.encode.assert_not_called()
        engine.session.run.assert_not_called()

    def test_line_limit_preserves_final_failure_without_inference(self):
        engine = make_engine()
        engine.tokenizer = Mock()
        engine.tokenizer.encode.side_effect = AssertionError(
            "over-limit lines were tokenized"
        )
        engine.session = Mock()
        text = "\n".join(
            ["routine"] * engine_module.MAX_INPUT_LINES + ["FAILED final test"]
        )
        result = engine.compress(text)
        self.assertEqual(result["compressed_text"], text)
        self.assertTrue(result["bypass_applied"])
        self.assertFalse(result["input_truncated"])
        engine.session.run.assert_not_called()

    def test_short_output_remains_verbatim(self):
        text = "one\ntwo\nthree\n"
        result = make_engine().compress(text)
        self.assertEqual(result["compressed_text"], text)
        self.assertTrue(result["bypass_applied"])
        self.assertEqual(result["raw_tokens"], result["kept_tokens"])

    def test_normal_multichunk_output_preserves_order(self):
        text = "\n".join(["line %d: %s" % (i, "data " * 20) for i in range(300)])
        result = make_engine().compress(text)
        self.assertEqual(result["compressed_text"], text)
        self.assertFalse(result["bypass_applied"])


class SequenceBudgetTests(unittest.TestCase):
    def test_long_intent_cannot_exceed_default_sequence_budget(self):
        tokenizer = CharacterTokenizer()
        chunks = chunk_tool_output(["x" * 1775], "i" * 512, tokenizer)
        inputs, spans = build_chunk_model_inputs(chunks[0], tokenizer)
        self.assertLessEqual(len(inputs["input_ids"]), 2048)
        self.assertEqual(len(inputs["input_ids"]), len(inputs["attention_mask"]))
        self.assertTrue(
            all(
                0 <= start < end <= len(inputs["input_ids"]) - 1 for start, end in spans
            )
        )

    def test_small_custom_budget_and_spans_are_enforced(self):
        tokenizer = CharacterTokenizer()
        lines = ["x" * 100, "last line"]
        chunks = chunk_tool_output(lines, "intent " * 80, tokenizer, max_tokens=32)
        self.assertEqual([line for chunk in chunks for line in chunk["lines"]], lines)
        for chunk in chunks:
            inputs, spans = build_chunk_model_inputs(chunk, tokenizer, max_length=32)
            self.assertLessEqual(len(inputs["input_ids"]), 32)
            self.assertEqual(inputs["input_ids"][-1], tokenizer.sep_token_id)
            encoded_text = tokenizer.decode(
                [token for token in inputs["input_ids"] if token > 2]
            )
            self.assertIn("Output:\n", encoded_text)
            self.assertTrue(
                all(
                    0 <= start < end <= len(inputs["input_ids"]) - 1
                    for start, end in spans
                )
            )

    def test_builder_rejects_over_budget_manual_chunk(self):
        with self.assertRaises(ValueError):
            build_chunk_model_inputs(
                {"lines": ["x" * 100]}, CharacterTokenizer(), max_length=32
            )


class StdioErrorTests(unittest.TestCase):
    def test_resource_bypass_metadata_is_explicit(self):
        text = "\n".join(["routine"] * engine_module.MAX_INPUT_LINES + ["FAILED tail"])
        with patch.object(server, "get_default_engine", return_value=make_engine()):
            result = server.handle_compress_tool_output({"text": text})
        self.assertEqual(result["content"][0]["text"], text)
        self.assertEqual(result["metadata"]["bypass_reason"], "input_line_limit")
        self.assertIsNone(result["metadata"]["raw_tokens"])
        self.assertFalse(result["metadata"]["truncated"])

    def test_cli_handles_unknown_token_counts(self):
        text = "\n".join(["routine"] * engine_module.MAX_INPUT_LINES + ["FAILED tail"])
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch.object(cli.sys, "argv", ["compressor-reflex-mcp", "compress", "-"]),
            patch.object(cli.sys, "stdin", io.StringIO(text)),
            patch.object(cli.sys, "stdout", output),
            patch.object(cli.sys, "stderr", errors),
            patch.object(cli, "get_default_engine", return_value=make_engine()),
        ):
            cli.main()
        self.assertEqual(output.getvalue(), text + "\n")
        self.assertIn("token counts unavailable", errors.getvalue())
        self.assertIn("0.0% saved", errors.getvalue())

    def test_model_failure_returns_error_and_next_request_succeeds(self):
        requests = [
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {
                    "name": "compress_tool_output",
                    "arguments": {"text": "hello"},
                },
            },
            {"jsonrpc": "2.0", "id": 43, "method": "ping"},
        ]
        output = io.StringIO()
        with (
            patch.object(
                server.sys,
                "stdin",
                io.StringIO("".join(json.dumps(r) + "\n" for r in requests)),
            ),
            patch.object(server.sys, "stdout", output),
            patch.object(server.sys, "stderr", io.StringIO()),
            patch.object(
                server,
                "get_default_engine",
                side_effect=RuntimeError("model unavailable"),
            ),
        ):
            server.run_stdio_server()
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["id"] for r in responses], [42, 43])
        self.assertTrue(responses[0]["result"]["isError"])
        self.assertEqual(responses[1]["result"], {})

    def test_inference_failure_returns_tool_error(self):
        broken_engine = Mock()
        broken_engine.compress.side_effect = RuntimeError("inference failed")
        request = {
            "jsonrpc": "2.0",
            "id": "inference",
            "method": "tools/call",
            "params": {"name": "compress_tool_output", "arguments": {"text": "hello"}},
        }
        with (
            patch.object(server, "get_default_engine", return_value=broken_engine),
            patch.object(server.sys, "stderr", io.StringIO()),
        ):
            response = server.process_request(request)
        self.assertEqual(response["id"], "inference")
        self.assertTrue(response["result"]["isError"])


class ProxyIntegrationTests(unittest.TestCase):
    def test_proxy_preserves_long_line_and_resource_bypass(self):
        # Exercise the actual bidirectional stdio proxy with a deterministic
        # engine and child server, keeping client stdin open until the reply.
        child = (
            "import json,sys\n"
            "for line in sys.stdin:\n"
            " r=json.loads(line)\n"
            " response={'jsonrpc':'2.0','id':r['id'],'result':{'content':["
            "{'type':'text','text':r['params']['arguments']['text']}]}}\n"
            " print(json.dumps(response),flush=True)\n"
        )
        runner = (
            "import sys; from compressor_reflex_mcp import proxy; "
            "from tests.test_output_preservation import make_engine; "
            "proxy.get_default_engine=make_engine; "
            "proxy.run_proxy([sys.executable,'-c'," + repr(child) + "])"
        )
        texts = [
            "\n".join(["x" * 3000 + " ERROR_TAIL"] + ["routine"] * 6),
            "\n".join(["routine"] * engine_module.MAX_INPUT_LINES + ["FAILED tail"]),
        ]
        for text in texts:
            with self.subTest(length=len(text)):
                proc = subprocess.Popen(
                    [sys.executable, "-c", runner],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
                replies = queue.Queue()
                reader = threading.Thread(
                    target=lambda: replies.put(proc.stdout.readline()), daemon=True
                )
                try:
                    reader.start()
                    request = {
                        "jsonrpc": "2.0",
                        "id": 7,
                        "method": "tools/call",
                        "params": {"name": "read_file", "arguments": {"text": text}},
                    }
                    proc.stdin.write(json.dumps(request) + "\n")
                    proc.stdin.flush()
                    reply = json.loads(replies.get(timeout=15))
                    self.assertEqual(reply["id"], 7)
                    self.assertEqual(reply["result"]["content"][0]["text"], text)
                finally:
                    proc.stdin.close()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                    reader.join(timeout=1)
                    proc.stdout.close()
                    proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
