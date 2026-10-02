"""
Compressor Reflex Inference Engine.
Runs ONNX INT8 line-level extraction model with calibrated threshold tau*=0.50
and strict fail-open bypass policy (<= 5 physical lines or <= 64 tokens).
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from compressor_reflex_mcp.model_manager import ensure_model_files
from compressor_reflex_mcp.serializer import chunk_tool_output, build_chunk_model_inputs

CALIBRATED_THRESHOLD = 0.50
BYPASS_MAX_LINES = 5
BYPASS_MAX_TOKENS = 64

class CompressorEngine:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = ensure_model_files(model_dir)
        self.onnx_path = self.model_dir / "model_int8.onnx"
        self._init_session()
        self._init_tokenizer()

    def _init_session(self):
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, os.cpu_count() // 2 if os.cpu_count() else 4)

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options,
            providers=providers
        )

    def _init_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            clean_up_tokenization_spaces=False
        )

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def check_bypass(self, text: str) -> Tuple[bool, int, int]:
        """
        Deployment Bypass Policy (per INSERTION.md):
        Outputs with <= 5 physical lines OR <= 64 tokens are passed through verbatim.
        Returns: (bypass_applied, physical_line_count, token_count)
        """
        if not text:
            return True, 0, 0

        # Physical lines in raw text
        physical_lines = text.count("\n") + (1 if not text.endswith("\n") else 0)
        tok_count = self.count_tokens(text)

        if physical_lines <= BYPASS_MAX_LINES or tok_count <= BYPASS_MAX_TOKENS:
            return True, physical_lines, tok_count

        return False, physical_lines, tok_count

    def compress(
        self,
        text: str,
        intent: str = "",
        threshold: float = CALIBRATED_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Compresses multi-line text using the calibrated Compressor Reflex ONNX model.
        """
        start_time = time.perf_counter()

        # Check fail-open bypass policy
        is_bypass, num_lines, raw_tokens = self.check_bypass(text)
        if is_bypass:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "compressed_text": text,
                "raw_tokens": raw_tokens,
                "kept_tokens": raw_tokens,
                "compression_ratio": 0.0,
                "bypass_applied": True,
                "compressor_latency_ms": round(elapsed_ms, 2),
                "threshold_used": threshold,
                "num_lines_original": num_lines,
                "num_lines_kept": num_lines
            }

        lines = text.splitlines()
        chunks = chunk_tool_output(lines, intent=intent, tokenizer=self.tokenizer)

        kept_lines: List[Tuple[int, str, float]] = [] # (original_index, line_str, score)

        for chunk in chunks:
            inputs, line_spans = build_chunk_model_inputs(chunk, self.tokenizer)
            input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
            attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)
            seq_len = input_ids.shape[1]
            num_lines_in_chunk = len(line_spans)

            line_mask = np.zeros((num_lines_in_chunk, seq_len), dtype=np.float32)
            for i, (s, e) in enumerate(line_spans):
                s = min(s, seq_len - 1)
                e = min(max(e, s + 1), seq_len)
                span_len = e - s
                if span_len > 0:
                    line_mask[i, s:e] = 1.0 / span_len

            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "line_mask": line_mask,
            }

            raw_scores = self.session.run(None, ort_inputs)[0]
            # Handle 1D or 2D output
            if raw_scores.ndim > 1:
                raw_scores = raw_scores[0]
            # If model outputs probabilities directly or logits:
            if np.all((raw_scores >= 0.0) & (raw_scores <= 1.0)):
                probs = raw_scores
            else:
                probs = 1.0 / (1.0 + np.exp(-raw_scores))

            chunk_lines = chunk["lines"]
            chunk_indices = chunk["line_indices"]

            chunk_kept = []
            for idx, orig_idx in enumerate(chunk_indices):
                score = float(probs[idx])
                if score >= threshold:
                    chunk_kept.append((orig_idx, chunk_lines[idx], score))

            # Safety fallback: if no line met threshold in chunk, keep highest scoring line
            if not chunk_kept and len(chunk_lines) > 0:
                best_local_idx = int(np.argmax(probs))
                chunk_kept.append((
                    chunk_indices[best_local_idx],
                    chunk_lines[best_local_idx],
                    float(probs[best_local_idx])
                ))

            kept_lines.extend(chunk_kept)

        # Sort kept lines by original index and deduplicate
        kept_lines.sort(key=lambda x: x[0])
        final_lines = [item[1] for item in kept_lines]
        compressed_text = "\n".join(final_lines)

        kept_tokens = self.count_tokens(compressed_text)
        comp_ratio = (1.0 - (kept_tokens / raw_tokens)) if raw_tokens > 0 else 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "compressed_text": compressed_text,
            "raw_tokens": raw_tokens,
            "kept_tokens": kept_tokens,
            "compression_ratio": round(comp_ratio, 4),
            "bypass_applied": False,
            "compressor_latency_ms": round(elapsed_ms, 2),
            "threshold_used": threshold,
            "num_lines_original": len(lines),
            "num_lines_kept": len(final_lines)
        }

_DEFAULT_ENGINE: Optional[CompressorEngine] = None

def get_default_engine() -> CompressorEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = CompressorEngine()
    return _DEFAULT_ENGINE
