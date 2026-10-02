"""
Serializer and Chunking Contract for Compressor Reflex.
Splits raw multi-line tool outputs into chunks within the ModernBERT token budget (2,048 tokens, max 128 lines),
tracking line token spans and formatting input prompts with security guardrails.
"""

import re
from typing import List, Dict, Any, Tuple

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_LINES = 128
MAX_INTENT_CHARS = 512
MAX_PREFIX_TOKENS = 256

def sanitize_intent(intent: str) -> str:
    """Sanitizes user/agent intent string against control character injection and excessive length."""
    if not intent:
        return ""
    # Strip null bytes and non-printable control characters (except common whitespace)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(intent))
    return cleaned[:MAX_INTENT_CHARS].strip()

def _encode_prefix(intent: str, tokenizer, max_tokens: int) -> List[int]:
    """Use the same bounded prefix for chunk budgeting and model input."""
    if max_tokens < 4:
        raise ValueError("Sequence budget must allow prefix, line, and separator tokens")
    prefix = f"Intent: {intent}\nOutput:\n" if intent else "Output:\n"
    tokens = tokenizer.encode(prefix, add_special_tokens=True)
    prefix_limit = min(MAX_PREFIX_TOKENS, max_tokens - 2)
    if len(tokens) > prefix_limit:
        # Shorten the intent's middle while retaining the Output marker and
        # final special token, as well as the leading special token.
        tail_length = min(
            len(tokenizer.encode("\nOutput:\n", add_special_tokens=False)) + 1,
            prefix_limit - 1,
        )
        tokens = tokens[:prefix_limit - tail_length] + tokens[-tail_length:]
    return tokens


def chunk_tool_output(
    lines: List[str],
    intent: str,
    tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> List[Dict[str, Any]]:
    """
    Chunks a list of lines with their token spans so they fit within the model's sequence budget.
    Guarantees available token budget is strictly positive.
    """
    chunks = []
    current_lines = []
    current_token_count = 0
    current_line_indices = []
    current_line_tokens = []

    safe_intent = sanitize_intent(intent)
    prefix_tokens = _encode_prefix(safe_intent, tokenizer, max_tokens)
    available_budget = max_tokens - len(prefix_tokens) - 1
    if max_lines < 1:
        raise ValueError("max_lines must be positive")

    for line_idx, line in enumerate(lines):
        # Only the scoring tokens are shortened; selected output stays original.
        line_tokens = tokenizer.encode(line + "\n", add_special_tokens=False)
        line_len = len(line_tokens)

        if line_len > available_budget:
            line_tokens = line_tokens[:available_budget]
            line_len = len(line_tokens)

        if (current_token_count + line_len > available_budget or len(current_lines) >= max_lines) and current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append({
                "lines": list(current_lines),
                "line_indices": list(current_line_indices),
                "line_tokens": list(current_line_tokens),
                "intent": safe_intent,
                "text": chunk_text
            })
            current_lines = [line]
            current_token_count = line_len
            current_line_indices = [line_idx]
            current_line_tokens = [line_tokens]
        else:
            current_lines.append(line)
            current_token_count += line_len
            current_line_indices.append(line_idx)
            current_line_tokens.append(line_tokens)

    if current_lines:
        chunk_text = "\n".join(current_lines)
        chunks.append({
            "lines": list(current_lines),
            "line_indices": list(current_line_indices),
            "line_tokens": list(current_line_tokens),
            "intent": safe_intent,
            "text": chunk_text
        })

    return chunks

def build_chunk_model_inputs(
    chunk: Dict[str, Any],
    tokenizer,
    max_length: int = DEFAULT_MAX_TOKENS
) -> Tuple[Dict[str, Any], List[Tuple[int, int]]]:
    """
    Encodes chunk text and computes exact (start_token, end_token) spans for each line.
    """
    intent = chunk.get("intent", "")
    lines = chunk.get("lines", [])

    input_ids = _encode_prefix(sanitize_intent(intent), tokenizer, max_length)
    line_spans = []
    scoring_tokens = chunk.get("line_tokens")
    if scoring_tokens is not None and len(scoring_tokens) != len(lines):
        raise ValueError("Each original line must have one scoring token span")

    for index, line in enumerate(lines):
        line_str = line + "\n"
        l_toks = (
            scoring_tokens[index] if scoring_tokens is not None
            else tokenizer.encode(line_str, add_special_tokens=False)
        )
        start_idx = len(input_ids)
        end_idx = start_idx + len(l_toks)
        input_ids.extend(l_toks)
        line_spans.append((start_idx, end_idx))

    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 2
    input_ids.append(sep_id)
    if len(input_ids) > max_length:
        raise ValueError("Chunk exceeds the model sequence budget")

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, line_spans
