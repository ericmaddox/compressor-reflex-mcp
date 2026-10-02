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
MIN_AVAILABLE_BUDGET = 512

def sanitize_intent(intent: str) -> str:
    """Sanitizes user/agent intent string against control character injection and excessive length."""
    if not intent:
        return ""
    # Strip null bytes and non-printable control characters (except common whitespace)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(intent))
    return cleaned[:MAX_INTENT_CHARS].strip()

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

    safe_intent = sanitize_intent(intent)
    prefix_str = f"Intent: {safe_intent}\nOutput:\n" if safe_intent else "Output:\n"
    intent_tokens = tokenizer.encode(prefix_str, add_special_tokens=False)
    
    # Cap intent token consumption so sequence budget is never exhausted by intent alone
    intent_len = min(len(intent_tokens), 256)
    safety_margin = 16
    available_budget = max(max_tokens - intent_len - safety_margin, MIN_AVAILABLE_BUDGET)

    for line_idx, line in enumerate(lines):
        # Truncate any single line that is excessively long
        line_tokens = tokenizer.encode(line + "\n", add_special_tokens=False)
        line_len = len(line_tokens)

        if line_len > available_budget:
            line = tokenizer.decode(line_tokens[:available_budget])
            line_tokens = line_tokens[:available_budget]
            line_len = len(line_tokens)

        if (current_token_count + line_len > available_budget or len(current_lines) >= max_lines) and current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append({
                "lines": list(current_lines),
                "line_indices": list(current_line_indices),
                "intent": safe_intent,
                "text": chunk_text
            })
            current_lines = [line]
            current_token_count = line_len
            current_line_indices = [line_idx]
        else:
            current_lines.append(line)
            current_token_count += line_len
            current_line_indices.append(line_idx)

    if current_lines:
        chunk_text = "\n".join(current_lines)
        chunks.append({
            "lines": list(current_lines),
            "line_indices": list(current_line_indices),
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

    prefix = f"Intent: {intent}\nOutput:\n" if intent else "Output:\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    cls_token = prefix_tokens[0]
    prefix_body = prefix_tokens[1:]

    input_ids = [cls_token] + prefix_body
    line_spans = []

    for line in lines:
        line_str = line + "\n"
        l_toks = tokenizer.encode(line_str, add_special_tokens=False)
        start_idx = len(input_ids)
        end_idx = start_idx + len(l_toks)
        input_ids.extend(l_toks)
        line_spans.append((start_idx, end_idx))

    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 2
    input_ids.append(sep_id)

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, line_spans