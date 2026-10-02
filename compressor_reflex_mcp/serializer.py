"""
Serializer and Chunking Contract for Compressor Reflex.
Splits raw multi-line tool outputs into chunks within the ModernBERT token budget (2,048 tokens, max 128 lines),
tracking line token spans and formatting input prompts.
"""

from typing import List, Dict, Any, Tuple

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_LINES = 128

def chunk_tool_output(
    lines: List[str],
    intent: str,
    tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> List[Dict[str, Any]]:
    """
    Chunks a list of lines with their token spans so they fit within the model's sequence budget.
    """
    chunks = []
    current_lines = []
    current_token_count = 0
    current_line_indices = []

    intent_tokens = tokenizer.encode(f"Intent: {intent}\nOutput:\n", add_special_tokens=False)
    intent_len = len(intent_tokens)
    safety_margin = 16
    available_budget = max_tokens - intent_len - safety_margin

    for line_idx, line in enumerate(lines):
        line_tokens = tokenizer.encode(line + "\n", add_special_tokens=False)
        line_len = len(line_tokens)

        # If a single line exceeds the budget, truncate its tokens
        if line_len > available_budget:
            line = tokenizer.decode(line_tokens[:available_budget])
            line_tokens = line_tokens[:available_budget]
            line_len = len(line_tokens)

        if (current_token_count + line_len > available_budget or len(current_lines) >= max_lines) and current_lines:
            # Finalize current chunk
            chunk_text = "\n".join(current_lines)
            chunks.append({
                "lines": list(current_lines),
                "line_indices": list(current_line_indices),
                "intent": intent,
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
            "intent": intent,
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
    intent = chunk["intent"]
    lines = chunk["lines"]

    prefix = f"Intent: {intent}\nOutput:\n" if intent else "Output:\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    # Special token handling for ModernBERT
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

    # Append separator/EOS token
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 2
    input_ids.append(sep_id)

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, line_spans
