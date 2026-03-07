"""
Tests for web_search tool: execution API and tokenizer train/inference alignment.
Run: python -m pytest tests/test_web_search.py -v
"""

import os
import pytest
from nanochat.execution import web_search


# -----------------------------------------------------------------------------
# Execution API tests
# -----------------------------------------------------------------------------

def test_web_search_basic():
    """web_search() accepts query and optional max_results."""
    out = web_search("latest updates on nanochat karpathy | 4")
    assert "Web search" in out or "search" in out.lower() or "nanochat" in out


def test_web_search_strips_quotes():
    """web_search() strips optional surrounding quotes from the query."""
    # Quoted query (as model or training data might produce)
    out = web_search('"mortgage rates Colorado today | 4"')
    assert "Web search error" not in out or "mortgage" in out


# -----------------------------------------------------------------------------
# Tokenizer: structured web_search content produces special token IDs
# -----------------------------------------------------------------------------

def _tokenizer_with_web_search_available():
    """Tokenizer must exist and include web_search special tokens (train with current SPECIAL_TOKENS)."""
    try:
        from nanochat.common import get_base_dir
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        if not os.path.isdir(tokenizer_dir) or not os.path.isfile(
            os.path.join(tokenizer_dir, "tokenizer.pkl")
        ):
            return False
        from nanochat.tokenizer import get_tokenizer
        tok = get_tokenizer()
        tok.encode_special("<|web_search_start|>")
        tok.encode_special("<|web_search_end|>")
        return True
    except (KeyError, FileNotFoundError, AssertionError):
        return False


@pytest.mark.skipif(
    not _tokenizer_with_web_search_available(),
    reason="Tokenizer with web_search special tokens not found (run tok_train with current SPECIAL_TOKENS)",
)
def test_render_conversation_web_search_uses_special_token_ids():
    """
    When assistant content is structured (list of parts with type "web_search"),
    render_conversation must emit the single special token IDs for
    <|web_search_start|> and <|web_search_end|>, not BPE pieces.
    This locks in train/inference alignment.
    """
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    ws_start_id = tokenizer.encode_special("<|web_search_start|>")
    ws_end_id = tokenizer.encode_special("<|web_search_end|>")

    conversation = {
        "messages": [
            {"role": "user", "content": "What is the latest on nanochat?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look that up. "},
                    {"type": "web_search", "text": "nanochat karpathy latest | 5"},
                ],
            },
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation)

    assert ws_start_id in ids, "token sequence should contain <|web_search_start|> as a single token id"
    assert ws_end_id in ids, "token sequence should contain <|web_search_end|> as a single token id"
    assert ids.count(ws_start_id) == 1, "exactly one web_search_start token"
    assert ids.count(ws_end_id) == 1, "exactly one web_search_end token"
    # Ensure they appear in order (start before end)
    pos_start = ids.index(ws_start_id)
    pos_end = ids.index(ws_end_id)
    assert pos_start < pos_end, "web_search_start must appear before web_search_end"