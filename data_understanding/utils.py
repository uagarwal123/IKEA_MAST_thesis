import json
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text) -> int:
    """Count tokens using the cl100k_base tokenizer (used by GPT-4 / Claude pricing)."""
    if not isinstance(text, str):
        text = json.dumps(text)
    return len(_enc.encode(text))
