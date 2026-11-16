"""Token estimation helpers for StarmentAI."""

from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Any

DEFAULT_CHARS_PER_TOKEN = 2.5
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = PROJECT_ROOT / "data" / "qwen3_tokenizer.json"


@lru_cache(maxsize=1)
def _load_tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None

    if not TOKENIZER_PATH.exists():
        return None

    try:
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    except Exception:
        return None


def count_text(
    text: str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Count tokens using a local Qwen tokenizer when available."""
    if not isinstance(text, str) or not text:
        return 0

    tokenizer = _load_tokenizer()

    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text).ids)
        except Exception:
            pass

    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero.")

    return max(1, ceil(len(text) / chars_per_token))


def count_message(
    message: dict[str, Any],
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    overhead: int = 6,
) -> int:
    """Estimate tokens used by one chat message."""
    if not isinstance(message, dict):
        raise ValueError("Message must be a dictionary.")

    content = str(message.get("content", ""))
    role = str(message.get("role", ""))

    return (
        overhead
        + count_text(content, chars_per_token)
        + count_text(role, chars_per_token)
    )


def count_messages(
    messages: list[dict[str, Any]],
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    overhead: int = 6,
) -> int:
    """Estimate total tokens used by a list of chat messages."""
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list.")

    return sum(
        count_message(message, chars_per_token, overhead)
        for message in messages
        if isinstance(message, dict)
    )