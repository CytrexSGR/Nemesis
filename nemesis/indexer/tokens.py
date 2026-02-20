"""Token-Counting fuer AST-aware Chunking."""

from __future__ import annotations

_encoder = None
_tiktoken_available: bool | None = None


def _get_encoder():
    global _encoder, _tiktoken_available
    if _tiktoken_available is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
            _tiktoken_available = True
        except ImportError:
            _tiktoken_available = False
    return _encoder


def count_tokens(text: str) -> int:
    if not text or not text.strip():
        if not text:
            return 0
        return estimate_tokens(text)
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
