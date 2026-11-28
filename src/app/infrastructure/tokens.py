"""Token counting behind one seam (conversation-compaction ARCH A5).

``count_tokens(text)`` is the only way the compaction budgets measure a
prompt: an exact count from tiktoken's encoding for the configured
``CHAT_MODEL`` (``o200k_base`` when the name is unknown), resolved once per
process. Measured on this project's text, the characters ÷ 4 estimate is
25% over for English and 20% under for Bangla, so the estimate is only the
last resort: when no encoding can load at all (an air-gapped first start),
every count degrades to length ÷ 4 with one warning. The function never
raises — a token budget is a safety net, not a new failure class on the
turn path.

tiktoken is imported here and nowhere else (allowed dependencies: tiktoken
and settings).
"""

import logging
import threading
from typing import Optional

import tiktoken

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

# tiktoken raises when the model name is not in its prefix table; this is
# the encoding that covers every current model family.
_FALLBACK_ENCODING = "o200k_base"

_lock = threading.Lock()
_encoding: Optional["tiktoken.Encoding"] = None
_resolved = False


def _load_encoding(model: str) -> "tiktoken.Encoding":
    """The model's own encoding, or ``o200k_base`` for an unknown name."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def _get_encoding() -> Optional["tiktoken.Encoding"]:
    """Resolve the encoding once per process; ``None`` means "use the
    estimate" (already logged)."""
    global _encoding, _resolved
    if _resolved:
        return _encoding
    with _lock:
        if not _resolved:
            try:
                model = Settings().CHAT_MODEL
            except Exception:
                model = ""
            try:
                _encoding = _load_encoding(model)
            except Exception:
                logger.warning(
                    "tiktoken encoding could not be loaded (model=%s); token "
                    "counts fall back to a character estimate for this process",
                    model,
                )
                _encoding = None
            _resolved = True
    return _encoding


def count_tokens(text: str) -> int:
    """Token count of ``text``; never raises.

    Special-token spellings in customer text (``<|endoftext|>`` and the
    like) are counted as ordinary characters, never rejected.
    """
    encoding = _get_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:
            logger.warning(
                "tiktoken could not encode text of %d characters; one count "
                "falls back to a character estimate",
                len(text),
            )
    return len(text) // 4
