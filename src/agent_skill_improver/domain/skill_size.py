"""The size of a candidate's SKILL.md text change: report-only, never a cost.

``review`` shows it before apply and the daily receipt records it for every
candidate the run made. The numbers are properties of the complete SKILL.md
text, base and replacement, each folded first as the hashes fold it
(:func:`~agent_skill_improver.hashing.normalize_text`), so a record that holds
CRLF line endings measures as the text both hashes cover. Nothing here reads a
tokenizer, a model, a price or a clock: the token figure is a declared estimate
for no particular model, and no record says which harness loaded the file or in
what form, so none of these numbers is a measured cost or a count of use.

Computed at display time from texts the caller already holds and has already
checked; never stored in a candidate or validation record, never gated.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..hashing import normalize_text

#: The tokenizer-free estimate's divisor: ceil(bytes / 4) per text.
ESTIMATED_BYTES_PER_TOKEN = 4

SIZE_SCOPE = "complete SKILL.md text only"
SIZE_BASIS = (
    "UTF-8 bytes of the text with line endings folded to LF, as hashed; "
    "words split on whitespace; lines as the diff counts them"
)
TOKEN_ESTIMATE = (
    "estimate: ceil(bytes/4) per text, then subtract; no tokenizer, no particular model"
)
NOT_CLAIMED = (
    "properties of the complete SKILL.md text, not a measured cost: a harness that injects "
    "the complete file exposes this text delta, metadata-only discovery does not; no prompt "
    "usage, per-session cost, price or runtime was measured"
)
CITED_BASIS = "distinct conversations in the candidate's claimed rubric failures"
WINDOW_BASIS = (
    "distinct scored conversations with a failure primarily attributed to this skill, "
    "including rubric-prior-only attribution; not a count of conversations that used or "
    "loaded it"
)

_METRICS = ("bytes", "words", "lines", "est_tokens")


def text_size(text: str) -> dict[str, int]:
    """Bytes, words, lines and the estimated tokens of ``text``, folded first.

    Every metric is taken from the folded text, none from ``text`` itself.
    """
    folded = normalize_text(text)
    size = len(folded.encode("utf-8"))
    return {
        "bytes": size,
        "words": len(folded.split()),
        "lines": len(folded.splitlines()),
        "est_tokens": -(-size // ESTIMATED_BYTES_PER_TOKEN),
    }


def size_change(base_text: str, new_text: str) -> dict[str, Any]:
    """The base and replacement sizes and their signed change (a shrink is negative).

    The token change is the difference of the two per-text estimates, never an
    estimate of the byte change, so base + delta = replacement in every column.
    """
    base = text_size(base_text)
    replacement = text_size(new_text)
    return {
        "scope": SIZE_SCOPE,
        "base": base,
        "replacement": replacement,
        "delta": {k: replacement[k] - base[k] for k in _METRICS},
        "basis": SIZE_BASIS,
        "token_estimate": TOKEN_ESTIMATE,
        "not_claimed": NOT_CLAIMED,
    }


def cited_conversations(rubric_failures: Iterable[Any]) -> int:
    """How many distinct conversations the candidate's claimed failures come from."""
    return len({r.conversation_id for r in rubric_failures})
