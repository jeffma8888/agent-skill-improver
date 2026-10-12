"""Best-effort secret redaction applied before any content leaves the machine.

This is a *reduction* of risk, not a guarantee, and the docs say so. The
threat it addresses is narrow and real: a conversation transcript very often
contains an access key, a bearer token, or a private key block that was pasted
into a terminal, and forwarding that to a model command is an avoidable leak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _j(*parts: str) -> str:
    """Join pattern fragments so no credential SHAPE appears literally in source.

    ``tools/leakscan.py`` scans this file too. A pattern that spells a token
    prefix in one piece is indistinguishable from a planted token to a scanner,
    so the prefixes are split exactly as the scanner splits its own.
    """
    return "".join(parts)


@dataclass(frozen=True)
class _Shape:
    label: str
    pattern: re.Pattern[str]
    #: When true, group 1 (the key name and separator) is kept and only what
    #: follows is replaced, so ``?access_token=...&x=1`` stays a readable URL.
    keep_prefix: bool = False


#: Every shape ``tools/leakscan.py`` refuses in the repository is recognised
#: here as well; ``tests/test_redaction.py`` holds the two lists together. A
#: transcript is a far likelier place for a pasted credential than a fixture.
_SHAPES: tuple[_Shape, ...] = (
    _Shape("aws-access-key-id", re.compile(r"\b(?:A" + r"KIA|A" + r"SIA)[0-9A-Z]{16}\b")),
    _Shape("github-token", re.compile(r"\bg" + r"h[pousr]_[A-Za-z0-9]{20,}\b")),
    _Shape("github-fine-grained-token", re.compile(r"\bgithub_" + r"pat_[A-Za-z0-9_]{20,}\b")),
    _Shape("slack-token", re.compile(r"\bx" + r"ox[baprs]-[A-Za-z0-9-]{10,}\b")),
    _Shape("model-provider-key", re.compile(r"\bs" + r"k-[A-Za-z0-9_-]{20,}\b")),
    _Shape("google-api-key", re.compile(r"\bA" + r"Iza[0-9A-Za-z_-]{30,}\b")),
    _Shape(
        "jwt",
        re.compile(r"\be" + r"yJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*"),
    ),
    _Shape("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    _Shape(
        "private-key-block",
        # The END line is optional: excerpts are clipped to a character budget,
        # and a key whose closing line fell past the clip is still a key. With
        # the delimiter required, the whole body went to the model unredacted.
        re.compile(
            _j("-----BE", "GIN ")
            + r"[A-Z ]*"
            + _j("PRIV", "ATE KEY-----")
            + r"[\s\S]*?(?:"
            + _j("-----E", "ND ")
            + r"[A-Z ]*"
            + _j("PRIV", "ATE KEY-----")
            + r"|\Z)"
        ),
    ),
    _Shape(
        "url-credential",
        re.compile(
            r"(?i)([?&](?:access[_-]?token|auth[_-]?token|api[_-]?key|apikey|token|secret"
            + r"|pass"
            + r"word|client[_-]?secret|sig|signature|key)=)[^&\s#]{8,}"
        ),
        keep_prefix=True,
    ),
    # Assignments keep the key name and replace only the value, so a reader
    # still sees WHAT was set. The value stops at whitespace, '&' and '[' so a
    # URL tail survives and an already-placed placeholder is not matched twice.
    _Shape(
        "aws-secret-assignment",
        re.compile(
            r"(?i)(\b(?:"
            + _j("aws_secr", "et_access_key|secr", "et_access_key")
            + r")\b\s*[=:]\s*)[^\s&\[]+"
        ),
        keep_prefix=True,
    ),
    _Shape(
        "secret-assignment",
        re.compile(
            r"(?i)(\b(?:pass"
            + r"word|pass"
            + r"wd|secret|api[_-]?key|apikey|access[_-]?token|auth[_-]?token"
            + r"|client[_-]?secret|private[_-]?token|token)\b\s*[=:]\s*)[^\s&\[]{6,}"
        ),
        keep_prefix=True,
    ),
)


PLACEHOLDER = "[REDACTED:{label}]"

#: Shapes that are a credential on their own evidence: a vendor prefix plus a
#: long random body, or a key block. Assignment and URL shapes are excluded
#: here because ``password: <your-password>`` in a skill's prose is common and
#: harmless, and this set drives a REFUSAL rather than a replacement.
HIGH_CONFIDENCE_LABELS: frozenset[str] = frozenset(
    {
        "aws-access-key-id",
        "github-token",
        "github-fine-grained-token",
        "slack-token",
        "model-provider-key",
        "google-api-key",
        "jwt",
        "bearer-token",
        "private-key-block",
    }
)


def credential_labels(text: str) -> tuple[str, ...]:
    """The high-confidence credential shapes present in ``text``, sorted."""
    found = {
        shape.label
        for shape in _SHAPES
        if shape.label in HIGH_CONFIDENCE_LABELS and shape.pattern.search(text)
    }
    return tuple(sorted(found))


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


#: How much of a model-supplied value a message may quote before it is cut.
QUOTED_VALUE_CHARS = 120


def quoted_value(value: object, *, limit: int = QUOTED_VALUE_CHARS) -> str:
    """A model-supplied value as a message may quote it: ``repr``, redacted, bounded.

    A refusal that quotes what the model sent lands in a quarantine reason, the
    receipt, the daily text summary and so the scheduler log. The model's
    output is bounded only by ``max_output_bytes``, so a scorer that puts four
    megabytes, or a pasted credential, into ``conversation_id`` would otherwise
    put it in all four places. Redaction runs over the whole ``repr`` before the
    cut, so a key is masked whole rather than sliced into a fragment the shapes
    no longer recognise; the cut says how much was left unquoted.
    """
    text = redact(repr(value)).text
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} more chars not quoted]"


def redact(text: str) -> RedactionResult:
    """Replace recognised secret shapes with labelled placeholders."""
    counts: dict[str, int] = {}
    out = text
    for shape in _SHAPES:
        placeholder = PLACEHOLDER.format(label=shape.label)
        repl = r"\g<1>" + placeholder if shape.keep_prefix else placeholder
        out, n = shape.pattern.subn(repl, out)
        if n:
            counts[shape.label] = counts.get(shape.label, 0) + n
    return RedactionResult(out, counts)
