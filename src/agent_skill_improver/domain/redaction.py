"""Best-effort secret redaction applied before any content leaves the machine.

This is a *reduction* of risk, not a guarantee, and the docs say so. The
threat it addresses is narrow and real: a conversation transcript very often
contains an access key, a bearer token, or a private key block that was pasted
into a terminal, and forwarding that to a model command is an avoidable leak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    (
        "private-key-block",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    (
        "aws-secret-assignment",
        re.compile(r"(?i)\b(aws_secret_access_key|secret_access_key)\b\s*[=:]\s*\S+"),
    ),
    (
        "password-assignment",
        re.compile(r"(?i)\b(password|passwd|api[_-]?key|token)\b\s*[=:]\s*\S{6,}"),
    ),
)

PLACEHOLDER = "[REDACTED:{label}]"


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    """Replace recognised secret shapes with labelled placeholders."""
    counts: dict[str, int] = {}
    out = text
    for label, pattern in _PATTERNS:
        out, n = pattern.subn(PLACEHOLDER.format(label=label), out)
        if n:
            counts[label] = counts.get(label, 0) + n
    return RedactionResult(out, counts)
