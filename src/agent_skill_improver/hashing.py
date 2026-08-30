"""SHA-256 helpers.

Every hash is taken over UTF-8 bytes of *line-ending normalized* text, so a
snapshot taken on two machines with different checkout settings still matches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HASH_PREFIX_LEN = 12


def normalize_text(text: str) -> str:
    """Normalize line endings only.

    Deliberately conservative: interior whitespace is content and is never
    altered, because skill files are prose whose spacing is meaningful.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(normalize_text(text).encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8", errors="replace"))


def short(digest: str) -> str:
    return digest[:HASH_PREFIX_LEN]


def tree_digest(entries: dict[str, str]) -> str:
    """Order-independent digest of a ``{relative_path: file_sha256}`` mapping."""
    lines = [f"{k}\x00{v}" for k, v in sorted(entries.items())]
    return sha256_bytes("\x01".join(lines).encode("utf-8"))
