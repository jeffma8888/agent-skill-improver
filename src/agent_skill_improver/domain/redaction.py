"""Best-effort secret redaction applied before any content leaves the machine.

This is a *reduction* of risk, not a guarantee, and the docs say so. The
threat it addresses is narrow and real: a conversation transcript very often
contains an access key, a bearer token, or a private key block that was pasted
into a terminal, and forwarding that to a model command is an avoidable leak.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
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
    #: For a shape whose first run of characters can hold a later start of the
    #: same shape: where a match can start, and the run that follows the start.
    #: A start with no match after it fails where that run ends, and so does
    #: every later start inside the run, so the scan resumes past the run (see
    #: :func:`_matches`). Both or neither.
    start: re.Pattern[str] | None = None
    run: re.Pattern[str] | None = None


#: Where a shape may begin: not right after an ASCII letter or digit, unless
#: that letter or digit ends an escape. Every shape below that began at ``\b``
#: (all but the key block and the URL parameter) begins here instead. ``\b``
#: needs a non-word character before a letter, so a credential glued to a word
#: character went through whole: after ``_`` (``MY_TOKEN_`` before a token,
#: ``DB_`` before an assignment), after a non-ASCII letter, and after the end
#: of an escape -- ``\n``, ``\t``, ``\x1b`` or ``\u00e9`` in a ``repr`` or JSON
#: text, ``%20`` in a URL, the ``m`` that ends a terminal colour code. Each
#: escape is spelled in full (a lookbehind has one width per branch); the
#: colour code is matched by its last two characters, a digit, ``;`` or ``[``
#: and then ``m``, since its length varies, so an ``m`` after a digit counts
#: whether or not the escape character came before it.
_LEAD = (
    r"(?:(?<![A-Za-z0-9])"
    r"|(?<=\\[nrtbfv])"
    r"|(?<=\\x[0-9A-Fa-f]{2})"
    r"|(?<=\\u[0-9A-Fa-f]{4})"
    r"|(?<=\\U[0-9A-Fa-f]{8})"
    r"|(?<=%[0-9A-Fa-f]{2})"
    r"|(?<=[0-9;\[]m))"
)
#: Where a vendor token may end: not right before an ASCII letter or digit.
#: ``\b`` there needed a non-word character after the token, so one followed by
#: a non-ASCII letter, or by ``_`` where its body cannot hold one, was left
#: unmasked (the Slack shape, cut back to a ``-``, masked only part of it).
_TAIL = r"(?![A-Za-z0-9])"


#: Where a URL parameter begins: ``?`` or ``&`` as written, ``&`` as JSON text
#: escapes it (``\u0026``), or either percent-encoded, as in a URL carried
#: inside another one's parameter (``%3F``, ``%26``), whose ``=`` is then
#: ``%3D``. A signed URL from a cloud store names its signature with the
#: store's prefix (``X-Amz-Signature``, ``X-Goog-Signature``), and its
#: ``X-Amz-Security-Token`` is a session credential.
_URL_PARAM = (
    r"(?:[?&]|\\u0026|%3F|%26)(?:x-(?:amz|goog)-)?"
    r"(?:access[_-]?token|auth[_-]?token|security[_-]?token|api[_-]?key|apikey|token|secret"
    + r"|pass"
    + r"word|client[_-]?secret|sig|signature|key)(?:=|%3D)"
)
#: A parameter's value. It ends at the next parameter (``&`` as written or as
#: ``\u0026``, or ``%26`` before a name and ``=`` or ``%3D``), at a fragment,
#: or at a character a URL cannot hold as written: whitespace, a double quote,
#: or a single quote that nothing a URL can hold follows. A backslash escape,
#: escaped once or more, ends it when it stands for ASCII whitespace, a double
#: quote, ``#`` or ``&`` (``\"``, ``\n``, ``\u0022``, ``\u0026``). Anything else
#: is part of the value, escapes included (``\/``, ``\u002B``): the assignment
#: shape cannot mask past a placeholder, so a value cut short at a character it
#: can hold would leave the rest of the secret in the text. It stopped only at
#: ``&``, ``#`` and whitespace before, so a quoted URL lost the rest of its line
#: to the placeholder: the closing quote, the JSON after it, every later
#: parameter of a nested or JSON-escaped URL. A run of backslashes is taken
#: whole, so a long one is read once.
_URL_VALUE = (
    r"(?:[^&\s#\\\"'%]"
    r"|\\++(?!(?-i:[\"nrtfv]|u(?:000[9a-dA-D]|001[c-fC-F]|002[0236])))"
    r"|'(?![\s\"<>\[\]^`{|}]|\Z)"
    r"|%(?!26[a-z0-9._~-]+(?:=|%3d)))"
)


def _lead(first: str, then: str = "") -> str:
    """``_LEAD`` for a shape whose first character is one of ``first``, then ``then``.

    The lookahead is there for speed alone. ``_LEAD`` tries seven branches,
    and a search tries it at every position; with nothing cheaper first, 4 MiB
    of prose took about 2.7 s to mask instead of 0.4 s (3.13). A character the
    lookahead leaves out is a shape that never matches, so each shape's
    tests give it a glued sample of every way it can begin. ``then`` is what
    follows that character in every match, for a shape whose first character
    is a common letter: with ``r`` and ``s`` alone, the Stripe shape tried
    ``_LEAD`` at every ``r`` and ``s`` in the text.
    """
    return f"(?=[{first}]{then})" + _LEAD


#: Every shape ``tools/leakscan.py`` refuses in the repository is recognised
#: here as well; ``tests/test_redaction.py`` holds the two lists together. A
#: transcript is a far likelier place for a pasted credential than a fixture.
_SHAPES: tuple[_Shape, ...] = (
    _Shape(
        "aws-access-key-id",
        re.compile(_lead("A") + r"(?:A" + r"KIA|A" + r"SIA)[0-9A-Z]{16}" + _TAIL),
    ),
    _Shape("github-token", re.compile(_lead("g") + r"g" + r"h[pousr]_[A-Za-z0-9]{20,}" + _TAIL)),
    _Shape(
        "github-fine-grained-token",
        re.compile(_lead("g") + r"github_" + r"pat_[A-Za-z0-9_]{20,}" + _TAIL),
    ),
    _Shape("slack-token", re.compile(_lead("x") + r"x" + r"ox[baprs]-[A-Za-z0-9-]{10,}" + _TAIL)),
    _Shape("model-provider-key", re.compile(_lead("s") + r"s" + r"k-[A-Za-z0-9_-]{20,}" + _TAIL)),
    _Shape("google-api-key", re.compile(_lead("A") + r"A" + r"Iza[0-9A-Za-z_-]{30,}" + _TAIL)),
    # Stripe's secret and restricted keys begin ``sk_`` or ``rk_`` and then the
    # mode, ``live_`` or ``test_`` (or ``prod_``, which the gitleaks rule also
    # takes); its publishable keys (``pk_``) are meant to be shown and are left
    # alone. At least 20 letters and digits follow, the floor the GitHub and
    # model-provider shapes use, and no ceiling: the gitleaks rule stops at 99,
    # and with a ceiling ``_TAIL`` would leave a longer key unmasked.
    _Shape(
        "stripe-secret-key",
        re.compile(
            _lead("rs", "k_") + r"(?:s|r)" + r"k_(?:live|test|prod)_[A-Za-z0-9]{20,}" + _TAIL
        ),
    ),
    # An npm access token is ``npm_`` and 36 letters and digits, the last six a
    # checksum; a Hugging Face token is ``hf_`` and 34 letters. Each is exactly
    # that long, so a longer run after either prefix is a name, not a token.
    _Shape("npm-token", re.compile(_lead("n", "pm_") + r"n" + r"pm_[A-Za-z0-9]{36}" + _TAIL)),
    _Shape("huggingface-token", re.compile(_lead("h", "f_") + r"h" + r"f_[A-Za-z]{34}" + _TAIL)),
    _Shape(
        "jwt",
        re.compile(_lead("e") + r"e" + r"yJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*"),
        # A JWT segment may hold ``-`` and ``_``, and neither is a letter or a
        # digit, so each ``-eyJ`` or ``_eyJ`` inside a run is a start of its
        # own (``_LEAD``). Measured before, for ``-eyJ``:
        # searched with the pattern alone, every such start was tried and ran
        # to the end of the run, so text full of them took time in the square
        # of its length (64 KiB: about 0.7 s on 3.11, 1.1 s on 3.13; each
        # doubling, four times as long).
        start=re.compile(_lead("e") + r"e" + r"yJ"),
        run=re.compile(r"[A-Za-z0-9_-]*"),
    ),
    _Shape("bearer-token", re.compile(r"(?i)" + _lead("b") + r"bearer\s+[A-Za-z0-9._~+/=-]{16,}")),
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
        re.compile(r"(?i)(" + _URL_PARAM + r")" + _URL_VALUE + r"{8,}"),
        keep_prefix=True,
    ),
    # Assignments keep the key name and replace only the value, so a reader
    # still sees WHAT was set. The value stops at whitespace, '&' and '[' so a
    # URL tail survives and an already-placed placeholder is not matched twice.
    _Shape(
        "aws-secret-assignment",
        re.compile(
            r"(?i)("
            + _lead("as")
            + r"(?:"
            + _j("aws_secr", "et_access_key|secr", "et_access_key")
            + r")\b\s*[=:]\s*)[^\s&\[]+"
        ),
        keep_prefix=True,
    ),
    _Shape(
        "secret-assignment",
        re.compile(
            r"(?i)("
            + _lead("psact")
            + r"(?:pass"
            + r"word|pass"
            + r"wd|secret|api[_-]?key|apikey|access[_-]?token|auth[_-]?token"
            + r"|client[_-]?secret|private[_-]?token|token)\b\s*[=:]\s*)[^\s&\[]{6,}"
        ),
        keep_prefix=True,
    ),
)


PLACEHOLDER = "[REDACTED:{label}]"

#: Each placeholder ``redact`` writes, one for each shape.
PLACEHOLDERS: frozenset[str] = frozenset(PLACEHOLDER.format(label=shape.label) for shape in _SHAPES)

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
        "stripe-secret-key",
        "npm-token",
        "huggingface-token",
        "jwt",
        "bearer-token",
        "private-key-block",
    }
)


def _matches(shape: _Shape, text: str) -> Iterator[re.Match[str]]:
    """``shape``'s matches in ``text``: the ones ``finditer`` finds, in time linear in the text.

    A shape without ``start`` is its pattern's ``finditer``. With it, each
    start is matched in place; one with no match after it fails where the
    ``run`` after it ends (the run's characters are the only ones the first
    segment takes, so no shorter segment can succeed), and a later start
    inside the same run meets the same end with less before it, so it fails
    too and is never tried. The scan resumes at the run's end, where a start
    may sit right after it. A match is never empty: it begins with its start.
    """
    if shape.start is None or shape.run is None:
        yield from shape.pattern.finditer(text)
        return
    pos = 0
    while (start := shape.start.search(text, pos)) is not None:
        found = shape.pattern.match(text, start.start())
        if found is not None:
            yield found
            pos = found.end()
            continue
        run = shape.run.match(text, start.end())
        pos = run.end() if run is not None else start.end()


def credential_starts(text: str) -> list[int]:
    """Where a credential the shapes know may begin in ``text``, ascending.

    Each shape is matched on ``text`` as written, not on what the shapes
    before it left as ``redact`` matches it, so these are places for a cut to
    try rather than what ``redact`` masks: where each match begins and, for a
    shape that keeps the name it follows (an assignment, a URL parameter),
    where that name ends.
    """
    starts: set[int] = set()
    for shape in _SHAPES:
        for found in _matches(shape, text):
            starts.add(found.start())
            if shape.keep_prefix:
                starts.add(found.end(1))
    return sorted(starts)


def credential_labels(text: str) -> tuple[str, ...]:
    """The high-confidence credential shapes present in ``text``, sorted."""
    found = {
        shape.label
        for shape in _SHAPES
        if shape.label in HIGH_CONFIDENCE_LABELS and next(_matches(shape, text), None) is not None
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


#: How long a path read from a record may be and still be named as written.
NAMED_PATH_CHARS = 512

#: A path that reads as one: letters, digits and ``_ . / ~ + -``, nothing to spell.
_PLAIN_PATH = re.compile(r"[A-Za-z0-9_./~+-]+")


def named_path(path: object) -> str:
    """A path read from a stored record as a refusal names it.

    Bare, as before, when it reads as a path (``_PLAIN_PATH``), is at most
    ``NAMED_PATH_CHARS`` long and nothing in it is masked, the way the config
    loader names a key; else ``quoted_value``, so a line break in it is spelled
    on its line, a credential in it is masked and a long one is cut. A record's
    canonical path is whatever was written to it: one holding a line break
    printed the text after the break as a line of its own, and one of a
    megabyte made a refusal of a megabyte.
    """
    text = str(path)
    if len(text) <= NAMED_PATH_CHARS and _PLAIN_PATH.fullmatch(text) and redact(text).text == text:
        return text
    return quoted_value(text, limit=NAMED_PATH_CHARS)


#: How long a stored record's skill id may be and still be listed whole. A
#: skill id is a directory name, which is at most 255 bytes on APFS and ext4,
#: so a cut past 255 characters never shortens a real one.
LISTED_SKILL_ID_CHARS = 255

#: How long a stored record's canonical path may be and still be listed whole:
#: ``PATH_MAX``, the longest path a system call takes by name, is 1,024 bytes on
#: macOS and 4,096 on Linux, so a cut past 4,096 characters leaves any such path
#: whole.
LISTED_PATH_CHARS = 4096


def listed_value(value: object, *, limit: int = QUOTED_VALUE_CHARS) -> str:
    """A value read from a stored record as a listing shows it: masked, then cut.

    ``candidates`` lists records that no integrity check has read, so what it
    shows of one is whatever was last written to it: a record edited to hold a
    credential in its skill id, path, status or creation time printed it whole,
    and one holding a megabyte made a listing of a megabyte. Unlike
    ``quoted_value`` this adds no quotes, so a value that holds no credential
    shape and is at most ``limit`` long lists exactly as it did; the listing
    keeps it on its line (``one_line`` in the text, the encoder in JSON).
    Masking runs before the cut, so a key the cut would split is masked whole,
    and the cut says how much it left out. The limit and the count are in
    characters: a name of at most 255 bytes is at most 255 characters.
    """
    text = redact(str(value)).text
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} more chars not shown]"


#: How many values a message lists before it says how many more there were.
QUOTED_VALUES_SHOWN = 5


def quoted_values(
    values: Iterable[object],
    *,
    shown: int = QUOTED_VALUES_SHOWN,
    name: Callable[[object], str] = quoted_value,
) -> str:
    """Values a message lists, each as ``name`` names it, the first few only.

    A refusal that lists keys or ids read from a record on disk quotes each one
    (``quoted_value`` unless the caller names them another way that also
    quotes what is not plain), so a key holding a credential is masked and one
    holding a line break stays on its line, and names only the first ``shown``
    with a count of the rest: a record holding ten thousand of them makes one
    short line, not one as long as the record.
    """
    items = list(values)
    parts = [name(item) for item in items[:shown]]
    if len(items) > shown:
        parts.append(f"and {len(items) - shown} more")
    return ", ".join(parts)


def kept_prose(value: object, *, limit: int | None = None) -> str:
    """Model prose a record keeps: a string or nothing, masked, then clipped.

    A scorer's rationale, a judge's rationale and the improver's hypothesis,
    intended change, risks and rollback are the model's own words, persisted
    in scorecards and candidates, printed by ``review`` and fed back into the
    improver prompt. The transcript the model saw was masked (when
    ``redact_secrets`` says so); a model role is a child process that may be
    able to read more than its prompt, so its prose gets the same masking
    whatever the flag says, as the diagnostics' quotes do. Masking runs before
    the clip so a key the cut would split is masked whole; the clip says it
    clipped. A non-string is kept as nothing, as before.
    """
    text = redact(value).text if isinstance(value, str) else ""
    if limit is not None and len(text) > limit:
        return text[:limit] + " ...[clipped]"
    return text


def redact(text: str) -> RedactionResult:
    """Replace recognised secret shapes with labelled placeholders.

    Each shape is applied to the text the shapes before it left, as ``subn``
    applied it, through :func:`_matches`, so the time it takes is in proportion
    to the text whatever the text holds.
    """
    return _redact(text, None)


#: Where a placeholder starts in the masked text, and the masks it holds by label.
Placed = tuple[int, dict[str, int]]


def redact_placed(text: str) -> tuple[RedactionResult, list[Placed]]:
    """``redact(text)``, and where each placeholder starts in its text, in order.

    A shape can mask a placeholder an earlier one placed, as the URL parameter's
    shape masks a GitHub token that was its value, so each placeholder holds its
    own mask and every mask it replaced: what they hold sums to the counts. So a
    cut can count the masks its shown part holds, the ones a later placeholder
    took in included, where counting the placeholders it shows lost those.
    """
    places: list[tuple[int, int, dict[str, int]]] = []
    result = _redact(text, places)
    return result, [(start, held) for start, _end, held in places]


def _redact(text: str, places: list[tuple[int, int, dict[str, int]]] | None) -> RedactionResult:
    """:func:`redact`, keeping ``places`` (start, end, masks held) when it is a list."""
    counts: dict[str, int] = {}
    out = text
    for shape in _SHAPES:
        placeholder = PLACEHOLDER.format(label=shape.label)
        pieces: list[str] = []
        moved: list[tuple[int, int, dict[str, int]]] = []
        last = n = shift = at = 0
        for found in _matches(shape, out):
            replacement = found.group(1) + placeholder if shape.keep_prefix else placeholder
            if places is not None:
                begin = found.end(1) if shape.keep_prefix else found.start()
                while at < len(places) and places[at][1] <= begin:
                    start, end, held = places[at]
                    moved.append((start + shift, end + shift, held))
                    at += 1
                holds: dict[str, int] = {}
                while at < len(places) and places[at][0] < found.end():
                    for label, k in places[at][2].items():
                        holds[label] = holds.get(label, 0) + k
                    at += 1
                holds[shape.label] = holds.get(shape.label, 0) + 1
                moved.append((begin + shift, begin + shift + len(placeholder), holds))
                shift += len(replacement) - (found.end() - found.start())
            pieces.append(out[last : found.start()])
            pieces.append(replacement)
            last = found.end()
            n += 1
        if n:
            pieces.append(out[last:])
            out = "".join(pieces)
            counts[shape.label] = counts.get(shape.label, 0) + n
            if places is not None:
                moved.extend((start + shift, end + shift, held) for start, end, held in places[at:])
                places[:] = moved
    return RedactionResult(out, counts)
