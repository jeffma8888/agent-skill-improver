#!/usr/bin/env python3
"""Privacy / leak scanner. Refuses to let private material reach a public repo.

Why this file looks the way it does
-----------------------------------
A scanner whose own source contains the tokens it bans will always flag itself,
and the usual fix is to exclude the scanner from the scan. That exclusion is a
fail-open hole: the one file guaranteed never to be checked is the file most
likely to be edited while adding a rule. So the banned literals here are
assembled from FRAGMENTS at import time. The scanner's own bytes are clean, it
is scanned like every other file, and there is no self-exclusion anywhere.

Two-sidedness is not assumed either. ``--selftest`` proves each rule fires on a
synthesised positive sample AND stays silent on a negative one. A rule that
matches everything is as useless as a rule that matches nothing, and only the
silent half catches the first case.

What is scanned
---------------
* every tracked file's CONTENT (``git ls-files``), text files only
* every tracked file's PATH, because a path can leak as loudly as a body
* the whole reachable HISTORY (``git log --all -p``), because deleting a secret
  in a later commit does not remove it from the repository
* commit metadata: author name, author email, subject and body
* configured Git remote URLs, which must be credential-free public HTTPS/SSH

Reporting
---------
Every finding for every file is reported. The scan never stops at the first hit
and never converts one hit into a verdict about the whole tree: a whole-batch
veto turns a filter into a ratchet, where one unresolved item blocks all
publication forever and nothing can be shipped.

Exit codes: 0 clean, 1 findings, 2 usage/environment error.
"""

from __future__ import annotations

import argparse
import re
from bisect import bisect_left
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from ipaddress import ip_address
from urllib.parse import urlsplit

MAX_FILE_BYTES = 2_000_000
CONTEXT_CHARS = 34


# --------------------------------------------------------------------------
# Fragment assembly. Keeps the banned literals out of this file's own bytes.
# --------------------------------------------------------------------------
def _j(*parts: str) -> str:
    return "".join(parts)


@dataclass(frozen=True)
class Rule:
    """One check. ``pattern`` is always a regex; literals are escaped by hand."""

    id: str
    pattern: str
    why: str
    remediation: str
    # A sample that MUST match, and one that must NOT. Both are used by
    # --selftest and neither is ever written to disk.
    positive: str
    negative: str
    flags: int = re.IGNORECASE
    # For a pattern whose first run of characters can hold a later start of the
    # same pattern: where a match can start, and the run that follows the
    # start. A start with no match after it fails where that run ends, and so
    # does every later start inside the run, so the scan resumes past the run
    # (see ``_matches``). Both or neither.
    start: str | None = None
    run: str | None = None

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


def _rules() -> tuple[Rule, ...]:
    # -- machine-specific absolute paths ---------------------------------
    home_root = _j("/", "Us", "ers", "/")
    home_alt = _j("/", "ho", "me", "/")
    # -- employer / internal namespaces ---------------------------------
    emp = _j("ama", "zon")
    emp_short = _j("am", "zn")
    internal_domain = _j("a2", "z", ".com")
    auth_tool = _j("mid", "way")
    auth_cmd = _j("mw", "init")
    build_tool = _j("bra", "zil", "-build")
    corp_host = _j(".", "co", "rp", ".")
    ticket_host = _j("t", ".", "co", "rp", ".")
    # -- the internal agent product this tool was developed against ------
    # A three-letter product name, so the boundary is load-bearing in BOTH
    # directions and \b is the wrong tool for it.
    #
    # An unbounded substring fires on ordinary English ("making", "taking",
    # "khaki"), and a rule that matches everything is as useless as one that
    # matches nothing. But \b alone is *fail-open* here, because an underscore
    # is a word character: a word-boundary pattern never matches a module named
    # "<product>_source.py", which is exactly the form the name took in this
    # tree before the rename. Camel case has the same hole.
    #
    # So the boundary is stated in terms of LETTERS rather than word
    # characters, in two alternatives:
    #
    #   any case, delimited by a non-letter    -> "<product>_source", "<product>
    #                                             export", 'kind = "<product>"'
    #   the mixed-case spelling, then a capital -> "<Product>Source"
    #
    # Both are preceded by "not a letter", which is what keeps ordinary English
    # silent: in "making", "taking", "khaki" and "kawasaki" the three letters
    # always follow another letter.
    #
    # The second alternative is deliberately restricted to the mixed-case
    # spelling. Widening it to any case would fire on the AWS access-key-id
    # prefix, which really does occur in this repository -- in the redaction
    # table and in a planted test sample -- so an over-broad rule here would
    # report two false positives on a clean tree and teach a reader to ignore
    # the scanner. That was measured, not predicted.
    #
    # (?-i: ... ) pins the case classes locally, because Rule applies IGNORECASE
    # globally and would otherwise make [A-Za-z] and the literal spelling mean
    # the same thing.
    agent_product = _j("a", "ki")
    agent_product_pattern = (
        "(?-i:(?<![A-Za-z])(?:[Aa][Kk][Ii](?![A-Za-z])|" + _j("A", "ki") + "(?=[A-Z])))"
    )
    # The one conversation title that was sampled from a real corpus while this
    # tool was being built. It reached fixtures, docs, config and a class name,
    # so it gets its own rule rather than relying on anyone remembering it.
    # The space around the optional dash or underscore has one way to be split:
    # with ``\s*[-_]?\s*`` it had one for each place in it, and a long run of
    # space after the first word took time in the square of the run.
    sampled_title = (
        _j("wo", "rk") + r"\s*(?:[-_]\s*)?" + _j("fl", "ow") + r"[\s_-]*" + _j("ag", "ent")
    )
    # -- operator identity ----------------------------------------------
    local_user = _j("jin", "cm")
    real_first = _j("Jin", "chen")
    real_last_pair = _j("Jin", "chen", r"\s+", "Ma")

    rules: list[Rule] = [
        Rule(
            id="abs-home-path",
            pattern=re.escape(home_root) + r"[A-Za-z0-9._-]+",
            why="an absolute path under a user home directory names the machine it was built on",
            remediation="use a path relative to the repository root, or a placeholder the config substitutes",
            positive=home_root + "someone/projects/x",
            negative="config/example.config.toml",
        ),
        Rule(
            id="abs-home-path-linux",
            pattern=re.escape(home_alt) + r"[A-Za-z0-9._-]+",
            why="same as abs-home-path, on Linux layouts",
            remediation="use a repository-relative path",
            positive=home_alt + "someone/projects/x",
            negative="examples/skills",
        ),
        Rule(
            id="employer-name",
            pattern=r"\b" + re.escape(emp) + r"\b",
            why="names the employer whose private corpus this tool was developed against",
            remediation="describe the conversation CLI by its command name only, with no organisation attached",
            positive="an internal " + emp + " service",
            negative="a conversation CLI",
        ),
        Rule(
            id="employer-short",
            pattern=r"\b" + re.escape(emp_short) + r"\b",
            why="short form of the employer namespace, common in internal hostnames",
            remediation="remove the hostname; public docs must not reference internal infrastructure",
            positive="host." + emp_short + " reachable",
            negative="amend the record",
        ),
        Rule(
            id="internal-domain",
            pattern=re.escape(internal_domain),
            why="an internal-only domain; its presence proves the repo was authored inside a private network",
            remediation="delete the URL. Cite a public source or cite nothing",
            positive="https://wiki." + internal_domain + "/page",
            negative="https://example.com/page",
        ),
        Rule(
            id="internal-auth-tool",
            pattern=r"\b" + re.escape(auth_tool) + r"\b",
            why="names an internal authentication system",
            remediation="remove it. This tool authenticates through nothing but the local CLI it shells out to",
            positive="refresh your " + auth_tool + " token",
            negative="mid-run checkpoint",
        ),
        Rule(
            id="internal-auth-command",
            pattern=r"\b" + re.escape(auth_cmd) + r"\b",
            why="names an internal authentication command",
            remediation="remove it",
            positive="run " + auth_cmd + " first",
            negative="run the CLI first",
        ),
        Rule(
            id="internal-build-tool",
            pattern=r"\b" + re.escape(build_tool) + r"\b",
            why="names an internal build system",
            remediation="use the public toolchain commands this repo actually ships with",
            positive="then " + build_tool + " release",
            negative="then python -m unittest",
        ),
        Rule(
            id="corp-hostname",
            # Started only where a run of host characters starts. Without the
            # lookbehind every position inside a long run was a start of its
            # own, each scanned to the end of the run: 32 KiB of letters and
            # dashes took 3 to 4 s, each doubling four times as long. A match
            # always began at the start of its run anyway, since one that
            # matches from inside a run matches from its start too.
            pattern=r"(?<![A-Za-z0-9-])[A-Za-z0-9-]+" + re.escape(corp_host) + r"[A-Za-z0-9.-]+",
            why="an internal corporate hostname",
            remediation="remove the hostname",
            positive="box" + corp_host + "example.net",
            negative="see docs/ARCHITECTURE.md",
        ),
        Rule(
            id="ticket-hostname",
            pattern=re.escape(ticket_host) + r"[A-Za-z0-9.-]+",
            why="an internal ticketing host",
            remediation="remove the link",
            positive="see " + ticket_host + "example",
            negative="see the issue tracker",
        ),
        Rule(
            id="internal-agent-product",
            pattern=agent_product_pattern,
            why=(
                "names the internal agent product whose conversation history this tool was "
                "developed against; the public artifact describes a generic conversation CLI "
                "and must not identify anybody's tooling"
            ),
            remediation=(
                "describe the interface, not the product: configure [source].discovery_argv and "
                "[source].export_argv in a gitignored config/local.toml, and refer to it in "
                "tracked text as a conversation CLI"
            ),
            # The positive sample is the IDENTIFIER form on purpose: it is the one
            # a \b-anchored rule silently missed, so it is the one the self-test
            # has to keep proving.
            positive="adapters/" + agent_product + "_source.py",
            negative="making a bounded selection, taking akin khaki kawasaki wiki",
        ),
        Rule(
            id="sampled-thread-title",
            pattern=sampled_title,
            why=(
                "a conversation title sampled from a real private corpus while this tool was "
                "being built; a real title is private content even when the transcript is not "
                "included"
            ),
            remediation=(
                "use an invented title. Fixtures must be generated by "
                "examples/build_fixtures.py, which invents every byte"
            ),
            positive='title = "Work' + 'flow Agent"',
            negative="an agent that runs one bounded pass",
        ),
        Rule(
            id="operator-local-username",
            pattern=r"\b" + re.escape(local_user) + r"\b",
            why="the local account name of the machine this was built on",
            remediation="remove it; no artifact should name a local account",
            positive="owner " + local_user + " ran it",
            negative="owner ran it",
        ),
        Rule(
            id="operator-real-name",
            pattern=r"\b" + real_last_pair + r"\b",
            why="the author's legal name; the repo is published under a handle, and linking the two is not this tool's business",
            remediation="use the publishing handle or no name at all",
            positive=real_first + " Ma wrote this",
            negative="the author wrote this",
        ),
        Rule(
            id="operator-given-name",
            pattern=r"\b" + re.escape(real_first) + r"\b",
            why="the author's given name",
            remediation="use the publishing handle or no name at all",
            positive="ask " + real_first + " about it",
            negative="ask the maintainer about it",
        ),
        Rule(
            id="private-classifier",
            pattern=r"Private\s*::\s*Do\s*Not\s*Upload",
            why="a packaging classifier that marks the project unpublishable; the whole point is that it is publishable",
            remediation="delete the classifier from pyproject.toml",
            positive='"' + _j("Priv", "ate :: Do ", "Not Upl", "oad") + '",',
            negative='"License :: OSI Approved :: MIT License",',
        ),
        # -- credential shapes ------------------------------------------
        Rule(
            id="aws-access-key-id",
            pattern=r"\b(?:A" + r"KIA|A" + r"SIA)[0-9A-Z]{16}\b",
            why="an AWS access key id",
            remediation="revoke the key, then remove it from the working tree AND from history",
            positive="A" + "KIA" + "ABCDEFGHIJKLMNOP",
            negative="A" + "KIA" + "short",
            flags=0,
        ),
        Rule(
            id="aws-secret-key-field",
            pattern=_j("aws_", "secr", "et_acc", "ess_key") + r"\s*[:=]",
            why="an AWS secret key assignment",
            remediation="move credentials into the environment of your model wrapper; never into this repo",
            positive=_j("aws_", "secr", "et_acc", "ess_key") + " = x",
            negative="model_label = x",
        ),
        Rule(
            id="private-key-block",
            pattern=_j("-----BE", "GIN ") + r"[A-Z ]*" + _j("PRIV", "ATE KEY", "-----"),
            why="an embedded private key",
            remediation="rotate the key and purge it from history",
            positive=_j("-----BE", "GIN RSA PRIV", "ATE KEY", "-----"),
            negative="-----BEGIN NOTHING-----",
        ),
        Rule(
            id="github-token",
            pattern=r"\b(?:g" + r"hp|g" + r"ho|g" + r"hu|g" + r"hs|g" + r"hr)_[A-Za-z0-9]{20,}\b",
            why="a GitHub personal access token",
            remediation="revoke the token, then purge it from history",
            positive="g" + "hp_" + "A" * 36,
            negative="g" + "hp_short",
            flags=0,
        ),
        Rule(
            id="github-fine-grained-token",
            pattern=r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
            why="a fine-grained GitHub token",
            remediation="revoke the token, then purge it from history",
            positive="github_pat_" + "B" * 30,
            negative="github_patient_notes",
            flags=0,
        ),
        Rule(
            id="slack-token",
            pattern=r"\bx" + r"ox[abprs]-[A-Za-z0-9-]{10,}\b",
            why="a Slack API token",
            remediation="revoke the token, then purge it from history",
            positive="x" + "oxb-1234567890-abcdef",
            negative="x" + "oxb-short",
            flags=0,
        ),
        Rule(
            id="openai-style-key",
            pattern=r"\bs" + r"k-[A-Za-z0-9]{24,}\b",
            why="a model-provider API key",
            remediation="revoke the key. Credentials belong in the model wrapper's environment",
            positive="s" + "k-" + "c" * 32,
            negative="s" + "k-short",
            flags=0,
        ),
        Rule(
            id="google-api-key",
            pattern=r"\bA" + r"Iza[0-9A-Za-z_-]{30,}\b",
            why="a Google API key",
            remediation="revoke the key, then purge it from history",
            positive="A" + "Iza" + "d" * 35,
            negative="A" + "Iza" + "short",
            flags=0,
        ),
        Rule(
            id="jwt",
            pattern=r"\be" + r"yJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",
            why="a JSON Web Token, which usually carries an identity and a signature",
            remediation="remove it; a captured token is a credential even after it expires",
            positive="e" + "yJ" + "a" * 12 + "." + "b" * 12 + ".sig",
            negative="e" + "yJshort.x.y",
            flags=0,
            # A segment may hold ``-``, which is not a word character, so each
            # ``-eyJ`` inside a run is a start of its own: searched with the
            # pattern alone, 256 KiB of them took 12 to 17 s, each doubling four
            # times as long. The redactor's jwt shape is searched the same way.
            start=r"\be" + r"yJ",
            run=r"[A-Za-z0-9_-]*",
        ),
        Rule(
            id="assigned-secret-literal",
            pattern=(
                r"\b(?:pass"
                + r"word|pass"
                + r"wd|secret|api[_-]?key|access[_-]?token|auth[_-]?token"
                r"|client[_-]?secret|private[_-]?token)\b\s*[:=]\s*"
                r"""["'][^"'\s]{8,}["']"""
            ),
            why="a secret assigned a literal value",
            remediation="read it from the environment at run time, and keep the value out of the repo",
            positive=_j("api_", "key") + ' = "abcdefgh12345678"',
            negative="redact_secrets = true",
        ),
        Rule(
            id="bearer-literal",
            pattern=r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}={0,2}",
            why="a bearer credential captured verbatim",
            remediation="remove it and rotate whatever issued it",
            positive="Authorization: Bearer " + "z" * 24,
            negative="Authorization: Bearer <redacted>",
        ),
    ]
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise SystemExit(f"duplicate rule id {rule.id!r}")
        seen.add(rule.id)
    return tuple(rules)


RULES = _rules()


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Finding:
    scope: str  # "tracked-content" | "tracked-path" | "history" | "commit-meta"
    location: str
    rule_id: str
    excerpt: str
    why: str
    remediation: str


@dataclass
class Allowance:
    """A narrow, deliberate exception. Printed in every report, never silent.

    Kept empty on purpose. A populated allowlist is how a scanner stops being
    evidence: the entries accumulate, nobody re-reads the reasons, and the rule
    they suppress is the one that mattered. If you must add one, scope it to a
    single path AND a single rule id, and write the reason for a stranger.
    """

    path: str
    rule_id: str
    reason: str


ALLOWANCES: tuple[Allowance, ...] = ()


def _allowed(path: str, rule_id: str) -> Allowance | None:
    for allowance in ALLOWANCES:
        if allowance.path == path and allowance.rule_id == rule_id:
            return allowance
    return None


def _excerpt(text: str, start: int, end: int) -> str:
    left = max(0, start - CONTEXT_CHARS)
    right = min(len(text), end + CONTEXT_CHARS)
    raw = text[left:right].replace("\n", "\\n").replace("\r", "")
    return ("..." if left > 0 else "") + raw + ("..." if right < len(text) else "")


def _line_breaks(text: str) -> list[int]:
    """The offset of every line break in ``text``, in order."""
    breaks: list[int] = []
    at = text.find("\n")
    while at != -1:
        breaks.append(at)
        at = text.find("\n", at + 1)
    return breaks


def _matches(rule: Rule, text: str) -> Iterator[re.Match[str]]:
    """``rule``'s matches in ``text``: the ones ``finditer`` finds, in time linear in the text.

    A rule without ``start`` is its pattern's ``finditer``. With it, each start
    is matched in place; one with no match after it fails where the ``run``
    after it ends (the run's characters are the only ones the first segment
    takes, so no shorter segment can succeed), and a later start inside the
    same run meets the same end with less before it, so it fails too and is
    never tried. The scan resumes at the run's end. A match is never empty: it
    begins with its start.
    """
    pattern = rule.compiled()
    if rule.start is None or rule.run is None:
        yield from pattern.finditer(text)
        return
    starts = re.compile(rule.start, rule.flags)
    runs = re.compile(rule.run, rule.flags)
    pos = 0
    while (start := starts.search(text, pos)) is not None:
        found = pattern.match(text, start.start())
        if found is not None:
            yield found
            pos = found.end()
            continue
        run = runs.match(text, start.end())
        pos = run.end() if run is not None else start.end()


def scan_text(
    text: str, *, scope: str, location: str, rules: Sequence[Rule] = RULES
) -> list[Finding]:
    """Report EVERY match of EVERY rule. Never returns early.

    A finding's line is found by bisecting the offsets of the text's line
    breaks, listed once, where counting them from the start for each finding
    took time in the product of the findings and the text (40,000 findings in
    1.5 MiB: about 10 s, each doubling four times as long).
    """
    findings: list[Finding] = []
    breaks: list[int] | None = None
    for rule in rules:
        if _allowed(location, rule.id) is not None:
            continue
        for match in _matches(rule, text):
            if breaks is None:
                breaks = _line_breaks(text)
            line = bisect_left(breaks, match.start()) + 1
            findings.append(
                Finding(
                    scope=scope,
                    location=f"{location}:{line}" if scope != "tracked-path" else location,
                    rule_id=rule.id,
                    excerpt=_excerpt(text, match.start(), match.end()),
                    why=rule.why,
                    remediation=rule.remediation,
                )
            )
    return findings


# --------------------------------------------------------------------------
# Git plumbing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RemoteUrl:
    name: str
    direction: str
    url: str


@dataclass
class GitInfo:
    is_repo: bool
    shallow: bool
    commit_count: int
    remotes: tuple[str, ...] = ()
    remote_urls: tuple[RemoteUrl, ...] = ()
    notes: list[str] = field(default_factory=list)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        errors="replace",
    )


def git_info(root: Path) -> GitInfo:
    probe = _git(root, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return GitInfo(is_repo=False, shallow=False, commit_count=0, notes=["not a git repository"])
    shallow = _git(root, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    count_proc = _git(root, "rev-list", "--all", "--count")
    count = int(count_proc.stdout.strip() or "0") if count_proc.returncode == 0 else 0
    remotes = tuple(x for x in _git(root, "remote").stdout.split() if x)
    remote_urls: list[RemoteUrl] = []
    for name in remotes:
        for direction, args in (
            ("fetch", ("remote", "get-url", "--all", name)),
            ("push", ("remote", "get-url", "--all", "--push", name)),
        ):
            result = _git(root, *args)
            if result.returncode != 0:
                continue
            for url in result.stdout.splitlines():
                if url.strip():
                    remote_urls.append(RemoteUrl(name=name, direction=direction, url=url.strip()))
    info = GitInfo(
        is_repo=True,
        shallow=shallow,
        commit_count=count,
        remotes=remotes,
        remote_urls=tuple(remote_urls),
    )
    if shallow:
        info.notes.append(
            "repository is SHALLOW: the history scan below covers only the commits present "
            "locally, so a clean history result here is not a statement about the full history. "
            "Fetch with full depth before trusting it."
        )
    if count == 0:
        info.notes.append("no commits yet, so there is no history to scan")
    return info


_SAFE_SCP_REMOTE = re.compile(
    r"\Agit@(?P<host>[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?):" r"(?P<path>[^/\s][^\s]*)\Z"
)


def _host_is_public(host: str) -> bool:
    lowered = host.rstrip(".").lower()
    if not lowered or lowered == "localhost" or lowered.endswith(".localhost"):
        return False
    if lowered.endswith(".local") or "." not in lowered:
        return False
    try:
        address = ip_address(lowered)
    except ValueError:
        return True
    return address.is_global


def remote_url_is_public_safe(url: str) -> bool:
    """Accept only credential-free HTTPS or git-only SSH to a public host."""
    if not url or any(ch.isspace() or ord(ch) < 32 for ch in url):
        return False
    if scan_text(url, scope="git-remote", location="configured remote"):
        return False
    scp = _SAFE_SCP_REMOTE.fullmatch(url)
    if scp:
        return _host_is_public(scp.group("host"))
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
        return False
    if not _host_is_public(parsed.hostname):
        return False
    if parsed.password is not None or parsed.query or parsed.fragment:
        return False
    if parsed.scheme == "https" and parsed.username is not None:
        return False
    if parsed.scheme == "ssh" and parsed.username not in {None, "git"}:
        return False
    return bool(parsed.path and parsed.path != "/" and not parsed.path.startswith("//"))


def unsafe_remote_urls(info: GitInfo) -> tuple[RemoteUrl, ...]:
    """Configured remotes that are local, credential-bearing, or otherwise unsafe."""
    return tuple(remote for remote in info.remote_urls if not remote_url_is_public_safe(remote.url))


def tracked_files(root: Path) -> list[str]:
    proc = _git(root, "ls-files", "-z")
    if proc.returncode != 0:
        return []
    return [p for p in proc.stdout.split("\0") if p]


def _read_text(path: Path) -> tuple[str | None, str]:
    """Return (text, note). ``None`` text means 'not scannable as text'."""
    try:
        raw = path.read_bytes()
    except OSError as exc:  # pragma: no cover - filesystem dependent
        return None, f"unreadable ({exc.__class__.__name__})"
    if not raw:
        return "", ""
    if b"\0" in raw:
        return None, "binary (NUL byte): content not scanned, review it by hand"
    if len(raw) > MAX_FILE_BYTES:
        return None, f"larger than {MAX_FILE_BYTES} bytes: content not scanned"
    try:
        return raw.decode("utf-8"), ""
    except UnicodeDecodeError:
        return raw.decode("utf-8", "replace"), "not valid UTF-8: decoded with replacement"


def scan_tracked(root: Path) -> tuple[list[Finding], list[str], int]:
    findings: list[Finding] = []
    notes: list[str] = []
    files = tracked_files(root)
    for rel in files:
        findings.extend(scan_text(rel, scope="tracked-path", location=rel))
        text, note = _read_text(root / rel)
        if note:
            notes.append(f"{rel}: {note}")
        if text is None:
            continue
        findings.extend(scan_text(text, scope="tracked-content", location=rel))
    return findings, notes, len(files)


def scan_history(root: Path, info: GitInfo) -> list[Finding]:
    if not info.is_repo or info.commit_count == 0:
        return []
    findings: list[Finding] = []
    meta = _git(
        root,
        "log",
        "--all",
        "--no-color",
        "--pretty=format:%H%x1f%an%x1f%ae%x1f%s%x1f%b%x1e",
    )
    if meta.returncode == 0:
        for record in meta.stdout.split("\x1e"):
            record = record.strip("\n")
            if not record:
                continue
            parts = record.split("\x1f")
            if len(parts) < 5:
                continue
            sha, author, email, subject, body = parts[0], parts[1], parts[2], parts[3], parts[4]
            blob = "\n".join([author, email, subject, body])
            for finding in scan_text(blob, scope="commit-meta", location=f"commit {sha[:12]}"):
                findings.append(finding)
    patch = _git(root, "log", "--all", "--no-color", "-p", "--unified=0")
    if patch.returncode == 0 and patch.stdout:
        findings.extend(scan_text(patch.stdout, scope="history", location="git log --all -p"))
    return findings


# --------------------------------------------------------------------------
# Self-test: prove every rule is two-sided
# --------------------------------------------------------------------------
def selftest() -> list[str]:
    problems: list[str] = []
    for rule in RULES:
        pattern = rule.compiled()
        if not pattern.search(rule.positive):
            problems.append(f"{rule.id}: FAILS TO FIRE on its own positive sample")
        if pattern.search(rule.negative):
            problems.append(f"{rule.id}: FIRES on its own negative sample (too broad)")
        if (rule.start is None) != (rule.run is None):
            problems.append(f"{rule.id}: names a start without a run, or a run without a start")
        elif next(_matches(rule, rule.positive), None) is None:
            problems.append(f"{rule.id}: its start and run skip its own positive sample")
    own = Path(__file__).read_text(encoding="utf-8")
    self_findings = scan_text(own, scope="tracked-content", location="tools/leakscan.py")
    if self_findings:
        problems.append(
            "this scanner's own source matches "
            f"{len(self_findings)} rule(s): {sorted({f.rule_id for f in self_findings})}. "
            "Assemble the literal from fragments instead of adding an exclusion."
        )
    return problems


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def group(findings: Iterable[Finding]) -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = {}
    for finding in findings:
        out.setdefault(finding.location.split(":")[0], []).append(finding)
    return out


def report(findings: list[Finding], *, files: int, info: GitInfo, notes: list[str]) -> None:
    print("leak scan")
    print(f"  rules            {len(RULES)}")
    print(f"  allowlist        {len(ALLOWANCES)} entr{'y' if len(ALLOWANCES) == 1 else 'ies'}")
    print(f"  tracked files    {files}")
    if info.is_repo:
        print(f"  commits scanned  {info.commit_count}" + ("  (SHALLOW)" if info.shallow else ""))
        print(f"  remotes          {', '.join(info.remotes) if info.remotes else 'none'}")
        print(f"  remote URLs      {len(info.remote_urls)} checked")
    else:
        print("  commits scanned  0  (not a git repository)")
    for note in info.notes:
        print(f"  note: {note}")
    for note in notes:
        print(f"  note: {note}")

    if not findings:
        print(
            "\nRESULT: clean. No banned pattern in tracked content, tracked paths, history, "
            "or configured remotes."
        )
        return

    print(f"\nRESULT: {len(findings)} finding(s) in {len(group(findings))} location(s).")
    print("Every finding is listed. The scan does not stop at the first hit, and one")
    print("unresolved finding does not veto the rest of the report.\n")
    for location in sorted(group(findings)):
        items = group(findings)[location]
        print(f"{location}  ({len(items)} finding(s))")
        for finding in items:
            print(f"    [{finding.rule_id}] {finding.location}  scope={finding.scope}")
            print(f"        matched : {finding.excerpt}")
            print(f"        why     : {finding.why}")
            print(f"        fix     : {finding.remediation}")
        print()
    if any(f.scope in {"history", "commit-meta"} for f in findings):
        print("At least one finding is in HISTORY. Editing the working tree does not remove it.")
        print("Either rewrite history before publishing, or start a fresh repository from the")
        print("current tree. Rotate any credential that was ever committed.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leakscan",
        description="Scan tracked files, tracked paths and git history for private material.",
    )
    parser.add_argument("root", nargs="?", default=".", help="repository root (default: .)")
    parser.add_argument(
        "--selftest", action="store_true", help="only prove the rules are two-sided"
    )
    parser.add_argument(
        "--no-history", action="store_true", help="skip the history scan (faster; less honest)"
    )
    args = parser.parse_args(argv)

    problems = selftest()
    if problems:
        print("SCANNER SELF-TEST FAILED. Not scanning anything, because a broken rule set")
        print("reports 'clean' for the wrong reason:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    if args.selftest:
        print(f"scanner self-test: {len(RULES)} rules, each fires on its positive sample and")
        print("stays silent on its negative one; this file's own source is clean.")
        return 0

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    info = git_info(root)
    if not info.is_repo:
        print("WARNING: no git repository here, so nothing is 'tracked' and there is no")
        print("history. Scanning the working tree would report a different set of files")
        print("than a publish would upload, so this is reported rather than guessed.\n")

    findings, notes, files = scan_tracked(root)
    for remote in unsafe_remote_urls(info):
        findings.append(
            Finding(
                scope="git-remote",
                location=f"remote {remote.name} ({remote.direction})",
                rule_id="unsafe-git-remote",
                excerpt="<redacted unsafe remote URL>",
                why="the remote is local, credential-bearing, non-HTTPS/SSH, or names private infrastructure",
                remediation="use a credential-free HTTPS URL or git@host:owner/repository SSH URL",
            )
        )
    if not args.no_history:
        findings.extend(scan_history(root, info))
    else:
        info.notes.append("history scan SKIPPED by --no-history")

    report(findings, files=files, info=info, notes=notes)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
