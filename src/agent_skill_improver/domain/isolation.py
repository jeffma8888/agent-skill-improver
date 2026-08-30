"""Source-aware execution boundary: the mechanical answer to the lethal trifecta.

The trifecta this package would otherwise assemble is textbook. A conversation
source hands the loop **private data**. That data is **attacker-influenceable**:
a transcript is full of text that was instruction to some other agent, and some
of it is hostile. The loop then hands that text to a **model command**, which is
an arbitrary subprocess that, left alone, inherits the parent environment and
whatever network and filesystem authority the parent had. Private data plus
untrusted content plus an external channel is exfiltration, and prompt fencing
does not touch it.

Fencing and redaction are mitigations for a *different* problem (the model being
talked into a wrong verdict). Neither removes a leg:

* a fence is a request to the model, and a request is not a boundary;
* redaction is shape matching over prose, so it is best effort by construction.

The leg removed here is **external communication**, and it is removed by the
operating system rather than by persuasion: for a private source, every model
role runs inside a local no-egress sandbox with a minimal environment, or the
run refuses before a single private byte is exported.

Two properties make that honest rather than decorative.

**The exception is derived, never declared.** A synthetic fixture corpus is not
private, so the offline fake may run without an OS sandbox and public CI stays
portable. That exception is computed from the *source kind* -- a closed
enumeration in the config schema -- and there is deliberately no configuration
field, environment variable or CLI flag that says "trust me". A trust label an
operator can type is the same class of defect as a cosmetic model label
satisfying an independence check: the gate ends up measuring a string somebody
chose rather than a property of what runs.

**Unavailable is a refusal, not a downgrade.** If the host has no supported
mechanism, there is no partial mode and no warning-and-continue. ``doctor``
reports a problem and ``daily`` raises before discovery. A safety check whose
failure path is "carry on with a note" is not a safety check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import IsolationRequired

# -- source classification -------------------------------------------------
#: A corpus this repository ships or generates. Contains no private content by
#: construction, because the builder invents every byte.
SOURCE_CLASS_SYNTHETIC = "synthetic-fixture"

#: Anything that reads a real conversation history. Private, and
#: attacker-influenceable, so it carries the isolation requirement.
SOURCE_CLASS_PRIVATE = "private-conversation"

#: The ONLY source kinds classified as synthetic. Deliberately a closed set and
#: deliberately not derived from a name pattern: a kind called
#: ``fixtures-from-prod`` must classify as private, and it does.
SYNTHETIC_SOURCE_KINDS: frozenset[str] = frozenset({"fixtures"})

# -- isolation modes -------------------------------------------------------
MODE_NONE = "none"
MODE_SANDBOX_EXEC = "sandbox-exec"
MODE_BUBBLEWRAP = "bubblewrap"

#: Modes that are an enforced OS boundary. Anything else -- including an
#: in-process fake or a bare subprocess -- is not.
OS_SANDBOX_MODES: tuple[str, ...] = (MODE_SANDBOX_EXEC, MODE_BUBBLEWRAP)

# -- environment policy ----------------------------------------------------
#: The only names copied from the parent environment. An allowlist rather than a
#: denylist, because a denylist has to enumerate every credential convention
#: that will ever exist and it is wrong the first time a new one appears.
ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
)

#: Substrings that must never appear in a child environment variable NAME. This
#: is not the mechanism -- the allowlist above is -- it is the assertion that
#: catches the two ways the allowlist gets bypassed: an operator putting a
#: credential in ``[model.<role>].env``, and a future edit widening the
#: allowlist. A denied name is a refusal, not a silent drop.
ENV_DENIED_SUBSTRINGS: tuple[str, ...] = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "SESSION",
    "COOKIE",
    "AUTH",
    "BEARER",
    "APIKEY",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "PROXY",
    "KEYCHAIN",
    "NETRC",
    "GPG",
    "SSH",
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "GCP_",
    "OPENAI",
    "ANTHROPIC",
    "BEDROCK",
    "GITHUB",
    "GITLAB",
    "SLACK",
    "NPM_",
    "PYPI_",
    "HF_",
)

#: A safe PATH when the parent has none at all.
FALLBACK_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def source_class(kind: str) -> str:
    """Classify a configured source kind. Fails CLOSED on anything unknown.

    An unrecognised kind is private, not synthetic. The failure mode of the
    opposite default is silent and total: a new source adapter lands, nobody
    updates this function, and the loop starts shipping private transcripts to
    an unsandboxed subprocess while every check still reports green.
    """
    return SOURCE_CLASS_SYNTHETIC if kind in SYNTHETIC_SOURCE_KINDS else SOURCE_CLASS_PRIVATE


def isolation_required(kind: str) -> bool:
    """True when the source is private and an OS sandbox is mandatory."""
    return source_class(kind) == SOURCE_CLASS_PRIVATE


def denied_env_names(env: Mapping[str, str]) -> tuple[str, ...]:
    """Names in ``env`` that match a denied substring, case-insensitively."""
    out: list[str] = []
    for name in env:
        upper = name.upper()
        if any(fragment in upper for fragment in ENV_DENIED_SUBSTRINGS):
            out.append(name)
    return tuple(sorted(out))


def minimal_env(
    parent: Mapping[str, str],
    *,
    workdir: str,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the child environment from the allowlist and refuse denied names.

    ``HOME`` is deliberately set to the per-call working directory rather than
    copied or dropped. Copying it points the child at every dotfile credential
    on the machine; dropping it makes ordinary tools crash. Pointing it at a
    fresh empty directory does neither.

    Raises :class:`IsolationRequired` when a denied name survives, which can
    only happen through ``extra`` (that is, through configuration).
    """
    env: dict[str, str] = {}
    for name in ENV_ALLOWLIST:
        value = parent.get(name)
        if value:
            env[name] = value
    env.setdefault("PATH", FALLBACK_PATH)
    env["HOME"] = workdir
    env["TMPDIR"] = workdir
    env["TMP"] = workdir
    env["TEMP"] = workdir
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    denied = denied_env_names(env)
    if denied:
        raise IsolationRequired(
            "refusing to run a model role: the child environment would carry "
            f"{len(denied)} name(s) matching a credential, token or proxy pattern "
            f"({', '.join(denied)}). For a private conversation source the model must be "
            "LOCAL and needs no credential, because the sandbox permits no egress; a "
            "credential in the child is an exfiltration channel with a config file in "
            "front of it. Remove it from [model.<role>].env."
        )
    return env


def dropped_env_count(parent: Mapping[str, str]) -> int:
    """How many parent names the allowlist discarded. Counted, never named.

    The count goes in the receipt; the names do not. A variable name can itself
    identify a system or a vendor, and a receipt is read far more casually than
    the tree it describes.
    """
    return sum(1 for name in parent if name not in ENV_ALLOWLIST)


def placeholders_in(argv: Sequence[str]) -> tuple[str, ...]:
    """Every ``{placeholder}`` token appearing anywhere in an argv template."""
    found: list[str] = []
    for part in argv:
        for match in _PLACEHOLDER.finditer(part):
            if match.group(0) not in found:
                found.append(match.group(0))
    return tuple(found)


@dataclass(frozen=True)
class IsolationDecision:
    """Whether this configuration is allowed to touch a conversation at all.

    Serialised into every run receipt. ``satisfied`` is the only field that
    decides anything; the rest exist so a reader can tell *why* without
    re-deriving it, and so a future reader can tell a synthetic run from a
    private one that happened to pass.
    """

    source_kind: str
    source_class: str
    required: bool
    mode: str
    verified: bool
    verification: Mapping[str, Any]
    roles: tuple[str, ...]
    roles_unisolated: tuple[str, ...]
    env_allowlist: tuple[str, ...]
    env_names_dropped: int
    satisfied: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_class": self.source_class,
            "os_sandbox_required": self.required,
            "mode": self.mode,
            "verified": self.verified,
            "verification": dict(self.verification),
            "roles": list(self.roles),
            "roles_unisolated": list(self.roles_unisolated),
            "env_allowlist": list(self.env_allowlist),
            "env_names_dropped_from_parent": self.env_names_dropped,
            "satisfied": self.satisfied,
            "reasons": list(self.reasons),
            "note": (
                "for a private conversation source the external-communication leg of the "
                "lethal trifecta is removed by an OS no-egress sandbox plus environment "
                "scrubbing, not by prompt fencing. A fence is a request to a model; a "
                "denied socket is a property of the process. The synthetic-fixture "
                "exception is derived from the source kind and cannot be granted by any "
                "configuration value."
            ),
        }

    def require(self) -> None:
        """Raise unless this configuration may proceed. The fail-closed point."""
        if self.satisfied:
            return
        raise IsolationRequired(
            "refusing to run: this configuration reads a PRIVATE conversation source "
            f"({self.source_kind!r} -> {self.source_class}) and the model roles are not "
            f"confined to an enforced local no-egress sandbox (mode={self.mode!r}).\n"
            + "\n".join(f"  - {reason}" for reason in self.reasons)
            + '\n  Fix one of: point [source] at kind = "fixtures" (synthetic, no private '
            "content), or run on a host with a supported sandbox mechanism "
            "(macOS sandbox-exec, Linux bubblewrap). There is deliberately no flag that "
            "waives this and no hosted-model opt-out: a hosted model needs egress, and "
            "egress is the leg being removed."
        )


def decide(
    *,
    source_kind: str,
    runner_isolation: Mapping[str, Any],
    roles: Sequence[str],
) -> IsolationDecision:
    """Combine the source classification with what the runner will actually do.

    ``runner_isolation`` is read off the object that runs the subprocesses, not
    off configuration, so this cannot pass because somebody described the runner
    correctly in a file.
    """
    cls = source_class(source_kind)
    required = cls == SOURCE_CLASS_PRIVATE
    mode = str(runner_isolation.get("mode", "") or "")
    verified = bool(runner_isolation.get("verified"))
    verification = runner_isolation.get("verification") or {}
    env_allowlist = tuple(runner_isolation.get("env_allowlist") or ())
    dropped = int(runner_isolation.get("env_names_dropped_from_parent") or 0)
    role_names = tuple(roles)

    reasons: list[str] = []
    is_os_sandbox = mode in OS_SANDBOX_MODES
    unisolated = () if is_os_sandbox else role_names

    if required:
        if not is_os_sandbox:
            reasons.append(
                f"isolation mode is {mode!r}, which is not one of {list(OS_SANDBOX_MODES)}. "
                "No supported no-egress mechanism is in force, so a model role could open a "
                "socket while holding private transcript text."
            )
        elif not verified:
            reasons.append(
                f"mode {mode!r} was selected but never VERIFIED on this host. A profile that "
                "is generated and not probed is a claim; the probe must show a refused socket "
                "and a refused write outside the private temp directory."
            )
        if unisolated:
            reasons.append("role(s) that would run outside the sandbox: " + ", ".join(unisolated))
        if not env_allowlist:
            reasons.append(
                "the runner reports no environment allowlist, so the child would inherit the "
                "parent environment including any credential in it"
            )
    satisfied = not required or not reasons
    return IsolationDecision(
        source_kind=source_kind,
        source_class=cls,
        required=required,
        mode=mode,
        verified=verified,
        verification=dict(verification),
        roles=role_names,
        roles_unisolated=tuple(unisolated),
        env_allowlist=env_allowlist,
        env_names_dropped=dropped,
        satisfied=satisfied,
        reasons=tuple(reasons),
    )
