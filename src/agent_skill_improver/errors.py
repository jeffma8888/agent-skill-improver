"""Exception taxonomy.

The split that matters operationally:

* :class:`RecordDefect` -- this *one* record is bad. Quarantine it and keep
  going. Never let it veto the batch.
* :class:`TransientFailure` -- we could not reach a verdict for reasons that
  are not the record's fault (timeout, rate limit, unreachable command).
  Defer it; "unreachable" is not a verdict.
* :class:`SafetyRefusal` -- the tool is refusing on purpose. Loud, terminal for
  the operation, and never retried into success.
"""

from __future__ import annotations


class AsiError(Exception):
    """Base class for every error this package raises."""


class ConfigError(AsiError):
    """Configuration or rubric definition is invalid."""


class RecordDefect(AsiError):
    """A single record is defective; quarantine it, do not veto the batch."""

    def __init__(self, reason: str, *, record_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record_id = record_id


class MalformedModelOutput(RecordDefect):
    """The model did not honour the JSON contract for its role."""


class EvidenceInvalid(RecordDefect):
    """Cited evidence does not resolve against the normalized conversation."""


class TransientFailure(AsiError):
    """Not the record's fault; defer and retry on a later run."""

    def __init__(self, reason: str, *, record_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.record_id = record_id


class SourceUnavailable(TransientFailure):
    """The configured conversation source could not be reached."""


class SafetyRefusal(AsiError):
    """A deliberate refusal enforcing one of the loop's safety invariants."""


class CredentialInSkill(SafetyRefusal):
    """A skill file holds a credential, so no model role may be shown it.

    The improver returns the COMPLETE new file, so a token in the base text
    would come back inside the candidate, its diff, its backup and the judge's
    prompt. Refusing leaks nothing; the operator removes the credential and the
    next pass proposes normally.
    """


class IsolationRequired(SafetyRefusal):
    """A private conversation source was configured without an enforced sandbox.

    The refusal that removes the external-communication leg of the lethal
    trifecta. It fires *before* any private byte is read or exported, and there
    is deliberately no flag that waives it: the whole point is that the
    exception is derived from the source classification rather than declared by
    whoever wrote the config.
    """


class ReadOnlyViolation(SafetyRefusal):
    """Something tried to write through a store opened read-only."""


class SelfMutationRefused(SafetyRefusal):
    """A proposal targeted this tool's own rubric, config, or source."""


class MirrorEditRefused(SafetyRefusal):
    """A proposal targeted a mirror copy instead of the canonical source."""


class NonDocumentWriteRefused(SafetyRefusal):
    """A write targeted something other than the skill's ``SKILL.md`` document.

    A skill directory can hold executable payloads, so an unrestricted write path
    into it is arbitrary code execution. This tool only ever replaces the
    document.
    """


class BaseHashMismatch(SafetyRefusal):
    """The live skill changed since the candidate was proposed."""


class BackupUnusable(SafetyRefusal):
    """A rollback was asked to restore a backup that cannot be trusted.

    Raised *before* the live file is touched. A recovery command that overwrites
    the file and only then discovers the backup was truncated, edited or
    unreadable has destroyed the one copy it existed to protect.
    """


class ApprovalRequired(SafetyRefusal):
    """Promotion attempted without a valid human approval token."""


class CandidateTampered(SafetyRefusal):
    """A candidate's derived fields do not recompute to the values it stores.

    Raised before any backup or skill write. A candidate record is a plain file,
    so a derived field -- the patch hash, the approval challenge, the diff and its
    counts, the status against the recorded gate -- can be edited to disagree with
    the input it was derived from, and an approval token derived from the stored
    value would then authorise bytes no human reviewed.
    """


class StateUnreadable(SafetyRefusal):
    """A state artifact could not be read or does not have the expected shape.

    A truncated, hand-edited or wrongly-keyed record must fail closed with a
    reason an operator can act on, rather than surfacing as a traceback from the
    middle of a promotion.
    """


class AutomationRefused(SafetyRefusal):
    """Promotion attempted from an automated (daily) context."""


class GateFailed(SafetyRefusal):
    """The candidate did not clear the regression / improvement gate."""


class LockHeld(AsiError):
    """Another run holds the daily lock."""
