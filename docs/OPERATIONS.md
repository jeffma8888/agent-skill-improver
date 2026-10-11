# Operations

## First run

```bash
asi --config config/example.config.toml doctor
```

`doctor` validates the configuration and reports what is actually wired: the
**source class and isolation mode**, state-directory permissions, source
reachability, model roles, skill roots present versus configured, how many skills
were found, how many **do not load**, how many carry executable payloads, the
selection bound, and whether the judge is independent of the improver. It exits
non-zero if the configuration is not runnable. Run it after every config change;
most problems in this system are configuration problems that otherwise surface as
a strange result three stages later.

### The isolation line

`doctor` decides the isolation question **before** it touches the source, and
prints it either way:

```
isolation      : source class private, roles confined by sandbox-exec (verified)
```

If the source classifies as private and no verified confinement mechanism is
available, that becomes a `PROBLEM`, `doctor` exits non-zero, and `daily` refuses
in the same way — before the lock, before the skill hash, before a single export.
There is no flag to proceed anyway. On macOS the mechanism is `sandbox-exec`; on
Linux it is `bubblewrap` (`bwrap`), which usually means installing a distribution
package. The alternative, if you cannot install one, is to run against the
synthetic fixture corpus, which classifies as synthetic and needs no sandbox.

`doctor` also repairs `state/` permissions on the way through and reports what it
changed. A repair that cannot take effect is a refusal, not a warning.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success, including "stood down for a stated reason" |
| 1 | the command ran and the answer is negative (`doctor` not ready, `selftest` failed, a gate not met) |
| 3 | a refusal or a domain error, with the exception class named on stderr (`SafetyRefusal`, `BaseHashMismatch`, `AutomationRefused`, `MirrorEditRefused`, `NonDocumentWriteRefused`, ...) |
| 4 | a required file was missing |

A refusal is exit 3, not a crash. If you are scripting around this, treat 3 as
"the tool declined and told you why".

## The daily pass

```bash
asi daily                       # DRY RUN. Reports what it would do, persists nothing
asi daily --commit              # persists to state/ ; still cannot touch a skill file
asi daily --commit --day 2026-01-02 --seed t1   # pinned day and seed: reproducible
```

Dry run is the default deliberately. `--commit` changes only whether `state/` is
written; it does not unlock any write to a skill file, and no flag does.

## Reading a receipt

One receipt per run, under `state/receipts/`. `--json` prints it. The fields worth
reading first, in order:

**`status`** — `ok`, `partial`, `stood-down`, or `blocked`. `partial` is normal: it
means at least one record or skill was quarantined or deferred while the rest
completed. `blocked` means the skill tree changed during the run; see below.

**`skill_tree.digest_before` / `digest_after`** — must be equal after any daily
run. If they differ, something changed a skill file during the run, from *any*
source. `mutation_detected` states it directly. This is the single most important
line in the receipt.

**`counts`** — the honest arithmetic:

```
conversations_ingested        what came back from the source
records_scored                what actually got a verdict
records_quarantined           malformed or unverifiable, one file each with a reason
records_deferred              retryable, e.g. the source was unreachable
skills_quarantined            a proposal target that is defective (implicated but absent), or
                              a skill that kept deferring past caps.max_defer_attempts
skills_deferred               a proposal that could not be made, e.g. the improver timed out
failure_cases                 FAIL verdicts with resolvable evidence
failure_cases_unattributed    real failures NOT tied to a skill (abstention, not absence)
candidates_proposed           proposals written
candidates_gate_passed        proposals with no-regression evidence
candidates_gate_failed        kept, with reasons, for a human to read
proposals_refused_by_policy   targets declined by a guard rule
open_candidates_after         the review backlog
```

`records_scored` will be lower than `conversations_ingested` whenever anything was
quarantined or deferred. That is the intended shape; it is not a bug to chase.

`records_*` count conversations only. Skill-level outcomes of the propose and
validate stages live in the receipt's `skills` table and the `skills_*` counts,
because the remedies differ: a quarantined record is a data or normalizer problem,
a deferred skill is almost always a model problem. Either kind makes the run
`partial`.

**`discovery.is_total`** — whether the source returned everything or a page. When
`false`, `any_page_truncated_by_limit` is `true` and `honesty_note` says so. Do not
compute a rate over a page and call it a rate over the corpus.

**`selection.bucket_coverage`** — fraction of buckets that contributed. A run that
touched one bucket of five looks fine by count and is not representative.

**`classes`** — the authorship classification per record, with reasons. Machine
traffic and human sessions are different populations; a loop reading mostly
generated threads is grading the orchestrator, not the skills a person uses.

**`evaluator_independence`** — whether the judge, improver and scorer commands are
distinct, plus a note on what that check cannot see.

**`isolation`** — the source class, the confinement mode the roles actually ran
under, and how many environment variables were dropped before the child started.
A receipt for a private source that does not name a confinement mode is a receipt
from a run that should not have happened; the run refuses first, so you should
never see one.

**`notes`** — refusals and standing-down reasons in plain language.

## Reviewing and applying

```bash
asi candidates
asi review <candidate-id>
```

`review` prints the failing cases with their evidence, the diff, the validation
result (fixes, regressions, holdouts, judge preference), and the approval token.
Read the *evidence*, not the summary. The question to ask is not "does this edit
look reasonable" but "does the cited turn actually show the problem this edit
claims to fix".

```bash
asi apply <candidate-id> --approve <approval-token> --i-reviewed-this
```

Refusals you should expect, and what each means:

| Refusal | Meaning |
| --- | --- |
| `AutomationRefused` | you are inside a scheduled run. Apply from an interactive shell |
| `BaseHashMismatch` | the skill file changed since the candidate was built. The evidence no longer describes this file; revalidate |
| `MirrorEditRefused` | the target is not a canonical editable root |
| `CredentialInSkill` | the skill file holds a token, key or key block, so no model role was shown it and nothing was proposed. Remove the credential from the skill; the next pass proposes normally |
| `NonDocumentWriteRefused` | the target is not a `SKILL.md` |
| `BackupUnusable` | rollback's backup or manifest does not verify, or the target is now a symlink. The live file was not touched; find an older backup under `state/backups/` |
| `BaseHashMismatch` on rollback | the skill was edited after this apply, so the backup would overwrite work the candidate never touched. Nothing was written; restore by hand from the backup path in the message if that is what you want |
| `SafetyRefusal` | the candidate did not pass the gate, the skill is protected, or a `state/` permission could not be repaired |
| `IsolationRequired` | the source is private and the model roles are not confined by a verified OS sandbox. Install the mechanism or use the fixture corpus |
| status already `applied` | re-applying is refused. Use `rollback` first if you want to redo it |

## Rollback

```bash
asi rollback <candidate-id>
```

Restores the pre-apply backup and verifies the restored hash against the recorded
base. Run it the moment an applied change looks wrong; there is no reason to
diagnose first.

The backup and its manifest are verified before anything is written, so a
`BackupUnusable` refusal means the applied file is still exactly as it was and
you can try a different backup directory under `state/backups/`.

Rollback overwrites only the text its own apply wrote. If the skill has been
edited since (by you, or by applying another candidate), it refuses with
`BaseHashMismatch` and leaves the file alone. Running it twice is safe: the
second run finds the previous text already in place and reports
`already_restored` without writing.

## Recovery

**A run crashed mid-pass.** Re-run it. The day key makes the pass idempotent:
existing records are re-used rather than re-scored. Nothing needs cleaning up.

**A stale lock.** `state/lock` is advisory and released when the process exits. If
a machine died hard, remove the file — but check first that no `asi` process is
running, because two concurrent runs interleaving writes is exactly what the lock
exists to prevent.

**The queue is full and runs keep standing down.** That is the backlog guard
working. Review or discard candidates, or raise `max_open_candidates` if you have
decided the queue can carry more. Do not remove the guard.

**Records keep landing in quarantine.** Read the reason files; each one names the
defect. A repeating shape defect usually means the source changed its export
format, which is a normalizer fix, not a data problem.

**The same record defers every day.** Deferral is for transient failures, and a
deferred record is retried first on the next run. That priority is a ratchet if
the failure never clears: enough such records hold every selection slot, every
day, and fresh conversations starve. So after `[caps].max_defer_attempts`
consecutive misses (default 3; `1` means quarantine on the first miss, never
retry) the record is quarantined under `<stage>:defer-exhausted` instead, with
the attempt count and last reason in the quarantine entry, and it stops claiming
priority. A completed score resets the streak. The same cap covers a skill whose
proposal or validation keeps deferring (a model that times out on that one
skill); a completed propose-and-validate resets that streak too. A persistent
miss is not transient; read the reason rather than waiting for the retry to work.
The quarantine *file name* flattens `:` to `_`, so match on the `stage` field
inside the JSON, not on the name.

**A skill tree mutation was detected.** Do not re-run. Find out what wrote to the
tree. The candidate evidence gathered during that run described a file state that
no longer exists.

## Housekeeping

`state/` grows: raw exports, records, receipts, backups. It is all local and all
gitignored, with `0700` directories and `0600` files. Keep backups as long as you
might want to roll back. Raw exports are the most sensitive thing in the tree and
are the safest to delete, because they are re-fetchable from the source.

`state/sandbox/` should be empty between runs: each model call gets a fresh
`0700` directory that is removed when the call ends. Leftovers mean a process was
killed mid-call; they are safe to delete, and worth reading first if you are
diagnosing a crash.

## Health, honestly

There is no dashboard, and one would be misleading. Two questions cover it:

1. Is the newest receipt recent? If not, the schedule is not firing, whatever the
   scheduler says about the job existing.
2. In the last few receipts, are `digest_before` and `digest_after` equal? If ever
   not, stop and investigate.
3. Does every receipt for a private source name a confinement mode? A missing one
   cannot happen through this code path, so its absence means somebody edited a
   receipt or ran an older build.

Everything else — counts, rates, verdict mixes — is input to your judgement about
the *rubrics*, not a health signal about the loop.
