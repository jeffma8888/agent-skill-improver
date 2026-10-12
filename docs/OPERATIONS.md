# Operations

## First run

```bash
asi --config config/example.config.toml doctor
```

`doctor` validates the configuration and reports what is actually wired: the
**source class and isolation mode**, state-directory permissions, source
reachability, model roles, skill roots present versus configured (a root whose path
holds something other than a directory, say a regular file or a dangling link, is
named as that in a `WARN` rather than reported as absent), how many skills
were found, how many **do not load**, how many carry executable payloads (a
`WARN` names any skill with a directory the payload inventory could not list,
since its count is then a floor), how
many paths discovery **passed over** (each one a `WARN` line naming
the path and the reason: a dangling link, a link while links are not followed,
a link that leaves its root, a document deeper than `max_depth`, a `SKILL.md` at
the root itself, a `SKILL.md` below a directory that carries its own `SKILL.md`
entry (whatever it says, it sits inside another skill's directory), a directory,
fifo, socket or device carrying the name, a directory that could not be listed,
a configured root whose path is not a directory),
any skill with **two documents in the root that would be edited** (a `WARN`
naming both; the resolver refuses to pick one by walk order, `skills` shows the
skill with no canonical copy and a `why:` line, and `daily` quarantines it at
propose until one copy remains), the selection bound, and whether the judge is
independent of the improver. `skills` lists the same passed-over documents under
its table. It exits
non-zero if the configuration is not runnable. Run it after every config change;
most problems in this system are configuration problems that otherwise surface as
a strange result three stages later. Every table in the config file is a closed
schema: a key this tool does not read is a config error naming the nearest key it
does (`[caps] has unknown key(s): max_open_candidate (did you mean
max_open_candidates?)`), never a setting that loads silently and leaves the
default in force.

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

`doctor` also applies the role-environment rules the run applies at its first
model call: a `[model.<role>].env` name matching a credential, token or proxy
pattern, or a value shaped like a credential whatever it is named, is a `PROBLEM`
here rather than an `IsolationRequired` refusal discovered by a scheduled run.
The report names the variable and the shape, never the value.

`doctor` also repairs `state/` permissions on the way through and reports what it
changed. A repair that cannot take effect is a refusal, not a warning. An entry
inside the tree that is not a regular file or directory (a symlink, a fifo or a
socket some other tool left there) is neither repaired nor removed: this tool
only creates regular files and directories, so its mode is not this tool's to
set; every state record read, the backup manifest read and the run lock refuse
such a leaf by name without blocking on it (the backup *file* is judged by its
hash instead, so a link to the right bytes is harmless), and a write replaces
rather than follows it. `doctor` lists such entries as `left_alone` and
prints a `WARN`, so a tree is never called clean while a foreign entry sits in it.

`doctor` also reads every record `daily` would refuse before `daily` has to. A
candidate record that does not parse is a `PROBLEM` (`candidate_records_unreadable`
in JSON): the next run stands down with the queue uncounted. A day checkpoint that
does not parse, or parses to the wrong shape, is listed under
`checkpoint_records_unreadable`; today's is a `PROBLEM`, because the next `daily`
refuses by name before it scores anything, and another day's is a `WARN`, because
only `daily --day` for that day would. A foreign entry at today's checkpoint path
is a `PROBLEM` for the same reason: it is the next run's first read, and a read
refuses it. The run lock, `state/locks/daily.lock`, is the very first thing `daily`
touches, so a link, a directory or a fifo at that path is a `PROBLEM` too (`daily`
refuses it by name; a regular file a finished run left behind is not a problem, the
lock is `flock`, not the file). Every one of these reads leaves the entry as found;
the remedy in each reason is to move it aside or repair it by hand.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success, including "stood down for a stated reason" |
| 1 | the command ran and the answer is negative (`doctor` not ready, `selftest` failed, a gate not met) |
| 2 | usage error: an unknown subcommand or flag, or a value argparse refuses before any config is read (a `--day` that could not name a candidate, a `--limit` below 1); argparse's code, never returned by a command |
| 3 | a refusal or a domain error, with the exception class named on stderr (`SafetyRefusal`, `BaseHashMismatch`, `AutomationRefused`, `MirrorEditRefused`, `NonDocumentWriteRefused`, `ConfigError`, ...) |
| 4 | a required file was missing |
| 5 | `daily` is `blocked`: the live skill tree changed while the run was reading it. Not a negative answer and not a refusal. Do not re-run until you know what wrote to the tree |
| 70 | an exception no command anticipated, with its traceback on stderr. A bug report, never a negative answer; 70 is `EX_SOFTWARE` from `sysexits.h` |

A refusal is exit 3, not a crash. If you are scripting around this, treat 3 as
"the tool declined and told you why", and 70 as "file a bug": no crash inside a
command borrows 1, so a negative answer from `doctor` or `selftest` stays unambiguous.
A configured conversation command that cannot be started is a 3 naming the
command and why, for the discovery program and the export program alike. Not on
PATH, not executable or a directory is caught by the source preflight, so
`doctor` reports it (exit 1, its negative verdict) with the file mode when the
executable bit is what is missing; a wrong binary format is only seen when the
start itself fails, and is the same 3. A model role command that cannot be
started is reported the same way: `doctor` lists it as a `PROBLEM`, `daily`
refuses with `ConfigError` before any model call, and a start that fails
mid-run is a transient failure of that role, never an internal error. The
definition files and the state directory are held to the same rule: a config
file or an enabled rubric that exists but cannot be read, or is not UTF-8, is a
`ConfigError` naming the file and the reason, and a `[run].state_dir` that
cannot be created (a read-only parent, a mount that refuses) is a
`ConfigError` naming the directory and the config key; `doctor` lists the
uncreatable state directory as a `PROBLEM` and still produces the rest of its
report, `daily` and `candidates` refuse with exit 3.

## The daily pass

```bash
asi daily                       # DRY RUN. Does the work, prints the receipt, writes NOTHING to state/
asi daily --commit              # persists to state/ ; still cannot touch a skill file
asi daily --commit --day 2026-01-02 --seed t1   # pinned day and seed: reproducible
```

Dry run is the default deliberately. `--commit` changes only whether `state/` is
written; it does not unlock any write to a skill file, and no flag does. "Written"
is meant literally and is tested on the whole tree: a dry run ingests, scores,
proposes and validates exactly as a committed run does, but the store withholds
every artifact (normalized conversations, raw exports, scorecards, failure cases,
quarantine, refusal and deferral records, candidates, checkpoint and receipt), so
the only things it leaves behind are the owner-only directory layout and the run
lock. Every `daily` run, dry or committed, also resets any existing state path whose
mode drifted from owner-only (`state_modes_repaired`, as `doctor` does); that repair
writes no content, so it is not withheld. The receipt still reports how many files it withheld
(`artifacts_withheld_dry_run`) and `--json` prints it. For a private source this is
the difference between a preview and a copy of the transcripts on disk.

`--day` is the run's idempotence key and the middle of every candidate id
(`CAND-2026-01-02-01-tdd`), so it is held to the id's alphabet: ASCII letters,
digits, dot, underscore or hyphen, at most 64 characters, starting with a letter
or digit. A value that cannot serve (`2026/01/02`, whitespace) is refused by the
command line with exit 2, before the config is read or a conversation ingested,
and a slash-spelled date is told the hyphenated spelling it wanted; an empty
`--day ""` is the same as none and means today. The skill directory name has no
length rule of its own, so once the skill tree is read (still before anything is
ingested) the pass forms, for every known skill, the longest id a first run of the
day could need and refuses with `ConfigError`, naming the day, the skill and both
lengths, if one would not fit the 204-character candidate id. Stated limit: a day
re-run more times than the candidate cap allows can push the sequence number one
digit wider than probed. Measured
before this check: `--day 2026/01/02` ran the whole pass, scored every
conversation at full model cost, wrote them under `--commit`, and then died
building its first candidate id with `invalid candidate id`, a message about
the wrong thing; no receipt or checkpoint was written, so the next run with the
same day repeated all of it. A library caller gets the same refusal as a
`ConfigError` from `run_daily`, before the source is touched.

## Reading a receipt

One receipt per run, under `state/receipts/`. `--json` prints it. The fields worth
reading first, in order:

**`status`** — `ok`, `partial`, `stood-down`, or `blocked`. `partial` is normal: it
means at least one record or skill was quarantined or deferred while the rest
completed. `stood-down` carries a `stand_down_reason`. One stand-down, "no skill
to improve" (no configured root is a directory, or the present roots hold no
skill document), is decided before discovery, so no conversation is read or
scored for a proposal that could not exist; `doctor` reports the same state as a
`PROBLEM`. `blocked` means the skill tree changed during the run; see below.

**`skill_tree.mutation_detected`** — must be false after any daily run. It is
true when any skill document's tree entry differed between the run's first and
last reading, from *any* source; `digest_before` / `digest_after` (over the
readable documents) then usually differ too, but not always, so read
`mutation_detected` and `changes`, not the digests. This is the single most
important line in the receipt.

**`skill_tree.changes`** — when `mutation_detected` is true, one entry per document
whose tree entry differed between the run's first and last reading, as
`{"key": "root:skill/SKILL.md", "kind": ...}`. The kind says what was seen:
`content-changed` is an edit (the hash moved); `added`, `removed`,
`became-unreadable`, `became-readable` and `added-unreadable` are changes to the
tree the run could not compare content across, and the note says so instead of
calling them an edit. Any kind blocks the run: something other than this tool was
in the tree while candidates were being judged against it. A document that is
unreadable for the whole run is in neither reading's digest and is not a change;
it is in `skill_documents_passed_over`.

**`skill_tree.skill_documents_passed_over`** — every `SKILL.md` discovery saw and
did not take this run, with its root and reason (a link, a fifo, a document too
deep, one at the root itself or below another skill's directory, a document this
process cannot read). A skill is `ROOT/NAME/SKILL.md`, or the same under a
grouping folder: a document at the root has no directory to be named by (measured
before: it was inventoried under the root folder's own basename), and one below
a directory that carries an entry named `SKILL.md`, readable or not, sits inside
that skill's directory and is treated as its material (measured before:
`tdd/examples/SKILL.md` was a skill called `examples`, and a proposal against it
would have written inside `tdd`). A skill whose every copy is here is in
none of the counts beside it, and this list is the only unattended surface that
says so; `doctor` and `skills` print the same entries. A copy passed over in an
editable root also withholds that skill's canonical slot: `skills` shows the skill
with no canonical copy and a `why:` line naming the copy, and `daily` refuses to
propose against it rather than point a proposal at another copy.

**`counts`** — the honest arithmetic:

```
conversations_ingested        what came back from the source
records_scored                what actually got a verdict
records_quarantined           malformed or unverifiable, one file each with a reason
records_deferred              retryable, e.g. the source was unreachable
skills_quarantined            a proposal target that is defective (implicated but absent,
                              or not valid UTF-8: `UndecodableDocument`, byte named), or
                              a skill that kept deferring past caps.max_defer_attempts
skills_deferred               a proposal that could not be made, e.g. the improver timed out
failure_cases                 FAIL verdicts with resolvable evidence
failure_cases_unattributed    real failures NOT tied to a skill (abstention, not absence)
skills_implicated             distinct skills cited by at least one attributed failure
skills_selected_for_proposal  of those, how many were handed to the improver this run, after
                              the open-candidate filter and the smaller of
                              caps.max_skills_per_run and caps.max_candidates_per_run
candidates_proposed           proposals this run made; written on a committed run, withheld
                              on a dry run (each one's `persisted` flag says which)
candidates_gate_passed        proposals with no-regression evidence
candidates_gate_failed        kept, with reasons, for a human to read
proposals_refused_by_policy   targets declined by a guard rule (includes the next line)
skills_with_open_candidate    implicated skills skipped because a candidate of theirs awaits review
candidate_records_unreadable  files under state/candidates/ that do not parse as a candidate; one
                              or more stands the run down (`queue guard`) because the open queue
                              cannot be counted, and the file is left exactly as found
open_candidates_after         open-status candidate files on disk, what a bare `asi review`
                              lists; a dry run's own proposals are never in it, and neither
                              is an unreadable record (see candidate_records_unreadable)
state_quarantine_files        files under state/quarantine/ after the run, ALL days, so this
state_refusal_files           and the two beside it are cumulative backlog rather than
state_deferred_files          this run's work; in a dry run they reflect earlier commits only
artifacts_withheld_dry_run    state writes (and deferral-record removals) a committed run would
                              have made and this dry run withheld; 0 on a committed run. Taken
                              before the receipt and the day checkpoint, which are withheld too
state_modes_repaired          existing paths under state/ whose mode was not owner-only and
                              was reset to it before the run; a dry run repairs them too,
                              since a repair writes no content (the note names them)
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

**`discovery.passed_over`** — every entry the source saw in its window and could
not take as a thread, with the reason. For the fixtures source: an export this
process cannot read, one that is not UTF-8 or not JSON, a JSON value that is not
an object, an object with no thread or a thread without an id; `doctor` lists the
same entries as `WARN` lines under the source, and a fetch that cannot find a
discovered thread while such entries exist quarantines the record as "no readable
fixture" naming them, not as a thread that was never there. A directory the
process cannot list is not an empty window: `doctor` reports it as a `PROBLEM`
and `daily` refuses with `SourceUnavailable`, since zero ingested with `is_total`
true would otherwise be the same receipt as a directory with nothing in it. For
the command-line source: a listed "Thread:" line whose id is option-shaped,
empty, over-long or carries
whitespace or control characters (shown as a cut-short repr, since a listing is
derived from conversation content and can carry a forged line), a listed entry
with no "Thread:" line the parser could read, or one whose "Thread:" line is
blank or not an id; an unsafe id is never placed on a command line, and `doctor`
cannot list these without running discovery, so the receipt is their only
surface. A listing is numbered per query, so for this source, when a workspace
was queried, the entry names it ("listing entry 1 ('x') in workspace /a"); the same number in
two workspaces is two entries. Each is in no count on the receipt and a note
names them. `is_total` is unaffected: the window was read whole; part of it did
not parse.

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

The hypothesis, intended change, risks and rollback note are the model's own
words, shown as written but quoted: every line of them opens with `  | `, or
with `  - ` for the first line of a risk. Nothing the tool says carries that
prefix. The real `asi apply ...` command is the two-space-indented line right
under the sentence that begins "To promote this change", which itself stands
at the left margin after `DIFF`. A `DIFF` or an apply
command behind a `  | ` or `  - ` prefix was written by the model and promotes
nothing; a skill id or path with a line break in it is shown with the break
spelled as `\n`, in the `skill` and `file` lines and in the diff's own
`---`/`+++` labels alike.

```bash
asi apply <candidate-id> --approve <approval-token> --i-reviewed-this
```

Refusals you should expect, and what each means. The name is what the CLI prints
on stderr before the message, and a refusal that reaches stderr exits 3. Two of
these are normally caught inside the daily pass instead and land in the receipt's
`skills` table as `refused` (exit 0 or 1): `CredentialInSkill`, and
`SelfMutationRefused` for a protected skill at proposal time.

| Refusal | Meaning |
| --- | --- |
| `GateFailed` | on `review`/`apply`: the candidate is not exactly `validated`, because it failed the gate or is closed (`applied`, `rolled-back`, `gate-failed`, `blocked-run`); a closed candidate never exposes a token or an apply command again. On `rollback`: no backup manifest exists for the candidate, so there is nothing to roll back |
| `CandidateTampered` | a derived field of the candidate record (patch hash, approval challenge, diff, target identity, status against the recorded gate, validation binding) does not recompute from its inputs. The file was edited after it was written; nothing was backed up or applied. Do not "fix" the record: re-propose |
| `ApprovalRequired` | no token was passed, `--i-reviewed-this` is missing, or the token does not match the challenge recomputed from the candidate id, base hash and patch text. Re-run `review` and copy the token it prints now |
| `AutomationRefused` | you are inside a scheduled run. Apply from an interactive shell |
| `BaseHashMismatch` | on `review` or `apply`: the skill file changed since the candidate was built, or what is at its path is not a regular file (a fifo or a device: refused on the open handle rather than read or waited on). The record is intact as far as it can be checked without the file (a record that fails its own recomputation is `CandidateTampered`, checked first); the evidence no longer describes this file, so re-run `asi daily --commit` to propose and validate against the new base |
| `MirrorEditRefused` | the target is not a canonical editable root |
| `CredentialInSkill` | the skill file holds a token, key or key block, so no model role was shown it and nothing was proposed. Remove the credential from the skill; the next pass proposes normally |
| `NonDocumentWriteRefused` | the target is not a `SKILL.md` |
| `UndecodableDocument` | on `review` or `apply`: the skill document is not valid UTF-8 at the byte named. In `daily` the same finding quarantines that one skill (`skills --json` carries the warning on its snapshot) and the run goes on; nothing is proposed, shown or applied against text the tool cannot read exactly. Fix the encoding and re-run |
| `LinkedTargetRefused` | on `review`, `apply` or `rollback`: the skill document (judged `O_NOFOLLOW` on every read of it, so `review` shows no token for it), or a directory beneath its root on the way to it, is now a symbolic link, or the target is spelled outside every configured root yet resolves into a canonical one, so the write would land wherever the link points (a same-text twin would pass the hash check). Nothing was read through the link, backed up or written; replace the link with the real file and re-run `review` |
| `BackupUnusable` | rollback's backup or manifest does not verify, or the target is now a symlink. The live file was not touched; find an older backup under `state/backups/` |
| `BaseHashMismatch` on rollback | the skill was edited after this apply, so the backup would overwrite work the candidate never touched. Nothing was written; restore by hand from the backup path in the message if that is what you want |
| `SelfMutationRefused` | the target is a protected skill (`[guard].protected_skill_ids`) or sits inside a protected path (this tool's own source, config or rubrics). Never a proposal target and never writable, by design |
| `ReadOnlyViolation` | something tried to write through a skill store opened read-only. The daily driver always opens it that way; only `apply` and `rollback` open it writable, so seeing this outside them is a bug report |
| `SafetyRefusal` | the residual class: the state directory (or one of its ancestors), a backup directory or the lock leaf is a symlink or not owner-only and could not be repaired; the daily driver was handed a writable skill store, or `apply` a read-only one; or promotion was attempted through a dry-run store. The message names which |
| `LockHeld` (`daily`) | another `daily` run holds `state/locks/daily.lock`; the message shows who. Wait for it, or see Recovery if the holder is dead |
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
you can try a different backup directory under `state/backups/`. Each directory
there is named `<candidate id>--<stamp>`, and rollback uses only the newest one
whose candidate id is exactly the one you named.

Rollback overwrites only the text its own apply wrote. If the skill has been
edited since (by you, or by applying another candidate), it refuses with
`BaseHashMismatch` and leaves the file alone. Running it twice is safe: the
second run finds the previous text already in place and reports
`already_restored` without writing.

## Recovery

**A run crashed mid-pass.** Re-run it. The day key makes the pass idempotent:
existing records are re-used rather than re-scored. Nothing needs cleaning up.

**A stale lock.** `state/locks/daily.lock` is an `flock`, released by the kernel
when the process exits, however it exits; the file that remains is inert and
`doctor` does not flag it. If `daily` reports `LockHeld`, another `asi` process is
alive and holds it: find it before anything else, because two concurrent runs
interleaving writes is exactly what the lock exists to prevent.

**The queue is full and runs keep standing down.** That is the backlog guard
working. Review or discard candidates, or raise `max_open_candidates` if you have
decided the queue can carry more. Do not remove the guard.

**A skill has a validated candidate waiting, and later runs propose nothing new
for it.** That is `max_open_candidates_per_skill` (default 1) holding: the
receipt's `skills` table shows the skill as `refused` with "open candidate(s)
already target". Two open drafts for one file cannot both be applied, so the run
waits for you to review or close the existing one rather than queue a competitor.
A `gate-failed` candidate is closed and does not hold the skill; a skill whose
proposals keep failing the gate is re-proposed each run.

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

**A skill tree mutation was detected.** The run is `blocked` and exits 5. Do not
re-run. Find out what wrote to the tree. The candidate evidence gathered during
that run described a file state that no longer exists, so every candidate the run
persisted was closed as `blocked-run` with that reason in its notes; `apply`
refuses them, and the proposal text stays readable for the post-mortem. If the
receipt says a candidate could NOT be rewritten, its file may still read
`validated`: do not apply it.

**`StateUnreadable: refusing to use the checkpoint record ...`** (exit 3). A state
file under `state/` is truncated, hand-edited, or not a JSON object. Nothing is
guessed at and the file is left as found; read it, then move it aside to continue
without it. For a checkpoint that means the day's already-processed set is
forgotten and those conversations are scored again. That re-spends model calls;
a skill whose candidate from the first run is still open is skipped by the
per-skill cap, and a skill whose candidate was closed gets a fresh `-02-`
proposal. Nothing on disk is overwritten or mutated.

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
2. In the last few receipts, is `skill_tree.mutation_detected` false? If ever not,
   stop, read `skill_tree.changes` for which documents and what was seen, and
   investigate.
3. Does every receipt for a private source name a confinement mode? A missing one
   cannot happen through this code path, so its absence means somebody edited a
   receipt or ran an older build.

Everything else — counts, rates, verdict mixes — is input to your judgement about
the *rubrics*, not a health signal about the loop.
