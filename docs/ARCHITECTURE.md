# Architecture

## The shape of the problem

The loop edits the instructions an agent obeys. That single fact decides the whole
design: every component is arranged so that the automated half can *argue* for a
change and only a human can *make* one, and so that the automated half cannot
quietly widen its own authority.

Three things follow, and they are worth stating before any module names.

1. **The examiner is not writable by the examinee.** Rubrics are files on disk,
   loaded read-only, hashed, and the hash is stamped on every scorecard. The
   tool's own source directory, its config file, its config directory, its state
   directory and its rubric directory are protected unconditionally, and that set
   is derived in code rather than read from configuration. A configurable
   protection list is not a protection list.
2. **Evidence is not the same as a verdict.** A model's `FAIL` is accepted only
   when it cites a turn that actually resolves in the transcript at the position
   claimed. Everything else becomes `INSUFFICIENT`, which earns no credit in
   either direction. A loop that treats *could not tell* as *broken* manufactures
   its own work.
3. **Refusals are per record.** One malformed export, one unfetchable thread, one
   unverifiable citation stops that record and nothing else.

## Layers

Dependencies point inward. `domain/` performs no I/O and imports neither
`adapters/` nor `app/`; that is what makes the rules testable without a
filesystem, a subprocess or a clock.

```
  cli.py                     argument parsing, exit codes, human-readable output
    │
    ▼
  app/                       use cases: sequence the stages, write state, decide
    │   daily.py             the automated pass (never promotes)
    │   promote.py           the human pass (the only skill writer)
    │   pipeline.py          score → isolate → propose
    │   validate.py          baseline vs candidate replay, blind judge, gate
    │   prompts.py           prompt assembly, untrusted content fenced
    │   state.py             on-disk records, quarantine, defer, refusals
    │   receipt.py           what the run actually did
    │   lock.py              single holder per state directory
    │   wiring.py            the only place that chooses adapters
    │
    ├──▶ ports.py            the interfaces app/ depends on
    │
    ├──▶ adapters/               processes, filesystem, clock
    │      conversation_cli.py   a configured discovery/export CLI
    │      transcript_export.py  export JSON → normalized conversation
    │      fixture_source.py     the same contract over a directory of JSON
    │      model_command.py      any command that reads one JSON object and writes one
    │      sandbox.py            OS no-egress confinement (sandbox-exec / bubblewrap)
    │      skill_fs.py           skill discovery, snapshots, the single write path
    │
    └──▶ domain/                 pure rules, no I/O
           conversation.py       normalized transcript, stable element ids
           authorship.py         human vs orchestrated vs unknown
           isolation.py          source classification and the confinement requirement
           discovery.py          a page of results, and what a full page means
           selection.py          bounded fair sampling
           rubric.py             criteria, weights, the three-way verdict
           scoring.py            parse the model, then distrust it correctly
           failure.py            isolate failures, attribute to a skill or abstain
           candidate.py          the only artifact the loop may emit
           validation.py         replay comparison and the promotion gate
           skills.py             skill identity, canonical vs mirror, payloads
           approval.py           review interlock, not a security boundary
           redaction.py          best-effort secret masking, defence in depth only
```

## The isolation boundary

The loop reads conversation history and hands it to model roles that are ordinary
child processes. Those two facts together are the whole risk: the content is
attacker-influenceable (anything a person or a tool pasted into a transcript ends
up in a prompt), the child can be any executable the operator configured, and a
child that inherits the parent environment and an open network is a complete
exfiltration path. Prompt fencing and regex redaction reduce the *chance* that a
model cooperates with an injected instruction; neither removes the *capability*,
so neither counts as a boundary here.

So the boundary is mechanical and it is decided from **source classification**,
not from a trust flag anybody can set:

- `domain/isolation.py` maps a configured `[source].kind` to a class.
  `fixtures` is `synthetic`; **every other kind, including an unknown one, is
  `private`** — the mapping fails closed, so adding a source without thinking
  about it lands on the strict side.
- A private source requires that every model role (scorer, improver, judge) runs
  under an OS confinement mode that is *verified*, not merely requested. If it
  does not, `IsolationRequired` is raised.
- `app/daily.py` raises it as the **first statement of `run_daily`**, before the
  lock, before the skill hash, before discovery — so a misconfigured private run
  cannot read one byte of private history, let alone write it to `state/raw/`.
- `asi doctor` performs the same decision and reports it as a PROBLEM, so the
  failure is visible before a scheduled run ever fires.

`adapters/sandbox.py` supplies the mechanism and the proof:

- **macOS** — `sandbox-exec` with a generated profile: default deny, then allow
  process execution and reads, then `(deny network*)` and `(deny file-write*)`
  with a single `(allow file-write* (subpath "<per-call temp dir>"))`. Rule order
  matters and a test asserts it, because in this profile language a later rule
  wins and a deny placed after the allow would silently confine nothing.
- **Linux** — `bubblewrap` with `--unshare-net`, a read-only root bind, and one
  writable bind for the per-call temp directory. The working directory is
  deliberately *not* changed, so repo-relative argv still resolves.
- **Neither available** — no fallback, no degraded mode. The decision refuses.

`verify()` is two-sided and runs before the mode is trusted: it starts a loopback
listener in the parent and executes a probe under the candidate wrapper that
tries (a) to connect to that listener and (b) to write a file outside the temp
directory. The mode is accepted only if **both attempts fail** and a control write
*inside* the temp directory succeeds. A confinement that blocks everything,
including the work, is not confinement either.

The environment is scrubbed independently of the sandbox, because
`--unshare-net` does nothing about a token that a child can write into its own
output. `minimal_env()` builds an allowlist (`PATH`, locale, `TZ`) plus `HOME`
and the temp variables pointed at the per-call directory, and drops everything
whose name matches a credential/token/proxy pattern. The count of dropped
variables and the resolved mode are recorded in every command id and every
receipt, so a run's confinement is auditable after the fact rather than assumed.

The synthetic exception exists so public CI stays portable: fixtures are files
this repository generates, contain no private content, and are the only class
allowed to run the deterministic fake model without an OS sandbox.

## The daily pass, stage by stage

### 1. Discover

The configured discovery command returns a page. The subtlety that decides whether the
whole loop is honest is that **a page is not a total**. The interface's own
default page size is small; a harvester that accepts the default reads the most
recently touched sliver of the corpus. So the source always passes its limit
explicitly, and a page that comes back exactly full is reported as
`truncated_by_limit = true` with `is_total = false`. The receipt carries that flag
and a human-readable honesty note. `discovery.page_from_threads` *derives* the
flag from the returned count rather than trusting a caller to set it.

### 2. Normalize

Export JSON becomes a `NormalizedConversation`: an ordered list of elements, each
with a stable id derived from its position and content, which is what makes a
citation checkable later. Shape defects (`elements` not a list, a thread without
an id, unparseable JSON) raise `RecordDefect` and that one record is quarantined
with the reason written next to it.

Truncation, when a cap binds, drops elements from the **middle** and says how many
were dropped. Head truncation would systematically grade only the opening of a
conversation, which is the same wrong-denominator error as sampling an
alphabetically sorted prefix.

### 3. Classify authorship

In an environment where an orchestrator drives agents continuously, generated
threads can be the large majority of everything the search returns. Grading them
as if they were human sessions moves the loop's attention away from the cases
where a skill actually mis-served a person.

Two classifiers exist on purpose, with different vocabularies:

- **Pre-fetch**, from a title alone: `orchestrated` or `not-agent-titled`.
  Deliberately *not* "human", because a title cannot prove human authorship.
- **Post-fetch**, from the transcript: `human`, `orchestrated`, or `unknown`.
  An unmatched thread is `unknown` and gets its own selection bucket; it is never
  folded into `human`.

The markers are configurable, because hard-coding one orchestrator's title makes
the classifier useless everywhere else.

### 4. Select

Bounded, and **never `sorted(items)[:limit]`**. Whenever the sort key correlates
with the property being selected, a prefix is a permanent filter rather than a
sample: an alphabetical cut over topic-prefixed names silently excludes whole
categories, and the result reads as "under-represented" rather than
"unreachable". Selection is round-robin across buckets of the configured fairness
dimension (`class` by default), best-first inside a bucket. A thin bucket
contributes everything it has and drops out, so no budget is reserved and wasted.
Unclassifiable records get their own bucket rather than being dropped, because an
unreadable record is exactly the one a human needs to see.

`selection.bucket_coverage` is recorded in every receipt: a run that touched one
bucket out of five looks healthy by count and is not.

### 5. Score

The scorer is asked one rubric at a time and must answer with a JSON object.
Then the answer is distrusted, in this order:

- unparseable output → `MalformedModelOutput`, that record quarantined
- a scorecard for the wrong conversation id → refused (guards a crossed or cached answer)
- a criterion the rubric does not define → recorded as `extraneous`, not silently dropped
- a criterion the model did not mention → `INSUFFICIENT`, never `PASS`; silence is not a pass
- a `PASS` or `FAIL` whose evidence reference does not resolve, or resolves to a
  different position than claimed → downgraded to `INSUFFICIENT` with
  `downgraded_from` recorded

`PASS` is held to exactly the same evidence standard as `FAIL`. Grading only the
accusations produces a scorer that learns to approve.

### 6. Isolate

Only `FAIL` becomes a `FailureCase`. `INSUFFICIENT` never does.

Attribution is where a plausible-looking loop does real damage. A failure is
attributed to a skill only when the transcript shows that skill was in play. When
it does not, the case is recorded with `ATTRIB_NONE` and a `rubric-prior-only`
marker: the failure is real, the cause is unproven, and no skill is edited on a
guess. Unattributed cases are counted in the receipt so that abstention is
visible rather than looking like a quiet absence of findings.

### 7. Propose

The improver sees the failing cases and the current skill text and returns a
replacement. Constraints:

- the target must be a **canonical, editable** root. A skill present only in a
  mirror has no editable copy, and a proposal against it is refused
  (`MirrorEditRefused`) rather than written somewhere that the next sync
  overwrites.
- a skill in `[guard].protected_skill_ids`, or one holding an executable payload,
  is refused as a target and the refusal is written to `state/refusals/`.
- a proposal whose canonical snapshot carries payload files is annotated
  `ELEVATED REVIEW RISK`, because a skill directory can hold scripts and hooks
  and write access there is arbitrary code execution rather than documentation
  editing.

### 8. Validate

Baseline and candidate are replayed over the failing cases plus a holdout set of
cases that previously passed. The judge is asked blind: option order is assigned
from a seeded shuffle, so a positional preference cannot be mistaken for a
judgement.

The gate then requires, all of them:

- at least `min_fixes` genuine `FAIL → PASS` transitions. `FAIL → INSUFFICIENT`
  earns nothing: a change that makes the scorer unable to decide is not a fix.
- no regressions, and no holdout degradations. `PASS → INSUFFICIENT` on a holdout
  is a loss of evidence, not a neutral outcome.
- at least `min_holdout_cases` holdouts. Without a holdout, "no regressions" is an
  unsupported claim.
- a judge preference ratio at or above the configured floor.
- **an independent judge.** If the judge's command matches the improver's, the
  gate refuses and says so. A judge sharing a model with the thing it grades is
  grading its own homework, and the resulting number mostly measures
  self-consistency. Every receipt carries the independence comparison, so an
  example that cannot satisfy it (all three roles wired to one offline script)
  says so out loud rather than passing quietly.

A candidate that fails the gate is kept with `STATUS_GATE_FAILED` and its reasons.
Nothing is deleted; a human can still read why.

## What separates automated from human

The interlock is structural, not a flag:

- `app/daily.py` does not import `app/promote.py`. `asi selftest` asserts this by
  parsing the module, not by searching its text — the first version searched for
  lines containing `import` and matched the docstring sentence stating the
  invariant, so the checker read the denial as evidence of the thing denied.
- `daily` refuses a writable skill store outright.
- `daily` hashes the entire skill tree before and after and records both digests,
  so a mutation from *any* source during the run is visible even if it came from
  outside this tool.
- `daily` sets `ASI_AUTOMATED_RUN`, and `approval.HumanApproval.from_cli` refuses
  when that marker is present. An apply reached from inside a scheduled run fails
  even with a valid token.
- The only filesystem write path for a skill is `skill_fs.write_canonical`, and
  its first action is `_assert_document`, which refuses any target whose filename
  is not `SKILL.md`.

## Apply and rollback

`apply` verifies, in order: the automation marker is absent; candidate status is
`validated` and the recorded gate passed; every derived field recomputes to the
value the record stores; the current file hash equals the recorded base hash; the
diff and its counts recompute against that live base; the approval token matches
the derivation over candidate id, base hash and the recomputed hash of the patch
text; the acknowledgement flag is present; the root is canonical and editable; the
skill is not protected; the target filename is `SKILL.md`. Then it writes a
backup, writes the file atomically, verifies the resulting hash, and marks the
candidate `applied`. Re-applying is refused because the status already moved.

Every check above happens **before** the backup directory exists and before any
byte is written, so a refused apply leaves the skill file, the candidate status,
the backups and the receipts exactly as they were. The recomputation lives in
`domain/integrity.py` and is shared by `review`, `apply` and validation, because a
derived field stored beside its input is a way for the two to disagree: the token
used to be derived from the *stored* patch hash, so editing `patch.new_text` alone
left the token matching and put unreviewed text into a skill file.

`rollback` restores the backup through the same `_assert_document` gate and
verifies the restored hash against the recorded base. Everything it can check
without touching the live file it checks first: the manifest must carry a target,
a backup path and a previous hash, and the backup on disk must hash to that
previous hash. A backup that was truncated, edited or lost is therefore a
`BackupUnusable` refusal with the skill still in its applied state, rather than an
overwrite followed by the discovery that the restored content was never the
original. Recovery must not be the step that loses the file.

**What the approval token is and is not.** It is a deterministic function of the
candidate id, the base hash and the hash recomputed from the candidate's patch
text (never the stored hash field), derived with the same code on
both sides. That makes it a *review interlock*: it cannot be produced for a
candidate whose base file has changed since it was proposed, so a stale approval
fails and a silent edit underneath a pending review is caught. It is **not** a
security boundary and must not be described as one — anyone who can run this tool
can derive it, because the derivation is in the source and takes no secret. The
things that actually stop an automated apply are structural: `daily` never
imports `promote`, `daily` refuses a writable skill store, and the automation
marker in the environment makes `from_cli` refuse outright.

## State

Everything the loop reads or produces lives under the configured state directory,
which is gitignored in its entirety:

```
  state/                       0700
    raw/          verbatim exports (PRIVATE conversation content)
    records/      normalized conversations and scorecards
    quarantine/   one file per record that could not be processed, with the reason
    deferred/     one file per record to retry, e.g. a source that was unreachable
    refusals/     one file per action refused by policy, with the rule that refused it
    candidates/   proposals awaiting review
    backups/      pre-apply copies, used by rollback
    receipts/     one receipt per run
    sandbox/      per-call confinement working directories, removed after each call
    lock          single-holder advisory lock
```

Every directory under the state root is `0700` and every file the loop writes is
`0600`, including backups and receipts, and the modes are set explicitly rather
than left to the process umask — a permissive umask is common in scheduled
environments and this tree holds verbatim private transcripts. `state.ensure()`
creates, then *repairs*, then **verifies** the modes, and raises `SafetyRefusal`
if a mode cannot be corrected. Refusing to run is the right outcome: an
already-existing world-readable state directory is exactly the case a fresh-run
check would miss.

Making that tree private is only meaningful if it is the tree that was
configured, and `mkdir(exist_ok=True)`, `chmod` and every later write follow a
symlink. So the layout is **verified before it is followed**: the state root and
every declared subdirectory must be a real directory, measured with `lstat`, or
`StateStore` refuses — before the first `mkdir` or `chmod`, because either one
would already have carried private content outside the configured directory.
`layout_problems()` reports the same verdict without following anything, so
`asi doctor` surfaces an unsafe layout as a `PROBLEM` and deliberately does *not*
repair it: repairing a redirected tree means chmodding a directory this tool does
not own. Directories whose names are computed at runtime — a `backups/<candidate
id>--<stamp>` directory, for instance — are proven to be real direct children of
an already-verified mandatory directory before they are created or chmodded,
because a guessable name is one an attacker with write access to `backups/` can
precreate as a link. An arbitrary *leaf* symlink someone else left in the tree is
still skipped rather than followed, which leaves its target untouched.

`quarantine`, `deferred` and `refusals` are three different states on purpose.
*Malformed* is not *temporarily unavailable* is not *deliberately declined*, and
collapsing them loses the distinction between a bug, a retry and a policy
decision.

## Idempotence and locking

A run is keyed by UTC day. Re-running the same day re-uses existing records rather
than re-scoring them, so a retry after a crash costs nothing and cannot double
the queue. `app/lock.py` holds an advisory lock on the state directory for the
duration; a second concurrent run stands down with a reason rather than
interleaving writes.

The backlog guard is the other half: when `max_open_candidates` proposals already
await review, the next run stands down instead of adding to the pile. Nothing is
discarded, and the receipt says why it stood down. An unattended producer with no
backlog guard is how a review queue becomes unbounded.

## Determinism

Given the same inputs, the same seed and the same day key, two runs produce
byte-identical artifacts. JSON is written with sorted keys and a trailing newline
through an atomic replace. The clock is injected. Nonces and blind A/B assignment
derive from the seed. This is what makes a diff between two runs meaningful.
