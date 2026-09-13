# Build receipt

What this file is: the exact commands that were run to verify this repository,
their real exit codes, and the counts those commands reported. It exists because
"tests pass" is a claim, and a claim about a test run is worth nothing without the
command and the code it returned.

Every exit code below was read from the process itself, unpiped. Nothing here is
estimated.

## Provenance

The design was set by a hostile architecture review whose brief was to find how a
skill self-improvement loop produces confident, well-documented, self-referential
garbage. Its verdict was **build the scoring half, do not build the closed loop**,
and that verdict is honoured: the daily path scores, isolates, proposes and
validates, and cannot write to a skill file at all.

A second, independent review then examined this tree for public-release and
security readiness. It found two release blockers and one privacy defect. All
three are fixed, and the section below states what each was, because a receipt
listing only successes is marketing.

## What the security review found, and what was done

### The lethal trifecta was assembled, not broken

The loop reads private conversation history (leg 1), that history is
attacker-influenceable content (leg 2), and the model roles were ordinary child
processes inheriting the full parent environment with open network and filesystem
authority (leg 3). Prompt fencing and regex redaction were the only mitigations,
and neither removes a capability, so neither is a boundary.

Leg 3 is now removed **mechanically, from source classification**:

- `domain/isolation.py` classifies `[source].kind`. Only `fixtures` is synthetic;
  every other kind, **including an unknown one**, is private. The mapping fails
  closed.
- A private source requires every model role to run under a *verified* OS
  confinement mode: `sandbox-exec` with a generated deny-network/deny-write
  profile on macOS, `bubblewrap` with `--unshare-net` and a read-only root bind
  on Linux. `adapters/sandbox.py` verifies it by executing a probe that must fail
  to reach a loopback listener this process owns, fail to write outside its
  private directory, and succeed at writing inside it.
- The check is the **first two statements of `run_daily`**, before the lock,
  before the skill-tree hash and before any export. `asi doctor` performs the same
  decision and reports it as a `PROBLEM`. If neither mechanism is available the
  run refuses; there is no degraded mode and no flag that waives it.
- The child does not inherit the environment. It receives `PATH`, locale, `TZ`,
  and `HOME`/`TMPDIR` pointed at a fresh `0700` per-call directory that is deleted
  afterwards; names matching credential, token, key, secret, session and proxy
  conventions are dropped, and a credential placed in a role's own configured env
  is a refusal rather than a silent drop.
- The resolved mode is part of every command id and every receipt.
- The synthetic exception is tied to the source kind in code. `[source]` is a
  **closed schema**, so a hopeful `trusted = true` is a hard configuration error
  rather than a new opt-out.

There is deliberately **no hosted-model opt-out** for a private source. A hosted
model needs egress, and egress is the leg being removed.

### The public artifact named a specific internal product

Tracked content and reachable history named a specific agent product, its CLI
subcommands, and a conversation title sampled from a real corpus, while the docs
claimed no internal tool names appeared. The whole artifact is now vendor-neutral:

- The two source adapters were renamed after the interface rather than after a
  product: `adapters/conversation_cli.py` (`ConversationCliSource`) and
  `adapters/transcript_export.py`. The old names are not repeated here, because
  this file is tracked content and the leak scanner is entitled to refuse it.
- The source kind is `conversation-cli`. Discovery, query, workspace and export
  are **argv templates** with `{since}`, `{limit}`, `{query}`, `{workspace}` and
  `{thread_id}` placeholders; the real executable belongs only in a gitignored
  `config/local.toml`. The line grammar and the export JSON shape are documented
  as generic contracts, and the parsing behaviour is unchanged.
- The authorship class that had been named after a sampled thread title is now
  `orchestrated`; fixtures, profile values, titles, URL schemes, docs, packaging
  text, tests and examples were renamed with it.
- `tools/leakscan.py` gained two rules for the product name and the sampled
  title, both assembled from fragments so the scanner's own source is scanned
  like any other file, and `tests/test_public_safety.py` proves each rule is
  two-sided and that a case-insensitive search over the **whole tracked tree**
  returns zero matches.
- **The first version of the product rule was fail-open, and that was measured
  rather than reasoned about.** It was anchored with `\b`, and an underscore is a
  word character, so it never fired on the module and filename spellings this
  tree actually used -- the very form the rename existed to remove. The rule is
  now bounded by *letters* instead of word characters, which is a strict superset
  of the word-boundary search. Widening the camel-case half to any case then
  produced two false positives on a clean tree, because the AWS access-key-id
  prefix contains the same three letters and genuinely occurs here in the
  redaction table; that half is therefore restricted to the mixed-case spelling.
  Both directions have their own test, because a rule that fires on everything
  and a rule that fires on nothing print the same word on a clean tree.
- The previous commits contained the name, so a cleanup commit would not have
  been enough. Local `main` was rewritten to a **single clean root commit**, the
  reflog was expired and unreachable objects pruned. No backup ref, no tag, no
  remote.

### State was created world-readable

`StateStore` created the state root and its subdirectories `0755`. `state/raw/`
holds verbatim private transcripts. Now: directories `0700`, files `0600`
including backups, receipts and the run lock, set explicitly rather than left to
the umask, with the mode applied to the temp file *before* the atomic rename so no
window exists at a wider mode. `state.ensure()` creates, then **repairs**, then
**verifies**, and raises `SafetyRefusal` if a mode cannot be corrected — because
an already-existing wide directory is the case a fresh-run check misses. Symlinks
inside the tree are skipped rather than followed.

### One misleading claim was corrected rather than defended

The docs described the approval token as unforgeable. It is not, and it was never
meant to be: it is a deterministic function of the candidate id, the base hash and
the patch hash, derived with no secret, so anyone who can run the tool can derive
one. It is a **review interlock** — an approval cannot outlive an edit to the base
or the patch — and `docs/ACCEPTANCE.md`, `docs/ARCHITECTURE.md` and
`docs/PRIVACY_AND_THREAT_MODEL.md` now say so. The things that actually stop an
automated apply are structural: `daily` does not import `promote`, `daily` refuses
a writable skill store, and the automation marker makes `from_cli` refuse.

## The approval token was not bound to the text it approved

A later review found and reproduced a bypass of the human gate, and it is recorded
here because a receipt that lists only successes is marketing.

**The reproduction.** A validated candidate was produced by a real `daily --commit`
pass, `asi review` printed its approval token, and then exactly one field of the
persisted candidate record was edited: `patch.new_text`. `patch.new_text_sha256`,
`approval_challenge`, `status` and the recorded gate were left byte-identical.
`asi apply` with the original token returned **0** and wrote the edited text into
the live skill file. The token was derived from the *stored* patch hash, so it was
never a function of the bytes a human had read; the one property the whole gate
exists to provide was absent.

**The fix.** `domain/integrity.py` recomputes every candidate field that claims
integrity and returns the disagreements: `new_text_sha256` from the normalized
`new_text`, the approval challenge from the candidate id and both hashes, the
unified diff and its added/removed counts against the **live** base text, the
canonical target identity (skill id against the path it sits in, `SKILL.md`, a
well-formed base digest, no canonical/mirror collision), and the recorded status
against the recorded gate. `review`, `apply` and `validate_candidate` all call it
before they act, and `Candidate.required_approval_token` now derives from the
recomputed hash, so even a caller that skipped the check cannot authorise changed
bytes. In `apply` the recomputation and the live-base comparison both run **before**
the backup directory is created and before any write, so a refusal leaves the skill
file, the candidate status, the backups and the receipts untouched. Candidate reads
in `app/state.py` fail closed: a truncated, wrongly-shaped or wrongly-keyed record,
or one that does not claim the id it was looked up by, is a named `SafetyRefusal`
rather than a traceback from the middle of a promotion.

**Two properties of the regression test, because a test that cannot fail is not
evidence.** `tests/test_candidate_integrity.py` drives the real CLI as a process
through the exact reproduction above and asserts the refusal, the untouched skill
hash, the unchanged candidate status, and the empty backup and receipt sets. Run
against the unwired call sites it fails 6 of its 10 end-to-end assertions --
including `'applied' != 'validated'` and a backup directory that should not
exist -- and its healthy controls then apply the same candidate successfully and
confirm the daily path is still proposal-only. Each derived field additionally has
its own unit test in both directions.

## Verification commands and their real exit codes

Run from the repository root. `<venv>` is a local virtual environment with the two
declared dev extras installed; the runtime itself needs nothing beyond the
standard library.

| Command | Exit | What it reported |
| --- | --- | --- |
| `python3 -m unittest discover -s tests -t .` | **0** | `Ran 461 tests` / `OK`, 0 skipped |
| `python3 -m compileall -q src tests examples tools` | **0** | silent, so every tracked tree byte-compiles |
| `<venv>/bin/black --check .` | **0** | `64 files would be left unchanged` |
| `<venv>/bin/mypy` | **0** | `Success: no issues found in 41 source files` |
| `python3 tools/leakscan.py --selftest` | **0** | 27 rules, each fires on its positive sample and stays silent on its negative one |
| `python3 tools/leakscan.py .` | **0** | `clean`: 111 tracked files, 4 commits scanned, 0 allowlist entries; configured fetch/push remote URLs checked |
| `python3 examples/build_fixtures.py --check` | **0** | `8 fixture(s) match the builder exactly` |
| `python3 tests/fixtures/build_fixtures.py` | **0** | `wrote 10 fixture file(s)`, and `git status` stayed clean, so they reproduce |
| `<venv>/bin/asi --config config/example.config.toml doctor` | **0** | `verdict: ready`, with the isolation and permission lines below |
| `<venv>/bin/asi --config config/example.config.toml selftest` | **0** | 16 architectural checks, all `ok`, `selftest: passed` |
| `<venv>/bin/asi ... daily --commit --day 2026-01-02 --seed t1` | **0** | first pass, see below |
| the same command again | **0** | second pass, idempotence, see below |
| `git diff --exit-code -- examples/skills examples/skills-mirror rubrics config` | **0** | no tracked skill, rubric or config file changed during either daily run |

The leak scan and the test suite were both re-run **after** the history rewrite,
because `tools/leakscan.py` scans reachable history and one test scans history for
banned literals. A history scan over a history that no longer exists proves
nothing about the one that does, so the numbers above are the post-rewrite ones.

## What `doctor` reports about the boundary

```
source       fixtures -> ok  [synthetic-fixture]
isolation    mode=none verified=False required=False satisfied=True
state perms  dirs 0o700 / files 0o600, 0 repaired, 0 still wrong
```

`required=False` is the synthetic fixture corpus, which is the one class allowed
to run unconfined. Point `[source].kind` at `conversation-cli` on a host with no
`sandbox-exec` and no `bwrap` and the same command reports a `PROBLEM` and exits
non-zero instead.

## The end-to-end daily run

Fixtures only, offline, with a deterministic fake model. First pass:

```
status=partial day=2026-01-02 mode=commit scored=6 quarantined=1 deferred=1
candidates=2 gate_passed=1 live_mutation=False isolation=none
```

Read it as designed behaviour rather than as damage:

- `quarantined=1` is the structurally malformed export, isolated **alone**. The
  other records still scored, which is the per-record verdict the whole design
  turns on.
- `deferred=1` is the record whose model call failed, retried by design rather
  than counted as a failure.
- `candidates=2` with `gate_passed=1` means one proposal carried no-regression
  evidence and one did not, and the one that did not was **kept with its reasons**
  rather than deleted.
- One proposal was refused by policy, reported in the run notes: the target skill
  is listed in `[guard].protected_skill_ids`.
- Selection reported `truncated: False` with both rubric buckets exhausted, so the
  bound was not reached and nothing was silently dropped.
- `live_mutation=False` is the tree hash taken before and after the run. It is the
  single most important line.
- `isolation=none` is honest: this corpus is synthetic, so no confinement was
  required and none is claimed.

Second pass, same day key, same seed:

```
status=partial ... scored=0 quarantined=0 deferred=1 candidates=0 gate_passed=0
live_mutation=False isolation=none
```

`scored=0` is the idempotence key working: records that reached a terminal state
for the day were re-used, not re-scored. `deferred=1` was retried on purpose.

### The private path was exercised too, not only the synthetic one

The run above uses the synthetic corpus, which is the class that runs
*unconfined*, so on its own it proves nothing about the boundary. The private
path was therefore run as well, against a throwaway stand-in CLI wired into a
gitignored local config (both deleted afterwards; neither is tracked):

```
source       conversation-cli -> ok  [private-conversation]
isolation    mode=sandbox-exec verified=True required=True satisfied=True
```

`daily --commit` then completed with `status=ok scored=1 isolation=sandbox-exec`,
and the receipt recorded the verification that had been performed before the
first model call, in both directions:

| Probe | Under the sandbox | Unsandboxed control |
| --- | --- | --- |
| outbound socket to a loopback listener this process owns | `blocked:PermissionError` | `allowed` |
| write outside the fresh private temp directory | `blocked:PermissionError` | `allowed` |
| write inside that directory | `allowed` | `allowed` |

32 environment names were dropped before the child started, and the persisted
scorecards carry the mode inside the command id
(`scorer:...:iso=sandbox-exec:...`), so the confinement of a past run is readable
from its artifacts rather than taken on trust. `scored=1` is the other half of
that result: the sandbox denied the socket without breaking the work.

### Determinism, measured and stated precisely

Two runs from an empty state directory, same day key and same seed, were compared
file by file. Every artifact matched except six fields, all of which are wall
clock or process identity rather than content: `.started_at`, `.finished_at`,
`.created_at`, `.sampled_at`, `.at`, `.run_id`, plus the pid inside
`state/locks/daily.lock`. Everything derived from the inputs and the seed —
records, scorecards, candidate text, patch hashes, judge assignment, refusal and
quarantine reasons — was byte-identical. The suite proves the stronger form with an
injected clock (`test_two_identical_runs_produce_identical_receipts`); this is the
weaker claim that survives a real wall clock, and it is stated as the weaker one
on purpose.

## What is in the tree

| Thing | Count |
| --- | --- |
| Tracked files | 111 |
| Source modules (`src/`) | 41, 9,806 lines |
| Test files | 20, 7,088 lines, of which 16 are test modules |
| Test methods | 461 |
| Documentation files under `docs/` | 6, 1,336 lines |
| Leak scanner | 784 lines, 27 rules, 0 allowances, configured remote URLs checked |
| Rubrics | 5 (2 enabled in the example configuration) |
| Synthetic conversation fixtures | 8 |
| Example skills | 4 across 2 roots, 1 carrying an executable payload |
| `asi selftest` structural checks | 16 |

## Honest limits of this verification

- **The OS-confinement tests skip where there is no mechanism.** The policy half
  (classification, refusal, ordering, environment scrubbing, generated profile and
  argv shape) is pure and runs everywhere. The half that executes a really
  confined child runs where `sandbox-exec` or `bwrap` exists, and skips otherwise
  rather than passing quietly. The gates above were run on a host where
  `sandbox-exec` is present, so that half really ran; the CI workflow prints which
  case its runner is in so a skip cannot be mistaken for a pass.
- **`bubblewrap` was not exercised on this host.** Its argv shape is asserted, its
  behaviour is not. That is stated rather than implied.
- **No real conversation was ingested.** Every fixture is invented by
  `examples/build_fixtures.py`. Nothing here is sanitized out of a real
  transcript, because sanitizing real prose takes one missed detail and a reviewer
  cannot tell what was removed.
- **Nothing was applied.** No `asi apply` ran against the tracked example skills,
  so no tracked file was modified by the loop. The apply, refusal, backup,
  rollback and stale-base paths are covered by the suite in throwaway directories.
- **A leak scanner is a lexical tool.** It proves the strings it knows about are
  absent. It cannot prove a paraphrase is absent, which is why no fixture is
  derived from real content in the first place.

## Deliberately not done

- **Publishing is complete to a private repository.** `origin` is configured with a
  credential-free HTTPS URL. The leak scanner checks every fetch and push URL and
  refuses embedded credentials, local paths, private hosts and non-HTTPS/SSH schemes.
- **No scheduler was installed.** `scheduler-template` prints to stdout; the
  templates under `packaging/` install nothing.
- **No hosted-model path for a private source.** Not an omission: a hosted model
  needs egress, and the whole boundary is the absence of egress.
- **No allowances in the leak scanner.** The mechanism exists, is printed in every
  report, and is empty.

## Public-flip readiness

Making this repository public should require no cleanup pass, and that is checked
rather than asserted: MIT licensed, no upload-blocking packaging classifier, no
absolute path under any user home directory, empty runtime dependencies, no
organisation, product or personal identifier in tracked content, tracked paths,
commit metadata **or reachable history**; configured remote URLs are credential-free
HTTPS or git-only SSH to public hosts. History begins at the clean public snapshot;
the later CI correction adds no private source material. `tools/leakscan.py`
covers all of those surfaces and its rules are two-sided, because on a clean tree
a rule that matches everything and a rule that matches nothing print the same
word.

## Post-publication CI correction

The first private-remote build exposed one incorrect pre-publication invariant: the
public-safety suite asserted that no Git remote existed. That was true before the
push and false by construction in every GitHub Actions checkout, where `origin` is
necessarily configured. All three Python matrix legs failed only that assertion;
the end-to-end job passed.

The replacement invariant checks the property that matters after publication: every
fetch and push URL must be credential-free HTTPS or git-only SSH to a public host.
It refuses embedded credentials, local/file paths, private or local hostnames,
non-global IPs, non-HTTPS/SSH schemes and unsafe SSH usernames. The policy is
two-sided, with accepted HTTPS/SSH controls and eight rejected URL shapes. CI's
checkout and Python setup actions were also upgraded to current Node 24 releases
and pinned to immutable commit SHAs, removing the Node 20 deprecation warnings.

Local verification after the correction: 389 tests pass, formatting and typing are
clean, and the leak scanner accepts the real credential-free `origin` after checking
both its fetch and push URLs. The remote CI result is recorded by the commit's
GitHub Actions run, not predicted in this receipt.

## Final state

| Property | Value |
| --- | --- |
| Commits | 5 (a clean public snapshot plus four fixes, the last of which bound the approval token to the reviewed text) |
| Tracked files | 111 |
| `git status --porcelain` | empty |
| `git remote -v` | credential-free HTTPS `origin`, fetch and push URLs checked |
| Reflog / refs | normal local and `origin/main` refs after publication; no backup refs or tags |
| Test methods | 461, all passing locally, none skipped |
| `python3 tools/leakscan.py` | `clean`, 111 tracked files, reachable history and configured fetch/push remote URLs scanned |
