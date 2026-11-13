# agent-skill-improver

A daily loop that reads your own agent conversations, finds places where a
**skill** (a Markdown instruction file an agent loads) actually mis-served you,
proposes a concrete edit, and then **stops and waits for a human**.

The automated half can ingest, score, isolate, propose and validate. The daily
driver does not import the promotion module and refuses a writable skill store,
so it does not write a skill file, and a rubric is outside the candidate write
path. The model commands it runs are separate processes: with a private source
the daily run refuses to start unless they run in a verified OS sandbox, and
with the synthetic fixture source they run unconfined
([what asi does, exactly](docs/PRIOR_ART.md#what-asi-does-exactly)).
Applying a change is a separate command that requires a typed acknowledgement, a
per-candidate approval token, an unchanged base hash (one exception is in
[Applying a change](#applying-a-change)), and a passed gate. By
default the gate needs at least one fix on a failure the improver was shown, no
regressions, no holdout degradations, at least one holdout, and a blind judge
preference for the candidate in at least half of the decided cases.

```
                         AUTOMATED (safe to schedule)                    HUMAN ONLY
  ┌────────┐   ┌──────────┐   ┌───────┐   ┌─────────┐   ┌─────────┐   │  ┌───────┐
  │discover│──▶│ normalize│──▶│ score │──▶│ isolate │──▶│ propose │──▶│  │ apply │
  └────────┘   └──────────┘   └───────┘   └─────────┘   └─────────┘   │  └───────┘
                                                             │        │      │
                                                        ┌────▼────┐   │  ┌───▼────┐
                                                        │validate │───┼─▶│rollback│
                                                        └─────────┘   │  └────────┘
                             writes only to state/  ─────────────────▶ │  writes SKILL.md
```

## Why the boring parts are the point

Most of this repository is refusal logic. That is deliberate: a loop that edits
the instructions an agent obeys is a loop that can edit its own guardrails, and
the failure is silent. The design notes in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explain each interlock; the short
version:

| Risk | What stops it |
| --- | --- |
| The improver rewrites its own examiner | Rubrics are versioned files, loaded read-only, hashed onto every scorecard. The candidate write path refuses any target inside the tool's package, the config file and its directory, the state directory or the rubrics directory; that list is built in code and is **not** configurable. This protects asi's write path; it does not confine model commands (see the private-transcript row below). |
| A scheduled run edits a skill | The daily driver does not import the promotion module at all (asserted structurally), refuses a writable skill store, and hashes the whole skill tree before and after so any mutation is visible in the receipt. |
| A skill ships executable code | The candidate write path replaces a file only if it is named `SKILL.md`, so scripts, hooks and binaries inside a skill directory are refused as write targets. When asi finds executable files in the skill's directory, or directories there it could not fully examine, a candidate's notes say how many, with up to five names each; with any such directory, the count of executable files is a floor. |
| One bad record blocks everything | Records are quarantined or deferred **individually**. There is no whole-batch veto, because a batch veto turns a filter into a ratchet where one unresolved item blocks all progress for ever. |
| Machine traffic drowns human sessions | Selection is round-robin across authorship classes, never a sorted prefix. In an orchestrated environment, generated threads can be the overwhelming majority of the corpus. |
| A pasted credential rides along to the model | When `redact_secrets` is set, which is the default, transcript text is masked for known credential shapes before it goes into a prompt (the same shapes the repo's own leak scanner bans); it can be turned off, and matching known shapes is best-effort. A skill file that itself holds a token or key block is refused as a proposal target outright, because the improver would hand it straight back inside the new file. |
| The improver copies a conversation into the skill | A replacement that repeats 25 or more consecutive words of a cited conversation excerpt as the improver was shown it, words the old skill text does not hold, is refused before the judge or the replay runs, and the refusal's reason interpolates no text of the replacement or the excerpt: its only variable parts are the skill's id and the digest that ends the name of the conversation's record under `state/conversations/` (after `asi daily --commit`, `ls state/conversations/*--<digest>.json` finds it; a dry run writes no record), never the conversation's id. The other output fields print what they always printed and are not checked for copied words ([what the reason and the output hold](docs/OPERATIONS.md#recovery)). It catches verbatim copying only: a paraphrase or an inserted token such as a list marker gets past it only when no exact 25-word run of one excerpt is left that the old text does not hold, and a copy of fewer than 25 words is not refused. |
| A conversation tells the agent what to do | Captured text is fenced as quoted data, forged delimiters are neutralized, and the model is told the block is evidence rather than instruction. This reduces the chance of a wrong verdict; it is **not** what stops exfiltration. |
| Private transcript text reaches a subprocess that can phone home | With a private source, the daily run refuses to start unless the model runner reports an OS sandbox that a probe verified on this host, every model role inside it, and an environment allowlist; the sandbox removes network access and does not restrict reads. The commands that list and export conversations are not model roles and run outside the sandbox. The exception for the synthetic fixture corpus is derived from the source kind, and `[source]` is a closed schema so no trust label can be typed into it. |
| A credential in the operator's shell reaches the model child | The child environment is built from a five-name allowlist, never inherited. A variable whose *name* matches a credential, token or proxy pattern is a refusal, not a silent drop. |
| Private artifacts are world-readable on a shared machine | The state root and every subdirectory are 0700 and every artifact is 0600, set explicitly rather than left to the umask. Existing state is repaired and then re-measured; a repair that did not take effect is a refusal. A state directory that is really a symlink is refused *before* anything follows or chmods it, so the tree that gets made private is the tree that was configured. |
| The judge agrees with the improver because it *is* the improver | By default the gate refuses when the judge's command matches the improver's; daily receipts record the command-independence check. |
| A candidate is applied against a changed file | Apply recomputes the base hash and refuses on mismatch, unless the live file is the candidate's own replacement from an earlier run of this apply that stopped before recording it, which apply then records without writing again. |
| An apply goes wrong | Before replacing a skill file, apply writes a backup, checks its hash, and writes a manifest; an apply that only records an earlier interrupted write reuses the backup and manifest that write left, once the manifest is checked to record this write; `asi rollback` verifies the restored hash. The backup and its manifest are verified *before* the restore overwrites anything, so an unusable backup is a refusal with the live file intact rather than a lost original. Rollback only ever overwrites the text its own apply wrote: a skill edited since is refused, not clobbered. |

## How this differs from other self-improving agent projects

asi is not the first loop to improve agent instructions. The closest precedents
are Anthropic's [skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
and the [writing-skills](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md)
skill in obra/superpowers, which test new or revised skill text against a
baseline, and GEPA's [gskill](https://github.com/gepa-ai/gepa/blob/main/src/gepa/gskill/README.md),
which learns skill text from an empty start and tests it against a no-skill
baseline. asi's case is the combination below, not the idea.

| Project | Stars (2026-11-13) | What it improves | What it does better | Where asi differs |
| --- | --- | --- | --- | --- |
| [anthropics/skills](https://github.com/anthropics/skills) (skill-creator) | 193,869 | One skill, with its author | [Runs the agent on test tasks](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md#running-and-evaluating-test-cases), tests triggering, reports mean and standard deviation | Scores recorded conversations instead of running test tasks |
| [obra/superpowers](https://github.com/obra/superpowers) (writing-skills) | 316,967 | Skills, written test-first | Pressure scenarios, a no-guidance control, [five or more runs per variant](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md#micro-test-wording-before-full-scenarios) | A rubric outside the candidate write path and, by default, a judge whose command differs from the improver's |
| [gepa-ai/gepa](https://github.com/gepa-ai/gepa) (gskill) | 7,418 | Prompts, code and skills, by reflection | [Verifiable tests and a test split](https://github.com/gepa-ai/gepa/blob/main/src/gepa/gskill/README.md), a population of candidates | Makes one candidate per selected skill |
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 39,028 | Instructions and examples in LM programs | [A general framework with many optimisers](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md) | Asks for one replacement text of one `SKILL.md` |
| [jennyzzt/dgm](https://github.com/jennyzzt/dgm) | 2,506 | Its own Python code | [Finds structural changes on coding benchmarks](https://arxiv.org/abs/2505.22954) | asi's candidate write path replaces a file only if it is named `SKILL.md` |
| [Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef) | 7,812 | Agent harnesses or weights, from scored live use | [Repeated task runs and a served-version history](https://reefinfra.ai/docs/user-guide/evolve-your-harness/) | With a private source, the daily run refuses to start unless a probe verified the OS sandbox on this host and every model role is inside it; Reef also supports held activation for human review and configurable executor isolation |
| [ace-agent/ace](https://github.com/ace-agent/ace) | 1,452 | A playbook of strategies used as context | [Learns many lessons at once](https://github.com/ace-agent/ace#how-it-works) | Asks for one replacement text and the smallest change |
| [letta-ai/letta-code](https://github.com/letta-ai/letta-code) | 3,828 | Memory and agent-owned skills | [A full agent runtime with versioned memory](https://docs.letta.com/letta-code/memory) | Apply is a separate command |

What asi does. Each sentence is from
[What asi does, exactly](docs/PRIOR_ART.md#what-asi-does-exactly), which links
the code:

- **Attribution.** A failure is linked to a skill by a mention of the skill or a
  path to it in the conversation, or by a prior the rubric declares. Attribution
  is heuristic.
- **Candidate write path.** The candidate write path replaces a file only if it
  is named `SKILL.md` and refuses any target inside asi's package (its source
  and prompts), the config file and its directory, the state directory or the
  rubrics directory. This protects asi's write path; it does not confine model
  commands.
- **Sandbox scope.** With a private source, the daily run refuses to start
  unless the model runner reports an OS sandbox that a probe verified on this
  host, every model role inside it, and an environment allowlist. The sandbox
  removes network access; it does not restrict reads. With the synthetic
  fixture source, model commands run unconfined.
- **Redaction.** Transcript text is masked for known credential shapes when
  `redact_secrets` is set, which is the default; it can be turned off, and
  matching known shapes is best-effort.
- **Apply and rollback.** The daily driver does not import the promotion module.
  Apply is a separate command, `asi apply`, that needs the candidate's review
  token, the `--i-reviewed-this` flag, and a live file whose hash equals the
  candidate's base hash, unless the live file is the candidate's own replacement
  from an earlier run of this apply that stopped before recording it. It does
  not authenticate who runs it. Apply writes a
  hash-verified backup before it replaces the file, and `asi rollback` reads
  the restored file back and compares it.

Limits:

- The fix count is in-sample: a fix counts FAIL -> PASS on a shown failure.
  An optional count of fixes on withheld failures is off by default and
  report-only: the gate never reads it.
  The holdouts are cases that scored PASS earlier in the run and can come from
  the same conversation as a target. Holdouts catch PASS -> FAIL and
  PASS -> `INSUFFICIENT` and say nothing about new fixes.
- Validation re-scores recorded conversations, caching one successful score per
  (conversation, rubric, arm); a failed attempt can be tried again for another
  criterion. It does not re-run the agent or collect repeated successful trials,
  so it reports no variance or confidence interval.
- A judge whose command differs from the improver's may still be the same model.
  asi does not test whether a skill is selected.

The full comparison, with a source for each claim, is in [docs/PRIOR_ART.md](docs/PRIOR_ART.md).

## Install and run, offline, in one minute

Requires Python 3.11+ and nothing else. There are **no runtime dependencies**.

```bash
git clone <this-repo> && cd agent-skill-improver
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

asi --config config/example.config.toml doctor
asi --config config/example.config.toml daily --commit --day 2026-01-02 --seed t1
```

The example configuration is entirely self-contained: synthetic conversations
from `examples/fixtures/`, a synthetic skill tree in `examples/skills/`, and a
deterministic offline fake model. No network, no credentials, no path outside the
repository. See [`examples/README.md`](examples/README.md) for what each fixture
is designed to exercise.

## Pointing it at your own conversations

A real run swaps two blocks in a copy of the config (`config/local.toml`, which
is gitignored): `[source]` to `kind = "conversation-cli"` with argv templates for
your own tool, and `[[skill_root]]` to point at your own skill directories.

```toml
[source]
kind = "conversation-cli"
discovery_argv = ["your-cli", "search", "--after", "{since}", "-n", "{limit}"]
export_argv    = ["your-cli", "export", "--thread", "{thread_id}"]
```

This repository names no conversation tool anywhere. It knows two documented
contracts, and any CLI that satisfies them works:

* **discovery** prints a line grammar -- a numbered `N. "TITLE" (YYYY-MM-DD)`
  header, an optional indented `Workspace: <path>`, and `Thread: <id>` where
  anything after the id is ignored;
* **export** prints the element-graph JSON described in
  [`transcript_export.py`](src/agent_skill_improver/adapters/transcript_export.py).

### That switch turns on a hard requirement, on purpose

`kind = "fixtures"` is synthetic. `kind = "conversation-cli"` is **private**, and
a private source assembles the lethal trifecta: private data, plus text that was
instruction to some other agent and may be hostile, plus a model subprocess with
whatever network and filesystem authority the parent had.

So for a private source the loop **refuses to run** unless every model role
executes inside a local no-egress sandbox that has been *verified on this host*,
with a minimal environment. `sandbox-exec` on macOS, `bubblewrap` on Linux. No
supported mechanism means `doctor` reports a problem and `daily` raises before a
single byte is exported -- there is no partial mode and no flag that waives it.
The commands configured in `[source]` that list and export conversations are not
model roles: they run outside the sandbox, with the parent's environment.

Prompt fencing and secret redaction are in the code and they are not the answer
here: a fence is a request to a model, and redaction is shape matching over
prose. Neither can deny a socket. Details, including why there is deliberately no
hosted-model option for a private source, are in
[`docs/PRIVACY_AND_THREAT_MODEL.md`](docs/PRIVACY_AND_THREAT_MODEL.md).

## The commands

| Command | Writes | Notes |
| --- | --- | --- |
| `doctor` | `state/` modes, and the sandbox probe's scratch under `state/sandbox/` | validates config, reports the source class and isolation mode, repairs state permissions, reads back the last run (day, status, age, whether the skill tree was mutated, confinement mode) from the newest daily receipt, exits non-zero if not runnable or if that run detected a mutation |
| `skills` | nothing | inventory: canonical vs mirror, files that never load, executable payloads |
| `rubrics` | nothing | loaded rubrics with versions and content hashes |
| `discover` | `state/` directory modes only (created empty and owner-only if absent) | what the source would ingest, with pagination honesty |
| `daily` | `state/` only | the whole automated pass. **Dry run by default**; `--commit` persists to `state/`. After a run that detected a skill tree mutation, the next one refuses (exit 5) until a committed run with `--commit --mutation-reviewed 2026-01-01` names that run's day |
| `candidates` | `state/` directory modes only (created empty and owner-only if absent) | the review queue |
| `review` | as `candidates` | a candidate's full argument, diff, validation evidence (per replayed case and judgement) and approval token |
| `set-aside` | as `candidates` | what runs set aside: quarantined, refused and deferred records, newest first, with stage and reason |
| `apply` | **a skill file**, plus `state/` (backup, candidate status, receipt) | the only skill writer. Review-gated, see below |
| `rollback` | a skill file, plus `state/` (candidate status, receipt) | restores a backup and verifies the hash |
| `scheduler-template` | nothing | prints a launchd or cron template. Installs nothing |
| `selftest` | nothing | re-checks the structural safety invariants on the deployed source |

## Applying a change

`apply` is deliberately awkward to invoke. All four of these must hold:

```bash
asi review CAND-...                     # read the argument and copy the token
asi apply CAND-... \
    --approve <approval-token> \
    --i-reviewed-this
```

1. **A typed acknowledgement.** `--i-reviewed-this` has no short form.
2. **An approval token** bound to the candidate id, the base file hash and the
   hash of the patch text *as it stands on disk*, recomputed on every read.
   Change any of the three and the token no longer verifies. Every other derived
   field -- the diff and its counts against the live base, the approval
   challenge, the target identity, the status against the recorded gate -- is
   recomputed and compared before the write, so a record edited after review is
   refused rather than authorised by the token printed before the edit.
3. **A matching base hash.** If the skill file changed since the candidate was
   built, apply refuses; the candidate must be revalidated against the new base.
   The one exception is the candidate's own replacement, written by an earlier
   run of this apply that stopped before recording it: apply then records that
   write and does not write again.
4. **No active automation marker.** The daily driver sets `ASI_AUTOMATED_RUN`
   only inside its own process; the scheduler templates set it for their job.
   Apply refuses while it is active, even with a valid token. Apply does not
   authenticate its caller: the token and acknowledgement are review interlocks,
   not a security boundary, and a separate unmarked process can still apply.

Apply also refuses a candidate that did not pass the gate, refuses a
non-canonical (mirror) root, refuses a protected skill, and refuses to write
anything other than `SKILL.md`.

## Scheduling

`asi scheduler-template launchd` and `... cron` print a template and install
nothing. Read [`docs/SCHEDULING.md`](docs/SCHEDULING.md) first. The scheduled job
runs the daily pass in commit mode and sets the automation marker for its whole
command, so an apply step appended to that command is refused. On cronie, Vixie and
OpenBSD, the cron environment line also marks later lines; BusyBox needs the
export prefix on each line. A separate job without the marker can still run
apply with a token and acknowledgement. **Do not schedule apply.**

## Verification

Everything is stdlib `unittest`, offline and deterministic.

```bash
python -m unittest discover -s tests -t .   # the suite
python -m compileall -q src tests examples tools
python tools/leakscan.py                    # tracked files, paths AND git history
python examples/build_fixtures.py --check    # fixtures are reproducible
asi --config config/example.config.toml selftest
```

`BUILD_RECEIPT.md` records the exact commands, counts and real exit codes, once
for the original publication build and once for the latest refresh, each tied to
the commit it was measured on.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the same commands on
three Python versions, plus the end-to-end daily pass against the fixtures. It
checks out with full history on purpose: a leak scan over a shallow clone would
report a clean history it had never seen.

## Privacy

Conversations are private content. Everything ingested or produced lands under
`state/`, which is gitignored in its entirety and created with `0700`
directories and `0600` files -- set explicitly, because a scheduled job often
runs with a permissive umask, and repaired on every run, because an
already-world-readable directory is the case a fresh-run check misses.

Nothing in this repository is derived from a real conversation: every fixture is
synthetic and generated by `examples/build_fixtures.py`. `tools/leakscan.py`
enforces that over tracked content, tracked paths **and reachable history**,
because deleting a string in a later commit does not remove it from the
repository -- which is why this tree has a single clean root commit rather than a
cleanup commit on top of an unclean one.

Read [`docs/PRIVACY_AND_THREAT_MODEL.md`](docs/PRIVACY_AND_THREAT_MODEL.md)
before pointing this at a real corpus, particularly the parts about what leaves
the machine in a model prompt and what redaction does not guarantee.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layers, data flow, every interlock and why it exists
- [`docs/PRIVACY_AND_THREAT_MODEL.md`](docs/PRIVACY_AND_THREAT_MODEL.md) — assets, adversaries, what is and is not defended
- [`docs/RUBRICS.md`](docs/RUBRICS.md) — writing a rubric that does not manufacture work
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — day-to-day running, reading a receipt, recovery
- [`docs/SCHEDULING.md`](docs/SCHEDULING.md) — unattended operation and its limits
- [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) — each design risk mapped to its implementation and its test
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) — how asi relates to other self-improving agent projects

## Licence

MIT. See [`LICENSE`](LICENSE).
