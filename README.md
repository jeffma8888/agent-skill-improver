# agent-skill-improver

A daily loop that reads your own agent conversations, finds places where a
**skill** (a Markdown instruction file an agent loads) actually mis-served you,
proposes a concrete edit, and then **stops and waits for a human**.

The automated half can ingest, score, isolate, propose and validate. It cannot
change a skill file, and it cannot change the rubric it is graded against.
Applying a change is a separate command that requires a typed acknowledgement, a
per-candidate approval token, an unchanged base hash, and validation evidence
that the change fixed something without breaking anything.

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
| The improver rewrites its own examiner | Rubrics are versioned files, loaded read-only, hashed onto every scorecard. The tool's own source, config and rubric directory are protected unconditionally and that list is **not** configurable. |
| A scheduled run edits a skill | The daily driver does not import the promotion module at all (asserted structurally), refuses a writable skill store, and hashes the whole skill tree before and after so any mutation is visible in the receipt. |
| A skill ships executable code | Only a file named `SKILL.md` is ever writable. Scripts, hooks and binaries inside a skill directory are inventoried, reported, and refused as write targets. |
| One bad record blocks everything | Records are quarantined or deferred **individually**. There is no whole-batch veto, because a batch veto turns a filter into a ratchet where one unresolved item blocks all progress for ever. |
| Machine traffic drowns human sessions | Selection is round-robin across authorship classes, never a sorted prefix. In an orchestrated environment, generated threads can be the overwhelming majority of the corpus. |
| A pasted credential rides along to the model | Transcripts are redacted for known credential shapes before any role sees them (the same shapes the repo's own leak scanner bans). A skill file that itself holds a token or key block is refused as a proposal target outright, because the improver would hand it straight back inside the new file. |
| A conversation tells the agent what to do | Captured text is fenced as quoted data, forged delimiters are neutralized, and the model is told the block is evidence rather than instruction. This reduces the chance of a wrong verdict; it is **not** what stops exfiltration. |
| Private transcript text reaches a subprocess that can phone home | A private source refuses to run unless every model role is confined to a verified OS no-egress sandbox with a minimal environment. The exception for the synthetic fixture corpus is derived from the source kind, and `[source]` is a closed schema so no trust label can be typed into it. |
| A credential in the operator's shell reaches the model child | The child environment is built from a five-name allowlist, never inherited. A variable whose *name* matches a credential, token or proxy pattern is a refusal, not a silent drop. |
| Private artifacts are world-readable on a shared machine | The state root and every subdirectory are 0700 and every artifact is 0600, set explicitly rather than left to the umask. Existing state is repaired and then re-measured; a repair that did not take effect is a refusal. A state directory that is really a symlink is refused *before* anything follows or chmods it, so the tree that gets made private is the tree that was configured. |
| The judge agrees with the improver because it *is* the improver | The gate refuses when the judge's command matches the improver's, and every receipt records the independence check. |
| A candidate is applied against a changed file | Apply recomputes the base hash and refuses on mismatch. |
| An apply goes wrong | Every apply writes a backup first; `asi rollback` restores it and verifies the restored hash. The backup and its manifest are verified *before* the restore overwrites anything, so an unusable backup is a refusal with the live file intact rather than a lost original. Rollback only ever overwrites the text its own apply wrote: a skill edited since is refused, not clobbered. |

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

Prompt fencing and secret redaction are in the code and they are not the answer
here: a fence is a request to a model, and redaction is shape matching over
prose. Neither can deny a socket. Details, including why there is deliberately no
hosted-model option for a private source, are in
[`docs/PRIVACY_AND_THREAT_MODEL.md`](docs/PRIVACY_AND_THREAT_MODEL.md).

## The commands

| Command | Writes | Notes |
| --- | --- | --- |
| `doctor` | `state/` modes only | validates config, reports the source class and isolation mode, repairs state permissions, exits non-zero if not runnable |
| `skills` | nothing | inventory: canonical vs mirror, files that never load, executable payloads |
| `rubrics` | nothing | loaded rubrics with versions and content hashes |
| `discover` | `state/` directory modes only (created empty and owner-only if absent) | what the source would ingest, with pagination honesty |
| `daily` | `state/` only | the whole automated pass. **Dry run by default**; `--commit` persists to `state/` |
| `candidates` | `state/` directory modes only (created empty and owner-only if absent) | the review queue |
| `review` | as `candidates` | a candidate's full argument, diff, validation evidence and approval token |
| `apply` | **a skill file**, plus `state/` (backup, candidate status, receipt) | the only skill writer. Human-gated, see below |
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
4. **No automation marker.** The daily driver always sets `ASI_AUTOMATED_RUN`, so
   an apply invoked from inside a scheduled run refuses even with a valid token.

Apply also refuses a candidate that did not pass the gate, refuses a
non-canonical (mirror) root, refuses a protected skill, and refuses to write
anything other than `SKILL.md`.

## Scheduling

`asi scheduler-template launchd` and `... cron` print a template and install
nothing. Read [`docs/SCHEDULING.md`](docs/SCHEDULING.md) first. The scheduled job
runs the daily pass in commit mode; there is no way to add an apply step to it
that works, and that is by design.

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

## Licence

MIT. See [`LICENSE`](LICENSE).
