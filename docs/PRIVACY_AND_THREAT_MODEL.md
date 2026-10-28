# Privacy and threat model

## What is at stake

The input to this tool is your conversation history with an agent. That is among
the most sensitive material on a developer's machine: it contains the code you
were debugging, the systems you were operating, the decisions you had not
announced, and often credentials pasted in a hurry. The output is an edit to the
instructions an agent obeys.

So there are two distinct concerns, and conflating them is how one of them gets
neglected:

- **Confidentiality.** Private content must not reach this repository, a public
  artifact, or a model endpoint it was not meant for.
- **Integrity.** The loop must not be talked into widening its own authority, and
  must not edit a guardrail on the strength of a guess.

## Assets

| Asset | Where it lives | Exposure |
| --- | --- | --- |
| Raw conversation exports | `state/raw/` | Local only. The whole `state/` tree is gitignored, the directories are `0700` and the files are `0600`. Written only by a committed run: `asi daily` without `--commit` withholds every artifact write, so a preview against a private source leaves no transcript on disk. |
| Normalized transcripts, scorecards, failure cases | `state/conversations/`, `state/scorecards/`, `state/failures/` | Local only, same modes, same dry-run rule. A scorecard keeps the scorer's rationale for each criterion, its own words; it is masked with the same credential shapes whatever `redact_secrets` says, then clipped, because a model role is a child process that may read more than its prompt. The criterion ids the scorer sent that the rubric does not define are recorded masked the same way and cut past 120 characters, the first 20 distinct ones. |
| Candidates | `state/candidates/` | Local only, same modes, same dry-run rule. A candidate holds the proposed skill text and its diff, and a hypothesis, intended change and risks written by the improver role, which may paraphrase or quote the transcript that motivated it; `asi review` prints them to your terminal. Those fields, the rollback note and the judge's rationale in the validation record are masked with the same credential shapes whatever `redact_secrets` says, when the record is written and again when it is read, as the record's notes are when it is read, and the rationales are clipped; the improver's fields are clipped too when the tool writes them, the hypothesis, intended change and rollback note past 4,000 characters and each risk past 2,000, and only the first 20 risks are kept, a note counting the rest; the proposed skill text is kept as written, its line endings folded as apply writes them, and a replacement that would introduce a credential is refused rather than masked. `asi candidates` lists each record's skill id, canonical path, status and creation time masked the same way, the skill id and the path cut only past the longest a real one can be and the status and the creation time past 120 characters. |
| Run diagnostics: quarantine, refusal and deferral records, receipts | `state/quarantine/`, `state/refusals/`, `state/deferred/`, `state/receipts/` | Local only, same modes, same dry-run rule. Their `reason` fields quote what failed, which can include the last 500 characters of a model role's stderr or of the conversation command's stderr (its stdout when stderr was blank) and the first 120 characters of model output that was not a JSON object, and the conversation command line that failed. Every one of those quotes passes through the same credential redaction the transcript gets before it is cut to length, whatever `redact_secrets` says, so a CLI that dies printing the key it was refused with, or an argv carrying a key as an option value, leaves a placeholder in the reason rather than the key (best effort: the shapes are the redactor's). A model role's command line is masked the same way wherever it is recorded (the command id in scorecards, candidates and receipts), as is the confinement probe's output in the receipt's isolation record; a receipt also carries each candidate's `diff_preview`. The dry-run rule covers the daily receipt; an `apply` or `rollback` receipt is never withheld, because promotion refuses to run through a withholding store at all. Treat them as transcript-derived and read one before pasting it anywhere. |
| Day checkpoints | `state/checkpoints/` | Local only, same modes, same dry-run rule. Per day: the run count, the path of the last receipt, and each record's status, stage and reason, the same `reason` string the row above describes. |
| Apply backups | `state/backups/` | Local only, same modes; never written by a dry run because `apply` refuses a withholding store. The skill text an apply replaced, plus a manifest naming the target path, the hashes, the operator and the rollback command. Nothing copied from a transcript, though a skill an earlier `apply` wrote is itself improver prose. |
| Model-role scratch | `state/sandbox/` | Local only, same modes. The per-call working directory a confined role may write to; removed when the call returns, whatever the outcome. Kept under `state/` so private scratch inherits the `0700` tree rather than the system temp directory. |
| Scheduler logs | `state/daily.out.log` and `state/daily.err.log`, only once a template from `scheduler-template` is installed | Local only. Both streams of a scheduled `daily` run, so they carry whatever the run printed: refusal and quarantine reasons, and therefore the same stderr and output excerpts as the run diagnostics above. The scheduler creates them under its own umask; the next `daily` run resets them to owner-only on its way in and counts the reset in its receipt. Treat them exactly like `state/raw/`: never committed, never pasted into a ticket. |
| Prompt text handed to a model role | in flight, on a pipe to a child process | For a **private** source the child is confined with no network and no writes outside a per-call temp directory, so this does not leave the machine. For the **synthetic** fixture corpus there is nothing private in it. |
| Skill files | your configured roots | Written only by `apply`, only `SKILL.md`, only after human approval. |
| Rubrics | `rubrics/` | Read-only to the loop. Protected unconditionally. |
| Approval tokens | derived, printed by `review` | A deterministic review interlock. **Not a credential, not a secret, and not a security boundary** -- anyone who can run this tool can derive one. |

## What leaves the machine

**For a private source: nothing.** That is a mechanical property, not an
intention, and the section below states the decision that makes it one.

The only content that moves at all is the prompt handed to a model role, over a
pipe to a child process. A prompt can contain:

- excerpts of the transcript being scored
- the current text of the skill under consideration
- the rubric criteria
- for validation, both the baseline and candidate skill text

Consequences worth being explicit about:

1. **A hosted model is not an option for a private source, and there is no
   opt-out.** A "model" here is any command that reads one JSON object on stdin
   and writes one JSON object on stdout. For a private source that command runs
   under an OS sandbox with no network, so a command that needs to reach an
   endpoint simply fails. Run a local model. This is deliberately not a
   configurable trade: an opt-out flag would be the whole defence, and the person
   who most needs the boundary is the person most likely to switch it off to make
   a run work.
2. **Credentials in your transcripts will be in the prompt** unless redaction
   catches them. `redact_secrets = true` masks common shapes, including one
   glued to an underscore, a non-ASCII letter or the end of an escape (`\n` in
   JSON or a `repr`, `%20` in a URL, a terminal colour code); otherwise one
   glued to an ASCII letter or digit is taken for part of a longer word. It is a reduction
   of risk and **not a guarantee**: a secret with an unusual shape, a credential
   split across lines or across messages, or a private value that simply looks
   like prose will pass through. A text a prompt cuts (a head field at 256
   characters, a message at `max_message_chars`) is masked before the cut,
   reading up to 4,096 characters past it, so a credential the cut would split
   is masked whole when it ends within them, and a private key whose header is
   in such a text is masked to its end line when that is within what is read,
   else to the end of what is read, never into the rest of its message; a
   tool's input and output are first cut when the export is read, before any
   mask, so one split there keeps the part before the cut. Treat it as defence
   in depth, never as a control --
   what keeps such a value on the machine is the sandbox, not the regex.
3. **Prompt budgets bind.** When a transcript exceeds the budget, blocks are
   dropped from the middle and the count of dropped blocks is stated in the
   prompt where they were dropped, so the model knows it is looking at an
   incomplete record. Each block
   (one message and its tool calls) is masked on its own before the budget is
   held, so a mask cannot carry it past the budget and a key block opened in one
   message cannot mask the messages after it; a private key whose body sits in a
   message of its own is therefore not masked. The first and the last block are
   never dropped, so when they alone exceed it they share it, each clip stating
   what it removed, and a head line shows at most 256 characters of an element's
   id, a message's role or a tool's name or status. The list of elements a
   scorer may cite is not held to the budget: it names every element, a dropped
   one too.

## The lethal trifecta, and the leg this repository removes

**Decision date: 2026-08-30.**

The failure pattern is well known: an agent system becomes an exfiltration
primitive when it holds all three of

1. access to **private data**,
2. exposure to **untrusted content**, and
3. the ability to **communicate externally**.

This tool starts with legs 1 and 2 unavoidably present. Reading conversation
history *is* the product, and a transcript is untrusted by construction --
anything anyone pasted into it, including a hostile document under discussion,
ends up inside a prompt. Removing either leg removes the tool.

So **leg 3 is the one removed, and it is removed mechanically for every private
source**:

- Every model role -- scorer, improver and judge -- runs under an OS confinement
  mode that is *verified before it is trusted*: `sandbox-exec` with a generated
  deny-network, deny-write profile on macOS, `bubblewrap` with `--unshare-net`
  and a read-only root on Linux.
- Verification is two-sided and runs at wiring time: a probe under the candidate
  wrapper must fail to connect to a loopback listener this process owns, must
  fail to write outside its private directory, and must succeed at writing
  *inside* it. A wrapper that blocks everything is not accepted either.
- The child does not inherit the parent environment. It gets `PATH`, locale and
  timezone, plus `HOME` and the temp variables pointed at a fresh `0700` per-call
  directory that is deleted afterwards. Names matching credential, token, key,
  secret, session and proxy conventions are dropped, and the count of dropped
  variables is recorded.
- **If the host offers neither mechanism, `doctor` and `daily` fail closed** --
  before any export, before any prompt. There is no degraded mode and no
  override.
- The resolved mode is written into every command id and every receipt, so a past
  run's confinement is auditable rather than assumed.

Two things this decision deliberately does **not** rely on:

- **Prompt fencing is not a removed leg.** The fence and the `NEUTRALIZED`
  substitution remain, and they still matter, but they lower the probability that
  a model cooperates with an injected instruction. They do not remove the child's
  ability to open a socket. A test asserts that the presence of a fence does not
  satisfy the isolation check.
- **Redaction is not a removed leg either**, for the same reason and by the same
  test.

**The synthetic exception**, and why it is not a hole: `[source].kind =
"fixtures"` reads a corpus this repository generates. Every byte is invented by
`examples/build_fixtures.py`, no file in it is derived from or sanitized out of a
real conversation, and a public-safety test asserts no fixture matches any leak
rule. That class -- and only that class -- may run the deterministic offline fake
without an OS sandbox, which is what keeps public CI portable on a runner that
has no confinement mechanism. The exception is derived from the source kind in
code; it is **not** a trust label a user can supply. The `[source]` table is a
closed schema, so a hopeful `trusted = true` is a hard configuration error rather
than a new opt-out, and an unrecognised source kind classifies as *private*, not
as synthetic.

## Permissions on the state tree

`state/` holds verbatim private transcripts, so the modes are set explicitly
rather than inherited from whatever umask a scheduler happened to run with:
directories `0700`, files `0600`, including backups, receipts and the run lock.
The atomic writer applies the mode to the temp file *before* the rename, so there
is no window at a wider mode and no stray temp file left behind.

`state.ensure()` creates, then **repairs**, then **verifies**, and raises
`SafetyRefusal` if a mode cannot be corrected. Repairing matters more than
creating: an already-existing world-readable state directory is exactly the case
a fresh-run check would miss. Symlinks inside the tree are skipped rather than
followed, because chasing one is how a permission repair becomes a way to chmod
something outside the tree.

Skipping a leaf link is not enough on its own, because the *directories* are the
load-bearing part of the claim that everything written stays under the configured
state directory. A state root, or a mandatory subdirectory such as `raw/`,
precreated as a symlink is followed by `mkdir`, `chmod` and every later write, so
verbatim transcripts would land outside the configured tree and the repair walk
would chmod a directory this tool does not own. `StateStore` therefore refuses
before it follows: the root and every declared subdirectory must be a real
directory, measured with `lstat`, and a directory whose name is computed at
runtime must be a real direct child of a mandatory directory that has already been
verified. `asi doctor` reports such a layout as a problem and repairs nothing, so
the mode and the contents of whatever the link pointed at are left exactly as they
were. Configuration loading keeps this check honest: `[run].state_dir` has its
ancestors resolved (they are the operator's own filesystem layout) but its last
component is kept exactly as written (a path ending in `.` or `..` names no leaf of its own and is resolved whole), because resolving the whole path would
follow a link planted at the state directory and hand the store the target as if
it had been configured.

## Prompt injection

A conversation is **untrusted input**. It routinely contains text that looks like
instructions, because it *is* instructions — to a different agent, at a different
time, from a user with different intent. Some of it will be adversarial, and the
interesting cases are not malicious: a transcript in which someone was discussing
prompt injection is indistinguishable, to a naive prompt builder, from an attack.

The defences, in order of how much they matter:

1. **Structural separation.** Captured content is never concatenated into the
   instruction region. It is placed inside an explicit fence and labelled as
   evidence to be graded, not as instruction to be followed.
2. **Delimiter neutralization.** Content that itself contains the fence delimiter
   is rewritten so it cannot close the fence early and escape into the
   instruction region. The substitution is marked `NEUTRALIZED` so a reader can
   see it happened rather than wondering why the text looks odd.
3. **A stated contract.** The prompt says what the block is and that instructions
   inside it are data.
4. **Nothing downstream trusts the answer.** This is the one that actually holds.
   Even a fully successful injection can only produce a scorecard, and a scorecard
   is checked against the rubric's criterion ids, checked against the conversation
   id, and every decided verdict must cite a turn that resolves. A model persuaded
   to declare a skill broken cannot make that stick without a resolvable citation,
   and it cannot cause a write at all, because the write path needs a human.

5. **The reviewer sees what applies.** The human is the gate, so what the human
   reads has to be what the file will hold. A proposal whose prose or replacement
   carries a terminal escape, a bidi override or an invisible tag character is
   quarantined at the propose stage with the code point named, never the byte
   echoed. And every text line any `asi` command writes to a terminal passes
   through one escape (`escape_for_terminal`), so a hostile byte quoted from a
   model's stderr, a file name or a configured path is shown as `\x1b` rather
   than obeyed, and the argument parser's own messages go through the same
   escape. `--json` output goes through its own choke point: the encoder
   escapes the C0 controls itself, and what it leaves raw (a bidi override, an
   invisible tag character, a line or paragraph separator, DEL, a C1 control, a
   lone surrogate) is written as the
   JSON `\uXXXX` escape it is, so a program parses the same object and a person
   reading the output in a terminal sees the escape rather than the effect.

The honest limitation: injection can waste a run, and it can push a record into
quarantine. It cannot silently edit a skill.

## Executable payloads inside skills

A skill is nominally a Markdown document. In practice a skill directory can hold
scripts, hooks and templates that something else executes. Write access to a
skill directory is therefore **arbitrary code execution**, not documentation
editing — and the most valuable target is a skill whose payload enforces a
guardrail.

What this repository does about it:

- Payload files are inventoried at discovery and counted in the receipt
  (`skills_with_executable_payloads`, `executable_payload_files`, and
  `skills_with_unlistable_directories`, non-empty when those counts are a floor).
- The only write path refuses any target whose filename is not `SKILL.md`
  (`NonDocumentWriteRefused`). This is a filename check on the resolved path, not
  a check on the extension of a caller-supplied string.
- A proposal against a skill that carries payload files is annotated
  `ELEVATED REVIEW RISK` so a human reviewer knows the directory is more than
  prose. A directory inside the skill's directory that the inventory could not list gets
  the same annotation, naming the directory and the error: what it holds is
  unknown, so the payload count is a floor, and a reviewer must not read the
  candidate as one for a documentation-only directory.
- Payload detection uses `pathlib` rather than shelling out, so filenames
  containing backslashes, spaces or quotes cannot become argument-splitting bugs.
  Filenames with embedded backslashes exist in the wild and break naive shell
  pipelines.

## Canonical source ambiguity

When the same skill name exists in more than one root, "edit the skill" is
underspecified. Editing the wrong copy fails in the quietest possible way: the
write succeeds, the tool reports success, and the file that is actually loaded
never changes — or the change is silently reverted by the next sync.

Handling: each root declares `precedence`, `canonical` and `editable`. Exactly one
root should be canonical and editable. Writes go only there. Two editable roots of
one kind at one precedence holding the same skill are refused as a tie, not
settled by root id, so the written copy is never the alphabetically first one. A
skill that exists only in a non-editable root is refused as a proposal target
(`MirrorEditRefused`) rather than written somewhere ineffective. Divergence
between a canonical file and its mirrors is reported
(`skills_with_divergent_mirrors`) because divergence means somebody's edit already
went to the wrong place.

The related trap, which this repository does not pretend to solve: a mirroring
script that reports success while copying a fraction of what it was given. If your
skill distribution has such a step, verify its per-item outcome; a summary line
saying "installed" is not evidence that each item arrived.

## Correlated evaluators

The gate's numbers are only meaningful if the judge is independent of the
improver. When both are the same model, agreement measures self-consistency, and
the gate becomes decoration that still emits a passing verdict.

`require_independent_judge = true` refuses the gate when the judge's command
matches the improver's, and every receipt records the full comparison
(`judge_vs_improver_distinct`, `judge_vs_scorer_distinct`,
`scorer_vs_improver_distinct`). Command inequality is a weak proxy — two different
commands can wrap the same model, or two different models from the same family
can share the failure — so the receipt states the limitation rather than implying
that a passing check means genuine independence. The example configuration wires
all three roles to one offline script and **says so in a comment**, because a
passing independence check on that config is a property of the example, not
evidence about the example.

## Publication safety

This repository is written to be public. Specifically:

- No absolute path under any user home directory. Configuration is
  repository-relative, and machine-specific values live in `config/local.toml`,
  which is gitignored.
- No organisation name, internal hostname, internal domain, internal or
  third-party product name, ticket link, account name or personal name anywhere
  in tracked content. The conversation source is described by its *interface* --
  a discovery command and an export command, configured as argv templates -- and
  the real executable belongs only in a gitignored `config/local.toml`.
- No thread title, id or excerpt sampled from a real corpus, in tracked content
  or in any reachable commit. History was rewritten to a single clean root commit
  rather than fixed forward, because a later deletion does not remove a string
  from a repository.
- No credentials, and no credential-shaped literals.
- Every fixture is synthetic and generated by `examples/build_fixtures.py`. No
  fixture is derived from, sanitized from, or paraphrased out of a real
  conversation. Sanitizing real prose is a losing strategy: it takes one missed
  detail, and the reviewer cannot tell what was removed.
- No packaging classifier that marks the project unpublishable.

`tools/leakscan.py` enforces all of it over three surfaces: tracked file
**content**, tracked file **paths** (a path leaks as loudly as a body), and
reachable **git history** (deleting a secret in a later commit does not remove it
from the repository). It also reports whether the repository is shallow, because a
history scan over a truncated clone that claims "clean" is worse than no scan.

Two design decisions in that scanner are worth copying:

- **Its banned literals are assembled from fragments**, so the scanner's own
  source is clean and it is scanned like every other file. The usual alternative
  is to exclude the scanner from its own scan, which makes the file most likely to
  be edited while adding a rule the one file guaranteed never to be checked.
- **Every rule is two-sided.** `--selftest` proves each rule fires on a positive
  sample *and* stays silent on a negative one. A rule that matches everything
  reports the same "clean" as a rule that matches nothing, and only the silent
  half catches the over-broad case.

There is an allowlist mechanism and it is **empty**. It exists because a real
repository eventually needs one narrow exception, and it is printed in every
report so entries cannot accumulate unread.

## What is not defended

Stated plainly, because a threat model that lists only wins is marketing.

- **A compromised model endpoint that is also your judge.** If one party controls
  both the improver and the judge, the gate's evidence is worthless. The command
  inequality check is a proxy for a property it cannot observe.
- **A malicious local user.** Anyone who can edit `rubrics/`, the config, or the
  package source can change what the loop considers a failure. Protection here is
  against the *loop* editing them, not against a human. Likewise `state/`: a
  candidate record and its backup manifest edited together are restored from as
  written, because rollback holds the manifest to the record and nothing signs it.
- **Redaction as a control.** Best effort only. See above.
- **A kernel-level sandbox escape.** The no-egress boundary is `sandbox-exec` or
  `bubblewrap`. Both are real enforcement points and neither is a hypervisor. A
  local privilege-escalation bug in the confinement mechanism defeats this, and
  nothing here would detect it.
- **What a confined wrapper writes into its own answer.** The child cannot reach
  the network, but it returns one JSON object and that object is read. A
  deliberately hostile local model could encode content there; nothing downstream
  ships it anywhere, but it lands in `state/`.
- **Anything at all when the source is synthetic.** The fixture corpus runs
  unconfined on purpose. It is public, invented data; if you point a `fixtures`
  source at a directory of real exports you have defeated the classification by
  hand, and no test can see that.
- **A human approving a bad change.** The gate raises the floor on evidence; it
  cannot supply judgement. `review` exists to make the argument readable, and
  `rollback` exists because approval is fallible.
- **Anything about the `state/` tree's confidentiality beyond gitignore and
  filesystem permissions.** It is plaintext private content on your disk. Back it
  up accordingly, or do not back it up at all.
