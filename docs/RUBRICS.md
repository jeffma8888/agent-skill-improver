# Rubrics

A rubric is the examiner. It decides what counts as a failure, and therefore what
the loop will try to change. Getting it wrong does not produce a wrong number; it
produces a stream of confident edits to files an agent obeys.

## The format

One TOML file per rubric, in the directory named by `[rubrics].dir`, with the file
stem equal to the declared id. Loading is read-only and every file is hashed; the
hash is recorded on every scorecard it produced. Every key is checked: one the
loader does not read is refused, quoted, with the nearest key it does when one is
close and every key it does listed, `requires_evidence`
must be `true` or `false`, `title` and `insufficient_when` must be strings, and
`implicates` must name each skill once.

```toml
[rubric]
id = "evidence-discipline"        # must equal the filename stem
version = 1                       # bump on ANY semantic edit
title = "Claims are backed by checkable evidence"
min_evidence_refs = 1

[[criterion]]
id = "cites-checkable-evidence"
title = "Assertions about the system cite something checkable"
weight = 3
requires_evidence = true
question = """
Find assertions the agent made about the state of the code, a file, or a command's
result. Did it cite the command it ran, the file it read, or the output it saw?
FAIL only if you can cite a specific turn making an unbacked claim.
"""
insufficient_when = "the transcript does not show whether the agent ran anything"
```

## Versioning, and why the loop cannot write these

The improver is graded by the rubric. If the improver could edit the rubric, the
cheapest way to pass would be to loosen the examiner, and nothing in the output
would look wrong. So:

- the rubric directory is in the protected set, derived in code, **not**
  configurable
- the loop has no write path to it at all
- every scorecard carries `rubric_id`, `rubric_version` and the file's content
  hash, so a rubric edit can never be mistaken for a behaviour change

**Bump `version` on any semantic edit.** Two scorecards with the same
`rubric_version` and different content hashes mean somebody edited a rubric without
declaring it, and comparing runs across that edit compares different examinations.

## Writing a criterion that does not manufacture work

The default answer must be `INSUFFICIENT`. That single calibration decision does
more for output quality than any wording.

**Write criteria about observables.** A transcript shows what was said and what
tool results came back. It does not show what the agent believed, whether it
"understood", whether it was "efficient in spirit", or what the user wanted but did
not say. A criterion about an unobservable cannot be evidenced, so a model asked to
score it will either guess or default to a verdict — and a guessed `FAIL` becomes a
real edit to a real file.

**Require a citation for both directions.** `requires_evidence = true` holds `PASS`
to the same standard as `FAIL`. Grading only the accusations trains a scorer to
approve.

**Say when the answer is `INSUFFICIENT`.** `insufficient_when` is the most
load-bearing field in the file. Without it, a model that cannot tell will pick.

**One decidable question per criterion.** A criterion asking two things gets a
verdict about whichever half was easier, and you cannot tell which from the result.

**Weights are about attention, not truth.** `weight` orders which skill gets
proposal budget first. It does not make a verdict more certain.

## What ships here

Five rubrics, in three families. The library is deliberately larger than any
enabled set, because every enabled rubric costs a model call per conversation:
enabling everything is not free, and a rubric you are not reading the output of is
pure cost.

| id | Family | Focus |
| --- | --- | --- |
| `instruction-compliance` | instruction adherence | explicit directives followed; the user did not have to repeat a correction |
| `instruction-following` | instruction adherence | compact pair: an explicit constraint was honoured; an irreversible action was gated |
| `verification-honesty` | evidence | claims of having tested or verified match what the transcript shows |
| `evidence-discipline` | evidence | compact pair: assertions cite something checkable; measured is separated from assumed |
| `tool-efficiency` | process | repeated or redundant tool work that a better instruction would have avoided |

`instruction-following` and `evidence-discipline` are the compact pair the offline
example is authored against: the synthetic fixtures in `examples/fixtures/` carry
control markers naming their criterion ids, which is what makes the example
deterministic. The other three are fuller and intended for a real corpus, and are
the ones the test harness exercises.

Two families cover overlapping ground on purpose, at different resolutions. Enable
one member of a family, not both, unless you specifically want the comparison.

## Enabling a rubric

```toml
[rubrics]
dir = "../rubrics"
enabled = ["evidence-discipline", "instruction-following"]
```

`asi rubrics` prints what is loaded, with versions and hashes. `asi doctor` fails
if an enabled id has no file, if a file's declared id disagrees with its filename,
or if a rubric has no criteria — all three are silent misconfigurations otherwise.

## Adding one

1. Write the file. Set `version = 1`.
2. Add its id to `enabled`, and consider what you are removing to pay for it.
3. Run a **dry-run** daily pass and read the scorecards before enabling it for real.
4. Check the `INSUFFICIENT` rate. A criterion that is almost never insufficient is
   probably being guessed; a criterion that is always insufficient is asking about
   something the transcript does not contain. Both are rubric bugs, not model bugs.
5. Check the *attribution* rate too. Failures that isolate with `ATTRIB_NONE` mean
   the criterion finds real problems it cannot tie to a skill. That is an honest
   outcome and it produces no proposals, so it tells you the rubric is measuring
   something outside this loop's reach.
