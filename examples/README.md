# Examples

A complete, offline, public-safe corpus. Everything under this directory is
**invented**. No file here contains, paraphrases, or is derived from a real
conversation, and nothing here reaches a network.

Because `[source].kind = "fixtures"` classifies as **synthetic**, this is the one
configuration allowed to run the model roles without an OS sandbox, which is what
keeps public CI portable. Point the tool at real conversations and the isolation
requirement applies; see the README.

Run the whole loop against it with no configuration of your own:

```bash
asi --config config/example.config.toml doctor
asi --config config/example.config.toml daily
```

`daily` is a dry run by default: it reports what it would do and persists
nothing. Add `--commit` to write candidates into `state/`. Neither form can
edit a skill file. `asi apply` writes a candidate with review interlocks, not
caller authentication; `asi rollback` restores a verified backup and refuses
under the automation marker, with no approval token or acknowledgement required.

## What is here

| Path | Purpose |
| --- | --- |
| `fixtures/` | 8 synthetic conversation exports in the documented export shape: a thread object plus an ordered `elements` list |
| `build_fixtures.py` | regenerates `fixtures/` deterministically; `--check` asserts the committed bytes still match |
| `fake_model.py` | an offline stand-in for the scorer, improver and judge, driven by markers in the transcript |
| `skills/` | a canonical, editable skill root with four skills |
| `skills-mirror/` | a read-only mirror of one of those skills, used to exercise the canonical-source interlock |

## The fixtures

Each fixture exercises one behaviour the loop must get right. Read the `title`
and the embedded markers to see what it is for.

| Fixture | Authorship class | What it exercises |
| --- | --- | --- |
| `conv-human-0001` | human-authored | a real, evidenced rubric failure with a fixable cause: the proposer should be able to build a candidate from it |
| `conv-human-0002` | human-authored | a clean session that passes every criterion, so the loop must propose nothing rather than manufacture work |
| `conv-human-0003` | human-authored | an instruction-following failure whose cited turn genuinely shows the break |
| `conv-human-0004` | human-authored | a pasted document containing an instruction-shaped payload; the loop must treat it as quoted data and score the *agent's* handling of it |
| `conv-human-0005` | human-authored | a record the model cannot judge, so the run defers this one record and keeps going |
| `conv-orchestrated-0001` | orchestrated | machine-driven iteration traffic, which is plentiful and must not crowd out the human-authored minority |
| `conv-orchestrated-0002` | orchestrated | a second iteration of the same loop, so bounded selection has a real majority class to balance against |
| `conv-broken-0001` | unclassified | a structurally malformed export: quarantined with a reason, while every other record still scores |

Two of the eight are deliberately *not* useful signal. That is the point: a
corpus in which every record produces a candidate cannot demonstrate
abstention, quarantine, or fair selection.

### Authorship classes

`conv-orchestrated-*` are titled so the discovery-time heuristic can see them, and
they carry an in-transcript marker so the post-fetch classifier can confirm it.
The two vocabularies are deliberately different words:

* **discovery time** knows only a title, so it can say `orchestrated` or
  `not-agent-titled` and nothing stronger;
* **after fetching** the transcript, the classifier can say `orchestrated`,
  `human-authored` or `unclassified`.

`not-agent-titled` is not a synonym for `human-authored`. Keeping the words
apart is what stops a title-only guess being reported as a fact about who wrote
the conversation.

## Scorer control markers

`fake_model.py` is deterministic because the fixtures tell it what to say. A
real scorer reads the conversation; these markers exist only so an end-to-end
run needs no model and no network.

| Marker | Effect |
| --- | --- |
| `[[PASS:<criterion_id>]]` | PASS, citing a real element id |
| `[[FAIL:<criterion_id>\|<rule text>]]` | FAIL, citing a real element id |
| `[[FIXABLE:<criterion_id>\|<marker>]]` | FAIL, but PASS once the skill text under evaluation contains `<marker>`. This is what makes a candidate measurably fix something |
| `[[BADCITE:<criterion_id>]]` | FAIL citing a **fabricated** element id. The loop downgrades it to INSUFFICIENT rather than believing it |
| `[[INSUFFICIENT:<criterion_id>]]` | INSUFFICIENT: the evidence does not decide the criterion |
| `[[DEFER]]` | the fake model exits non-zero. The loop defers **this one record** and scores the rest |

`[[BADCITE:]]` and `[[DEFER]]` are the two that matter most. The first proves a
model claim with no verifiable citation earns nothing. The second proves one bad
record does not veto a batch: a per-record verdict, not a whole-batch veto.

## The skills

| Skill | Root | Purpose |
| --- | --- | --- |
| `example-evidence-check` | `skills/` | the ordinary target: a canonical, editable `SKILL.md` |
| `example-writing-guide` | `skills/` **and** `skills-mirror/` | the same skill id under two roots. Only the canonical one is editable; a write aimed at the mirror is refused |
| `example-guardrails` | `skills/` | ships `scripts/block-risky-action.sh`, so the inventory reports an executable payload. The loop may still edit the document, never the script |
| `example-broken-frontmatter` | `skills/` | a `SKILL.md` whose frontmatter has no opening delimiter, so it loads with a reported warning instead of silently reading as empty |

`skills-mirror/` exists so the canonical-source ambiguity is a *test*, not a
paragraph in a document. Point the config at the mirror as if it were canonical
and `asi apply` refuses with `MirrorEditRefused`.

## Regenerating

```bash
python examples/build_fixtures.py            # rewrite fixtures/
python examples/build_fixtures.py --check    # assert the committed bytes match
```

The `--check` form runs in CI. If it fails, the fixtures were hand-edited: fix
the builder and regenerate, rather than editing the JSON.
