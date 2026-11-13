"""A replacement that copies a cited excerpt is refused.

The improver is shown the cited conversation excerpts (``prompts.cited_excerpts``)
and returns a whole replacement for the skill file. A skill file is shared and
often committed, so transcript text copied into it carries private conversation
content out of state/, and a rule that quotes the failing cases fits the fix to
those cases. Measured before (3.11 and 3.13, ``daily --commit`` with an improver
that appended 30 words of the cited user message): the run exited 0, the
candidate was validated, and its patch held the words. A replacement that
repeats ``MIN_COPIED_RUN_WORDS`` consecutive words of one excerpt, words the base
does not hold, is now a defect of the skill for this run, raised after the
structural and size checks and before the patch is made; the reason interpolates
neither the replacement nor the excerpt: its only variable parts are the skill id, as
in every refusal, and the digest that ends the conversation's record name, never its id.
The check reads each excerpt as the evidence
fence shows it (``prompts.shown_excerpts``: the nonce stripped, any forged
delimiter neutralized), the texts the fence joins, one excerpt at a time.

Words are whitespace-separated tokens compared exactly, so the check catches
verbatim copying only: not a paraphrase, not a shorter run, and not text taken
from the scorer's rationale.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest import mock

from agent_skill_improver.adapters.fixture_source import FixtureConversationSource
from agent_skill_improver.adapters.skill_fs import FilesystemSkillStore
from agent_skill_improver.app import pipeline, prompts
from agent_skill_improver.app.daily import run_daily
from agent_skill_improver.app.pipeline import (
    MIN_COPIED_RUN_WORDS,
    _word_runs,
    copied_conversation_named,
    copied_excerpt_source,
)
from agent_skill_improver.app.prompts import (
    _masked,
    cited_excerpts,
    improver_user_prompt,
    render_transcript,
    wrap_untrusted,
)
from agent_skill_improver.app.state import StateStore, keyed_name
from agent_skill_improver.clock import FrozenClock
from agent_skill_improver.config import PromptConfig, load_config
from agent_skill_improver.domain.redaction import quoted_value
from tests import fake_model
from tests.support import (
    FAILING_THREAD,
    FIXTURE_EXPORTS,
    REPO_ROOT,
    TDD_SKILL,
    InProcessRunner,
    build_env,
    run_cli,
)

DAY = "2026-02-01"
COMPLIANT_THREAD = "1700000000001-1"
#: Appended to the user's directive in the failing export, the message the fake
#: scorer cites for ``explicit-directive-followed``, so it is in a cited excerpt.
PRIVATE = (
    "my sister Ottilie Brackenridge lives at fourteen Wexcombe Lane and her surgery is booked "
    "for the ninth so please keep the Halvorsen payroll export out of every log you write "
    "today because legal has not cleared it yet thanks a lot"
)
#: In the compliant export, which no failure cites.
UNCITED = (
    "the quarterly board pack for Fenwick Marlowe holds the acquisition price and the names "
    "of the three people being let go next month so nobody outside the deal team may read "
    "any of it before the announcement"
)
#: In the failing export's correction, a message no failure case cites.
UNCITED_MESSAGE = (
    "our landlord Perpetua Osei wants the deposit of the flat on Calloway Street returned by "
    "Friday and my bank details for that are in the note I pasted yesterday so please do not "
    "lose it"
)
#: 24 words of the cited user message; ``fenced_passage`` puts a 25th between them.
AROUND = (
    "the vendor portal for Quillfeather Dental keeps the patient recall list and the clinic "
    "door code which changes on the first of every month"
).split()
#: The evidence fence's nonce in the improver's prompt (seed c241, skill tdd).
EVIDENCE_NONCE = prompts.nonce_for("c241", "evidence:tdd")
#: The two rewrites the evidence fence makes: (what the transcript holds, what it shows).
DELIMITER = ("<<<UNTRUSTED-DATA>>>", prompts.NEUTRALIZED)
NONCE = (EVIDENCE_NONCE, "[nonce-stripped]")
#: The skill's own words, which the agent loaded and repeated in the conversation.
SHARED = (
    "Before declaring a migration done run the whole suite from a clean checkout and paste "
    "the summary line of its output so the reader can see the count of tests that passed"
)


def fenced_passage(marker: str) -> str:
    """25 words, the 13th of them ``marker``."""
    assert len(AROUND) == 24
    return " ".join(AROUND[:12] + [marker] + AROUND[12:])


def words(text: str, n: int) -> str:
    out = text.split()[:n]
    assert len(out) == n, f"the passage has {len(text.split())} words, fewer than {n}"
    return " ".join(out)


def id_of(n: int) -> str:
    """A fixture thread id of ``n`` short words, ``pa pb pc ...``."""
    return " ".join("p" + chr(97 + i) for i in range(n))


def json_paths(value: Any, text: str, at: str = "") -> set[str]:
    """Where ``text`` stands in a JSON value: a dotted path, ``<key>`` for a key holding it."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            part = "<key>" if text in key else key
            here = f"{at}.{part}" if at else part
            if text in key:
                found.add(here)
            found |= json_paths(item, text, here)
    elif isinstance(value, list):
        for item in value:
            found |= json_paths(item, text, f"{at}[]")
    elif isinstance(value, str) and text in value:
        found.add(at)
    return found


def only(paths: Any) -> Path:
    (path,) = list(paths)
    return path


def clock() -> FrozenClock:
    return FrozenClock(datetime.datetime(2026, 2, 1, 9, 0, 0, tzinfo=datetime.timezone.utc))


def refusal(conversation_id: str, *, skill_id: str = "tdd") -> str:
    return (
        f"improver returned a replacement for skill {skill_id!r} that repeats "
        f"{MIN_COPIED_RUN_WORDS} or more consecutive words of the cited excerpt of "
        f"{copied_conversation_named(conversation_id)}, words the base skill text does not hold: "
        "a skill file "
        "is shared and often committed, so copied transcript text would carry private "
        "conversation content out of state/, and a rule that quotes the failing cases fits the "
        "fix to them. The improver was asked; the judge and the replay were not. Nothing is "
        "proposed against it this run, and neither the replacement nor the excerpt is "
        "quoted here."
    )


def add_to_export(exports: Path, name: str, element_id: str, text: str) -> None:
    path = exports / name
    data = json.loads(path.read_text(encoding="utf-8"))
    (element,) = [e for e in data["elements"] if e.get("id") == element_id]
    element["content"] = element["content"] + " " + text
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_env(base: Path, *, subprocess_model: bool = False, skill_text: str = "") -> Path:
    """``build_env`` over a copy of the fixture exports holding the passages above."""
    exports = base / "exports"
    shutil.copytree(FIXTURE_EXPORTS, exports)
    add_to_export(exports, "failing.json", "1700000000001-13", PRIVATE)
    add_to_export(exports, "failing.json", "1700000000001-19", UNCITED_MESSAGE)
    add_to_export(exports, "compliant.json", "1700000000001-13", UNCITED)
    if skill_text:
        add_to_export(exports, "failing.json", "1700000000001-13", SHARED)
    config = build_env(
        base / "env", seed="c241", fixtures_dir=exports, use_subprocess_model=subprocess_model
    )
    if skill_text:
        for root in ("skills-canonical", "skills-mirror"):
            (base / "env" / root / "tdd" / "SKILL.md").write_text(skill_text, encoding="utf-8")
    return config


class CopyingImprover(InProcessRunner):
    """The fake's answers, with ``line`` appended to the improver's replacement."""

    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line

    def run(self, *, role: str, system: str, user: str, response_schema: str) -> dict[str, Any]:
        payload = super().run(role=role, system=system, user=user, response_schema=response_schema)
        if role == "improver" and self.line:
            payload["new_skill_text"] = payload["new_skill_text"] + self.line + "\n"
        return payload


def run_pass(
    config_path: Path, line: str, *, day: str = DAY
) -> tuple[dict[str, Any], CopyingImprover, Path]:
    config = load_config(config_path)
    runner = CopyingImprover(line)
    store = FilesystemSkillStore(
        config.skill_roots, read_only=True, protected_paths=config.protected_paths
    )
    receipt = run_daily(
        config,
        source=FixtureConversationSource(config.source.fixtures_dir),
        runner=runner,
        skill_store=store,
        state=StateStore(config.state_dir, clock=clock()),
        clock=clock(),
        day=day,
        commit=True,
    )
    return receipt.to_dict(), runner, config.state_dir


def roles(runner: InProcessRunner) -> list[str]:
    return [role for role, _ in runner.calls]


def improver_prompts(runner: InProcessRunner) -> list[str]:
    return [p for (role, _), p in zip(runner.calls, runner.prompts) if role == "improver"]


def holders(state_dir: Path, text: str) -> list[str]:
    """Every file under ``state_dir`` holding ``text``, as written or JSON-escaped."""
    escaped = json.dumps(text)[1:-1]
    found = []
    for path in sorted(state_dir.rglob("*")):
        if path.is_file():
            body = path.read_text(encoding="utf-8", errors="replace")
            if text in body or escaped in body:
                found.append(path.relative_to(state_dir).as_posix())
    return found


def skill_id_of(n: int) -> str:
    """A skill id of ``n`` short words, ``sa sb sc ...``: its directory and frontmatter name."""
    return " ".join("s" + chr(97 + i) for i in range(n))


#: In the cited message after "skill '<name>'": the excerpt goes on past the name.
AFTER_NAME = "is the one to load before any migration of the billing tables"

#: Where the receipt, which ``daily --json`` prints, holds a conversation's id: the keys of
#: the conversation's entries, as it does every conversation's.
RECEIPT_ID_KEYS = frozenset({"classes.<key>", "records.<key>"})


def stands_only_at(value: Any, text: str, paths: frozenset[str]) -> bool:
    """True when ``text`` stands in the JSON ``value`` at exactly ``paths`` and nowhere else."""
    return json_paths(value, text) == paths


def copied_runs(text: str, *, shown: str, base: str) -> set[tuple[str, ...]]:
    """The ``MIN_COPIED_RUN_WORDS``-word runs of ``text`` that ``shown`` holds and ``base`` not."""
    n = MIN_COPIED_RUN_WORDS
    return _word_runs(text, n) & (_word_runs(shown, n) - _word_runs(base, n))


def roles_script(
    path: Path, *, line: str, thread: str = "", prompt_log: Path | None = None
) -> Path:
    """A model command for every role: the fake's answers, ``line`` added to the replacement.

    ``thread`` is the failing export's thread id when the test changed it (a fixture id
    may hold spaces); the improver's prompt is written to ``prompt_log`` when given.
    """
    lines = ["import json, sys", f"sys.path.insert(0, {str(REPO_ROOT / 'tests')!r})"]
    lines.append("import fake_model")
    if thread:
        lines.append(f"fake_model.FAILING_THREAD = {thread!r}")
        lines.append(
            "fake_model._scored_thread = lambda user: user.split("
            "'Score conversation ', 1)[1].split(' against every criterion', 1)[0]"
        )
    lines += [
        "request = json.loads(sys.stdin.read())",
        "role, user = request['role'], request['user']",
        "if role == 'scorer':",
        "    payload = fake_model.scorer(user, '')",
        "elif role == 'improver':",
    ]
    if prompt_log is not None:
        lines.append(f"    open({str(prompt_log)!r}, 'w', encoding='utf-8').write(user)")
    lines += [
        "    payload = fake_model.improver(user)",
        f"    payload['new_skill_text'] += {line!r} + '\\n'",
        "else:",
        "    payload = fake_model.judge(user)",
        "sys.stdout.write(json.dumps(payload))",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def use_roles(config: Path, script: Path, names: Sequence[str]) -> None:
    """Point each named role's ``argv`` in ``config`` at ``script``."""
    text = config.read_text(encoding="utf-8")
    for role in names:
        text, k = re.subn(
            rf'(\[model\.{role}\]\nargv = \[)"([^"]+)", "([^"]+)"',
            lambda m: f'{m.group(1)}"{sys.executable}", "{script}"',
            text,
        )
        assert k == 1, f"the {role}'s argv was not found"
    config.write_text(text, encoding="utf-8")


def only_the_conversations_records(found: Sequence[str]) -> bool:
    """The normalized conversation (and a raw export, when kept) hold the transcript by design."""
    return all(p.startswith(("conversations/", "raw/")) for p in found)


class TheRuleTests(unittest.TestCase):
    def test_the_run_length_is_the_one_the_docs_state(self) -> None:
        self.assertEqual(MIN_COPIED_RUN_WORDS, 25)

    def test_twenty_five_words_of_one_excerpt_are_a_copy(self) -> None:
        excerpt = "head line " + PRIVATE
        new = "---\nname: tdd\n---\n- " + words(PRIVATE, 25) + "\n"
        self.assertEqual(copied_excerpt_source(new, base_text="", excerpts=[("c1", excerpt)]), "c1")

    def test_twenty_four_words_are_not(self) -> None:
        new = "---\nname: tdd\n---\n- " + words(PRIVATE, 24) + "\n"
        self.assertIsNone(copied_excerpt_source(new, base_text="", excerpts=[("c1", PRIVATE)]))

    def test_a_run_the_base_holds_is_kept_not_copied(self) -> None:
        run = words(PRIVATE, 30)
        self.assertIsNone(
            copied_excerpt_source(run, base_text="x " + run + " y", excerpts=[("c1", PRIVATE)])
        )

    def test_a_run_that_reaches_past_what_the_base_holds_is_a_copy(self) -> None:
        # The base holds the first 25 words; the window that ends on word 26 is not the base's.
        base = words(PRIVATE, 25)
        new = words(PRIVATE, 26)
        self.assertEqual(
            copied_excerpt_source(new, base_text=base, excerpts=[("c1", PRIVATE)]), "c1"
        )

    def test_words_are_split_by_any_whitespace_and_compared_exactly(self) -> None:
        run = PRIVATE.split()[:25]
        spread = "\n".join(" ".join(run[i : i + 5]) for i in range(0, 25, 5)).replace(" ", "\t  ")
        self.assertEqual(
            copied_excerpt_source(spread, base_text="", excerpts=[("c", PRIVATE)]), "c"
        )
        changed = " ".join(run[:12] + [run[12].upper()] + run[13:])
        self.assertIsNone(copied_excerpt_source(changed, base_text="", excerpts=[("c", PRIVATE)]))

    def test_the_run_must_stand_in_one_excerpt_and_the_first_one_is_named(self) -> None:
        run = PRIVATE.split()[:25]
        halves = [("a", " ".join(run[:13])), ("b", " ".join(run[13:]))]
        self.assertIsNone(copied_excerpt_source(" ".join(run), base_text="", excerpts=halves))
        both = [("a", "nothing alike"), ("b", PRIVATE), ("c", PRIVATE)]
        self.assertEqual(copied_excerpt_source(" ".join(run), base_text="", excerpts=both), "b")

    def test_unicode_whitespace_separates_words_as_a_space_does(self) -> None:
        # str.split's whitespace: a no-break, line separator, ideographic or em space.
        run = PRIVATE.split()[:25]
        gaps = ["\u00a0", "\u2028", "\u3000", "\u2003"]
        spread = "".join(w + gaps[i % 4] for i, w in enumerate(run))
        self.assertEqual(
            copied_excerpt_source(spread, base_text="", excerpts=[("c", PRIVATE)]), "c"
        )
        excerpt = "\u00a0".join(PRIVATE.split())
        self.assertEqual(
            copied_excerpt_source(" ".join(run), base_text="", excerpts=[("c", excerpt)]), "c"
        )

    def test_a_changed_punctuation_mark_is_a_different_word(self) -> None:
        run = PRIVATE.split()[:25]
        marked = [w + {3: ",", 12: ":", 17: ".", 22: "!"}.get(i, "") for i, w in enumerate(run)]
        excerpt = " ".join(marked)
        self.assertEqual(
            copied_excerpt_source(excerpt, base_text="", excerpts=[("c", excerpt)]), "c"
        )
        dropped = " ".join(marked[:12] + [run[12]] + marked[13:])
        self.assertIsNone(copied_excerpt_source(dropped, base_text="", excerpts=[("c", excerpt)]))
        added = " ".join(marked[:6] + [marked[6] + ";"] + marked[7:])
        self.assertIsNone(copied_excerpt_source(added, base_text="", excerpts=[("c", excerpt)]))

    def test_a_run_that_ends_on_the_excerpts_last_word_is_a_copy(self) -> None:
        excerpt = "[conversation c / case k]\n" + words(PRIVATE, 30)
        last = " ".join(excerpt.split()[-25:])
        self.assertEqual(copied_excerpt_source(last, base_text="", excerpts=[("c", excerpt)]), "c")

    def test_the_excerpts_head_and_body_are_read_as_one_text(self) -> None:
        # The improver reads the head and the cited blocks as one text.
        head = "[conversation c / case k]"
        excerpt = head + "\n" + PRIVATE
        across = head + " " + words(PRIVATE, 20)
        self.assertEqual(len(across.split()), 25)
        self.assertEqual(
            copied_excerpt_source(across, base_text="", excerpts=[("c", excerpt)]), "c"
        )
        self.assertIsNone(copied_excerpt_source(head, base_text="", excerpts=[("c", excerpt)]))


class ThroughDailyTests(unittest.TestCase):
    def test_a_copied_run_is_refused_before_the_judge_and_leaves_no_copy(self) -> None:
        for n in (MIN_COPIED_RUN_WORDS, 30):
            with self.subTest(words=n), tempfile.TemporaryDirectory() as tmp:
                copied = words(PRIVATE, n)
                payload, runner, state_dir = run_pass(
                    make_env(Path(tmp).resolve()), "- Keep in mind: " + copied
                )
                (prompt,) = improver_prompts(runner)
                self.assertIn(copied, prompt, "the improver was shown the words it copied")
                self.assertEqual(payload["counts"]["candidates_proposed"], 0)
                self.assertEqual(list(state_dir.glob("candidates/*.json")), [])
                self.assertIn("improver", roles(runner))
                self.assertNotIn("judge", roles(runner))
                after = roles(runner)[roles(runner).index("improver") + 1 :]
                self.assertEqual(after, [], "no replay scored the replacement")
                skill = payload["skills"]["tdd"]
                self.assertEqual(
                    (skill["status"], skill["stage"], skill["reason"]),
                    ("quarantined", "propose", refusal(FAILING_THREAD)),
                )
                (record,) = state_dir.glob("quarantine/*.json")
                self.assertEqual(
                    json.loads(record.read_text(encoding="utf-8"))["reason"],
                    refusal(FAILING_THREAD),
                )
                self.assertNotIn(copied, json.dumps(payload))
                found = holders(state_dir, copied)
                self.assertTrue(found, "the transcript itself still holds the words")
                self.assertTrue(only_the_conversations_records(found), found)

    def test_the_next_run_asks_the_improver_again(self) -> None:
        # The quarantine holds for the run that made it: nothing is kept against the skill.
        with tempfile.TemporaryDirectory() as tmp:
            config = make_env(Path(tmp).resolve())
            first, _, _ = run_pass(config, "- " + words(PRIVATE, 30))
            self.assertEqual(first["skills"]["tdd"]["status"], "quarantined")
            second, runner, _ = run_pass(config, "", day="2026-02-02")
        self.assertIn("improver", roles(runner))
        self.assertEqual(second["counts"]["candidates_proposed"], 1)
        self.assertNotIn("tdd", second["skills"])

    def test_twenty_four_copied_words_are_proposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied = words(PRIVATE, MIN_COPIED_RUN_WORDS - 1)
            payload, runner, _ = run_pass(make_env(Path(tmp).resolve()), "- " + copied)
            self.assertEqual(payload["counts"]["candidates_proposed"], 1)
            self.assertNotIn("tdd", payload["skills"])
            self.assertIn("judge", roles(runner))

    def test_a_run_split_by_line_breaks_is_still_refused(self) -> None:
        run = PRIVATE.split()[:MIN_COPIED_RUN_WORDS]
        # Five lines of five words, one indented by a tab: the same 25 words.
        paragraph = "\n".join(" ".join(run[i : i + 5]) for i in range(0, len(run), 5))
        with tempfile.TemporaryDirectory() as tmp:
            payload, runner, _ = run_pass(
                make_env(Path(tmp).resolve()), paragraph.replace("\n", "\n\t", 1)
            )
            self.assertEqual(payload["skills"]["tdd"]["reason"], refusal(FAILING_THREAD))
            self.assertNotIn("judge", roles(runner))

    def test_the_control_a_bullet_between_the_lines_is_a_word_and_breaks_the_run(self) -> None:
        # Verbatim only: a "-" written between the copied words is a word of its own.
        run = PRIVATE.split()[:MIN_COPIED_RUN_WORDS]
        bullets = "\n".join("- " + " ".join(run[i : i + 5]) for i in range(0, len(run), 5))
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, _ = run_pass(make_env(Path(tmp).resolve()), bullets)
            self.assertEqual(payload["counts"]["candidates_proposed"], 1)

    def test_a_run_the_base_skill_also_holds_is_proposed(self) -> None:
        # The agent loaded the skill and repeated its words; the replacement keeps them.
        skill = TDD_SKILL + "- " + SHARED + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            payload, runner, _ = run_pass(make_env(Path(tmp).resolve(), skill_text=skill), "")
            (prompt,) = improver_prompts(runner)
            self.assertIn(SHARED, fake_model._fenced(prompt, "evidence"))
            self.assertEqual(payload["counts"]["candidates_proposed"], 1)
            self.assertNotIn("tdd", payload["skills"])

    def test_a_run_copied_from_text_the_prompt_did_not_cite_is_proposed(self) -> None:
        for name, passage in (
            ("a conversation no failure cites", UNCITED),
            ("an uncited message of the cited conversation", UNCITED_MESSAGE),
        ):
            with self.subTest(source=name), tempfile.TemporaryDirectory() as tmp:
                copied = words(passage, 30)
                payload, runner, _ = run_pass(make_env(Path(tmp).resolve()), "- " + copied)
                (prompt,) = improver_prompts(runner)
                self.assertNotIn(copied, prompt)
                self.assertEqual(payload["counts"]["candidates_proposed"], 1)
                self.assertNotIn("tdd", payload["skills"])


def run_with_passage(
    tmp: str, marker: str, line: str
) -> tuple[dict[str, Any], CopyingImprover, Path]:
    """``run_pass`` with ``fenced_passage(marker)`` added to the cited user message."""
    base = Path(tmp).resolve()
    config = make_env(base)
    add_to_export(base / "exports", "failing.json", "1700000000001-13", fenced_passage(marker))
    return run_pass(config, line)


class TheFenceRewritesTests(unittest.TestCase):
    """The check reads each excerpt as the evidence fence shows it to the improver."""

    def assert_shown_copy_refused(self, raw: str, shown: str) -> None:
        copied = fenced_passage(shown)
        with tempfile.TemporaryDirectory() as tmp:
            payload, runner, state_dir = run_with_passage(tmp, raw, "- " + copied)
            (prompt,) = improver_prompts(runner)
            self.assertIn(f"<<<UNTRUSTED-DATA evidence {EVIDENCE_NONCE}>>>", prompt)
            self.assertIn(copied, fake_model._fenced(prompt, "evidence"), "shown as copied")
            self.assertNotIn(fenced_passage(raw), fake_model._fenced(prompt, "evidence"))
            self.assertEqual(payload["skills"]["tdd"]["reason"], refusal(FAILING_THREAD))
            self.assertNotIn("judge", roles(runner))
            self.assertEqual(list(state_dir.glob("candidates/*.json")), [])
            self.assertTrue(only_the_conversations_records(holders(state_dir, copied)))

    def test_a_run_holding_a_neutralized_delimiter_copied_as_shown_is_refused(self) -> None:
        self.assert_shown_copy_refused(*DELIMITER)

    def test_a_run_holding_the_stripped_nonce_copied_as_shown_is_refused(self) -> None:
        self.assert_shown_copy_refused(*NONCE)

    def test_the_raw_run_the_fence_rewrote_was_not_shown_and_is_proposed(self) -> None:
        # The improver read the rewritten run, never the raw one: a replacement
        # holding the raw run copies no 25 words it was shown, and nothing refuses it.
        for name, (raw, shown) in (("delimiter", DELIMITER), ("nonce", NONCE)):
            with self.subTest(rewrite=name), tempfile.TemporaryDirectory() as tmp:
                payload, runner, _ = run_with_passage(tmp, raw, "- " + fenced_passage(raw))
                (prompt,) = improver_prompts(runner)
                self.assertIn(fenced_passage(shown), fake_model._fenced(prompt, "evidence"))
                self.assertEqual(payload["counts"]["candidates_proposed"], 1)
                self.assertNotIn("tdd", payload["skills"])
                self.assertIn("judge", roles(runner))


class TheOrderOfTheChecksTests(unittest.TestCase):
    """The copy check runs after the structural and size checks, before the patch."""

    def test_an_unsafe_replacement_that_also_copies_is_refused_as_unsafe(self) -> None:
        token = "g" + "hp_" + "B" * 36
        line = "- Keep in mind: " + words(PRIVATE, 30) + f"\n- Authenticate with GH_TOKEN={token}"
        with tempfile.TemporaryDirectory() as tmp:
            payload, runner, _ = run_pass(make_env(Path(tmp).resolve()), line)
        reason = payload["skills"]["tdd"]["reason"]
        self.assertTrue(reason.startswith("proposed replacement text is unsafe: "), reason)
        self.assertNotIn("consecutive words of the cited excerpt", reason)
        self.assertNotIn("judge", roles(runner))

    def test_a_replacement_past_the_cap_that_also_copies_is_refused_for_its_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            config = make_env(base)
            skill = (base / "env" / "skills-canonical" / "tdd" / "SKILL.md").read_text("utf-8")
            text = config.read_text(encoding="utf-8")
            self.assertEqual(text.count("max_skill_text_chars = 24000\n"), 1)
            # The cap is at its floor of 500, above the base; the replacement is past it.
            self.assertLess(len(skill), 500)
            config.write_text(
                text.replace("max_skill_text_chars = 24000\n", "max_skill_text_chars = 500\n"),
                "utf-8",
            )
            line = "- Keep in mind: " + words(PRIVATE, 30) + "\n- " + "keep it short " * 40
            payload, runner, _ = run_pass(config, line)
        reason = payload["skills"]["tdd"]["reason"]
        self.assertRegex(
            reason, r"^improver returned a \d+-char replacement for skill 'tdd', above "
        )
        self.assertNotIn("consecutive words of the cited excerpt", reason)
        self.assertNotIn("judge", roles(runner))


class TheCheckAndThePromptShareTheExcerptsTests(unittest.TestCase):
    def test_the_check_reads_the_excerpts_the_prompt_carried(self) -> None:
        # One more excerpt, made by the shared function only: the prompt must carry
        # it and the check must refuse a copy of it. A check that rendered its own
        # excerpts would not hold it, and a prompt that did would not show it.
        marker = words(UNCITED, 30)
        calls: list[int] = []

        def with_one_more(*args: Any, **kwargs: Any) -> list[tuple[str, str]]:
            calls.append(1)
            return [*cited_excerpts(*args, **kwargs), ("forged-conversation", marker)]

        with tempfile.TemporaryDirectory() as tmp:
            config = make_env(Path(tmp).resolve())
            with mock.patch.object(prompts, "cited_excerpts", with_one_more):
                payload, runner, _ = run_pass(config, "- " + marker)
        (prompt,) = improver_prompts(runner)
        self.assertIn(marker, fake_model._fenced(prompt, "evidence"))
        self.assertEqual(calls, [1], "one rendering serves the prompt and the check")
        self.assertEqual(payload["skills"]["tdd"]["reason"], refusal("forged-conversation"))

    def test_the_check_reads_each_excerpt_as_the_evidence_fence_shows_it(self) -> None:
        seen: list[list[tuple[str, str]]] = []
        cited: list[list[tuple[str, str]]] = []
        real = pipeline.copied_excerpt_source

        def recording(new_text: str, *, base_text: str, excerpts: Any) -> str | None:
            seen.append(list(excerpts))
            return real(new_text, base_text=base_text, excerpts=excerpts)

        def citing(*args: Any, **kwargs: Any) -> list[tuple[str, str]]:
            cited.append(cited_excerpts(*args, **kwargs))
            return cited[-1]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            config = make_env(base)
            for raw, _ in (DELIMITER, NONCE):
                add_to_export(
                    base / "exports", "failing.json", "1700000000001-13", fenced_passage(raw)
                )
            with (
                mock.patch.object(pipeline, "copied_excerpt_source", recording),
                mock.patch.object(prompts, "cited_excerpts", citing),
            ):
                _, runner, _ = run_pass(config, "")
        (prompt,) = improver_prompts(runner)
        (excerpts,) = seen
        (made,) = cited
        # One text per cited excerpt, each as the fence shows it, never two joined.
        self.assertGreater(len(made), 1)
        self.assertEqual(excerpts, prompts.shown_excerpts(made, seed="c241", skill_id="tdd"))
        self.assertEqual(
            "\n\n".join(text for _, text in excerpts), fake_model._fenced(prompt, "evidence")
        )
        for _, shown in (DELIMITER, NONCE):
            self.assertIn(fenced_passage(shown), "\n\n".join(text for _, text in excerpts))

    def test_the_fence_shows_what_it_showed_when_the_excerpts_were_rewritten_joined(
        self,
    ) -> None:
        # Rewritten one at a time, the texts join to what the fence made of them
        # joined: each excerpt begins with its head's "[", which no delimiter or
        # nonce runs on into. Delimiters and the nonce at both ends of each text:
        n = prompts.nonce_for("s", "evidence:tdd")
        texts = [
            f"[conversation a / case 1]\n{n} ends open <<",
            f"[conversation b / case 2]\nUNTRUSTED-DATA>>> then <<<UNTRUSTED-DATA evidence {n}",
            f"[conversation c / case 3]\n<<</END-UNTRUSTED-DATA>> x{n}",
            f"[conversation d / case 4]\n{n[:6]}",
            f"[conversation e / case 5]\n{n[6:]} <<<",
        ]
        excerpts = [(t[14], t) for t in texts]
        prompt = improver_user_prompt(
            skill_id="tdd",
            skill_path="tdd/SKILL.md",
            base_text="x",
            cases=[],
            conversations={},
            cfg=PromptConfig(),
            seed="s",
            excerpts=excerpts,
        )
        before = wrap_untrusted("\n\n".join(texts), label="evidence", nonce=n)
        self.assertIn("## CITED CONVERSATION EXCERPTS (untrusted data)\n" + before + "\n\n", prompt)
        shown = prompts.shown_excerpts(excerpts, seed="s", skill_id="tdd")
        self.assertEqual([cid for cid, _ in shown], ["a", "b", "c", "d", "e"])
        self.assertEqual(
            before.split("\n", 1)[1].rsplit("\n", 1)[0], "\n\n".join(t for _, t in shown)
        )


class TheConversationIsNamedByItsRecordDigestTests(unittest.TestCase):
    """The reason names the conversation by the digest its record's name ends in, never its id.

    An excerpt's head holds the id, and the conversation's own text can hold it too
    with the reason's words around it, so a quoted id of any length can complete a
    run. Measured before (3.11 and 3.13, ``daily --commit``): an id of 24 short words,
    quoted after the word "conversation", put a cited 25-word run in stdout and the
    receipt.
    """

    def test_any_id_is_named_by_its_record_digest_only(self) -> None:
        for cid in (FAILING_THREAD, id_of(1), id_of(24), id_of(25), " Mixed Case id "):
            with self.subTest(words=len(cid.split()), id=cid[:12]):
                digest = keyed_name(cid).rsplit("--", 1)[1]
                self.assertEqual(digest, hashlib.sha256(cid.encode("utf-8")).hexdigest())
                named = copied_conversation_named(cid)
                self.assertEqual(named, f"the conversation whose record name ends in --{digest}")
                self.assertNotIn(quoted_value(cid), named)
                self.assertEqual(set(cid.split()) & set(named.split()), set(), "a word of the id")

    def test_through_daily_a_copied_id_is_not_repeated_in_the_reason(self) -> None:
        # A fixture thread id may hold spaces; conversation-cli discovery refuses one.
        def scored_thread(user: str) -> str:
            return user.split("Score conversation ", 1)[1].split(" against every criterion", 1)[0]

        for n in (24, 25):
            cid = id_of(n)
            copied = f"conversation {quoted_value(cid)}"
            with self.subTest(words=n), tempfile.TemporaryDirectory() as tmp:
                config = make_env(Path(tmp).resolve())
                exports = Path(tmp).resolve() / "exports"
                add_to_export(exports, "failing.json", "1700000000001-13", copied)
                path = exports / "failing.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                data["thread"]["id"] = cid
                path.write_text(json.dumps(data), encoding="utf-8")
                with (
                    mock.patch.object(fake_model, "FAILING_THREAD", cid),
                    mock.patch.object(fake_model, "_scored_thread", scored_thread),
                ):
                    payload, runner, state_dir = run_pass(config, "- " + copied)
                skill = payload["skills"]["tdd"]
                self.assertEqual((skill["status"], skill["stage"]), ("quarantined", "propose"))
                self.assertEqual(skill["reason"], refusal(cid))
                self.assertNotIn(cid, skill["reason"])
                (record,) = state_dir.glob("quarantine/*.json")
                self.assertNotIn(cid, json.loads(record.read_text(encoding="utf-8"))["reason"])
                self.assertNotIn("judge", roles(runner))
                # How an operator finds the conversation the reason names.
                digest = skill["reason"].split("record name ends in --", 1)[1].split(",", 1)[0]
                (found,) = state_dir.glob(f"conversations/*--{digest}.json")
                self.assertEqual(found.name, keyed_name(cid) + ".json")


def pre_refactor_evidence(cases, conversations, cfg) -> list[str]:
    """The excerpt loop ``improver_user_prompt`` ran before ``cited_excerpts``, as written."""
    excerpts: list[str] = []
    for case in cases:
        conversation = conversations.get(case.conversation_id)
        if conversation is None:
            continue
        positions = [e.position for e in case.evidence]
        render = render_transcript(conversation, cfg=cfg, only_positions=positions)
        if render.text.strip():
            head = _masked(f"[conversation {case.conversation_id} / case {case.id}]", cfg)
            excerpts.append(f"{head}\n{render.text}")
    return excerpts


class ThePromptIsUnchangedTests(unittest.TestCase):
    def test_the_improver_prompt_is_the_one_the_pre_refactor_loop_made(self) -> None:
        captured: list[dict[str, Any]] = []
        real = prompts.improver_user_prompt

        def recording(**kwargs: Any) -> str:
            captured.append(kwargs)
            return real(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            config = build_env(Path(tmp).resolve() / "env", seed="c241", use_subprocess_model=False)
            with mock.patch.object(prompts, "improver_user_prompt", recording):
                payload, runner, _ = run_pass(config, "")
        self.assertEqual(payload["counts"]["candidates_proposed"], 1)
        (kwargs,) = captured
        (prompt,) = improver_prompts(runner)
        args = (kwargs["cases"], kwargs["conversations"], kwargs["cfg"])
        old = pre_refactor_evidence(*args)
        self.assertEqual([text for _, text in kwargs["excerpts"]], old)
        self.assertEqual(
            [cid for cid, _ in kwargs["excerpts"]], [FAILING_THREAD] * len(old), "one per case"
        )
        rest = {k: v for k, v in kwargs.items() if k != "excerpts"}
        self.assertEqual(prompt, prompts.improver_system_prompt() + "\n" + real(**rest))
        self.assertEqual(fake_model._fenced(prompt, "evidence"), "\n\n".join(old))


class TheCommandLineTests(unittest.TestCase):
    """``daily --commit`` as an operator runs it, with an improver that copies the excerpt."""

    def test_the_summary_the_receipt_and_state_carry_no_copy(self) -> None:
        copied = words(PRIVATE, 30)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            improver = base / "improver.py"
            improver.write_text(
                "import json, subprocess, sys\n"
                f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})\n"
                "from agent_skill_improver.adapters.model_command import extract_json_object\n"
                f"fake = {str(REPO_ROOT / 'tests' / 'fake_model.py')!r}\n"
                "raw = sys.stdin.read()\n"
                "out = subprocess.run([sys.executable, fake, *sys.argv[1:]], input=raw,"
                " capture_output=True, text=True).stdout\n"
                "if json.loads(raw).get('role') == 'improver':\n"
                "    obj = extract_json_object(out)\n"
                f"    obj['new_skill_text'] += '- Keep in mind: ' + {copied!r} + '\\n'\n"
                "    out = json.dumps(obj)\n"
                "sys.stdout.write(out)\n",
                encoding="utf-8",
            )
            config = make_env(base, subprocess_model=True)
            text, n = re.subn(
                r'(\[model\.improver\]\nargv = \[)"([^"]+)", "([^"]+)"',
                lambda m: f'{m.group(1)}"{sys.executable}", "{improver}"',
                config.read_text(encoding="utf-8"),
            )
            self.assertEqual(n, 1, "the improver's argv was not found")
            config.write_text(text, encoding="utf-8")
            daily = run_cli(["--config", str(config), "daily", "--commit", "--day", DAY])
            self.assertEqual(daily.returncode, 0, daily.stderr)
            state_dir = base / "env" / "state"
            self.assertIn("candidates=0", daily.stdout)
            self.assertIn("skills_quarantined=1", daily.stdout)
            self.assertNotIn(copied, daily.stdout + daily.stderr)
            self.assertEqual(list(state_dir.glob("candidates/*.json")), [])
            (receipt,) = state_dir.glob("receipts/*.json")
            record = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(record["skills"]["tdd"]["reason"], refusal(FAILING_THREAD))
            found = holders(state_dir, copied)
            self.assertTrue(only_the_conversations_records(found), found)

    def test_an_id_quoted_after_the_word_conversation_reaches_no_output_or_record(self) -> None:
        # Measured before (3.11 and 3.13): the reason quoted an id of 24 short words
        # after "conversation", a cited 25-word run, and stdout and the receipt held it.
        for n in (24, 25):
            cid = id_of(n)
            copied = f"conversation {quoted_value(cid)}"
            with self.subTest(words=n), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp).resolve()
                model = base / "roles.py"
                model.write_text(
                    "import json, sys\n"
                    f"sys.path.insert(0, {str(REPO_ROOT / 'tests')!r})\n"
                    "import fake_model\n"
                    f"fake_model.FAILING_THREAD = {cid!r}\n"
                    "fake_model._scored_thread = lambda user: user.split("
                    "'Score conversation ', 1)[1].split(' against every criterion', 1)[0]\n"
                    "request = json.loads(sys.stdin.read())\n"
                    "role, user = request['role'], request['user']\n"
                    "if role == 'scorer':\n"
                    "    payload = fake_model.scorer(user, '')\n"
                    "elif role == 'improver':\n"
                    "    payload = fake_model.improver(user)\n"
                    f"    payload['new_skill_text'] += '- ' + {copied!r} + '\\n'\n"
                    "else:\n"
                    "    payload = fake_model.judge(user)\n"
                    "sys.stdout.write(json.dumps(payload))\n",
                    encoding="utf-8",
                )
                config = make_env(base, subprocess_model=True)
                add_to_export(base / "exports", "failing.json", "1700000000001-13", copied)
                path = base / "exports" / "failing.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                data["thread"]["id"] = cid
                path.write_text(json.dumps(data), encoding="utf-8")
                text = config.read_text(encoding="utf-8")
                for role in ("scorer", "improver"):
                    text, k = re.subn(
                        rf'(\[model\.{role}\]\nargv = \[)"([^"]+)", "([^"]+)"',
                        lambda m: f'{m.group(1)}"{sys.executable}", "{model}"',
                        text,
                    )
                    self.assertEqual(k, 1, f"the {role}'s argv was not found")
                config.write_text(text, encoding="utf-8")
                daily = run_cli(["--config", str(config), "daily", "--commit", "--day", DAY])
                self.assertEqual(daily.returncode, 0, daily.stderr)
                self.assertIn("candidates=0", daily.stdout)
                self.assertIn("skills_quarantined=1", daily.stdout)
                state_dir = base / "env" / "state"
                self.assertEqual(list(state_dir.glob("candidates/*.json")), [])
                files = {
                    name: only(state_dir.glob(f"{name}/*.json"))
                    for name in ("receipts", "checkpoints", "quarantine")
                }
                records = {
                    name: json.loads(f.read_text(encoding="utf-8")) for name, f in files.items()
                }
                self.assertEqual(records["receipts"]["skills"]["tdd"]["reason"], refusal(cid))
                self.assertEqual(records["quarantine"]["reason"], refusal(cid))
                # The receipt and the checkpoint key each conversation's entry by its id,
                # as they do every conversation's, before this check and after; the
                # reason never names it.
                keyed = {
                    "receipts": {"classes.<key>", "records.<key>"},
                    "checkpoints": {"records.<key>"},
                }
                for label, needle in (
                    ("the copied run", copied),
                    ("the run unquoted", "conversation " + cid),
                    ("the id", cid),
                ):
                    self.assertNotIn(needle, daily.stdout, f"{label} in stdout")
                    self.assertNotIn(needle, daily.stderr, f"{label} in stderr")
                    for name, record in records.items():
                        expected = keyed.get(name, set()) if needle == cid else set()
                        self.assertEqual(
                            json_paths(record, needle), expected, f"{label} in the {name}"
                        )
                        if needle != cid:
                            self.assertNotIn(needle, files[name].read_text(encoding="utf-8"))

    def test_a_skill_named_by_the_copied_words_is_repeated_only_as_its_name(self) -> None:
        # Measured (audit 3, 3.11 and 3.13): a skill named by 24 short words, and a cited
        # message holding "skill '<name>'" (25 words) that the improver copied. The run is
        # refused, and the reason, which names the skill by its id as every refusal does,
        # and the summary's line for the skill hold that run: the word "skill" and the
        # skill's own name. It is the one copied run the output may hold.
        sid = skill_id_of(24)
        named = f"skill {sid!r}"
        self.assertEqual(len(named.split()), MIN_COPIED_RUN_WORDS)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            config = make_env(base, subprocess_model=True)
            # The message goes on past the name, so a reason that quoted the excerpt's
            # next word would hold a second copied run.
            add_to_export(
                base / "exports", "failing.json", "1700000000001-13", f"{named} {AFTER_NAME}"
            )
            for root in ("skills-canonical", "skills-mirror"):
                skill_dir = (base / "env" / root / "tdd").rename(base / "env" / root / sid)
                doc = skill_dir / "SKILL.md"
                doc.write_text(
                    doc.read_text(encoding="utf-8").replace("name: tdd\n", f"name: {sid}\n", 1),
                    encoding="utf-8",
                )
            skill_text = (base / "env" / "skills-canonical" / sid / "SKILL.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(f"name: {sid}\n", skill_text)
            prompt_log = base / "improver-prompt.txt"
            script = roles_script(base / "roles.py", line="- " + named, prompt_log=prompt_log)
            use_roles(config, script, ("improver",))
            daily = run_cli(["--config", str(config), "daily", "--commit", "--day", DAY])
            self.assertEqual(daily.returncode, 0, daily.stderr)
            self.assertIn("candidates=0", daily.stdout)
            self.assertIn("skills_quarantined=1", daily.stdout)
            state_dir = base / "env" / "state"
            self.assertEqual(list(state_dir.glob("candidates/*.json")), [])
            files = {
                name: only(state_dir.glob(f"{name}/*.json"))
                for name in ("receipts", "checkpoints", "quarantine")
            }
            records = {name: json.loads(f.read_text(encoding="utf-8")) for name, f in files.items()}
            skill = records["receipts"]["skills"][sid]
            self.assertEqual((skill["status"], skill["stage"]), ("quarantined", "propose"))
            # The reason is the fixed words, the skill id and the conversation's digest.
            expected = refusal(FAILING_THREAD, skill_id=sid)
            self.assertEqual(skill["reason"], expected)
            self.assertEqual(records["checkpoints"]["skills"][sid]["reason"], expected)
            self.assertEqual(records["quarantine"]["reason"], expected)
            self.assertIn(f"quarantined skill {sid!r} at propose: {expected}", daily.stdout)
            # What the improver was shown holds the run; the base does not.
            shown = fake_model._fenced(prompt_log.read_text(encoding="utf-8"), "evidence")
            run = tuple(named.split())
            self.assertIn(run, _word_runs(shown, MIN_COPIED_RUN_WORDS))
            self.assertNotIn(run, _word_runs(skill_text, MIN_COPIED_RUN_WORDS))

            def holds_only_the_name(output: str, body: str) -> bool:
                # The copied runs of ``body`` are exactly the one of the word skill and the
                # skill's name, and in stderr none.
                allowed = set() if output == "stderr" else {run}
                return copied_runs(body, shown=shown, base=skill_text) == allowed

            outputs = {"stdout": daily.stdout, "stderr": daily.stderr}
            outputs.update({name: f.read_text(encoding="utf-8") for name, f in files.items()})
            for name, body in outputs.items():
                with self.subTest(output=name):
                    self.assertTrue(
                        holds_only_the_name(name, body),
                        f"a copied run in {name} other than the word skill and the skill's name",
                    )
            # The check is strict: a run that goes on past the name to the excerpt's next
            # word, as a reason quoting that word would print, holds the name's words and
            # still fails it.
            next_word = AFTER_NAME.split()[0]
            went_on = (*run[1:], next_word)
            planted = f"{daily.stdout}\n{named} {next_word}\n"
            self.assertEqual(copied_runs(planted, shown=shown, base=skill_text), {run, went_on})
            self.assertFalse(holds_only_the_name("stdout", planted))

    def test_the_json_summary_prints_a_long_id_only_as_the_receipts_keys(self) -> None:
        # ``daily --json`` prints the receipt, which keys each conversation's entry by its
        # id (``classes``, ``records``), as it keys every conversation's; the default text
        # output prints neither. Measured (audit 3, 3.11 and 3.13): a 25-word id stood in
        # the JSON stdout, not in the reason.
        cid = id_of(25)
        copied = f"conversation {quoted_value(cid)}"
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            config = make_env(base, subprocess_model=True)
            add_to_export(base / "exports", "failing.json", "1700000000001-13", copied)
            path = base / "exports" / "failing.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["thread"]["id"] = cid
            path.write_text(json.dumps(data), encoding="utf-8")
            script = roles_script(base / "roles.py", line="- " + copied, thread=cid)
            use_roles(config, script, ("scorer", "improver"))
            daily = run_cli(["--config", str(config), "daily", "--json", "--commit", "--day", DAY])
            self.assertEqual(daily.returncode, 0, daily.stderr)
            out = json.loads(daily.stdout)
            self.assertEqual(out["counts"]["candidates_proposed"], 0)
            self.assertEqual(list((base / "env" / "state").glob("candidates/*.json")), [])
            self.assertEqual(out["skills"]["tdd"]["reason"], refusal(cid))
            self.assertTrue(stands_only_at(out, cid, RECEIPT_ID_KEYS), sorted(json_paths(out, cid)))
            for needle in (copied, "conversation " + cid):
                self.assertEqual(json_paths(out, needle), set(), needle[:30])
                self.assertNotIn(needle, daily.stdout)
            self.assertNotIn(cid, daily.stderr)
            # The check is strict: the id as a value, even inside its own entry, fails it.
            planted = json.loads(daily.stdout)
            planted["records"][cid]["planted"] = cid
            self.assertFalse(stands_only_at(planted, cid, RECEIPT_ID_KEYS))


if __name__ == "__main__":
    unittest.main()
