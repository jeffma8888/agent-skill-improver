"""A candidate's SKILL.md text size change is shown before apply, never claimed as a cost.

The tests of c249 (DESIGN v2, 4.2). Measured before: a replacement that doubles
``tdd``'s SKILL.md (275 -> 608 bytes, 46 -> 104 words, 10 -> 16 lines) passed the
growth cap, the gate and ``review`` with no line or key naming its size. Now
``review`` prints the complete SKILL.md text's change after ``base sha256``,
``review --json`` carries it in ``size_change`` and the daily receipt's entry for
every candidate carries it with the run's window. The numbers are properties of
the complete SKILL.md text, never a measured cost or a count of use.

Every expected number is recomputed here with plain operations, never imported
from the helper under test: the fold is written out
(``raw.replace("\\r\\n", "\\n").replace("\\r", "\\n")``), then ``len(encode)``,
``split``, ``splitlines`` and ``-(-b // 4)``. Conversation counts come from the
record's claimed failures and from ``state/failures/*.json``, and are checked
against the fixed numbers measured before (DESIGN 1.7, 1.8) as well.

The withheld-failure module is imported as a module only, so unittest does not
collect its classes a second time from here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tests.test_a_failure_withheld_from_the_improvers_prompt_is_replayed_and_reported_but_never_gated as withheld  # noqa: E501
from agent_skill_improver.domain.candidate import Candidate, make_patch
from tests.support import REPO_ROOT, build_env

DAY = "2026-04-01"
SIZE_PREFIX = "size          SKILL.md "
SECOND_LINE = (
    "  full SKILL.md text change; {n} conversation(s) cited as failure evidence; "
    "not a measured cost (token est.:"
)
CITED_BASIS_TEXT = "distinct conversations in the candidate's claimed rubric failures"
WINDOW_BASIS_TEXT = (
    "distinct scored conversations with a failure primarily attributed to this skill, "
    "including rubric-prior-only attribution; not a count of conversations that used or "
    "loaded it"
)
GATE_FAILED = "CAND-2026-01-02-03-example-writing-guide"
#: A `candidates --json` row, exactly: the keys it had before c249.
LISTING_ROW_KEYS = [
    "added_lines",
    "canonical_path",
    "created_at",
    "gate_passed",
    "id",
    "removed_lines",
    "skill_id",
    "status",
]
REFUSAL_MARKERS = ("size_change", "bytes (", "est. ", "not a measured cost")


def fold(raw: str) -> str:
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def expected(raw: str) -> dict[str, int]:
    text = fold(raw)
    size = len(text.encode("utf-8"))
    return {
        "bytes": size,
        "words": len(text.split()),
        "lines": len(text.splitlines()),
        "est_tokens": -(-size // 4),
    }


def expected_change(base_raw: str, new_raw: str) -> tuple[dict, dict, dict]:
    base, new = expected(base_raw), expected(new_raw)
    return base, new, {k: new[k] - base[k] for k in base}


def size_line(base: dict, new: dict, delta: dict) -> str:
    return (
        f"{SIZE_PREFIX}{base['bytes']} -> {new['bytes']} bytes ({delta['bytes']:+d}), "
        f"{base['words']} -> {new['words']} words ({delta['words']:+d}), "
        f"{base['lines']} -> {new['lines']} lines ({delta['lines']:+d}), "
        f"est. {delta['est_tokens']:+d} tokens"
    )


def keys_anywhere(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_anywhere(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_anywhere(v)}
    return set()


def pool(failures_dir: Path, skill_id: str) -> set[str]:
    """The distinct conversations whose failure is primarily attributed to ``skill_id``."""
    return {
        case["conversation_id"]
        for path in sorted(failures_dir.glob("*.json"))
        for case in json.loads(path.read_text(encoding="utf-8"))["cases"]
        if case.get("primary_skill_id") == skill_id
    }


def cli(cfg: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("ASI_CONFIG", None)
    env.pop("ASI_AUTOMATED_RUN", None)
    return subprocess.run(
        [sys.executable, "-m", "agent_skill_improver", "--config", str(cfg), *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=300,
    )


def doubling_env(base: Path) -> Path:
    """``build_env``'s tree with the improver doubling the skill (``grow-double``)."""
    cfg = build_env(base, seed="c249")
    text = cfg.read_text(encoding="utf-8")
    heading = "[model.improver]\n"
    assert text.count(heading) == 1
    cfg.write_text(
        text.replace(heading, heading + 'env = { ASI_FAKE_MODE = "grow-double" }\n'),
        encoding="utf-8",
    )
    return cfg


class TheSkillTextSizeChangeIsShownBeforeApply(unittest.TestCase):
    """One committed doubling run in ``build_env``'s tree, put back before each test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="asi-c249-")
        root = Path(cls._tmp.name).resolve()
        cls.base = root / "env"
        cls.cfg = doubling_env(cls.base)
        daily = cli(cls.cfg, "--json", "daily", "--commit", "--day", DAY)
        assert daily.returncode == 0, daily.stderr
        cls.daily_json = json.loads(daily.stdout)
        listing = json.loads(cli(cls.cfg, "--json", "candidates").stdout)
        cls.cid = sorted(c["id"] for c in listing["candidates"])[0]
        cls.pristine = root / "pristine"
        shutil.copytree(cls.base, cls.pristine, symlinks=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.restore()

    def restore(self) -> None:
        shutil.rmtree(self.base)
        shutil.copytree(self.pristine, self.base, symlinks=True)

    # -- the fixture ---------------------------------------------------------

    @property
    def record_path(self) -> Path:
        (path,) = sorted((self.base / "state" / "candidates").rglob(f"{self.cid}.json"))
        return path

    def raw_record(self) -> dict[str, Any]:
        return json.loads(self.record_path.read_text(encoding="utf-8"))

    def skill_path(self) -> Path:
        return Path(self.raw_record()["target"]["canonical_path"])

    def change(self) -> tuple[dict, dict, dict]:
        """Recomputed from the live SKILL.md bytes and the record's replacement."""
        base_raw = self.skill_path().read_bytes().decode("utf-8")
        return expected_change(base_raw, self.raw_record()["patch"]["new_text"])

    def entry(self) -> dict[str, Any]:
        (entry,) = [e for e in self.daily_json["candidates"] if e["candidate_id"] == self.cid]
        return entry

    def review(self, *args: str) -> subprocess.CompletedProcess:
        return cli(self.cfg, *args, "review", self.cid)

    def assert_refused_without_a_size(self, run) -> None:
        """``run(*args)`` reviews the record; both forms exit 3 and print no number."""
        for form in ((), ("--json",)):
            proc = run(*form)
            self.assertEqual(proc.returncode, 3, (form, proc.stdout, proc.stderr))
            self.assertEqual(proc.stdout, "", form)
            for marker in REFUSAL_MARKERS:
                self.assertNotIn(marker, proc.stdout + proc.stderr, (form, marker))

    def sized(self, payload: dict[str, Any], where: Any = None) -> dict[str, Any]:
        """``payload["size_change"]``, asserted present (the build before c249 has none)."""
        self.assertIn("size_change", payload, where)
        return payload["size_change"]

    def helper(self) -> Any:
        """The module under test, imported here so its absence fails one test, not the module."""
        try:
            from agent_skill_improver.domain import skill_size
        except ImportError as exc:
            self.fail(f"agent_skill_improver.domain.skill_size is missing: {exc}")
        return skill_size

    # -- 1-3: what review says, and what it never says ----------------------

    def test_review_names_the_full_skill_text_change_of_a_doubled_skill(self):
        shown = self.review()
        self.assertEqual(shown.returncode, 0, shown.stderr)
        base, new, delta = self.change()
        self.assertEqual(
            (base["bytes"], new["bytes"], base["words"], new["words"]), (275, 608, 46, 104)
        )
        self.assertEqual((base["lines"], new["lines"], delta["est_tokens"]), (10, 16, 83))
        lines = shown.stdout.splitlines()
        sized = [i for i, line in enumerate(lines) if line.startswith(SIZE_PREFIX)]
        self.assertEqual(len(sized), 1, shown.stdout)
        (at,) = sized
        self.assertEqual(lines[at], size_line(base, new, delta))
        self.assertTrue(lines[at + 1].startswith(SECOND_LINE.format(n=1)), lines[at + 1])
        base_at = next(i for i, line in enumerate(lines) if line.startswith("base sha256   "))
        gate_at = next(i for i, line in enumerate(lines) if line.startswith("gate          "))
        self.assertEqual(at, base_at + 1)
        self.assertLess(at + 1, gate_at)

    def test_review_json_carries_the_size_change_and_its_basis(self):
        shown = self.review("--json")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        size = self.sized(json.loads(shown.stdout))
        base, new, delta = self.change()
        self.assertEqual(size["base"], base)
        self.assertEqual(size["replacement"], new)
        self.assertEqual(size["delta"], delta)
        for column in ("bytes", "words", "lines", "est_tokens"):
            self.assertEqual(
                size["base"][column] + size["delta"][column], size["replacement"][column]
            )
        for side in ("base", "replacement"):
            self.assertEqual(size[side]["est_tokens"], -(-size[side]["bytes"] // 4))
        self.assertEqual((size["base"]["est_tokens"], size["delta"]["est_tokens"]), (69, 83))
        self.assertIn("estimate", size["token_estimate"])
        self.assertIn("no particular model", size["token_estimate"])
        self.assertEqual(size["scope"], "complete SKILL.md text only")
        for phrase in (
            "not a measured cost",
            "metadata-only discovery does not",
            "no prompt usage",
        ):
            self.assertIn(phrase, size["not_claimed"])
        self.assertEqual(size["cited_basis"], CITED_BASIS_TEXT)

    def test_the_line_claims_no_price_runtime_or_exact_token_count(self):
        shown = self.review()
        self.assertEqual(shown.returncode, 0, shown.stderr)
        lines = shown.stdout.splitlines()
        sized = [i for i, line in enumerate(lines) if line.startswith(SIZE_PREFIX)]
        self.assertEqual(len(sized), 1, shown.stdout)
        (at,) = sized
        first, second = lines[at], lines[at + 1]
        self.assertIn("est.", first)
        self.assertLess(first.index("est."), first.index("tokens"))
        self.assertIn("token est.", second)
        self.assertIn("not a measured cost", second)
        for line in (first, second):
            for word in (
                "$",
                "USD",
                "second",
                " ms",
                "per session",
                "every session",
                "loaded",
                "used by",
            ):
                self.assertNotIn(word, line, (word, line))
        size = self.sized(json.loads(self.review("--json").stdout))
        for key in keys_anywhere(size):
            for word in ("cost", "price", "usd", "seconds", "duration", "session"):
                self.assertNotIn(word, key.lower(), key)

    # -- 4: the two conversation counts -------------------------------------

    def assert_counts(self, *, review_json, review_text, claims, entry, pool_ids, cited, window):
        size = self.sized(review_json)
        self.assertEqual(size["cited_conversations"], len({r["conversation_id"] for r in claims}))
        self.assertEqual(size["cited_conversations"], cited)
        self.assertIn(SECOND_LINE.format(n=cited), review_text)
        attributed = self.sized(entry)["window"]["conversations_with_a_failure_attributed"]
        self.assertEqual(attributed, len(pool_ids))
        self.assertEqual(attributed, window)

    def test_cited_conversations_count_the_claims_and_the_window_the_whole_pool(self):
        with self.subTest("tdd: 2 claimed failures from 1 conversation"):
            claims = self.raw_record()["rubric_failures"]
            self.assertEqual(len(claims), 2)
            shown = self.review()
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assert_counts(
                review_json=json.loads(self.review("--json").stdout),
                review_text=shown.stdout,
                claims=claims,
                entry=self.entry(),
                pool_ids=pool(
                    self.base / "state" / "failures", self.raw_record()["target"]["skill_id"]
                ),
                cited=1,
                window=1,
            )
        for label, options, claimed, cited in (
            ("withheld k = 0: 5 claimed failures from 4 conversations", {"k": 0}, 5, 4),
            ("(a) a per-candidate cap of 1 leaves one claim", {"k": 0, "max_failures": 1}, 1, 1),
            ("(b) k = 1 withholds conversation 0006 whole", {"k": 1}, 3, 3),
        ):
            with self.subTest(label):
                run = withheld.daily(**options)
                claims = [r.to_dict() for r in run.candidate.rubric_failures]
                self.assertEqual(len(claims), claimed)
                shown = run.cli("review", withheld.CID)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                as_json = run.cli("--json", "review", withheld.CID)
                self.assertEqual(as_json.returncode, 0, as_json.stderr)
                pool_ids = pool(run.state.dir("failures"), "example-evidence-check")
                self.assertEqual(pool_ids, set(withheld.FAILURE_CONVERSATIONS))
                self.assert_counts(
                    review_json=json.loads(as_json.stdout),
                    review_text=shown.stdout,
                    claims=claims,
                    entry=run.entry,
                    pool_ids=pool_ids,
                    cited=cited,
                    window=4,
                )

    # -- 5-6: the helper's arithmetic (unit) --------------------------------

    def test_a_shrinking_replacement_shows_a_negative_change(self):
        base = "---\nname: tidy\n---\n\n" + "- Keep every rule short, plain and checkable.\n" * 6
        shorter = "---\nname: tidy\n---\n\n- Keep it short.\n"
        change = self.helper().size_change(base, shorter)
        exp_base, exp_new, exp_delta = expected_change(base, shorter)
        self.assertEqual((change["base"], change["replacement"]), (exp_base, exp_new))
        for column, value in exp_delta.items():
            self.assertLess(change["delta"][column], 0, column)
            self.assertEqual(change["delta"][column], exp_new[column] - exp_base[column], column)
        self.assertEqual(change["delta"], exp_delta)

    def test_bytes_are_utf8_bytes_and_lines_are_the_diffs_lines(self):
        text_size = self.helper().text_size
        accented = "é" * 10
        self.assertEqual(len(accented), 10)
        self.assertEqual(text_size(accented)["bytes"], len(accented.encode("utf-8")))
        self.assertEqual(text_size(accented)["bytes"], 20)
        self.assertEqual(text_size("a\nb")["lines"], 2)
        self.assertEqual(text_size("a\n\nb\n")["lines"], 3)
        spaced = "a  b\tc\nd"
        self.assertEqual(len(spaced.split(" ")), 3)
        self.assertEqual(text_size(spaced)["words"], 4)
        self.assertEqual(text_size("a\r\nb\r\n"), text_size("a\nb\n"))
        self.assertEqual(text_size("a\nb\n"), expected("a\nb\n"))

    # -- 7: line endings ------------------------------------------------------

    def test_a_crlf_skill_and_a_crlf_replacement_are_counted_as_hashed(self):
        run, candidate, _ = withheld.TheFrozenV3Record.frozen_run(self)
        live = Path(candidate.target.canonical_path)
        lf = live.read_bytes()
        with self.subTest("a CRLF live base"):
            live.write_bytes(lf.replace(b"\n", b"\r\n"))
            raw = live.read_bytes()
            self.assertTrue(raw.endswith(b"\r\n"))
            shown = run.cli("--json", "review", withheld.CID)
            self.assertEqual(shown.returncode, 0, shown.stderr)
            size = self.sized(json.loads(shown.stdout))
            self.assertEqual(size["base"]["bytes"], len(raw.replace(b"\r\n", b"\n")))
            self.assertEqual(len(raw) - size["base"]["bytes"], raw.count(b"\r\n"))
            self.assertEqual((len(raw), size["base"]["bytes"], raw.count(b"\r\n")), (1007, 980, 27))
        live.write_bytes(lf)
        with self.subTest("a self-consistent CRLF replacement record"):
            base_text = lf.decode("utf-8")
            raw = candidate.to_dict()
            raw["patch"] = make_patch(
                base_text,
                candidate.patch.new_text.replace("\n", "\r\n"),
                path_label=candidate.target.canonical_path,
            ).to_dict()
            crlf = Candidate.from_dict(raw)
            self.assertEqual(crlf.patch.new_text_sha256, candidate.patch.new_text_sha256)
            run.state.write_candidate(crlf)
            shown = run.cli("--json", "review", withheld.CID)
            self.assertEqual(shown.returncode, 0, shown.stderr)
            size = self.sized(json.loads(shown.stdout))
            stored = json.loads(withheld.candidate_file(run).read_text(encoding="utf-8"))
            new_raw = stored["patch"]["new_text"]
            folded = len(fold(new_raw).encode("utf-8"))
            self.assertEqual(size["replacement"]["bytes"], folded)
            self.assertEqual(len(new_raw.encode("utf-8")) - folded, new_raw.count("\r\n"))
            self.assertEqual(
                (len(new_raw.encode("utf-8")), folded, new_raw.count("\r\n")), (1104, 1073, 31)
            )
            self.assertIn("folded to LF", size["basis"])

    # -- 8: a refused review prints no number --------------------------------

    def test_a_refused_review_prints_no_size_and_a_frozen_v3_record_gets_one(self):
        with self.subTest("an edited record"):
            raw = self.raw_record()
            raw["patch"]["new_text"] += "edited\n"
            self.record_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
            self.assert_refused_without_a_size(self.review)
        self.restore()
        with self.subTest("a moved base"):
            skill = self.skill_path()
            skill.write_text(skill.read_text(encoding="utf-8") + "live edit\n", encoding="utf-8")
            self.assert_refused_without_a_size(self.review)
        self.restore()
        with self.subTest("a closed record: gate-failed"):
            run = withheld.daily(k=0)
            self.assertEqual(run.state.read_candidate(GATE_FAILED).status, "gate-failed")
            self.assert_refused_without_a_size(lambda *form: run.cli(*form, "review", GATE_FAILED))
        with self.subTest("a closed record: declined"):
            declined = cli(self.cfg, "decline", self.cid, "--reason", "-", stdin="too broad\n")
            self.assertEqual(declined.returncode, 0, declined.stderr)
            self.assertEqual(self.raw_record()["status"], "declined")
            self.assert_refused_without_a_size(self.review)
        self.restore()
        with self.subTest("a validated frozen v3 record gets a size"):
            run, candidate, _ = withheld.TheFrozenV3Record.frozen_run(self)
            self.assertEqual(candidate.validation["schema"], "asi.validation.v3")
            text = run.cli("review", withheld.CID)
            self.assertEqual(text.returncode, 0, text.stderr)
            base_raw = Path(candidate.target.canonical_path).read_bytes().decode("utf-8")
            base, new, delta = expected_change(base_raw, candidate.patch.new_text)
            sized = [line for line in text.stdout.splitlines() if line.startswith(SIZE_PREFIX)]
            self.assertEqual(sized, [size_line(base, new, delta)])
            as_json = run.cli("--json", "review", withheld.CID)
            self.assertEqual(as_json.returncode, 0, as_json.stderr)
            size = self.sized(json.loads(as_json.stdout))
            self.assertEqual((size["base"], size["replacement"], size["delta"]), (base, new, delta))

    # -- 9-11: the receipt, the listings, the record --------------------------

    def test_daily_records_the_size_and_the_window_for_every_candidate(self):
        entry = self.entry()
        review = self.sized(json.loads(self.review("--json").stdout))
        size = self.sized(entry)
        self.assertEqual(size["delta"], review["delta"])
        self.assertEqual(size["delta"], self.change()[2])
        window = size["window"]
        self.assertEqual(len(entry["addressed_failure_case_ids"]), 2)
        self.assertEqual(window["conversations_with_a_failure_attributed"], 1)
        self.assertEqual(window["basis"], WINDOW_BASIS_TEXT)
        self.assertIn("including rubric-prior-only attribution", window["basis"])
        self.assertIn("not a count of conversations that used or loaded it", window["basis"])
        with self.subTest("a gate-failed candidate's entry"):
            run = withheld.daily(k=0)
            (failed,) = [e for e in run.receipt.candidates if e["candidate_id"] == GATE_FAILED]
            self.assertFalse(failed["gate_passed"])
            record = run.state.read_candidate(GATE_FAILED)
            base_raw = Path(record.target.canonical_path).read_bytes().decode("utf-8")
            base, new, delta = expected_change(base_raw, record.patch.new_text)
            size = self.sized(failed)
            self.assertEqual((size["base"], size["replacement"], size["delta"]), (base, new, delta))
            self.assertEqual(
                size["window"]["conversations_with_a_failure_attributed"],
                len(pool(run.state.dir("failures"), record.target.skill_id)),
            )
        with self.subTest("a conversation attributed on the rubric prior alone"):
            receipt, state = self.prior_only_run()
            cases = [
                case
                for path in sorted(state.dir("failures").glob("*.json"))
                for case in json.loads(path.read_text(encoding="utf-8"))["cases"]
                if case["conversation_id"] == "conv-human-0009"
            ]
            self.assertTrue(cases)
            for case in cases:
                self.assertEqual(case["primary_skill_id"], "example-evidence-check")
                bases = {
                    a["basis"]
                    for a in case["attributions"]
                    if a["skill_id"] == "example-evidence-check"
                }
                self.assertEqual(bases, {"rubric-prior-only"})
            pool_ids = pool(state.dir("failures"), "example-evidence-check")
            self.assertEqual(pool_ids, {*withheld.FAILURE_CONVERSATIONS, "conv-human-0009"})
            (prior,) = [e for e in receipt.candidates if e["candidate_id"] == withheld.CID]
            window = self.sized(prior)["window"]
            self.assertEqual(window["conversations_with_a_failure_attributed"], len(pool_ids))
            self.assertEqual(window["conversations_with_a_failure_attributed"], 5)
            self.assertEqual(window["basis"], WINDOW_BASIS_TEXT)

    def prior_only_run(self):
        """``withheld.make_tree(k=0)`` plus 0009, a clone of 0001 naming another skill."""
        tree = withheld.make_tree(k=0)
        fixtures = tree / "examples" / "fixtures"
        source = (fixtures / "conv-human-0001.json").read_text(encoding="utf-8")
        clone = source.replace("conv-human-0001", "conv-human-0009").replace(
            "example-evidence-check", "unrelated-notes"
        )
        (fixtures / "conv-human-0009.json").write_text(clone, encoding="utf-8")
        cfg = tree / "config" / "example.config.toml"
        cfg.write_text(
            withheld._set(cfg.read_text(encoding="utf-8"), "max_conversations_per_run", 12),
            encoding="utf-8",
        )
        config = withheld.load_config(cfg)
        state = withheld.build_state(config)
        clock = withheld.FrozenClock(datetime(2026, 1, 2, tzinfo=timezone.utc))
        state._clock = clock
        receipt = withheld.run_daily(
            config,
            source=withheld.build_source(config, state),
            runner=withheld.Runner(),
            skill_store=withheld.build_skill_store(config),
            state=state,
            clock=clock,
            day="2026-01-02",
            commit=True,
        )
        return receipt, state

    def test_daily_and_candidates_text_are_as_before(self):
        with tempfile.TemporaryDirectory(prefix="asi-c249-text-") as name:
            cfg = doubling_env(Path(name).resolve() / "env")
            daily = cli(cfg, "daily", "--commit", "--day", DAY)
            self.assertEqual(daily.returncode, 0, daily.stderr)
            for line in daily.stdout.splitlines():
                self.assertFalse(line.startswith("size"), line)
            self.assertNotIn("est. ", daily.stdout)
            listing = cli(cfg, "--json", "candidates")
            self.assertEqual(listing.returncode, 0, listing.stderr)
            rows = json.loads(listing.stdout)["candidates"]
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(sorted(row), LISTING_ROW_KEYS)
                self.assertNotIn("size_change", row)
            text = cli(cfg, "candidates")
            self.assertEqual(text.returncode, 0, text.stderr)
            for row in rows:
                shown = [line for line in text.stdout.splitlines() if row["id"] in line]
                self.assertEqual(len(shown), 1, text.stdout)
                self.assertIn(
                    f"gate={'pass' if row['gate_passed'] else 'fail'} "
                    f"+{row['added_lines']}/-{row['removed_lines']}",
                    shown[0],
                )
            self.assertNotIn("est. ", text.stdout)

    def test_nothing_new_is_written_into_the_record(self):
        with tempfile.TemporaryDirectory(prefix="asi-c249-record-") as name:
            base = Path(name).resolve() / "env"
            cfg = doubling_env(base)
            daily = cli(cfg, "--json", "daily", "--commit", "--day", DAY)
            self.assertEqual(daily.returncode, 0, daily.stderr)
            records = sorted((base / "state" / "candidates").rglob("*.json"))
            self.assertTrue(records)
            for record in records:
                raw = json.loads(record.read_text(encoding="utf-8"))
                self.assertNotIn("size_change", keys_anywhere(raw), record.name)
                self.assertEqual(raw["validation"]["schema"], "asi.validation.v3", record.name)
            cid = json.loads(records[0].read_text(encoding="utf-8"))["id"]
            shown = cli(cfg, "--json", "review", cid)
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertIn("size_change", json.loads(shown.stdout))
            raw = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertNotIn("size_change", keys_anywhere(raw))
            self.assertEqual(raw["validation"]["schema"], "asi.validation.v3")


if __name__ == "__main__":
    unittest.main()
