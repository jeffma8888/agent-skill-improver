"""A record named by two ids fits one file name.

Scorecards and failures are named by the conversation and the rubric, and
quarantine and refusal records by the record and the stage, which for a score
carries the rubric id. Each id is turned into a keyed name of up to 162
characters, so the two together reached 331, and a file name holds 255.

Measured before (3.11 and 3.13): a 100-character conversation id scored against
a 100-character rubric id passed ``doctor``, then made a committed ``daily``
fall at its first scorecard with ``OSError: [Errno 63] File name too long``,
exit 70, after it had written the conversation, with no receipt.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.app.state import (
    FILE_NAME_MAX,
    StateStore,
    keyed_name,
    keyed_pair_name,
    safe_name,
)
from agent_skill_improver.clock import FrozenClock

from tests.support import FIXTURE_EXPORTS, build_env, run_cli
from tests.test_non_finite_numbers import _set_field

DAY = "2026-01-02"
LONG_CONVERSATION = "1700000000001-" + "9" * 86
LONG_RUBRIC = "r" * 100


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ARecordNamedByTwoIdsFitsOneFileNameTests(unittest.TestCase):
    def test_a_long_conversation_and_rubric_id_run_to_a_receipt(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = Path(build_env(base, seed="pair-names"))
            exports = base / "exports-long"
            shutil.copytree(FIXTURE_EXPORTS, exports)
            export = json.loads((exports / "compliant.json").read_text(encoding="utf-8"))
            export["thread"]["id"] = LONG_CONVERSATION
            (exports / "compliant.json").write_text(json.dumps(export), encoding="utf-8")
            rubric = (base / "rubrics" / "instruction-compliance.toml").read_text(encoding="utf-8")
            self.assertEqual(rubric.count('id = "instruction-compliance"'), 1)
            (base / "rubrics" / f"{LONG_RUBRIC}.toml").write_text(
                rubric.replace('id = "instruction-compliance"', f'id = "{LONG_RUBRIC}"'),
                encoding="utf-8",
            )
            _set_field(cfg, "rubrics", "enabled", json.dumps([LONG_RUBRIC]))
            _set_field(cfg, "source", "fixtures_dir", json.dumps(str(exports)))
            proc = run_cli(["--config", str(cfg), "daily", "--day", DAY, "--commit"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            state = base / "state"
            self.assertTrue(any((state / "receipts").glob("*.json")), "a receipt was written")
            names = [p.name for p in state.rglob("*")]
            self.assertLessEqual(max(len(n.encode()) for n in names), FILE_NAME_MAX)
            scorecard = [
                p.name
                for p in (state / "scorecards").glob("*.json")
                if _digest(LONG_CONVERSATION) in p.name
            ]
            self.assertEqual(len(scorecard), 1, "the long conversation was scored")
            self.assertIn(_digest(LONG_RUBRIC), scorecard[0])

    def test_a_pair_that_fits_keeps_the_name_it_had(self):
        for first, second in (
            ("conv-1", "instruction-compliance"),
            ("1700000000001-1", "score:instruction-compliance"),
            ("a" * 58, "b" * 58),
        ):
            with self.subTest(first=first[:12], second=second[:12]):
                self.assertEqual(
                    keyed_pair_name(first, second),
                    f"{keyed_name(first)}--{keyed_name(second)}.json",
                )
        # Written out, so a change to keyed_name itself, which would rename
        # records already on disk, fails here too.
        self.assertEqual(
            keyed_pair_name("conv-1", "instruction-compliance"),
            "conv-1--36524fd8f6747fc2712506d01fee0e18b48cd6261295e2f7e79106460a79899f"
            "--instruction-compliance--"
            "747bb6ed90a052edc1eecda4502bfc32a2e46acf32c17d8a973b5a91131199c3.json",
        )
        self.assertEqual(
            keyed_pair_name("a" * 96, "b"),
            "a" * 96
            + "--ee4caa5518a866f33e174d6e71ba3961a86ca00a7486b132e5a9f01bfaa1d794"
            + "--b--3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d.json",
        )

    def test_a_cut_name_fits_keeps_both_digests_and_stays_distinct(self):
        lengths = (1, 10, 57, 58, 59, 96, 97, 150, 400)
        seen: dict[str, tuple[str, str]] = {}
        for i in lengths:
            for j in lengths:
                first, second = "a" * i, "b" * j
                name = keyed_pair_name(first, second)
                with self.subTest(i=i, j=j):
                    self.assertLessEqual(len(name.encode()), FILE_NAME_MAX)
                    self.assertIn(_digest(first), name)
                    self.assertIn(_digest(second), name)
                    self.assertTrue(name.startswith(safe_name(first)[:58]), name[:20])
                    self.assertTrue(name.endswith(".json"))
                    self.assertNotIn(name, seen, f"{seen.get(name)} has the same name")
                seen[name] = (first, second)
        # Two pairs that differ only past where the stems are cut, and a pair
        # and its swap, are still apart.
        tail = "x" * 120
        self.assertNotEqual(keyed_pair_name(tail + "1", tail), keyed_pair_name(tail + "2", tail))
        self.assertNotEqual(keyed_pair_name(tail, tail + "1"), keyed_pair_name(tail, tail + "2"))
        self.assertNotEqual(
            keyed_pair_name("c" * 99, "d" * 99), keyed_pair_name("d" * 99, "c" * 99)
        )
        # A cut gives what one stem leaves to the other, so a short id is kept
        # whole and the name uses all 255 bytes: 30 + 86 = 116 stem characters.
        for first, second, kept_first, kept_second in (
            ("s" * 30, "t" * 200, "s" * 30, "t" * 86),
            ("t" * 200, "s" * 30, "t" * 86, "s" * 30),
        ):
            with self.subTest(first=len(first), second=len(second)):
                self.assertEqual(
                    keyed_pair_name(first, second),
                    f"{kept_first}--{_digest(first)}--{kept_second}--{_digest(second)}.json",
                )
                self.assertEqual(len(keyed_pair_name(first, second)), FILE_NAME_MAX)

    def test_every_writer_named_by_a_pair_takes_a_long_pair(self):
        with TemporaryDirectory() as tmp:
            clock = FrozenClock(
                datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
            )
            state = StateStore(Path(tmp).resolve() / "state", clock=clock)
            state.ensure()
            stage = f"score:{LONG_RUBRIC}"
            for kind, write, second in (
                (
                    "scorecards",
                    lambda: state.write_scorecard(LONG_CONVERSATION, LONG_RUBRIC, {"k": 1}),
                    LONG_RUBRIC,
                ),
                (
                    "failures",
                    lambda: state.write_failures(LONG_CONVERSATION, LONG_RUBRIC, []),
                    LONG_RUBRIC,
                ),
                (
                    "quarantine",
                    lambda: state.quarantine(LONG_CONVERSATION, stage=stage, reason="probe"),
                    stage,
                ),
                (
                    "refusals",
                    lambda: state.refuse(LONG_CONVERSATION, stage=stage, reason="probe"),
                    stage,
                ),
            ):
                with self.subTest(kind=kind):
                    path = write()
                    self.assertEqual(path.parent.name, kind)
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.name, keyed_pair_name(LONG_CONVERSATION, second))
                    self.assertLessEqual(len(path.name.encode()), FILE_NAME_MAX)
                    self.assertIn(_digest(LONG_CONVERSATION), path.name)
                    self.assertIn(_digest(second), path.name)
            record = json.loads(
                (state.dir("quarantine") / keyed_pair_name(LONG_CONVERSATION, stage)).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual((record["record_id"], record["stage"]), (LONG_CONVERSATION, stage))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
