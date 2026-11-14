"""A decline frees its skill's slot and records who said no.

A declined candidate is closed: it holds neither the queue cap nor its skill's
slot, so the next ``daily`` proposes for that skill again. The record changes
only in its status, a note and the ``decline`` object; the decline receipt holds
the decision, the gate and the recomputed patch hash, mode 0600. The operator is
a label the output says asi does not check: an omitted or empty ``--operator``
takes ``$USER``, and a label empty after trimming is shown as "unspecified". A
reason is trimmed, then masked to its fixed point before it is stored, so the
record, the receipt, the output and a re-read hold one string. A reason never
reaches a model prompt or the rejection memory. ``daily`` counts a declined
record by its status and the shape of its decision, not its integrity check, as
it counts every closed record: one edited after its decline still frees its
skill's slot, while review, apply, rollback and decline refuse it.
Measured before (p01): with no decline the next day's run counted the skill as
holding an open candidate (1) and proposed nothing (0), and a queue cap of 1
stood the run down.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import agent_skill_improver.domain.candidate as candidate_module
from agent_skill_improver import cli


def _absent_before_c247b(name: str) -> Any:
    """What a name c247b adds stands for on a tree without it: a failure, by assertion."""

    def absent(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{name} does not exist: this tree has no decline")

    return absent


try:
    from agent_skill_improver.domain.candidate import settled_mask
except ImportError:  # a tree without c247b: its tests fail by assertion
    settled_mask = _absent_before_c247b("settled_mask")
from agent_skill_improver.domain.integrity import verify_full_integrity
from agent_skill_improver.errors import CandidateTampered, SafetyRefusal

from tests.test_a_declined_candidate_cannot_be_applied_or_rolled_back import DeclineTreeCase
from tests.test_end_to_end_cli import margin_lines_outside_their_section

NEXT_DAY = "2026-02-02"
ISO = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
#: Words that would claim asi checked who declined.
CLAIMS = ("authenticated", "verified operator", "human-approved", "by the owner")
#: Synthetic secret shapes, as p04 builds them (none is a real credential).
SHAPES = {
    "aws_access_key": ("AK" + "IA" + "A" * 16, 508),
    "github_token": ("gh" + "p_" + "a" * 36, 483),
    "bearer": ("Bearer " + "x" * 24, 492),
}
#: A URL parameter whose value is a second one: masked once, it reads back with one mask fewer.
MERGING = "see ?" + "token=" + "\\u0026" + "secret=" + "x" * 20


class ADeclineFreesItsSkillsSlotTests(DeclineTreeCase):
    def daily(self, day: str = NEXT_DAY) -> dict[str, Any]:
        rc, out, err = self.cli("--json", "daily", "--commit", "--day", day)
        self.assertEqual(rc, 0, err)
        return json.loads(out)

    def test_the_next_daily_proposes_for_a_skill_whose_candidate_was_declined(self) -> None:
        held = self.daily()["counts"]
        self.assertEqual(
            (held["skills_with_open_candidate"], held["candidates_proposed"]), (1, 0), held
        )
        self.setUp()
        self.assertEqual(self.decline()[0], 0)
        freed = self.daily()["counts"]
        self.assertEqual(freed["skills_with_open_candidate"], 0, freed)
        self.assertEqual(freed["candidates_proposed"], 1, freed)

    def test_a_declined_record_edited_after_its_decline_is_refused_yet_still_frees_its_skills_slot(
        self,
    ) -> None:
        # Pinned as documented ('Declining a candidate', and the c247b row's "Not
        # covered"): the queue, candidates and doctor count a declined record by
        # its status and its decision's shape, as every closed record, so a later
        # cycle that checks its integrity there changes this test on purpose.
        self.assertEqual(self.decline()[0], 0)
        self.tamper_patch()
        with self.assertRaises(CandidateTampered):
            verify_full_integrity(
                self.state.read_candidate(self.cid), rubrics=None, action="decline"
            )
        before = self.tree()
        refusals = {
            "review": (("review", self.cid), "status is 'declined', not 'validated'"),
            "apply": (tuple(self.apply_argv[2:]), "has status 'declined'"),
            "rollback": (tuple(self.rollback_argv[2:]), f"no backup manifest found for {self.cid}"),
            "decline": (("decline", self.cid, "--reason", "too broad"), "CandidateTampered"),
        }
        for name, (argv, said) in refusals.items():
            with self.subTest(name):
                rc, out, err = self.cli(*argv)
                self.assertEqual(rc, cli.RC_REFUSED, out + err)
                self.assertIn(said, err)
                self.assertNotIn("Traceback", err)
                self.assertEqual(self.tree(), before)
        self.assertEqual(self.skill.read_bytes(), self.original)
        rc, out, err = self.cli("--json", "candidates", "--all")
        self.assertEqual(rc, 0, err)
        listed = {c["id"]: c["status"] for c in json.loads(out)["candidates"]}
        self.assertEqual(listed[self.cid], "declined")
        freed = self.daily()["counts"]
        self.assertEqual(freed["skills_with_open_candidate"], 0, freed)
        self.assertEqual(freed["candidates_proposed"], 1, freed)

    def test_an_omitted_or_empty_operator_takes_user_and_only_an_empty_label_is_unspecified(
        self,
    ) -> None:
        cases: dict[str, tuple[tuple[str, ...], str, str]] = {
            "omitted": ((), "alice", "alice"),
            "empty": (("--operator", ""), "alice", "alice"),
            "given, trimmed": (("--operator", "  bob\t"), "alice", "bob"),
            "omitted, USER empty": ((), "", ""),
            "blanks only": (("--operator", "   "), "alice", ""),
        }
        for name, (extra, user, stored) in cases.items():
            with self.subTest(name):
                self.setUp()
                rc, out, err = self.decline("too broad", *extra, env={"USER": user})
                self.assertEqual(rc, 0, err)
                record = self.raw_record()
                self.assertEqual(record["decline"]["operator"], stored)
                shown = stored or "unspecified"
                self.assertTrue(record["notes"][-1].endswith(f" by operator:{shown}"), record)
                self.assertIn(
                    f"  operator        {shown} (a label; asi does not check who ran this)", out
                )
        with self.subTest("a reason is trimmed before its rules are applied"):
            for given in ("\nwhy", "why\t", " " * 10_000 + "why"):
                self.setUp()
                rc, out, err = self.decline(given)
                self.assertEqual(rc, 0, err)
                self.assertEqual(self.raw_record()["decline"]["reason"], "why")

    def test_a_decline_changes_only_the_status_the_notes_and_the_decline_of_the_record(
        self,
    ) -> None:
        before = self.raw_record()
        rc, out, err = self.decline("too broad", "--operator", "alice")
        self.assertEqual(rc, 0, err)
        after = self.raw_record()
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        self.assertEqual(changed, {"status", "notes", "decline"})
        # The whole validation block, an ``unseen`` block included where a run wrote one.
        self.assertEqual(json.dumps(after["validation"]), json.dumps(before["validation"]))
        self.assertEqual(after["status"], "declined")
        self.assertEqual(after["notes"][: len(before["notes"])], before["notes"])
        (note,) = after["notes"][len(before["notes"]) :]
        self.assertRegex(note, rf"\Adeclined {ISO} by operator:alice\Z")
        self.assertEqual(
            after["decline"],
            {"declined_at": note.split()[1], "operator": "alice", "reason": "too broad"},
        )

    def test_the_receipt_holds_the_decision_the_gate_and_the_recomputed_patch_hash(self) -> None:
        rc, out, err = self.decline("too broad", "--operator", "alice")
        self.assertEqual(rc, 0, err)
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        record = self.raw_record()
        gate = self.read_before.validation["gate"]
        self.assertEqual(
            receipt,
            {
                "schema": "asi.decline_receipt.v1",
                "candidate_id": self.cid,
                **record["decline"],
                "status_before": "validated",
                "skill_id": self.read_before.target.skill_id,
                "canonical_path": self.read_before.target.canonical_path,
                "base_sha256": self.read_before.target.base_sha256,
                "patch_sha256": self.read_before.recomputed_patch_sha256,
                "validation_schema": self.read_before.validation.get("schema"),
                "gate": {k: gate.get(k) for k in ("passed", "reasons", "metrics")},
                "improver_command_id": self.read_before.improver_command_id,
                "finishes_an_earlier_run": False,
            },
        )
        self.assertEqual(stat.S_IMODE(self.receipt_path.stat().st_mode), 0o600)

    def test_the_output_calls_the_operator_a_label_and_claims_no_check(self) -> None:
        rc, out, err = self.decline("too broad", "--operator", "alice")
        self.assertEqual(rc, 0, err)
        self.assertIn(f"declined {self.cid} (it was validated)", out)
        self.assertIn("  operator        alice (a label; asi does not check who ran this)", out)
        self.setUp()
        rc, as_json, err = self.decline("too broad", "--json", "--operator", "alice")
        self.assertEqual(rc, 0, err)
        payload = json.loads(as_json)
        self.assertEqual(payload["operator"], "alice")
        self.assertEqual(payload["outcome"], "declined")
        for said in (out, as_json):
            for claim in CLAIMS:
                self.assertNotIn(claim, said.lower())

    def test_a_500_character_reason_ending_in_a_secret_shape_is_one_string_everywhere(
        self,
    ) -> None:
        cases = {
            label: ("x" * (500 - len(token) - 1) + " " + token, token, stored)
            for label, (token, stored) in SHAPES.items()
        }
        cases["url_parameter_merge"] = (MERGING, "x" * 20, 36)
        for label, (reason, token, stored_len) in cases.items():
            with self.subTest(label):
                self.setUp()
                rc, out, err = self.decline(reason)
                self.assertEqual(rc, 0, err)
                stored = self.raw_record()["decline"]["reason"]
                self.assertEqual(len(stored), stored_len)
                self.assertEqual(stored, settled_mask(reason)[0])
                receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
                (line,) = [ln for ln in out.splitlines() if ln.startswith("  reason  ")]
                self.assertEqual(line, "  reason          " + stored)
                self.setUp()
                rc, as_json, err = self.decline(reason, "--json")
                self.assertEqual(rc, 0, err)
                read = self.state.read_candidate(self.cid)
                for seen in (
                    self.raw_record()["decline"]["reason"],
                    receipt["reason"],
                    json.loads(as_json)["reason"],
                    read.decline.reason,
                ):
                    self.assertEqual(seen, stored)
                for text in (self.record_path.read_text(encoding="utf-8"), out, as_json):
                    self.assertNotIn(token, text)
                self.assertNotIn(token, self.receipt_path.read_text(encoding="utf-8"))
                on_disk = self.record_path.read_bytes()
                self.state.write_candidate(read)
                self.assertEqual(self.record_path.read_bytes(), on_disk)

    def test_a_declined_reason_never_reaches_a_prompt_or_the_rejection_memory(self) -> None:
        seen: dict[str, tuple[list[str], bytes | None]] = {}
        for marker in ("quokkaword", "pangolinword"):
            self.setUp()
            memory = self.state.rejection_memory_path()
            had = memory.read_bytes() if memory.exists() else None
            rc, out, err = self.decline(f"not this, {marker}")
            self.assertEqual(rc, 0, err)
            self.assertEqual(memory.read_bytes() if memory.exists() else None, had)
            log = self.base / "role-calls.jsonl"
            body = self.config_path.read_text(encoding="utf-8")
            heading = "[model.improver]\n"
            self.assertEqual(body.count(heading), 1)
            line = "env = { ASI_TEST_ROLE_CALL_LOG = " + json.dumps(str(log)) + " }\n"
            self.config_path.write_text(body.replace(heading, heading + line), encoding="utf-8")
            self.daily()
            calls = [json.loads(r) for r in log.read_text(encoding="utf-8").splitlines()]
            prompts = [c["user_sha256"] for c in calls if c["role"] == "improver"]
            self.assertTrue(prompts, calls)
            texts = b"".join(p.read_bytes() for p in self.base.rglob("*") if p.is_file())
            self.assertEqual(texts.count(marker.encode()), 2)  # its record and its receipt
            seen[marker] = (prompts, memory.read_bytes() if memory.exists() else None)
        first, second = seen.values()
        self.assertEqual(first, second)

    def test_review_offers_the_decline_command_indented_under_the_apply_command(self) -> None:
        rc, out, err = self.cli("review", self.cid)
        self.assertEqual(rc, 0, err)
        lines = out.splitlines()
        offer = f"  or decline it on the record:  asi decline {self.cid} --reason -"
        self.assertIn(offer, lines)
        at = lines.index(offer)
        self.assertIn("--approve", lines[at - 1])
        self.assertTrue(lines[at - 1].lstrip().startswith("asi"), lines[at - 1])
        self.assertEqual(margin_lines_outside_their_section(lines), [])


class ADeclinedCandidateNoLongerHoldsTheQueueTests(DeclineTreeCase):
    env_options = {"max_open_candidates": 1}

    def test_a_declined_candidate_no_longer_holds_the_queue_cap(self) -> None:
        rc, out, err = self.cli("--json", "daily", "--commit", "--day", NEXT_DAY)
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out)["status"], "stood-down")
        self.setUp()
        self.assertEqual(self.decline()[0], 0)
        rc, out, err = self.cli("--json", "daily", "--commit", "--day", NEXT_DAY)
        self.assertEqual(rc, 0, err)
        ran = json.loads(out)
        self.assertNotEqual(ran["status"], "stood-down")
        self.assertEqual(ran["counts"]["candidates_proposed"], 1)


class _Redact:
    """A counting stand-in for ``redact`` that returns the texts it is given, in turn."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls = 0

    def __call__(self, text: str) -> Any:
        out = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        return mock.Mock(text=out, total=1)


class TheMaskingLoopTests(unittest.TestCase):
    def test_the_masking_loop_refuses_after_four_calls_when_the_mask_never_settles(self) -> None:
        self.assertTrue(
            hasattr(candidate_module, "redact"), "no masking loop in domain/candidate.py"
        )
        flip = _Redact(["one", "two"])
        with mock.patch.object(candidate_module, "redact", flip):
            with self.assertRaisesRegex(SafetyRefusal, "does not settle"):
                settled_mask("x")
        self.assertEqual(flip.calls, 4)
        late = _Redact(["first", "second", "third", "third"])
        with mock.patch.object(candidate_module, "redact", late):
            self.assertEqual(settled_mask("x"), ("third", 1))
        self.assertEqual(late.calls, 4)
        self.assertIs(cli.main, cli.main)


if __name__ == "__main__":
    unittest.main()
