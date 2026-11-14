"""A declined record is written first, holds its decision and keeps its first reason.

Decline writes the record (``declined``, its ``decline`` object and a note) and
then its receipt, never over a leaf, both under the run lock. A stop between the
two leaves a closed record that apply and rollback refuse; a rerun with the same
reason (a stored one over 500 given back included) writes the missing receipt
from the record, a rerun with another is refused and the first kept, and a rerun
with both in place writes nothing. A declined record without a well-shaped
decision, or a record that is not declined but holds one, is unreadable and
untrusted. Whatever stands at the receipt path that is not a regular file, or
goes missing after its ``lstat``, is refused naming the path, never read as no
receipt and never blocked on. A receipt that is not one this version writes (its
schema, a field missing, one it never writes, one of another type) or that
disagrees with the record is refused the same way, quoting none of it, and what
a decline prints is built from the record, never from a receipt. Measured
before (3.11 and 3.13): a rerun beside a receipt without ``skill_id`` was a
KeyError traceback (exit 70), and one with another schema, a 100,000-character
skill id or a planted field was printed as it stood. The receipt and the record
are compared as JSON, by type and value, nested fields included: measured
before FIX 2 (3.11 and 3.13), a receipt whose gate held ``1`` for ``true`` or
``false`` for ``0`` agreed by Python's ``==`` and the rerun exited 0.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import agent_skill_improver.app.promote as promote
from agent_skill_improver import cli
from agent_skill_improver.app.lock import RunLock
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.domain import redaction


def _absent_before_c247b(name: str) -> Any:
    """What a name c247b adds stands for on a tree without it: a failure, by assertion."""

    def absent(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{name} does not exist: this tree has no decline")

    return absent


try:
    from agent_skill_improver.domain.candidate import (
        MAX_STORED_DECLINE_OPERATOR_CHARS,
        MAX_STORED_DECLINE_REASON_CHARS,
        Decline,
        settled_mask,
    )
except ImportError:  # a tree without c247b: its tests fail by assertion
    MAX_STORED_DECLINE_OPERATOR_CHARS = MAX_STORED_DECLINE_REASON_CHARS = -1
    Decline = _absent_before_c247b("Decline")
    settled_mask = _absent_before_c247b("settled_mask")
from agent_skill_improver.domain.integrity import verify_full_integrity
from agent_skill_improver.errors import CandidateTampered, SafetyRefusal, StateUnreadable

from tests.test_a_declined_candidate_cannot_be_applied_or_rolled_back import (
    HAND_DECISION,
    DeclineTreeCase,
    clock,
)

AWS = "AK" + "IA" + "A" * 16
SRC = Path(__file__).resolve().parents[1] / "src"
REAL_RECEIPT_WRITE = StateStore.write_action_receipt
#: Every field a decline prints, text and ``--json``: spelled out, not imported.
PRINTED = {
    "candidate_id",
    "status",
    "outcome",
    "declined_at",
    "operator",
    "reason",
    "skill_id",
    "receipt_path",
    "masked",
}
BOUND = ("candidate_id", "declined_at", "operator", "reason", "status_before")
#: What the fixture's gate holds that a refusal must not show: field names and a
#: case key, each long enough not to occur in a temporary path by chance.
GATE_CONTENT = (
    "holdout_degradations",
    "judge_independent",
    "fixed_case_keys",
    "judge_candidate_preference_ratio",
    "correction-not-repeated",
)


def _stop_at_the_decline_receipt(self: StateStore, action: str, *args: Any, **kw: Any) -> Path:
    if action == "decline":
        raise cli.Stopped(signal.SIGTERM)
    return REAL_RECEIPT_WRITE(self, action, *args, **kw)


def _is(given: Any, path: Path) -> bool:
    return isinstance(given, (str, os.PathLike)) and os.fspath(given) == str(path)


STOP_RECEIPT = mock.patch.object(StateStore, "write_action_receipt", _stop_at_the_decline_receipt)


class ADeclinedRecordIsWrittenFirstTests(DeclineTreeCase):
    def stopped_after_the_record(self, reason: str = "too broad", *extra: str) -> bytes:
        self._main([*self.cfg, "decline", self.cid, "--reason", reason, *extra], STOP_RECEIPT)
        self.assertEqual(self.raw_record()["status"], "declined")
        self.assertFalse(os.path.lexists(self.receipt_path))
        return self.record_path.read_bytes()

    def refused(self, outcome: tuple[int, str, str], before: Any) -> str:
        rc, out, err = outcome
        self.assertEqual(rc, cli.RC_REFUSED, out + err)
        self.assertEqual(self.tree(), before)
        return err

    def rerun_json(self, reason: str, *extra: str, **kw: Any) -> tuple[int, dict[str, Any], str]:
        rc, out, err = self.decline(reason, "--json", *extra, **kw)
        return rc, (json.loads(out) if rc == 0 else {}), err

    def test_a_stop_between_record_and_receipt_leaves_a_decline_that_apply_and_rollback_refuse(
        self,
    ) -> None:
        self.stopped_after_the_record()
        self.assertEqual(set(self.raw_record()["decline"]), {"declined_at", "operator", "reason"})
        before = self.tree()
        self.refused(self._main(self.apply_argv), before)
        self.refused(self._main(self.rollback_argv), before)

    def test_a_rerun_with_the_same_reason_writes_the_missing_receipt_from_the_record(self) -> None:
        left = self.stopped_after_the_record("too broad", "--operator", "alice")
        later = mock.patch.object(cli, "SystemClock", lambda: clock(48))
        argv = [*self.cfg, "--json", "decline", self.cid, "--reason", "too broad"]
        rc, out, err = self._main([*argv, "--operator", "bob"], later)
        self.assertEqual(rc, 0, err)
        payload = json.loads(out)
        self.assertEqual(payload["outcome"], "receipt-written")
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        decision = json.loads(left)["decline"]
        self.assertEqual({k: receipt[k] for k in decision}, decision)
        self.assertEqual(decision["operator"], "alice")
        self.assertIs(receipt["finishes_an_earlier_run"], True)
        self.assertEqual(self.record_path.read_bytes(), left)

    def test_a_rerun_with_another_reason_is_refused_and_the_first_is_kept(self) -> None:
        for state in ("receipt missing", "receipt present"):
            with self.subTest(state):
                self.setUp()
                if state == "receipt missing":
                    self.stopped_after_the_record("first words", "--operator", "alice")
                else:
                    self.assertEqual(self.decline("first words", "--operator", "alice")[0], 0)
                decision = self.raw_record()["decline"]
                before = self.tree()
                err = self.refused(self.decline("second words"), before)
                self.assertIn(decision["declined_at"], err)
                self.assertIn("operator:alice", err)
                self.assertIn("another reason", err)
                self.assertNotIn("first words", err)
                self.assertNotIn("second words", err)

    def test_a_rerun_when_record_and_receipt_both_exist_is_a_no_op_that_says_so(self) -> None:
        self.assertEqual(self.decline()[0], 0)
        stamps = [
            (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.record_path, self.receipt_path)
        ]
        rc, out, err = self.decline()
        self.assertEqual(rc, 0, err)
        self.assertIn("nothing written", out)
        self.assertEqual(
            [(p.read_bytes(), p.stat().st_mtime_ns) for p in (self.record_path, self.receipt_path)],
            stamps,
        )

    def test_the_decline_receipt_write_never_replaces_a_file(self) -> None:
        try:
            where = self.state.action_receipt_path("decline", self.cid)
        except SafetyRefusal as refused:
            self.fail(f"the state store has no decline receipt: {refused}")
        self.assertEqual(where, self.receipt_path)
        payload = {"schema": "asi.decline_receipt.v1", "candidate_id": self.cid}
        path = self.receipt_path
        target = self.base / "a-target.json"
        target.write_bytes(b'{"kept": true}')
        leaves = {
            "a regular file": lambda: path.write_bytes(b'{"there": 1}'),
            "a directory": lambda: path.mkdir(),
            "a link to a file": lambda: path.symlink_to(target),
            "a dangling link": lambda: path.symlink_to(self.base / "nothing-here"),
            "a FIFO": lambda: os.mkfifo(path),
        }
        for name, make in leaves.items():
            with self.subTest(name):
                self.setUp()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'{"kept": true}')
                path.parent.mkdir(parents=True, exist_ok=True)
                make()
                before = self.tree()
                with self.assertRaises(SafetyRefusal):
                    self.state.write_action_receipt("decline", self.cid, payload, replace=False)
                self.assertEqual(self.tree(), before)
                self.assertEqual(target.read_bytes(), b'{"kept": true}')
        self.setUp()
        for action in ("apply", "rollback"):
            with self.subTest(replaced=action):
                first = self.state.write_action_receipt(action, self.cid, {"n": 1})
                again = self.state.write_action_receipt(action, self.cid, {"n": 2})
                self.assertEqual(first, again)
                self.assertEqual(json.loads(again.read_text(encoding="utf-8")), {"n": 2})

    def test_a_decline_receipt_beside_a_validated_record_is_refused_and_left_in_place(
        self,
    ) -> None:
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_path.write_text(json.dumps({"stray": True}), encoding="utf-8")
        before = self.tree()
        err = self.refused(self.decline(), before)
        self.assertIn("StateUnreadable", err)
        self.assertIn("exists but the record says validated", err)

    def test_a_receipt_that_disagrees_with_its_declined_record_is_refused_naming_both_paths(
        self,
    ) -> None:
        self.assertEqual(self.decline()[0], 0)
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        receipt["operator"] = "someone-else"
        self.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        before = self.tree()
        err = self.refused(self.decline(), before)
        self.assertIn("StateUnreadable", err)
        self.assertIn(str(self.record_path), err)
        self.assertIn(str(self.receipt_path), err)

    # -- a receipt that stands is checked before it is relied on --------------

    def declined_with_its_receipt(self, change: Callable[[dict[str, Any]], dict[str, Any]]) -> Any:
        """A decline done, then its receipt rewritten as ``change`` returns it; the tree then."""
        self.assertEqual(self.decline("too broad", "--operator", "alice")[0], 0)
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.receipt_path.write_text(json.dumps(change(receipt)), encoding="utf-8")
        return self.tree()

    def rerun_refused_unreadable(self, before: Any, *never: str) -> list[str]:
        """A rerun, in text and in ``--json``: exit 3 and ``StateUnreadable`` naming the
        receipt, no traceback, nothing on standard output, nothing written and none
        of ``never`` shown. The two error outputs, text first."""
        said: list[str] = []
        for extra in ((), ("--json",)):
            rc, out, err = self.decline("too broad", *extra, "--operator", "alice")
            self.assertEqual(rc, cli.RC_REFUSED, (extra, out[:300], err[:300]))
            self.assertEqual(out, "", extra)
            self.assertIn("StateUnreadable", err)
            self.assertIn(str(self.receipt_path), err)
            self.assertNotIn("Traceback", err)
            self.assertLess(len(err), 2_000, extra)
            self.assertEqual(self.tree(), before)
            for text in never:
                self.assertNotIn(text, out + err)
            said.append(err)
        return said

    def test_a_decline_receipt_missing_a_field_is_refused_unreadable_without_a_traceback(
        self,
    ) -> None:
        cases: dict[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str]] = {
            "skill_id missing": (
                lambda r: {k: v for k, v in r.items() if k != "skill_id"},
                "skill_id missing",
            ),
            "the schema and the bound five only": (
                lambda r: {k: r[k] for k in ("schema", *BOUND)},
                "skill_id, canonical_path, base_sha256",
            ),
            "the schema missing": (
                lambda r: {k: v for k, v in r.items() if k != "schema"},
                "schema",
            ),
        }
        for name, (change, named) in cases.items():
            with self.subTest(name):
                self.setUp()
                before = self.declined_with_its_receipt(change)
                for err in self.rerun_refused_unreadable(before):
                    self.assertIn(named, err)
                    self.assertIn("none of it is shown or used", err)

    def test_a_decline_receipt_with_another_schema_is_refused_and_not_echoed(self) -> None:
        for schema in ("wrong-schema", "asi.decline_receipt.v2", "forged " + AWS, 1, None):
            with self.subTest(schema=schema):
                self.setUp()
                before = self.declined_with_its_receipt(lambda r: {**r, "schema": schema})
                never = (schema,) if isinstance(schema, str) else ()
                for err in self.rerun_refused_unreadable(before, *never):
                    self.assertIn("its schema is not asi.decline_receipt.v1", err)

    def test_a_decline_receipt_field_of_the_wrong_type_is_refused(self) -> None:
        cases: dict[str, tuple[str, Any]] = {
            "finishes_an_earlier_run a string": ("finishes_an_earlier_run", "yes"),
            "finishes_an_earlier_run a number": ("finishes_an_earlier_run", 0),
            "the gate a list": ("gate", ["passed"]),
            "the skill id a number": ("skill_id", 7),
            "the operator null": ("operator", None),
            "the improver command id a number": ("improver_command_id", 1),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name):
                self.setUp()
                before = self.declined_with_its_receipt(lambda r: {**r, field: value})
                for err in self.rerun_refused_unreadable(before):
                    self.assertIn(f"({field} of the wrong type)", err)

    def test_an_oversized_decline_receipt_field_is_refused_without_printing_it(self) -> None:
        for field in ("skill_id", "reason", "canonical_path"):
            with self.subTest(field):
                self.setUp()
                before = self.declined_with_its_receipt(lambda r: {**r, field: "z" * 100_000})
                for err in self.rerun_refused_unreadable(before, "z" * 40):
                    self.assertIn(f"({field} differs)", err)
                    self.assertIn(str(self.record_path), err)

    def test_an_extra_receipt_field_holding_a_credential_shape_is_refused_and_absent_from_output(
        self,
    ) -> None:
        cases = {
            "a planted field": {"planted_field": "key " + AWS},
            "a credential as the field's name": {AWS: "x"},
            "two planted fields": {"planted_field": AWS, "planted_too": {"deep": AWS}},
        }
        for name, extra in cases.items():
            with self.subTest(name):
                self.setUp()
                before = self.declined_with_its_receipt(lambda r: {**r, **extra})
                for err in self.rerun_refused_unreadable(before, AWS, "planted"):
                    self.assertIn(f"({len(extra)} field(s) this version does not write)", err)

    def test_a_decline_receipt_naming_another_candidate_is_refused_naming_both_paths(
        self,
    ) -> None:
        other = "cand-planted-" + "0" * 12
        cases: dict[str, tuple[str, Any]] = {
            "another candidate": ("candidate_id", other),
            "another patch hash": ("patch_sha256", "0" * 64),
            "another gate": ("gate", {"passed": False, "reasons": [], "metrics": {}}),
            "status_before declined": ("status_before", "declined"),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name):
                self.setUp()
                before = self.declined_with_its_receipt(lambda r: {**r, field: value})
                for err in self.rerun_refused_unreadable(before, other):
                    self.assertIn(f"({field} differs)", err)
                    self.assertIn(str(self.record_path), err)
                    self.assertIn("does not agree with the record", err)

    # -- compared as JSON: Python's == takes true for 1, 0 for false, 1 for 1.0 --

    def a_gate_edit_is_refused(
        self,
        path: tuple[str, ...],
        was: type,
        make: Callable[[Any], Any],
        *,
        python_equal: bool,
    ) -> None:
        """The receipt's ``gate`` value at ``path`` (of type ``was``) replaced by
        ``make`` of it, Python's ``==`` saying ``python_equal`` of the two; then a
        rerun refused as differing from the record, in text and ``--json``."""

        def change(receipt: dict[str, Any]) -> dict[str, Any]:
            *parents, leaf = ("gate", *path)
            node = receipt
            for key in parents:
                node = node[key]
            self.assertIs(type(node[leaf]), was, path)
            new = make(node[leaf])
            self.assertIs(node[leaf] == new, python_equal, path)
            node[leaf] = new
            return receipt

        before = self.declined_with_its_receipt(change)
        for err in self.rerun_refused_unreadable(before, *GATE_CONTENT, AWS, "planted"):
            self.assertIn("(gate differs)", err)
            self.assertIn(str(self.record_path), err)
            self.assertIn("does not agree with the record", err)

    def test_a_receipt_gate_with_a_number_for_a_boolean_or_a_boolean_for_a_number_is_refused(
        self,
    ) -> None:
        cases: dict[str, tuple[tuple[str, ...], type, Any]] = {
            "gate.passed true to 1": (("passed",), bool, 1),
            "gate.passed true to 1.0": (("passed",), bool, 1.0),
            "a nested metric 0 to false": (("metrics", "holdout_degradations"), int, False),
            "a nested metric true to 1": (("metrics", "judge_independent"), bool, 1),
        }
        for name, (path, was, value) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.a_gate_edit_is_refused(path, was, lambda _old: value, python_equal=True)

    def test_a_receipt_gate_with_a_float_for_an_int_or_an_int_for_a_float_is_refused(
        self,
    ) -> None:
        cases: dict[str, tuple[tuple[str, ...], type, Callable[[Any], Any]]] = {
            "an int metric to the same float": (("metrics", "holdout_cases"), int, float),
            "a metric 0 to 0.0": (("metrics", "holdout_degradations"), int, float),
            "a float metric to the same int": (
                ("metrics", "judge_candidate_preference_ratio"),
                float,
                int,
            ),
        }
        for name, (path, was, make) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.a_gate_edit_is_refused(path, was, make, python_equal=True)

    def test_a_receipt_gate_with_a_list_reordered_or_a_key_added_or_removed_is_refused(
        self,
    ) -> None:
        def reordered(keys: list[str]) -> list[str]:
            self.assertGreaterEqual(len(set(keys)), 2, "the fixture's list holds two keys")
            return list(reversed(keys))

        cases: dict[str, tuple[tuple[str, ...], type, Callable[[Any], Any]]] = {
            "a nested list reordered": (("metrics", "fixed_case_keys"), list, reordered),
            "a nested metric added": (
                ("metrics",),
                dict,
                lambda metrics: {**metrics, "planted_metric": AWS},
            ),
            "a nested metric removed": (
                ("metrics",),
                dict,
                lambda metrics: {k: v for k, v in metrics.items() if k != "holdout_cases"},
            ),
            "a gate key added": ((), dict, lambda gate: {**gate, "planted": AWS}),
            "a gate key removed": (
                (),
                dict,
                lambda gate: {k: v for k, v in gate.items() if k != "reasons"},
            ),
        }
        for name, (path, was, make) in cases.items():
            with self.subTest(name):
                self.setUp()
                self.a_gate_edit_is_refused(path, was, make, python_equal=False)

    def test_a_decline_prints_only_what_the_record_holds_and_fixed_strings(self) -> None:
        rc, payload, err = self.rerun_json("too broad", "--operator", "alice")
        self.assertEqual(rc, 0, err)
        self.assertEqual(set(payload), PRINTED)
        self.assertEqual(payload["outcome"], "declined")
        record = self.raw_record()
        first = dict(payload)
        rc, again, err = self.rerun_json("too broad", "--operator", "bob")
        self.assertEqual(rc, 0, err)
        self.assertEqual(set(again), PRINTED)
        self.assertEqual(
            again,
            {
                "candidate_id": self.cid,
                "status": "declined",
                "outcome": "already-declined",
                **record["decline"],
                "skill_id": self.read_before.target.skill_id,
                "receipt_path": str(self.receipt_path),
                "masked": 0,
            },
        )
        self.assertEqual(
            {k: v for k, v in first.items() if k != "outcome"},
            {k: v for k, v in again.items() if k != "outcome"},
        )
        rc, text, err = self.decline("too broad")
        self.assertEqual(rc, 0, err)
        for said in (json.dumps(again), text):
            self.assertNotIn("asi.decline_receipt", said)
            self.assertNotIn("finishes_an_earlier_run", said)
        self.setUp()
        self.stopped_after_the_record("too broad", "--operator", "alice")
        rc, written, err = self.rerun_json("too broad")
        self.assertEqual(rc, 0, err)
        self.assertEqual(set(written), PRINTED)
        self.assertEqual(written["outcome"], "receipt-written")

    def test_a_declined_record_without_a_well_shaped_decision_is_unreadable_and_untrusted(
        self,
    ) -> None:
        shapes: dict[str, tuple[Any, str, Any]] = {
            "absent": (None, "decline", None),
            "a list": (["no"], "decline", ["no"]),
            "an empty object": ({}, "decline", {}),
            "a blank reason": ({"reason": "  "}, "reason", "  "),
            "a reason with a line feed": ({"reason": "mark\nword"}, "reason", "mark\nword"),
            "an operator that is a number": ({"operator": 7}, "operator", 7),
            "a time not in iso form": ({"declined_at": "yesterday"}, "declined_at", "yesterday"),
            "a reason of 18,001 characters": ({"reason": "q" * 18_001}, "reason", "q" * 18_001),
            "U+FEFF in the reason": ({"reason": "mark\ufeffword"}, "U+FEFF", "mark\ufeffword"),
            "U+FEFF in the operator": ({"operator": "op\ufeffname"}, "U+FEFF", "op\ufeffname"),
        }
        for name, (decision, field, value) in shapes.items():
            with self.subTest(name):
                self.setUp()

                def declined(raw: dict[str, Any]) -> None:
                    raw["status"] = "declined"
                    if decision is None:
                        raw.pop("decline", None)
                    elif isinstance(decision, dict) and decision:
                        raw["decline"] = {**HAND_DECISION, **decision}
                    else:
                        raw["decline"] = decision

                self.edit_record(declined)
                before = self.tree()
                rc, listed, err = self.cli("--json", "candidates")
                self.assertIn(self.cid, listed + err)
                self.assertIn("unreadable", (listed + err).lower())
                for argv in (("review", self.cid), ("decline", self.cid, "--reason", "x")):
                    err = self.refused(self.cli(*argv), before)
                    self.assertIn("StateUnreadable", err)
                    self.assertIn(field, err)
                    if isinstance(value, str) and value.strip():
                        self.assertNotIn(value, err)
                rc, out, err = self.cli("doctor")
                self.assertIn(self.cid, out + err)
                if isinstance(decision, dict) and decision:
                    shaped = {**HAND_DECISION, **decision}
                    bad = Decline(**shaped)
                elif decision is None:
                    bad = None
                else:
                    bad = decision
                handed = dataclasses.replace(self.read_before, status="declined", decline=bad)
                with self.assertRaises(CandidateTampered):
                    verify_full_integrity(handed, rubrics=None, action="decline")
        for field in ("reason", "operator"):
            with self.subTest(surrogate=field):
                shaped = {**HAND_DECISION, field: "left\udcffright"}
                handed = dataclasses.replace(
                    self.read_before, status="declined", decline=Decline(**shaped)
                )
                with self.assertRaises(CandidateTampered) as caught:
                    verify_full_integrity(handed, rubrics=None, action="decline")
                said = str(caught.exception)
                self.assertIn("U+DCFF", said)
                self.assertIn(field, said)
                self.assertNotIn("left", said)
        with self.subTest("the escape \\udcff written into the JSON"):
            self.setUp()
            self.hand_decline()
            text = self.record_path.read_text(encoding="utf-8")
            self.record_path.write_text(
                text.replace('"declined elsewhere"', '"declined \\udcff elsewhere"'),
                encoding="utf-8",
            )
            read = self.state.read_candidate(self.cid)
            self.assertEqual(read.decline.reason, "declined \ufffd elsewhere")
            self.assertFalse(any(0xD800 <= ord(c) <= 0xDFFF for c in read.decline.reason))

    def test_a_record_that_is_not_declined_but_holds_a_decision_is_refused(self) -> None:
        for status in ("validated", "applied"):
            with self.subTest(status):
                self.setUp()

                def holding(raw: dict[str, Any]) -> None:
                    raw["status"] = status
                    raw["decline"] = dict(HAND_DECISION)

                self.edit_record(holding)
                with self.assertRaises(StateUnreadable):
                    self.state.read_candidate(self.cid)
                handed = dataclasses.replace(
                    self.read_before, status=status, decline=Decline(**HAND_DECISION)
                )
                with self.assertRaises(CandidateTampered):
                    verify_full_integrity(handed, rubrics=None, action="decline")

    def test_the_stored_reason_ceiling_is_500_times_the_longest_mask(self) -> None:
        longest = max(len(redaction.PLACEHOLDER.format(label=s.label)) for s in redaction._SHAPES)
        self.assertEqual(MAX_STORED_DECLINE_REASON_CHARS, 500 * longest)
        self.assertEqual(MAX_STORED_DECLINE_OPERATOR_CHARS, 200 * longest)

    def test_a_stop_after_the_record_of_a_508_character_reason_is_repaired_by_its_copied_stored_reason(
        self,
    ) -> None:
        reason = 'a "quoted" word ' + "x" * (500 - 16 - len(AWS) - 1) + " " + AWS
        self.assertEqual(len(reason), 500)
        left = self.stopped_after_the_record(reason)
        self.assertIn(b'\\"quoted\\"', left)
        stored = json.loads(left)["decline"]["reason"]
        self.assertEqual(len(stored), 508)
        rc, payload, err = self.rerun_json("-", stdin=(stored + "\n").encode())
        self.assertEqual(rc, 0, err)
        self.assertEqual(payload["outcome"], "receipt-written")
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        decision = json.loads(left)["decline"]
        self.assertEqual({k: receipt[k] for k in decision}, decision)
        self.assertEqual(self.record_path.read_bytes(), left)
        with self.subTest("the bound"):
            self.setUp()
            wide = "\U00020000" * 18_000
            self.hand_decline(reason=wide)
            data = (wide + "\r\n").encode()
            self.assertEqual(len(data), cli.REASON_LINE_BOUND)
            rc, payload, err = self.rerun_json("-", stdin=data)
            self.assertEqual(rc, 0, err)
            self.assertEqual(payload["outcome"], "receipt-written")
            self.assertTrue(self.receipt_path.is_file())

    def test_a_long_reason_that_is_not_the_stored_one_is_refused_and_writes_nothing(self) -> None:
        for size in (501, 18_000):
            with self.subTest(declined_with_no_receipt=size):
                self.setUp()
                self.hand_decline()
                text = "w" * size
                before = self.tree()
                err = self.refused(self.decline("-", stdin=(text + "\n").encode()), before)
                self.assertIn("another reason", err)
                self.assertNotIn("w" * 40, err)
        with self.subTest("a validated record given a 508-character fixed point"):
            self.setUp()
            fixed = settled_mask("x" * (500 - len(AWS) - 1) + " " + AWS)[0]
            self.assertEqual(len(fixed), 508)
            before = self.tree()
            err = self.refused(self.decline("-", stdin=(fixed + "\n").encode()), before)
            self.assertIn("a new decline takes at most 500", err)
        over = {
            "501 holding a secret shape": "x" * (501 - len(AWS) - 1) + " " + AWS,
            "18,001": "w" * 18_001,
        }
        for name, text in over.items():
            for held in (False, True):
                with self.subTest(name, lock_held=held):
                    self.setUp()
                    before = self.tree()
                    lock = RunLock(self.state.lock_path(), owner="daily 2026-02-02 pid=1")
                    with lock if held else contextlib.nullcontext():
                        outcome = self.decline("-", stdin=(text + "\n").encode())
                    err = self.refused(outcome, before)
                    self.assertIn(f"is {len(text)} characters; the limit is 500", err)
                    self.assertNotIn("LockHeld", err)

    def test_both_decline_writes_happen_while_the_run_lock_is_held(self) -> None:
        seen: list[tuple[str, bool]] = []
        real_record = StateStore.write_candidate
        real_receipt = StateStore.write_action_receipt
        case = self

        def record(self: StateStore, *args: Any, **kw: Any) -> Any:
            seen.append(("record", case.lock_is_held()))
            return real_record(self, *args, **kw)

        def receipt(self: StateStore, action: str, *args: Any, **kw: Any) -> Any:
            seen.append((f"{action} receipt", case.lock_is_held()))
            return real_receipt(self, action, *args, **kw)

        watch = (
            mock.patch.object(StateStore, "write_candidate", record),
            mock.patch.object(StateStore, "write_action_receipt", receipt),
        )
        rc, out, err = self._main([*self.cfg, "decline", self.cid, "--reason", "too broad"], *watch)
        self.assertEqual(rc, 0, err)
        self.assertEqual(seen, [("record", True), ("decline receipt", True)])
        self.setUp()
        self.stopped_after_the_record()
        seen.clear()
        rc, out, err = self._main([*self.cfg, "decline", self.cid, "--reason", "too broad"], *watch)
        self.assertEqual(rc, 0, err)
        self.assertEqual(seen, [("decline receipt", True)])

    def run_child(self, *argv: str) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k != "ASI_AUTOMATED_RUN"}
        env.update(PYTHONPATH=f"{SRC}{os.pathsep}{SRC.parent}", PYTHONDONTWRITEBYTECODE="1")
        try:
            return subprocess.run(
                [sys.executable, "-m", "agent_skill_improver", *self.cfg, *argv],
                capture_output=True,
                text=True,
                env=env,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            self.fail("decline blocked on the leaf at its receipt path")

    def test_a_decline_receipt_leaf_that_is_not_a_regular_file_is_refused_without_blocking(
        self,
    ) -> None:
        path = self.receipt_path
        target = self.base / "a-target.json"

        def bind_socket() -> None:
            here = os.getcwd()
            os.chdir(path.parent)
            try:
                with socket.socket(socket.AF_UNIX) as sock:
                    sock.bind(path.name)
            finally:
                os.chdir(here)

        leaves = {
            "a FIFO with no writer": lambda: os.mkfifo(path),
            "a directory": lambda: path.mkdir(),
            "a link to a regular file": lambda: path.symlink_to(target),
            "a socket": bind_socket,
            "a dangling link": lambda: path.symlink_to(self.base / "nothing-here"),
        }
        for name, make in leaves.items():
            with self.subTest(name):
                self.setUp()
                self.hand_decline()
                target.write_text(json.dumps({"kept": True}), encoding="utf-8")
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    make()
                except OSError as exc:
                    self.skipTest(f"{name} cannot be made here: {exc}")
                before = self.tree()
                done = self.run_child("decline", self.cid, "--reason", HAND_DECISION["reason"])
                self.assertEqual(done.returncode, cli.RC_REFUSED, done.stderr)
                self.assertIn("StateUnreadable", done.stderr)
                self.assertIn(str(path), done.stderr)
                self.assertNotIn("Traceback", done.stderr)
                self.assertEqual(self.tree(), before)
                with self.assertRaises(StateUnreadable):
                    self.state.read_action_receipt("decline", self.cid)
        real_lstat = os.lstat
        with self.subTest("the leaf removed as soon as its lstat returns"):
            self.setUp()
            self.hand_decline()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema": "asi.decline_receipt.v1"}), encoding="utf-8")

            removed: list[str] = []

            def lstat_then_unlink(p: Any, *args: Any, **kw: Any) -> os.stat_result:
                info = real_lstat(p, *args, **kw)
                if _is(p, path) and not removed:
                    removed.append(str(p))
                    os.unlink(path)
                return info

            with mock.patch.object(os, "lstat", lstat_then_unlink):
                with self.assertRaises(StateUnreadable) as caught:
                    self.state.read_action_receipt("decline", self.cid)
            self.assertEqual(removed, [str(path)])
            self.assertIn(str(path), str(caught.exception))
        with self.subTest("the lstat refused"):
            self.setUp()
            self.hand_decline()

            def lstat_refused(p: Any, *args: Any, **kw: Any) -> os.stat_result:
                if _is(p, path):
                    raise PermissionError(13, "Permission denied", str(p))
                return real_lstat(p, *args, **kw)

            with mock.patch.object(os, "lstat", lstat_refused):
                rc, out, err = self.decline(HAND_DECISION["reason"])
            self.assertEqual(rc, cli.RC_REFUSED, err)
            self.assertNotIn("Traceback", err)
            with mock.patch.object(os, "lstat", lstat_refused):
                with self.assertRaises(StateUnreadable) as caught:
                    self.state.read_action_receipt("decline", self.cid)
            self.assertIn(str(path), str(caught.exception))

    def test_a_reason_changed_at_the_lock_boundary_is_judged_against_the_record_read_under_the_lock(
        self,
    ) -> None:
        self.hand_decline(reason="the first reason")
        real = promote._holding_the_run_lock
        case = self

        @contextlib.contextmanager
        def then_rewritten(*args: Any, **kw: Any) -> Any:
            with real(*args, **kw) as held:
                case.hand_decline(reason="a reason written at the boundary")
                yield held

        with mock.patch.object(promote, "_holding_the_run_lock", then_rewritten):
            rc, out, err = self.decline("the first reason")
        self.assertEqual(rc, cli.RC_REFUSED, err)
        self.assertIn("another reason", err)
        self.assertFalse(os.path.lexists(self.receipt_path))
        self.assertEqual(self.raw_record()["decline"]["reason"], "a reason written at the boundary")


if __name__ == "__main__":
    unittest.main()
