"""Optional cost memory is bounded and dispensable; candidate integrity is not."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_skill_improver import cli
from agent_skill_improver.app import rejections
from agent_skill_improver.app.state import StateStore
from agent_skill_improver.app.wiring import build_runner
from agent_skill_improver.config import load_config
from agent_skill_improver.domain.candidate import Candidate
from agent_skill_improver.errors import ReadOnlyViolation
from agent_skill_improver.hashing import sha256_text
from agent_skill_improver.jsonio import dumps
from tests.support import FIXTURE_EXPORTS, build_env, run_cli
from tests.test_a_rejected_replacement_is_not_replayed_again import (
    SEED,
    _RoleOverride,
    role_counts,
    wire_modes,
)


def sample_entry(index: int = 0) -> rejections.RejectionEntry:
    return rejections.RejectionEntry(
        skill_id=f"skill-{index % 5}",
        base_sha256=sha256_text(f"base-{index}"),
        replacement_sha256=sha256_text(f"replacement-{index}"),
        cases_sha256=sha256_text(f"cases-{index}"),
        evaluation_context_sha256=sha256_text(f"examiner-{index}"),
        candidate_id=f"CAND-2026-01-01-{index + 1:02d}-skill-{index % 5}",
        rejected_at="2026-01-01T00:00:00Z",
    )


def memory_payload(entries):
    return {"schema": rejections.REJECTION_MEMORY_SCHEMA, "entries": [e.to_dict() for e in entries]}


class _MemoryHarness(unittest.TestCase):
    """No inherited tests: reusable real-CLI fixture for the two five-method modules."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c242-memory-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.env_index = 0
        self.fresh()

    def fresh(self, *, mode="repeat-no-fix", **kwargs):
        self.env_index += 1
        self.env = self.base / f"env-{self.env_index}"
        self.config = build_env(self.env, seed=SEED, **kwargs)
        self.log = self.env / "calls.jsonl"
        wire_modes(self.config, self.log, mode=mode)
        self.state = self.env / "state"
        self.memory = self.state / rejections.REJECTION_MEMORY_NAME
        self.skill = self.env / "skills-canonical" / "tdd" / "SKILL.md"
        self.last_store = None
        self.last_output = ""
        return self.config

    def calls(self):
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def invoke(self, day="2026-02-01", *, commit=True, transform=None, stack=None, rc=0):
        args = ["--config", str(self.config), "--json", "daily", "--day", day, "--seed", SEED]
        if commit:
            args.append("--commit")
        offset = len(self.calls())
        if transform is None and stack is None:
            result = run_cli(args)
            actual, out = result.returncode, result.stdout
            diagnostic = result.stdout + result.stderr
        else:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.ExitStack() as active:
                if transform is not None:
                    active.enter_context(
                        patch.object(
                            cli,
                            "build_runner",
                            side_effect=lambda cfg, state: _RoleOverride(
                                build_runner(cfg, state), transform
                            ),
                        )
                    )
                if stack is not None:
                    stack(active)
                active.enter_context(contextlib.redirect_stdout(stdout))
                active.enter_context(contextlib.redirect_stderr(stderr))
                actual = cli.main(args)
            out = stdout.getvalue()
            diagnostic = out + stderr.getvalue()
        self.last_output = diagnostic
        self.assertEqual(actual, rc, diagnostic)
        return (json.loads(out) if out.strip().startswith("{") else None), self.calls()[offset:]

    def capture_store(self, active):
        def create(cfg):
            self.last_store = StateStore(cfg.state_dir)
            return self.last_store

        active.enter_context(patch.object(cli, "build_state", side_effect=create))

    def candidates(self):
        return sorted((self.state / "candidates").glob("*.json"))

    def seed(self):
        receipt, calls = self.invoke()
        candidate = Candidate.from_dict(json.loads(self.candidates()[-1].read_text()))
        self.assertEqual(candidate.status, "gate-failed")
        self.assertEqual(
            role_counts(calls), {"initial_scorer": 5, "replay_scorer": 8, "improver": 1, "judge": 2}
        )
        self.assertTrue(self.memory.is_file(), receipt)
        return candidate

    def write_memory(self, entries):
        self.state.mkdir(parents=True, exist_ok=True)
        self.memory.write_text(dumps(memory_payload(entries)))

    def context(self):
        cfg = load_config(self.config)
        return rejections.evaluation_context_sha256(
            cfg, {r.id: r for r in cfg.rubrics}, build_runner(cfg)
        )

    def mode(self, role, mode):
        body = self.config.read_text()
        heading = f"[model.{role}]\n"
        first = body.index(heading)
        end = body.find("\n[", first + len(heading))
        end = len(body) if end < 0 else end
        section = body[first:end]
        before = json.loads(section.split("ASI_FAKE_MODE = ", 1)[1].split(" }", 1)[0])
        body = body[:first] + section.replace(json.dumps(before), json.dumps(mode)) + body[end:]
        self.config.write_text(body)


class ADamagedRejectionMemoryDoesNotBlockAProposalTests(_MemoryHarness):
    def test_bad_memory_is_named_ignored_and_left_untouched(self):
        self.seed()
        original = json.loads(self.memory.read_text())
        valid = self.memory.read_bytes()
        invalid_entry = copy.deepcopy(original)
        invalid_entry["entries"][0]["base_sha256"] = "not-a-hash"
        duplicate = copy.deepcopy(original)
        duplicate["entries"] *= 2
        unknown = copy.deepcopy(original)
        unknown["schema"] = "asi.rejection-memory.v999"
        variants = {
            "truncated": b'{"schema":',
            "unknown-schema-matching-entry": dumps(unknown).encode(),
            "bad-digest": dumps(invalid_entry).encode(),
            "duplicate": dumps(duplicate).encode(),
            "over-count": dumps(memory_payload([sample_entry(i) for i in range(65)])).encode(),
            "over-bytes": b" " * (rejections.MAX_REJECTION_MEMORY_BYTES + 1),
            "bad-utf8": b"\xff",
            "symlink": None,
            "directory": None,
        }
        target = self.base / "foreign.json"
        target.write_bytes(valid)
        for index, (name, body) in enumerate(variants.items(), 2):
            with self.subTest(damage=name):
                self.memory.unlink()
                if name == "symlink":
                    self.memory.symlink_to(target)
                elif name == "directory":
                    self.memory.mkdir()
                else:
                    self.memory.write_bytes(body)
                before_count = len(self.candidates())
                receipt, calls = self.invoke(f"2026-02-{index:02d}")
                self.assertEqual(receipt["counts"]["rejection_memory_ignored"], 1)
                self.assertEqual(receipt["counts"]["rejection_memory_write_failed"], 0)
                self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 0)
                self.assertEqual(len(self.candidates()), before_count + 1)
                self.assertEqual(
                    role_counts(calls),
                    {"initial_scorer": 5, "replay_scorer": 8, "improver": 1, "judge": 2},
                )
                note = next(n for n in receipt["notes"] if "Rejection memory" in n)
                self.assertIn(rejections.REJECTION_MEMORY_NAME, note)
                self.assertIn("ignored", note)
                self.assertIn("left untouched", note)
                self.assertEqual(target.read_bytes(), valid)
                if name == "directory":
                    self.assertTrue(self.memory.is_dir())
                    self.assertEqual(list(self.memory.iterdir()), [])
                    self.memory.rmdir()
                elif name == "symlink":
                    self.assertTrue(self.memory.is_symlink())
                    self.memory.unlink()
                else:
                    self.assertEqual(self.memory.read_bytes(), body)
                    self.memory.unlink()
                self.memory.write_bytes(valid)
        self.memory.unlink()
        receipt, calls = self.invoke("2026-02-12")
        self.assertEqual(receipt["counts"]["rejection_memory_ignored"], 0)
        self.assertFalse(any("Rejection memory" in n and "ignored" in n for n in receipt["notes"]))
        self.assertGreater(role_counts(calls)["replay_scorer"], 0)
        self.assertTrue(self.memory.is_file())
        malformed = []
        for key in ("candidate_id", "rejected_at", "skill_id", "cases_sha256"):
            for value in (None, 1, [], ""):
                raw = copy.deepcopy(original)
                raw["entries"][0][key] = value
                malformed.append(raw)
        for edits in (
            {"extra": "text"},
            {"rejected_at": "2026-02-30T00:00:00Z"},
            {"rejected_at": "2026-01-01T00:00:00+00:00"},
            {"base_sha256": "A" * 64},
        ):
            raw = copy.deepcopy(original)
            raw["entries"][0].update(edits)
            malformed.append(raw)
        missing = copy.deepcopy(original)
        del missing["entries"][0]["skill_id"]
        malformed += [
            missing,
            {**original, "extra": 1},
            {"schema": original["schema"], "entries": {}},
        ]
        for raw in malformed:
            with self.subTest(parser=raw):
                parsed = rejections.parse_rejection_memory(raw)
                self.assertTrue(parsed.problem)
                self.assertEqual(parsed.entries, ())
                self.assertFalse(parsed.writable)
        self.memory.write_text('{"schema":"asi.rejection-memory.v1","entries":[],"extra":NaN}')
        self.assertEqual(StateStore(self.state).read_rejection_memory().problem, "invalid JSON")

    def test_the_optional_reader_never_reads_more_than_its_byte_budget(self):
        self.state.mkdir(parents=True)
        self.memory.write_bytes(b" " * (rejections.MAX_REJECTION_MEMORY_BYTES + 20))
        real_fdopen = os.fdopen
        streams, sizes = [], []

        class Instrumented:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def fileno(self):
                return self.stream.fileno()

            def read(self, size=-1):
                sizes.append(size)
                self_outer.assertGreaterEqual(size, 0, "an unbounded read is forbidden")
                self_outer.assertLessEqual(size, rejections.MAX_REJECTION_MEMORY_BYTES + 1)
                return self.stream.read(size)

        self_outer = self

        def wrap(fd, *args, **kwargs):
            stream = real_fdopen(fd, *args, **kwargs)
            streams.append(stream)
            return Instrumented(stream)

        with (
            patch("agent_skill_improver.app.state.os.fdopen", side_effect=wrap),
            patch("agent_skill_improver.app.state.loads_untrusted") as parse,
        ):
            result = StateStore(self.state).read_rejection_memory()
        self.assertEqual(sizes, [131073])
        self.assertTrue(all(stream.closed for stream in streams))
        parse.assert_not_called()
        self.assertEqual(result.problem, "over byte limit")
        self.assertEqual(result.entries, ())

    def test_unwritable_memory_does_not_lose_the_candidate_or_daily_receipt(self):
        real_persist = StateStore._persist
        for fault in ("oserror", "directory", "readonly", "serialization", "merge"):
            with self.subTest(fault=fault):
                self.fresh()
                writes, published = [], []

                def persist(store, path, write):
                    writes.append(path.relative_to(store.root).as_posix())
                    if path.name == rejections.REJECTION_MEMORY_NAME:
                        self.assertEqual(len(self.candidates()), 1, "candidate must persist first")
                        if fault == "oserror":
                            raise OSError("fixture optional write failed")
                        if fault == "readonly":
                            raise ReadOnlyViolation("fixture structural refusal")
                        if fault == "serialization":
                            raise ValueError("fixture serialization error")
                        self.memory.mkdir()
                    return real_persist(store, path, write)

                real_publish = rejections.RejectionMemoryRun.publish_if_stable

                def publish(run, state, **kwargs):
                    published.append(kwargs["receipt"])
                    return real_publish(run, state, **kwargs)

                def configure(active):
                    self.capture_store(active)
                    active.enter_context(patch.object(StateStore, "_persist", persist))
                    active.enter_context(
                        patch.object(rejections.RejectionMemoryRun, "publish_if_stable", publish)
                    )
                    if fault == "merge":
                        active.enter_context(
                            patch.object(
                                rejections,
                                "remember_rejections",
                                side_effect=ValueError("fixture merge error"),
                            )
                        )

                expected = (
                    0 if fault in ("oserror", "directory") else 3 if fault == "readonly" else 70
                )
                receipt, _ = self.invoke(stack=configure, rc=expected)
                self.assertEqual(len(self.candidates()), 1)
                self.assertEqual(len(published), 1)
                if expected == 0:
                    self.assertEqual(receipt["counts"]["rejection_memory_write_failed"], 1)
                    self.assertEqual(receipt["status"], "ok")
                    self.assertTrue(any("write failed" in n for n in receipt["notes"]))
                    self.assertEqual(len(list((self.state / "receipts").glob("*.json"))), 1)
                    self.assertTrue(
                        StateStore(self.state).read_checkpoint("2026-02-01")["completed"]
                    )
                    self.assertEqual(sum(p == rejections.REJECTION_MEMORY_NAME for p in writes), 1)
                    if fault == "directory":
                        self.assertTrue(self.memory.is_dir())
                else:
                    self.assertIsNone(receipt)
                    self.assertEqual(published[0].counts["rejection_memory_write_failed"], 0)
                    self.assertFalse(any("write failed" in n for n in published[0].notes))
                    self.assertEqual(list((self.state / "receipts").glob("*.json")), [])
                    self.assertIn(
                        "ReadOnlyViolation" if fault == "readonly" else "ValueError",
                        self.last_output,
                    )

    def test_an_unreadable_candidate_still_stands_the_daily_run_down(self):
        self.seed()
        before = self.memory.read_bytes()
        (path,) = self.candidates()
        path.write_bytes(b"{broken")
        receipt, calls = self.invoke("2026-02-02")
        self.assertEqual(receipt["status"], "stood-down")
        self.assertEqual(receipt["counts"]["candidate_records_unreadable"], 1)
        self.assertEqual(receipt["counts"]["records_scored"], 0)
        self.assertEqual(calls, [])
        self.assertEqual(path.read_bytes(), b"{broken")
        self.assertEqual(self.memory.read_bytes(), before)
        self.assertEqual(len(self.candidates()), 1)
        self.assertIn("queue", receipt["stand_down_reason"])

    def test_the_cited_excerpt_guard_precedes_the_repeat_guard(self):
        exports = self.base / "exports"
        shutil.copytree(FIXTURE_EXPORTS, exports)
        self.fresh(fixtures_dir=exports)
        candidate = self.seed()
        passage = " ".join(f"citedword{i:02d}" for i in range(30))
        path = exports / "failing.json"
        export = json.loads(path.read_text())
        element = next(e for e in export["elements"] if e.get("id") == "1700000000001-13")
        element["content"] += " " + passage
        path.write_text(json.dumps(export))
        replacement = candidate.patch.new_text + passage + "\n"
        raw = json.loads(self.memory.read_text())
        raw["entries"][0]["replacement_sha256"] = sha256_text(replacement)
        self.memory.write_text(dumps(raw))
        before = self.memory.read_bytes()
        self.assertEqual(raw["entries"][0]["cases_sha256"], rejections.cases_sha256(candidate))
        self.assertEqual(raw["entries"][0]["evaluation_context_sha256"], self.context())

        def transform(kwargs, payload):
            if kwargs["role"] == "improver":
                payload["new_skill_text"] = replacement
            return payload

        with patch.object(
            rejections.RejectionMemoryRun, "refuse_if_repeated", autospec=True
        ) as guard:
            receipt, calls = self.invoke("2026-02-02", transform=transform)
        guard.assert_not_called()
        self.assertEqual(receipt["counts"]["skills_quarantined"], 1)
        self.assertEqual(receipt["counts"]["replacements_refused_as_repeats"], 0)
        self.assertEqual(
            role_counts(calls), {"initial_scorer": 5, "replay_scorer": 0, "improver": 1, "judge": 0}
        )
        self.assertEqual(len(self.candidates()), 1)
        self.assertEqual(self.memory.read_bytes(), before)
        reason = receipt["skills"]["tdd"]["reason"]
        self.assertIn("cited excerpt", reason)
        self.assertIn("conversation", reason)
        self.assertNotIn(passage, reason)
        self.assertNotIn("citedword00", reason)


if __name__ == "__main__":
    unittest.main()
