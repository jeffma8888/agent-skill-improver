"""A long workspace is a bounded selection bucket key.

With ``[selection].fairness_dimension = "workspace"``, ``daily`` buckets the
discovered threads by the workspace each one names, and the selection report
writes every bucket key three times into the receipt (``buckets``,
``selected_per_bucket`` and ``exhausted_buckets``). The workspace is the
export's ``workspacePath`` (or a listing's ``Workspace:`` line), untrusted and
unbounded. Measured before (3.11 and 3.13, the same on both): one fixture export
with a 200,000-character ``workspacePath`` made a receipt of about 610 KB where
the healthy fixtures make one of about 10 KB (the receipt names its temporary
directory, so its size moves by a few bytes with that path). A key is now kept
whole up to 512 characters and past that cut and fingerprinted
(``domain/selection.py`` ``bucket_key``), so it stays a bucket of its own. A
key kept whole is the value as written, neither trimmed nor case-folded, so two
workspaces that differ only in case or surrounding spaces stay two buckets.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_skill_improver.domain.selection import BUCKET_KEY_CHARS, bucket_key, select_fair
from tests.support import FIXTURE_EXPORTS, build_env, run_cli

LONG = "/ws/" + "w" * 200_000


class BucketKeyTests(unittest.TestCase):
    def test_a_key_up_to_the_limit_is_kept_whole(self) -> None:
        for value in (
            "",
            "/tmp/asi-fixture-workspace-a",
            " /a ",
            "/Ws",
            "\t/Ws/Mixed Case/ ",
            "p" * BUCKET_KEY_CHARS,
        ):
            with self.subTest(value=value[:40], length=len(value)):
                self.assertEqual(bucket_key(value), value)

    def test_one_character_past_the_limit_is_cut_and_fingerprinted(self) -> None:
        value = "p" * (BUCKET_KEY_CHARS + 1)
        key = bucket_key(value)
        self.assertNotEqual(key, value)
        self.assertTrue(key.startswith("p" * BUCKET_KEY_CHARS + "... ["))
        self.assertIn(f"{BUCKET_KEY_CHARS + 1} characters, sha256 ", key)

    def test_a_long_key_is_bounded(self) -> None:
        key = bucket_key(LONG)
        self.assertLess(len(key), BUCKET_KEY_CHARS + 60)
        self.assertTrue(key.startswith(LONG[:BUCKET_KEY_CHARS]))
        self.assertIn(f"{len(LONG)} characters", key)

    def test_a_fingerprinted_key_never_equals_one_kept_whole(self) -> None:
        self.assertGreater(len(bucket_key(LONG)), BUCKET_KEY_CHARS)

    def test_two_long_keys_sharing_their_first_characters_stay_two_buckets(self) -> None:
        a, b = LONG + "-a", LONG + "-b"
        self.assertNotEqual(bucket_key(a), bucket_key(b))
        _, report = select_fair([a, b], limit=10, bucket_of=bucket_key)
        self.assertEqual(sorted(report.buckets.values()), [1, 1])

    def test_the_same_long_key_is_the_same_bucket(self) -> None:
        again = "".join(list(LONG))
        self.assertIsNot(again, LONG)
        self.assertEqual(bucket_key(again), bucket_key(LONG))

    def test_keys_differing_only_in_case_or_spaces_are_separate_buckets(self) -> None:
        values = ["/ws", "/Ws", "/ws ", " /ws"]
        _, report = select_fair(values, limit=10, bucket_of=bucket_key)
        self.assertEqual(report.buckets, {v: 1 for v in values})

    def test_long_keys_differing_only_in_case_or_spaces_stay_separate(self) -> None:
        variants = [LONG, LONG.upper(), " " + LONG, LONG + " ", LONG + "/Proj", LONG + "/proj"]
        self.assertEqual(len({bucket_key(v) for v in variants}), len(variants))

    def test_a_lone_surrogate_in_a_long_key_is_fingerprinted_not_a_crash(self) -> None:
        key = bucket_key("\ud800" * (BUCKET_KEY_CHARS + 1))
        self.assertIn("sha256 ", key)


class TheReceiptStaysSmallTests(unittest.TestCase):
    def daily(self, workspaces: dict[str, str]) -> dict:
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            exports = base / "exports"
            shutil.copytree(FIXTURE_EXPORTS, exports)
            for name, workspace in workspaces.items():
                path = exports / f"{name}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["thread"]["workspacePath"] = workspace
                path.write_text(json.dumps(payload), encoding="utf-8")
            config = build_env(
                base / "env", fixtures_dir=exports, seed="c178", fairness_dimension="workspace"
            )
            proc = run_cli(
                ["--config", str(config), "--json", "daily", "--commit", "--day", "2026-02-01"]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            (receipt,) = (base / "env" / "state" / "receipts").glob("2026-02-01*.json")
            text = receipt.read_text(encoding="utf-8")
            self.assertEqual(proc.stdout, text)
            return json.loads(text) | {"_size": len(text)}

    def long_key(self, receipt: dict) -> str:
        (key,) = [k for k in receipt["selection"]["buckets"] if k.startswith("/ws/")]
        return key

    def test_a_long_workspace_path_makes_a_small_receipt(self) -> None:
        receipt = self.daily({"compliant": LONG})
        self.assertLess(receipt["_size"], 20_000)
        self.assertEqual(receipt["counts"]["conversations_ingested"], 5)
        key = self.long_key(receipt)
        self.assertLess(len(key), BUCKET_KEY_CHARS + 60)
        self.assertEqual(receipt["selection"]["buckets"][key], 1)
        self.assertEqual(receipt["selection"]["selected_per_bucket"][key], 1)
        self.assertIn(key, receipt["selection"]["exhausted_buckets"])

    def test_two_threads_in_one_long_workspace_share_its_bucket(self) -> None:
        receipt = self.daily({"compliant": LONG, "failing": LONG})
        key = self.long_key(receipt)
        self.assertLess(len(key), BUCKET_KEY_CHARS + 60)
        self.assertEqual(receipt["selection"]["buckets"][key], 2)

    def test_workspaces_differing_only_in_case_or_spaces_are_separate_buckets(self) -> None:
        written = {
            "compliant": "/tmp/Asi-Case",
            "failing": "/tmp/asi-case",
            "injection": "/tmp/asi-space",
            "second-workspace": " /tmp/asi-space ",
        }
        buckets = self.daily(written)["selection"]["buckets"]
        for workspace in written.values():
            with self.subTest(workspace=workspace):
                self.assertEqual(buckets.get(workspace), 1)

    def test_the_control_short_workspaces_are_the_keys_as_written(self) -> None:
        receipt = self.daily({})
        self.assertEqual(
            receipt["selection"]["buckets"],
            {
                "/tmp/asi-fixture-workspace-a": 1,
                "/tmp/asi-fixture-workspace-b": 1,
                "/tmp/asi-fixture-workspace-c": 1,
                "/tmp/asi-fixture-workspace-d": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
