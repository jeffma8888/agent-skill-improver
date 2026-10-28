"""A scorer reply leaves a masked, bounded scorecard.

``parse_scorecard`` ignores the ``criteria`` entries it cannot use (not an
object, no criterion id, an id the rubric does not define, a criterion scored
again) and records the unknown ids as ``extraneous_criterion_ids``. The scorer
decides how many entries its reply holds, bounded only by the role's
``max_output_bytes``, and the scorecard is persisted under ``state/``. Measured
before (3.11 and 3.13, through ``daily --commit`` with a scorer that adds
entries to the fake model's reply): 200,000 entries that were not objects made
each scorecard about 9.9 MB with 200,000 notes, 50,000 invented ids one of about
5.2 MB, one 500,000-character id one of about 502 KB, and a GitHub token sent as
an id was recorded whole in every scorecard while its note masked it; the healthy
scorecards are about 2 KB. The first ``MAX_IGNORED_SHOWN`` ignored entries are
now noted one by one and one note counts the rest, and the unknown ids are
recorded as a listing shows a record's value (masked, then cut), the first
``MAX_IGNORED_SHOWN`` distinct ones in sorted order, a note counting the rest.
"""

from __future__ import annotations

import json
import unittest

from agent_skill_improver.domain.redaction import QUOTED_VALUE_CHARS
from agent_skill_improver.domain.scoring import MAX_IGNORED_SHOWN
from tests.test_scoring_trust import card, rubric

GITHUB_TOKEN = "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
MASK = "[REDACTED:github-token]"


def noted(result, fragment: str) -> list[str]:
    return [n for n in result.notes if fragment in n]


def stored(result) -> str:
    return json.dumps(result.to_dict())


class AnUnknownIdIsRecordedMaskedAndCutTests(unittest.TestCase):
    def test_a_credential_sent_as_a_criterion_id_is_recorded_masked(self) -> None:
        result = card([{"criterion_id": GITHUB_TOKEN, "verdict": "PASS"}])
        self.assertEqual(result.extraneous_criterion_ids, (MASK,))
        self.assertNotIn(GITHUB_TOKEN, stored(result))

    def test_a_long_unknown_id_is_recorded_cut(self) -> None:
        result = card([{"criterion_id": "x" * 500_000}])
        left_out = 500_000 - QUOTED_VALUE_CHARS
        self.assertEqual(
            result.extraneous_criterion_ids,
            ("x" * QUOTED_VALUE_CHARS + f"... [{left_out} more chars not shown]",),
        )
        self.assertLess(len(stored(result)), 5_000)

    def test_a_short_unknown_id_is_recorded_as_written(self) -> None:
        result = card([{"criterion_id": "invented"}, {"criterion_id": " Invented "}])
        self.assertEqual(result.extraneous_criterion_ids, (" Invented ", "invented"))

    def test_a_repeated_unknown_id_is_recorded_once(self) -> None:
        result = card([{"criterion_id": "invented"}] * 3)
        self.assertEqual(result.extraneous_criterion_ids, ("invented",))


class IgnoredEntriesAreNotedBoundedTests(unittest.TestCase):
    def test_up_to_the_limit_every_ignored_entry_is_noted(self) -> None:
        result = card([0] * MAX_IGNORED_SHOWN)
        self.assertEqual(len(noted(result, "is not an object; ignored")), MAX_IGNORED_SHOWN)
        self.assertEqual(noted(result, "more criteria entries ignored"), [])

    def test_past_the_limit_one_note_counts_the_rest(self) -> None:
        result = card([0] * 200_000)
        self.assertEqual(len(noted(result, "is not an object; ignored")), MAX_IGNORED_SHOWN)
        self.assertEqual(
            noted(result, "more criteria entries ignored"),
            [
                f"{200_000 - MAX_IGNORED_SHOWN} more criteria entries ignored; only the first "
                f"{MAX_IGNORED_SHOWN} are noted"
            ],
        )
        self.assertEqual(len(result.scores), len(rubric().criterion_ids))
        self.assertLess(len(stored(result)), 10_000)

    def test_an_unknown_id_past_the_note_limit_is_still_recorded_masked(self) -> None:
        result = card([0] * MAX_IGNORED_SHOWN + [{"criterion_id": GITHUB_TOKEN}])
        self.assertNotIn(GITHUB_TOKEN, stored(result))
        self.assertEqual(result.extraneous_criterion_ids, (MASK,))
        self.assertEqual(noted(result, "is not in rubric"), [])
        self.assertEqual(
            noted(result, "more criteria entries ignored"),
            [f"1 more criteria entries ignored; only the first {MAX_IGNORED_SHOWN} are noted"],
        )

    def test_each_kind_of_ignored_entry_counts_toward_the_limit(self) -> None:
        known = rubric().criterion_ids[0]
        entries = (
            [0] * 5
            + [{"verdict": "PASS"}] * 5
            + [{"criterion_id": f"invented-{i}"} for i in range(6)]
            + [{"criterion_id": known, "verdict": "PASS"}] * 6
        )
        result = card(entries)
        ignored_notes = [
            n for n in result.notes if n.endswith("; ignored") or "scored more than once" in n
        ]
        self.assertEqual(len(ignored_notes), MAX_IGNORED_SHOWN)
        self.assertEqual(
            noted(result, "more criteria entries ignored"),
            [f"1 more criteria entries ignored; only the first {MAX_IGNORED_SHOWN} are noted"],
        )


class UnknownIdsAreRecordedBoundedTests(unittest.TestCase):
    def test_up_to_the_limit_every_unknown_id_is_recorded(self) -> None:
        ids = [f"invented-{i:02d}" for i in range(MAX_IGNORED_SHOWN)]
        result = card([{"criterion_id": i} for i in ids])
        self.assertEqual(result.extraneous_criterion_ids, tuple(sorted(ids)))
        self.assertEqual(noted(result, "more unknown criterion ids not recorded"), [])

    def test_past_the_limit_the_first_ids_in_sorted_order_are_recorded(self) -> None:
        ids = [f"invented-{i}" for i in range(50_000)]
        result = card([{"criterion_id": i} for i in reversed(ids)])
        self.assertEqual(result.extraneous_criterion_ids, tuple(sorted(ids)[:MAX_IGNORED_SHOWN]))
        self.assertEqual(
            noted(result, "more unknown criterion ids not recorded"),
            [
                f"{50_000 - MAX_IGNORED_SHOWN} more unknown criterion ids not recorded; only "
                f"the first {MAX_IGNORED_SHOWN} in sorted order are"
            ],
        )
        self.assertLess(len(stored(result)), 10_000)


class TheControlTests(unittest.TestCase):
    def test_the_control_a_reply_naming_each_criterion_once_has_no_ignored_note(self) -> None:
        result = card([{"criterion_id": c, "verdict": "PASS"} for c in rubric().criterion_ids])
        self.assertEqual(result.extraneous_criterion_ids, ())
        self.assertEqual([n for n in result.notes if "ignored" in n or "recorded" in n], [])


if __name__ == "__main__":
    unittest.main()
