"""Structural reasons a proposed replacement is never offered for review.

``skill_text_defects`` is the last check between the improver's output and a
candidate record. Everything it refuses is something a human reviewer would
have to notice unaided in a diff: a lost frontmatter delimiter, a renamed
skill, a rewrite too large to review, or, added here, a credential that the
model wrote into the file. The transcript the model saw was redacted and the
base skill was refused if it held a token, so a token in the output can only
have come from the model itself; it must not become a reviewable candidate, a
backup, or a prompt to the judge.
"""

from __future__ import annotations

import unittest

from agent_skill_improver.app.pipeline import skill_text_defects

BASE = """---
name: example
description: An example skill.
---

# Example

- Do the thing.
"""


def _j(*parts: str) -> str:
    return "".join(parts)


class ExistingStructuralRulesTests(unittest.TestCase):
    def test_a_faithful_small_edit_has_no_defects(self):
        new = BASE + "- Ask before anything destructive.\n"
        self.assertEqual(skill_text_defects(new, base_text=BASE, skill_id="example"), [])

    def test_losing_the_opening_delimiter_is_a_defect(self):
        new = BASE.replace("---\nname", "name", 1)
        problems = skill_text_defects(new, base_text=BASE, skill_id="example")
        self.assertTrue(any("opening '---'" in p for p in problems), problems)

    def test_renaming_the_skill_is_a_defect(self):
        new = BASE.replace("name: example", "name: renamed")
        problems = skill_text_defects(new, base_text=BASE, skill_id="example")
        self.assertTrue(any("renaming" in p for p in problems), problems)

    def test_dropping_the_description_is_a_defect(self):
        new = BASE.replace("description: An example skill.\n", "")
        problems = skill_text_defects(new, base_text=BASE, skill_id="example")
        self.assertTrue(any("description" in p for p in problems), problems)

    def test_an_unreviewably_large_rewrite_is_a_defect(self):
        new = BASE + ("- padding line that says nothing new\n" * 2000)
        problems = skill_text_defects(new, base_text=BASE, skill_id="example")
        self.assertTrue(any("cannot be reviewed" in p for p in problems), problems)


class TruncatedReplacementTests(unittest.TestCase):
    """A model that hits its output limit, or that was shown a clipped base,
    returns the first part of the document and stops. Measured before this
    rule: a replacement that removed 6006 of a base's lines and added 3 passed
    every structural check and became a candidate."""

    LARGE = BASE + ("- a line of guidance that is part of the skill\n" * 200)

    def test_a_replacement_that_drops_more_than_half_of_a_large_base_is_a_defect(self):
        self.assertGreaterEqual(len(self.LARGE), 2000)
        new = self.LARGE[: len(self.LARGE) // 3] + "\n- Ask first, every time.\n"
        problems = skill_text_defects(new, base_text=self.LARGE, skill_id="example")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("more than half the document is gone", problems[0])
        self.assertIn(f"{len(self.LARGE)}-char base", problems[0])

    def test_the_control_a_large_base_cut_to_sixty_percent_is_still_reviewable(self):
        keep = self.LARGE.split("\n")
        new = "\n".join(keep[: int(len(keep) * 0.6)]) + "\n- Ask first, every time.\n"
        self.assertGreater(len(new), len(self.LARGE) * 0.5)
        self.assertEqual(skill_text_defects(new, base_text=self.LARGE, skill_id="example"), [])

    def test_the_control_a_base_under_the_floor_may_lose_more_than_half(self):
        self.assertLess(len(BASE), 2000)
        new = "---\nname: example\ndescription: A.\n---\n"
        self.assertLess(len(new), len(BASE) * 0.5)
        self.assertEqual(skill_text_defects(new, base_text=BASE, skill_id="example"), [])


class ByteOrderMarkTests(unittest.TestCase):
    """A UTF-8 byte-order mark before the frontmatter is read past by this
    tool's parser. Measured before: a base carrying one was
    inventoried as unloadable and every replacement that kept it was
    quarantined as having no opening delimiter, so the skill could never be
    improved and the model calls were spent for nothing."""

    def test_a_mark_kept_from_the_base_is_not_a_defect(self):
        base = "\ufeff" + BASE
        new = base + "- Ask before anything destructive.\n"
        self.assertEqual(skill_text_defects(new, base_text=base, skill_id="example"), [])

    def test_a_mark_the_model_added_is_a_defect_by_name(self):
        new = "\ufeff" + BASE + "- Ask before anything destructive.\n"
        problems = skill_text_defects(new, base_text=BASE, skill_id="example")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("byte-order mark (U+FEFF) the base does not have", problems[0])

    def test_dropping_the_base_mark_is_allowed_as_a_visible_first_line_change(self):
        base = "\ufeff" + BASE
        new = BASE + "- Ask before anything destructive.\n"
        self.assertEqual(skill_text_defects(new, base_text=base, skill_id="example"), [])


class CredentialInReplacementTests(unittest.TestCase):
    def test_a_token_written_into_the_replacement_is_a_defect(self):
        token = "g" + "hp_" + "B" * 36
        new = BASE + f"- Authenticate with GH_TOKEN={token}\n"
        problems = skill_text_defects(new, base_text=BASE, skill_id="example")
        self.assertTrue(any("credential" in p for p in problems), problems)
        self.assertFalse(
            any(token in p for p in problems),
            "the defect names the shape, never repeats the credential into a receipt",
        )

    def test_a_key_block_written_into_the_replacement_is_a_defect(self):
        new = BASE + _j("-----BE", "GIN RSA PRIV", "ATE KEY-----") + "\nMIIE" + "x" * 40 + "\n"
        problems = skill_text_defects(new, base_text=BASE, skill_id="example")
        self.assertTrue(any("credential" in p for p in problems), problems)

    def test_placeholder_prose_in_the_replacement_is_not_a_defect(self):
        new = BASE + "- Set `password: <your-password>` and pass `--token $TOKEN`.\n"
        self.assertEqual(skill_text_defects(new, base_text=BASE, skill_id="example"), [])


if __name__ == "__main__":
    unittest.main()
