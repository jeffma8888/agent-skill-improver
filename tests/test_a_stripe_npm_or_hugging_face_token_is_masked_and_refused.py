"""A Stripe secret or restricted key, an npm token or a Hugging Face token is masked and refused.

Measured before (3.11 and 3.13): neither the redactor nor the leak scanner knew
any of the three. Of 8 samples (``sk_`` and ``rk_`` in each of the modes
``live_``, ``test_`` and ``prod_``, an npm token and a Hugging Face token),
each in prose after a space, ``redact``, ``quoted_value`` and ``listed_value``
masked none, ``credential_labels`` named none, so a model-written replacement
adding one was not refused and a configured child environment value holding
one was passed on, and the leak scanner reported none. A skill file holding
one was shown to the improver and the judge. Only an assignment the
assignment shape knows (``token=``) masked one.

Every sample is assembled at run time from fragments, so no credential shape
appears literally in tracked text.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import tests.test_daily_isolation as daily
from agent_skill_improver.app.pipeline import skill_text_defects
from agent_skill_improver.domain.isolation import credential_bearing_env_values
from agent_skill_improver.domain.redaction import (
    PLACEHOLDER,
    credential_labels,
    listed_value,
    quoted_value,
    redact,
)
from tests.support import InProcessRunner
from tests.test_redaction import leakscan


def _j(*parts: str) -> str:
    return "".join(parts)


B24 = "Ab3dEf6hIj9kLm2nOp5qRs8t"
LETTERS = "AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"
NPM_BODY = "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
LABELS = ("stripe-secret-key", "npm-token", "huggingface-token")

#: Every prefix and mode each new shape takes, by label.
TOKENS: tuple[tuple[str, str], ...] = (
    ("stripe-secret-key", _j("s", "k_", "live_") + B24),
    ("stripe-secret-key", _j("r", "k_", "live_") + B24),
    ("stripe-secret-key", _j("s", "k_", "test_") + B24),
    ("stripe-secret-key", _j("r", "k_", "test_") + B24[:20]),
    ("stripe-secret-key", _j("s", "k_", "prod_") + (B24 * 5)[:99]),
    ("stripe-secret-key", _j("r", "k_", "prod_") + B24 * 5),
    ("npm-token", _j("n", "pm_") + NPM_BODY),
    ("huggingface-token", _j("h", "f_") + LETTERS),
)

#: Text that is none of the three, by why.
NOT_TOKENS: tuple[tuple[str, str], ...] = (
    ("a publishable key, which is meant to be shown", _j("p", "k_", "live_") + B24),
    ("a publishable test key", _j("p", "k_", "test_") + B24),
    ("a mode Stripe does not issue", _j("s", "k_", "staging_") + B24),
    ("19 characters after the mode", _j("r", "k_", "live_") + B24[:19]),
    ("a name that ends in the prefix", _j("ta", "sk_", "test_") + B24),
    ("an upper-case prefix, which Stripe does not issue", _j("S", "K_", "LIVE_") + B24),
    (
        "a placeholder in snake case",
        _j("S", "TRIPE_SECRET_KEY=s", "k_", "test_") + "your_secret_key_here",
    ),
    ("an npm config variable", _j("n", "pm_") + "config_cache"),
    ("an npm placeholder in snake case", _j("n", "pm_") + "your_access_token_goes_right_here_ok"),
    ("an upper-case npm prefix", _j("N", "PM_") + NPM_BODY),
    ("35 characters after the npm prefix", _j("n", "pm_") + NPM_BODY[:35]),
    ("37 characters after the npm prefix", _j("n", "pm_") + NPM_BODY + "s"),
    ("33 letters after the Hugging Face prefix", _j("h", "f_") + LETTERS[:33]),
    ("35 letters after the Hugging Face prefix", _j("h", "f_") + LETTERS + "Z"),
    ("a digit among the 34", _j("h", "f_") + LETTERS[:33] + "7"),
    ("an upper-case Hugging Face prefix", _j("H", "F_") + LETTERS),
    (
        "a Hugging Face placeholder in snake case",
        _j("h", "f_") + "your_hugging_face_access_token_now",
    ),
    (
        "an Objective-C name",
        "- (id)" + _j("h", "f_") + "requiredCharacteristicTypesForDisplayMetadata;",
    ),
    ("a C name", "&" + _j("h", "f_") + "servingcellpriorityparametersdescription_h_prio"),
)


def _rules():
    return [rule for rule in leakscan.RULES if rule.id in LABELS]


class EachIsMaskedAndNamedTests(unittest.TestCase):
    def test_the_samples_are_as_long_as_the_formats_say(self):
        bodies = {token.rsplit("_", 1)[1] for _, token in TOKENS}
        self.assertEqual(len(NPM_BODY), 36)
        self.assertEqual(len(LETTERS), 34)
        self.assertTrue(LETTERS.isalpha() and NPM_BODY.isalnum())
        self.assertFalse(NPM_BODY.islower() or NPM_BODY.isupper(), "Base62 mixes the cases")
        self.assertEqual({len(b) for b in bodies}, {20, 24, 34, 36, 99, 120})

    def test_each_is_masked_where_it_stands_and_named_by_its_shape(self):
        for label, token in TOKENS:
            with self.subTest(token=token[:8], length=len(token)):
                text = f"key is {token} here"
                result = redact(text)
                self.assertEqual(result.text, f"key is {PLACEHOLDER.format(label=label)} here")
                self.assertEqual(result.counts, {label: 1})
                self.assertEqual(credential_labels(text), (label,))

    def test_an_assignment_holding_one_is_masked_once_by_its_shape(self):
        for label, token in TOKENS:
            with self.subTest(token=token[:8]):
                result = redact(_j("api_", "key") + f" = {token}")
                self.assertEqual(
                    result.text, _j("api_", "key") + f" = {PLACEHOLDER.format(label=label)}"
                )
                self.assertEqual(result.counts, {label: 1})

    def test_a_quoted_or_listed_value_holding_one_shows_its_placeholder(self):
        for label, token in TOKENS:
            with self.subTest(token=token[:8]):
                placeholder = PLACEHOLDER.format(label=label)
                self.assertEqual(quoted_value(f"see {token}"), repr(f"see {placeholder}"))
                self.assertEqual(listed_value(f"see {token}"), f"see {placeholder}")

    def test_a_replacement_adding_one_is_refused_by_its_shape(self):
        base = "---\nname: tdd\ndescription: Test first.\n---\n\nKeep it short.\n"
        for label, token in TOKENS:
            with self.subTest(token=token[:8]):
                problems = skill_text_defects(
                    base + f"\nexport KEY={token}\n", base_text=base, skill_id="tdd"
                )
                self.assertEqual(
                    problems,
                    [
                        f"replacement contains a credential ({label}); a model-written "
                        "token or key never becomes a reviewable candidate"
                    ],
                )

    def test_a_configured_child_environment_value_holding_one_is_named(self):
        for label, token in TOKENS:
            with self.subTest(token=token[:8]):
                found = credential_bearing_env_values({"MODEL_FLAG": token, "OTHER": "plain"})
                self.assertEqual(found, (("MODEL_FLAG", label),))

    def test_the_leak_scanner_reports_each_by_its_shape(self):
        for label, token in TOKENS:
            with self.subTest(token=token[:8]):
                found = leakscan.scan_text(f"key is {token} here", scope="test", location="t")
                self.assertEqual([f.rule_id for f in found], [label])


class ASkillHoldingOneIsRefusedTests(daily.DailyHarness):
    def test_a_skill_holding_one_is_refused_before_any_model_role_sees_it(self):
        for label, token in TOKENS:
            with self.subTest(token=token[:8]), TemporaryDirectory() as tmp:
                self.base = Path(tmp).resolve()
                config = self.env()
                path = self.base / "skills-canonical" / "tdd" / "SKILL.md"
                path.write_text(
                    path.read_text(encoding="utf-8") + f"\n- Example: export KEY={token}\n",
                    encoding="utf-8",
                )
                runner = InProcessRunner()
                receipt, _, _ = self.run_pass(config, runner=runner)
                payload = receipt.to_dict()
                self.assertEqual(payload["counts"]["candidates_proposed"], 0)
                self.assertEqual([c for c in runner.calls if c[0] in ("improver", "judge")], [])
                for sent in runner.prompts_seen():
                    self.assertNotIn(token, sent)
                refused = [s for s in payload["skills"].values() if s["status"] == "refused"]
                self.assertEqual(len(refused), 1, payload["skills"])
                self.assertIn(f"contains a credential ({label})", refused[0]["reason"])
                self.assertNotIn(token, refused[0]["reason"])


class TheControlWhatIsNoneOfThemIsLeftAloneTests(unittest.TestCase):
    def test_the_control_what_is_none_of_them_is_left_alone(self):
        for why, text in NOT_TOKENS:
            with self.subTest(why=why):
                prose = f"key is {text} here"
                self.assertEqual(redact(prose).text, prose)
                self.assertEqual(credential_labels(prose), ())
                self.assertEqual(
                    leakscan.scan_text(prose, scope="test", location="t", rules=_rules()), []
                )


if __name__ == "__main__":
    unittest.main()
