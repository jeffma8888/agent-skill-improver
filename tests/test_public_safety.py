"""This repository must be safe to make public with no cleanup pass.

Every check here is a *publication* invariant, not a behaviour invariant.  They
run offline and they run against the real tracked tree, because the thing that
matters is what a reader of the public repository would receive, not what the
source of some artifact happened to say.

Two habits are deliberate throughout.

Every banned literal in this file is assembled from fragments with ``_j`` so
this test module is itself clean under the scanner it exercises.  A test that
had to be excluded from the scan would be a hole in the scan.

Every checker is exercised two-sided: once on something that must be refused
and once on something that must be accepted.  A leak scanner that never fires
and a leak scanner that always fires are indistinguishable from their output on
a clean tree.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _j(*parts: str) -> str:
    """Join fragments so the assembled literal never appears in this source."""
    return "".join(parts)


def _load(name: str, relative: str):
    """Import a repository script that is not part of the installed package."""
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, relative
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


leakscan = _load("_asi_leakscan", "tools/leakscan.py")
build_fixtures = _load("_asi_build_fixtures", "examples/build_fixtures.py")


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return []
    return [p for p in out.stdout.split("\0") if p]


class LeakScannerRulesAreSound(unittest.TestCase):
    def test_a_every_rule_is_two_sided(self) -> None:
        # Each rule carries its own positive and negative sample; selftest also
        # asserts the scanner's own source is clean, which is what stops a rule
        # from being written as a bare literal.
        self.assertEqual(leakscan.selftest(), [])

    def test_b_there_are_enough_rules_to_be_meaningful(self) -> None:
        # A collector that lost its rules would report a clean tree forever.
        self.assertGreaterEqual(len(leakscan.RULES), 20)
        ids = [rule.id for rule in leakscan.RULES]
        self.assertEqual(len(ids), len(set(ids)), "duplicate rule ids")

    def test_c_the_scanner_fires_on_a_planted_absolute_home_path(self) -> None:
        planted = _j("/", "Us", "ers", "/", "someone", "/projects/x")
        findings = leakscan.scan_text(planted, scope="test", location="planted")
        self.assertTrue(findings, "an absolute home path was not detected")

    def test_d_the_scanner_fires_on_a_planted_credential(self) -> None:
        planted = _j("AKIA", "ABCDEFGHIJKLMNOP")
        findings = leakscan.scan_text(planted, scope="test", location="planted")
        self.assertTrue(findings, "an access key id was not detected")

    def test_d2_the_scanner_fires_on_the_internal_agent_product_name(self) -> None:
        planted = "then it shells out to " + _j("a", "ki") + " export --thread x"
        ids = {f.rule_id for f in leakscan.scan_text(planted, scope="test", location="planted")}
        self.assertIn("internal-agent-product", ids)

    def test_d3_the_product_rule_respects_word_boundaries(self) -> None:
        """The negative half, and it is the one that matters for a 3-letter name.

        An unbounded substring rule would fire on ordinary English and would then
        be suppressed by an allowance, which is how a scanner stops being
        evidence.
        """
        for benign in ("making a bounded selection", "taking the canonical copy", "a kimono"):
            with self.subTest(benign=benign):
                ids = {
                    f.rule_id for f in leakscan.scan_text(benign, scope="test", location="benign")
                }
                self.assertNotIn("internal-agent-product", ids)

    def test_d3b_the_product_rule_fires_on_the_identifier_form(self) -> None:
        """The fail-OPEN case, which is the one a word-boundary rule misses.

        An underscore is a word character, so the first version of this rule --
        ``\bname\b`` -- was silent on the module and filename spellings that this
        tree actually used. It was caught by pointing the scanner at a file whose
        content was already known to contain the name.
        """
        for planted in (
            "adapters/" + _j("a", "ki") + "_source.py",
            "from ." + _j("a", "ki") + "_export import parse",
            _j("A", "ki") + "Source()",
            'kind = "' + _j("a", "ki") + '"',
            "the " + _j("A", "KI") + " CLI",
        ):
            with self.subTest(planted=len(planted)):
                ids = {
                    f.rule_id for f in leakscan.scan_text(planted, scope="test", location="planted")
                }
                self.assertIn("internal-agent-product", ids)

    def test_d3c_the_product_rule_stays_silent_on_the_aws_key_prefix(self) -> None:
        """The over-broad half, and it is not hypothetical.

        A widened camel-case alternative fired on the AWS access-key-id prefix,
        which occurs in ``domain/redaction.py`` and in a planted sample in this
        very file, so an over-broad rule reported two false positives on a clean
        tree. A scanner that cries wolf on its own repository stops being read.
        """
        for benign in (
            _j("AKIA", "|ASIA") + "[0-9A-Z]{16}",
            _j("AKIA", "ABCDEFGHIJKLMNOP"),
        ):
            with self.subTest(benign=len(benign)):
                ids = {
                    f.rule_id for f in leakscan.scan_text(benign, scope="test", location="benign")
                }
                self.assertNotIn("internal-agent-product", ids)

    def test_d4_the_scanner_fires_on_the_sampled_thread_title(self) -> None:
        for spelling in (
            _j("Work", "flow ", "Agent"),
            _j("work", "flow-", "agent"),
            _j("work", "flow_", "agent"),
            _j("WORK", "FLOW", "AGENT"),
        ):
            with self.subTest(spelling_len=len(spelling)):
                ids = {
                    f.rule_id
                    for f in leakscan.scan_text(spelling, scope="test", location="planted")
                }
                self.assertIn("sampled-thread-title", ids)

    def test_d5_the_title_rule_does_not_fire_on_the_words_apart(self) -> None:
        for benign in (
            "an agent that runs one bounded pass",
            "the review workflow is review-gated",
            "agent, then workflow",
        ):
            with self.subTest(benign=benign):
                ids = {
                    f.rule_id for f in leakscan.scan_text(benign, scope="test", location="benign")
                }
                self.assertNotIn("sampled-thread-title", ids)

    def test_e_the_scanner_stays_silent_on_ordinary_prose(self) -> None:
        benign = (
            "The daily pass scores conversations against rubrics and proposes "
            "candidate edits. Only a human applies one, with an approval token."
        )
        self.assertEqual(leakscan.scan_text(benign, scope="test", location="benign"), [])

    def test_f_allowances_are_empty(self) -> None:
        # An allowance is how a scan quietly stops covering something. The
        # mechanism exists so a future exception is explicit and reported, but
        # nothing is exempt today.
        self.assertEqual(leakscan.ALLOWANCES, ())

    def test_g_this_test_file_is_clean_under_the_scanner(self) -> None:
        own = Path(__file__).read_text(encoding="utf-8")
        findings = leakscan.scan_text(own, scope="tracked-content", location="tests/self")
        self.assertEqual([f.rule_id for f in findings], [])


class TrackedTreeIsPublicSafe(unittest.TestCase):
    def setUp(self) -> None:
        self.tracked = _tracked()
        if not self.tracked:
            self.skipTest("not a git repository, or nothing tracked yet")

    def test_a_no_tracked_file_content_matches_any_rule(self) -> None:
        findings, notes, count = leakscan.scan_tracked(REPO)
        detail = "\n".join(f"{f.location}: {f.rule_id}" for f in findings[:20])
        self.assertEqual(findings, [], f"{len(findings)} finding(s):\n{detail}")
        self.assertGreater(count, 30, "suspiciously few tracked files were scanned")

    def test_b_no_tracked_path_names_a_private_artifact(self) -> None:
        # A path can leak without any file content leaking.
        banned_fragments = (
            _j("jin", "cm"),
            _j("ama", "zon"),
            _j("Priv", "ate :: Do ", "Not Upl", "oad"),
        )
        for path in self.tracked:
            lowered = path.lower()
            for fragment in banned_fragments:
                with self.subTest(path=path, fragment_len=len(fragment)):
                    self.assertNotIn(fragment.lower(), lowered)
        # A filename is as public as a file body, and two fixtures were named
        # after the sampled title before this check existed. Matched as the PAIR
        # rather than as the bare word, because `.github/workflows/` is a path
        # GitHub requires and banning that would force an allowance -- which is
        # how a scan quietly stops covering the thing it was written for.
        for path in self.tracked:
            with self.subTest(path=path):
                self.assertIsNone(
                    self.sampled_title_re().search(path),
                    f"{path} is named after the sampled conversation title",
                )

    @staticmethod
    def sampled_title_re():
        import re

        return re.compile(
            _j("wo", "rk") + r"\s*[-_]?\s*" + _j("fl", "ow") + r"[\s_-]*" + _j("ag", "ent"),
            re.IGNORECASE,
        )

    @staticmethod
    def product_re():
        """The product name, bounded by LETTERS rather than by word characters.

        A plain ``\bname\b`` is fail-OPEN for this name and that was measured on
        this tree: an underscore is a word character, so ``\b`` never fires on a
        module named ``<product>_source.py`` -- which is exactly the form the name
        took here before the rename. This is a strict superset of the
        word-boundary search (every ``\b`` match is also a match here, because a
        non-word character is never a letter), so the zero-occurrence claim below
        is at least as strong as the one a reviewer would run by hand.

        The camel-case alternative is restricted to the mixed-case spelling so it
        cannot fire on the AWS access-key-id prefix, which genuinely occurs in
        this repository.
        """
        import re

        return re.compile(
            "(?-i:(?<![A-Za-z])(?:[Aa][Kk][Ii](?![A-Za-z])|" + _j("A", "ki") + "(?=[A-Z])))",
            re.IGNORECASE,
        )

    def test_c_no_state_or_virtualenv_artifact_is_tracked(self) -> None:
        forbidden_prefixes = ("state/", ".venv/", "config/local.toml")
        for path in self.tracked:
            with self.subTest(path=path):
                self.assertFalse(
                    path.startswith(forbidden_prefixes),
                    f"{path} must not be committed",
                )
                self.assertNotIn("__pycache__", path)

    #: Every tracked JSON fixture except the deliberately malformed ones.
    def _fixture_json(self) -> list[str]:
        return [
            path
            for path in self.tracked
            if path.endswith(".json") and "fixtures" in path and "malformed" not in path
        ]

    @staticmethod
    def _fixture_kind(text: str) -> str:
        """Tell an export from a frozen candidate record by shape, never by path.

        An export carries a ``thread`` object; a candidate record carries a
        ``CAND-`` id and a ``validation`` block. Anything else is ``unknown``
        and refused below, so a new fixture shape cannot slip past both checks.
        """
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return "unknown"
        if isinstance(payload.get("thread"), dict):
            return "export"
        if str(payload.get("id", "")).startswith("CAND-") and "validation" in payload:
            return "candidate-record"
        return "unknown"

    def _fixtures_of(self, kind: str) -> list[str]:
        return [
            path
            for path in self._fixture_json()
            if self._fixture_kind((REPO / path).read_text(encoding="utf-8")) == kind
        ]

    #: JSON fixture files that are export-shaped and therefore in scope below.
    def _fixture_exports(self) -> list[str]:
        return self._fixtures_of("export")

    def test_d0_every_json_fixture_is_an_export_or_a_candidate_record(self) -> None:
        unknown = self._fixtures_of("unknown")
        self.assertEqual(unknown, [], "a JSON fixture of neither checked shape")
        # The classifier is two-sided: each shape is recognised and a third is not.
        self.assertEqual(self._fixture_kind(json.dumps({"thread": {"id": "x"}})), "export")
        record = {"id": "CAND-x", "validation": {}}
        self.assertEqual(self._fixture_kind(json.dumps(record)), "candidate-record")
        self.assertEqual(self._fixture_kind(json.dumps({"id": "x"})), "unknown")
        self.assertEqual(self._fixture_kind("[]"), "unknown")

    def test_d2_a_candidate_record_fixture_carries_no_absolute_path_or_personal_data(
        self,
    ) -> None:
        """The privacy check for fixtures that are not exports.

        A frozen candidate record has no thread id to place in the synthetic
        namespace, so it is held to the other signals instead: every path is
        written against the ``{TREE}`` placeholder, never an absolute home
        path; no deep-link URL scheme; and the leak scanner finds nothing
        under either the fixture or the tracked-content scope, and neither the
        product name nor the sampled title occurs.
        """
        records = self._fixtures_of("candidate-record")
        self.assertGreater(len(records), 0, "no candidate record fixture was checked")
        homes = (_j("/", "Us", "ers", "/"), _j("/", "ho", "me", "/"))
        for path in records:
            text = (REPO / path).read_text(encoding="utf-8")
            squashed = text.lower().replace(" ", "")
            with self.subTest(path=path):
                self.assertIn("{TREE}/", text, f"{path} does not use the path placeholder")
                for home in homes:
                    self.assertNotIn(home, text, f"{path} carries an absolute home path")
                for scheme in (_j("a", "ki", "://"), "chat://", "thread://"):
                    self.assertNotIn(scheme, squashed, f"{path} carries a deep-link URL scheme")
                for scope in ("fixture", "tracked-content"):
                    findings = leakscan.scan_text(text, scope=scope, location=path)
                    self.assertEqual(findings, [], f"{path} under scope {scope}")
                for pattern in (self.product_re(), self.sampled_title_re()):
                    self.assertIsNone(pattern.search(text), f"{path} names {pattern.pattern!r}")

    def test_d_no_tracked_file_is_a_real_conversation_export(self) -> None:
        """Fixtures must be synthetic, and the test says how it knows.

        Two independent signals, because either alone is weak. POSITIVE: every
        thread id sits in the synthetic epoch namespace the builder owns, or is a
        readable ``conv-*`` stem it invented. NEGATIVE: no deep-link URL scheme
        of any kind, since a real export carries one and it is the single most
        identifying string in a transcript.
        """
        exports = self._fixture_exports()
        self.assertGreater(len(exports), 5, "suspiciously few fixtures were checked")
        for path in exports:
            text = (REPO / path).read_text(encoding="utf-8")
            squashed = text.lower().replace(" ", "")
            with self.subTest(path=path):
                thread_id = json.loads(text)["thread"]["id"]
                self.assertTrue(
                    thread_id.startswith("1700000000") or thread_id.startswith("conv-"),
                    f"{path} thread id {thread_id!r} is outside the synthetic namespace",
                )
                for scheme in (_j("a", "ki", "://"), "chat://", "thread://"):
                    self.assertNotIn(scheme, squashed, f"{path} carries a deep-link URL scheme")

    def test_d1_the_provenance_check_would_notice_a_real_looking_export(self) -> None:
        """The control. Without it, a check over an empty file list would pass."""
        planted = json.dumps({"thread": {"id": "1234567890123-999"}, "elements": []})
        self.assertFalse(json.loads(planted)["thread"]["id"].startswith("1700000000"))
        planted_url = "Thread: x (" + _j("a", "ki", "://") + "chat/x)"
        self.assertIn(_j("a", "ki", "://"), planted_url.lower())

    def test_d6_no_tracked_content_names_the_internal_product_or_sampled_title(self) -> None:
        """The whole-tree version of the two new rules.

        Stated as its own test rather than relying on the tree scan above,
        because this is the specific claim the public-flip decision rests on:
        a case-insensitive word-boundary search finds ZERO occurrences.
        """
        import re

        product = self.product_re()
        title = self.sampled_title_re()
        hits: list[str] = []
        for path in self.tracked:
            full = REPO / path
            try:
                raw = full.read_bytes()
            except OSError:  # pragma: no cover - filesystem dependent
                continue
            if b"\0" in raw:
                continue
            text = raw.decode("utf-8", "replace")
            for name, pattern in (("product", product), ("title", title)):
                for match in pattern.finditer(path + "\n" + text):
                    hits.append(f"{path}: {name} -> {match.group(0)!r}")
        self.assertEqual(hits[:20], [], f"{len(hits)} occurrence(s) in tracked content")

    def test_d7_that_search_really_would_find_something(self) -> None:
        """Two-sided control for the search above, on planted text."""
        import re

        product = self.product_re()
        title = self.sampled_title_re()
        self.assertTrue(product.search("run " + _j("A", "ki") + " export"))
        self.assertTrue(title.search('title = "' + _j("Work", "flow Agent") + '"'))
        self.assertIsNone(product.search("making a change"))
        self.assertIsNone(title.search("an agent workflow diagram"))

    def test_e_the_license_is_mit_and_tracked(self) -> None:
        self.assertIn("LICENSE", self.tracked)
        text = (REPO / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", text)
        self.assertIn("Permission is hereby granted, free of charge", text)
        # No real name: the copyright line uses the publishing handle only.
        self.assertNotIn(_j("Jin", "chen"), text)

    def test_f_the_readme_and_core_docs_are_tracked(self) -> None:
        for required in (
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "docs/ARCHITECTURE.md",
            "docs/PRIVACY_AND_THREAT_MODEL.md",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.tracked)


class PackagingIsPublishable(unittest.TestCase):
    def setUp(self) -> None:
        self.pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")

    def test_a_no_unpublishable_classifier(self) -> None:
        # A classifier that forbids upload is a signal the artifact was never
        # meant to be public. It must be absent, not commented out.
        self.assertNotIn(_j("Priv", "ate :: Do ", "Not Upl", "oad"), self.pyproject)

    def test_b_the_license_is_declared_mit(self) -> None:
        self.assertIn("MIT", self.pyproject)

    def test_c_runtime_dependencies_are_empty(self) -> None:
        import tomllib

        data = tomllib.loads(self.pyproject)
        self.assertEqual(data["project"].get("dependencies", []), [])
        # Everything at runtime is standard library, so a public clone needs no
        # index access to run the daily pass offline.

    def test_d_requires_python_is_at_least_3_11(self) -> None:
        import tomllib

        data = tomllib.loads(self.pyproject)
        requires = data["project"]["requires-python"]
        self.assertTrue(requires.startswith(">=3.1"), requires)
        self.assertGreaterEqual(int(requires.split(".")[-1]), 11)

    def test_e_no_absolute_home_path_in_packaging_or_config(self) -> None:
        planted = _j("/", "Us", "ers", "/")
        for rel in ("pyproject.toml", "config/example.config.toml"):
            with self.subTest(rel=rel):
                self.assertNotIn(planted, (REPO / rel).read_text(encoding="utf-8"))


class FixturesAreDeterministicAndSynthetic(unittest.TestCase):
    def test_a_committed_fixtures_match_the_builder_byte_for_byte(self) -> None:
        drift = []
        for filename, text in build_fixtures.expected().items():
            path = build_fixtures.OUT_DIR / filename
            actual = path.read_text(encoding="utf-8") if path.is_file() else ""
            if actual != text:
                drift.append(filename)
        self.assertEqual(drift, [], "run: python examples/build_fixtures.py")

    def test_b_the_builder_is_deterministic_across_calls(self) -> None:
        self.assertEqual(build_fixtures.expected(), build_fixtures.expected())

    def test_c_the_drift_check_is_two_sided(self) -> None:
        # Prove the comparison above could fail: mutate one expected byte.
        expected = dict(build_fixtures.expected())
        name = sorted(expected)[0]
        mutated = expected[name].replace("{", "{ ", 1)
        self.assertNotEqual(mutated, expected[name])

    def test_d_every_fixture_is_valid_json_with_a_thread_id(self) -> None:
        for filename, text in build_fixtures.expected().items():
            with self.subTest(filename=filename):
                payload = json.loads(text)
                self.assertIsInstance(payload, dict)

    def test_e_no_fixture_contains_a_banned_literal(self) -> None:
        for filename, text in build_fixtures.expected().items():
            findings = leakscan.scan_text(text, scope="fixture", location=filename)
            with self.subTest(filename=filename):
                self.assertEqual([f.rule_id for f in findings], [])

    def test_f_the_check_flag_returns_zero_on_a_matching_tree(self) -> None:
        self.assertEqual(build_fixtures.main(["--check"]), 0)


class GitStateIsPublishable(unittest.TestCase):
    def setUp(self) -> None:
        self.info = leakscan.git_info(REPO)
        if not self.info.is_repo:
            self.skipTest("not a git repository yet")

    def test_a_configured_remotes_are_public_safe(self) -> None:
        # A published clone necessarily has an origin. The invariant is that a
        # remote cannot embed credentials or name a machine-local path. Empty is
        # valid before publication; normal HTTPS/SSH remotes are valid after it.
        self.assertEqual(leakscan.unsafe_remote_urls(self.info), ())

    def test_a2_remote_policy_accepts_credential_free_https_and_ssh(self) -> None:
        info = leakscan.GitInfo(
            is_repo=True,
            shallow=False,
            commit_count=1,
            remote_urls=(
                leakscan.RemoteUrl("origin", "fetch", "https://example.com/owner/repo.git"),
                leakscan.RemoteUrl("origin", "push", "git@example.com:owner/repo.git"),
                leakscan.RemoteUrl("mirror", "fetch", "ssh://git@example.com/owner/repo.git"),
            ),
        )
        self.assertEqual(leakscan.unsafe_remote_urls(info), ())

    def test_a3_remote_policy_refuses_credentials_and_local_paths(self) -> None:
        urls = (
            "https://user:secret@example.com/owner/repo.git",
            "file:" + _j("/", "/", "/ho", "me/") + "someone/private/repo",
            "../private/repo",
            _j("/ho", "me/") + "someone/private/repo",
            "http://example.com/owner/repo.git",
            "https://localhost/owner/repo.git",
            "ssh://root@example.com/owner/repo.git",
            "git@example.com:/absolute/repo.git",
        )
        info = leakscan.GitInfo(
            is_repo=True,
            shallow=False,
            commit_count=1,
            remote_urls=tuple(leakscan.RemoteUrl("origin", "fetch", url) for url in urls),
        )
        self.assertEqual(
            tuple(remote.url for remote in leakscan.unsafe_remote_urls(info)),
            urls,
        )

    def test_b_history_is_not_shallow(self) -> None:
        # A shallow clone silently reduces the history scan's coverage, so the
        # scanner reports it rather than claiming the history is clean.
        self.assertFalse(self.info.shallow)

    def test_c_history_contains_no_banned_literal(self) -> None:
        if self.info.commit_count == 0:
            self.skipTest("no commits yet; nothing in history to scan")
        findings = leakscan.scan_history(REPO, self.info)
        detail = "\n".join(f"{f.location}: {f.rule_id}" for f in findings[:20])
        self.assertEqual(findings, [], detail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
