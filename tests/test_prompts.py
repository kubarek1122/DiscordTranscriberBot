"""Unit tests for type-aware summarization prompt selection and parsing.

Stdlib unittest, no third-party deps — run from the repo root:

    python -m unittest tests/test_prompts.py
"""

import unittest

from src.artifacts import extract_actions_section
from src.prompts import (
    DEFAULT_KIND,
    DISCUSSION_KINDS,
    PROMPTS,
    parse_kind,
    resolve_prompt,
)


class ParseKindTests(unittest.TestCase):
    def test_exact_known_kinds(self):
        for kind in DISCUSSION_KINDS:
            self.assertEqual(parse_kind(kind), kind)

    def test_messy_replies(self):
        self.assertEqual(parse_kind("Design"), "design")
        self.assertEqual(parse_kind('"design".'), "design")
        self.assertEqual(parse_kind("  organizational\n"), "organizational")
        self.assertEqual(parse_kind("brainstorm — bo padały pomysły"), "brainstorm")

    def test_unknown_and_empty_fall_back(self):
        self.assertEqual(parse_kind("cokolwiek"), DEFAULT_KIND)
        self.assertEqual(parse_kind(""), DEFAULT_KIND)
        self.assertEqual(parse_kind("   "), DEFAULT_KIND)


class PromptRegistryTests(unittest.TestCase):
    def test_all_kinds_present_and_distinct(self):
        self.assertEqual(set(PROMPTS), set(DISCUSSION_KINDS))
        bodies = list(PROMPTS.values())
        self.assertEqual(len(bodies), len(set(bodies)), "prompts must be distinct")

    def test_every_prompt_has_summary_and_rules(self):
        for kind, prompt in PROMPTS.items():
            self.assertIn("## Podsumowanie", prompt, kind)
            self.assertIn("skrybo zapisz", prompt, kind)  # trigger preserved

    def test_resolve_prompt_fallback(self):
        self.assertEqual(resolve_prompt(None), PROMPTS[DEFAULT_KIND])
        self.assertEqual(resolve_prompt("bogus"), PROMPTS[DEFAULT_KIND])
        self.assertEqual(resolve_prompt("design"), PROMPTS["design"])


class ActionsExtractionPerKindTests(unittest.TestCase):
    # A minimal summary fixture per kind, mirroring the section layout each
    # kind's prompt asks for. The extractor must find the actionable section.
    FIXTURES = {
        "general": "## Podsumowanie\nx\n\n## Decyzje i zadania\n- zrobić X\n",
        "brainstorm": "## Podsumowanie\nx\n\n## Pomysły\n- y\n\n## Decyzje i zadania\n- zrobić X\n",
        "organizational": (
            "## Podsumowanie\nx\n\n## Ustalenia i decyzje\n- d\n\n"
            "## Zadania\n- zrobić X\n\n## Terminy\n- piątek\n"
        ),
        "design": (
            "## Podsumowanie\nx\n\n## Omawiane podejścia\n- a\n\n"
            "## Decyzje projektowe\n- d\n\n## Otwarte pytania\n- q\n\n"
            "## Zadania\n- zrobić X\n"
        ),
    }

    def test_actions_extracted_for_every_kind(self):
        for kind, summary in self.FIXTURES.items():
            section = extract_actions_section(summary)
            self.assertTrue(section.strip(), f"empty actions for {kind}")
            self.assertIn("zrobić X", section, kind)

    def test_organizational_picks_tasks_not_decisions(self):
        section = extract_actions_section(self.FIXTURES["organizational"])
        self.assertTrue(section.lstrip().startswith("## Zadania"), section)
        self.assertNotIn("Terminy", section)  # stops at the next header


if __name__ == "__main__":
    unittest.main()
