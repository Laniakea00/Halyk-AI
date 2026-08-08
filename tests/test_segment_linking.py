"""Offline tests for resolution/segment_linking.py — the secondary,
narrower document-to-scenario linking mechanism used only for documents
accounts.py's exact ACC-token match already failed to place anywhere.

No API calls, no real dataset needed — pure synthetic text.
"""

from __future__ import annotations

import unittest

from covenant_agent.resolution.segment_linking import (
    is_trustworthy_borrower_name,
    references_borrower_as_segment,
)


class ReferencesBorrowerAsSegmentTest(unittest.TestCase):
    def test_matches_the_real_confirmed_case(self) -> None:
        # The actual text (abridged) that surfaced this gap: P5's covenant
        # references Group-consolidated capex, and the real consolidated
        # report for the Group's parent (Sarybel Energy Holding JSC) names
        # the borrower as a segment, in English.
        text = (
            "Note 6 — Segment Information\n"
            "The Group's thermal generation and network services segment is "
            "conducted through Ekibastuz Power Services JSC, which operates "
            "and maintains generating plant."
        )
        self.assertTrue(references_borrower_as_segment(text, "Ekibastuz Power Services JSC"))

    def test_no_match_when_borrower_name_absent(self) -> None:
        text = "The Group's segment is conducted through Some Other Entity JSC."
        self.assertFalse(references_borrower_as_segment(text, "Ekibastuz Power Services JSC"))

    def test_no_match_when_name_present_but_no_segment_marker_nearby(self) -> None:
        # Bare mention, far from any subsidiary/segment language — not enough.
        text = "A brief unrelated mention of Ekibastuz Power Services JSC in a footnote."
        self.assertFalse(references_borrower_as_segment(text, "Ekibastuz Power Services JSC"))

    def test_does_not_match_on_the_reporting_companys_own_name(self) -> None:
        # The explicit thing this must NOT do: link a document to a scenario
        # just because the *issuer's* name sounds Group-affiliated. Sarybel
        # Energy Holding JSC is not, and must not be treated as, evidence
        # for a scenario named e.g. "Sarybel Capital LLP" by sound-alike.
        text = (
            "CONSOLIDATED ANNUAL REPORT — SARYBEL ENERGY HOLDING JSC — "
            "for the year ended 31 December 2025."
        )
        self.assertFalse(references_borrower_as_segment(text, "Sarybel Energy Holding JSC"))

    def test_whitespace_noise_in_borrower_name_is_normalized(self) -> None:
        # PDF extraction routinely breaks names across lines — accounts.py's
        # own extracted company_names carry this same noise.
        text = "The Group's segment is conducted through Ekibastuz Power Services JSC."
        noisy_name = "Ekibastuz Power\n      Services JSC"
        self.assertTrue(references_borrower_as_segment(text, noisy_name))

    def test_known_noise_name_never_matches(self) -> None:
        # "Kazakhstan JSC" is a confirmed universal false-positive source
        # (a fragment of every document's "Halyk Bank of Kazakhstan JSC"
        # letterhead) — must never be trusted as a borrower name, even when
        # a segment marker happens to sit right next to it.
        text = "The Group's segment is conducted through Kazakhstan JSC for local operations."
        self.assertFalse(references_borrower_as_segment(text, "Kazakhstan JSC"))


class IsTrustworthyBorrowerNameTest(unittest.TestCase):
    def test_known_noise_name_is_untrustworthy(self) -> None:
        self.assertFalse(is_trustworthy_borrower_name("Kazakhstan JSC"))
        self.assertFalse(is_trustworthy_borrower_name("kazakhstan jsc"))  # case-insensitive

    def test_real_name_is_trustworthy(self) -> None:
        self.assertTrue(is_trustworthy_borrower_name("Ekibastuz Power Services JSC"))

    def test_blank_name_is_untrustworthy(self) -> None:
        self.assertFalse(is_trustworthy_borrower_name(""))
        self.assertFalse(is_trustworthy_borrower_name("   "))


if __name__ == "__main__":
    unittest.main()
