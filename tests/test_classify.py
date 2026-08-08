"""Offline unit tests for resolution/classify.py's document-kind detection,
including the H4 (red-team) Kazakh-language markers.
"""

from __future__ import annotations

import unittest

from covenant_agent.resolution.classify import classify_kind


class ClassifyKindTest(unittest.TestCase):
    def test_credit_agreement_by_russian_markers(self) -> None:
        kind, score = classify_kind("Договор банковского займа. Заёмщик обязуется перед Кредитором...")
        self.assertEqual(kind, "credit_agreement")
        self.assertGreaterEqual(score, 1)

    def test_no_markers_is_other(self) -> None:
        kind, score = classify_kind("Служебная переписка без ключевых терминов.")
        self.assertEqual(kind, "other")
        self.assertEqual(score, 0)

    def test_credit_agreement_by_kazakh_markers(self) -> None:
        # H4: a Kazakh-only credit agreement previously scored 0 against
        # every kind (all markers were Russian/English) and fell through to
        # "other" — invisible to every downstream layer.
        kind, score = classify_kind(
            "Несиелік келісім-шарт бойынша Қарыз алушының міндеттемелері белгіленеді."
        )
        self.assertEqual(kind, "credit_agreement")
        self.assertGreaterEqual(score, 1)

    def test_financial_notes_by_kazakh_markers(self) -> None:
        kind, score = classify_kind(
            "Қаржылық есептілікке ескертпелер. 2025 жылғы табыс туралы мәліметтер келтірілген."
        )
        self.assertEqual(kind, "financial_notes")
        self.assertGreaterEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
