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


class TemplateVariationRobustnessTest(unittest.TestCase):
    """Organizers confirmed (2026-08-09) the private dataset uses multiple
    financial-report templates — a title synonym must not repeat Sarybel's
    pre-fix letter-spacing failure mode (whole document kind scores 0,
    falls through to "other", invisible downstream).
    """

    def test_financial_notes_title_synonyms(self) -> None:
        for phrase in (
            "Пояснения к финансовой отчётности за 2025 год",
            "Финансовые примечания. Компания раскрывает следующую информацию.",
            "Пояснительная записка к отчётности заёмщика",
        ):
            with self.subTest(phrase=phrase):
                kind, score = classify_kind(phrase)
                self.assertEqual(kind, "financial_notes")
                self.assertGreaterEqual(score, 1)

    def test_numbered_notes_alone_classifies_as_financial_notes(self) -> None:
        # No title marker at all — a "Consolidated Financial Statements"
        # style title (real Sarybel shape) that none of the literal
        # markers would catch, but the body has the numbered-note
        # structure real financial_notes documents consistently have.
        text = (
            "Sarybel Energy Holding JSC Consolidated Financial Statements\n\n"
            "Note 1 — Basis of preparation\n...\n"
            "Note 2 — Summary of accounting policies\n...\n"
            "Note 9 — Related party transactions\n..."
        )
        kind, score = classify_kind(text)
        self.assertEqual(kind, "financial_notes")

    def test_single_numbered_note_does_not_trigger_structural_marker(self) -> None:
        # Only one numbered note — below the 2-distinct threshold, and no
        # title marker either — must stay "other", not a false positive.
        text = "Some memo. See Note 1 for details."
        kind, score = classify_kind(text)
        self.assertEqual(kind, "other")

    def test_numbered_notes_do_not_override_a_stronger_credit_agreement_match(self) -> None:
        # A credit agreement that happens to reference "Примечание 1" and
        # "Примечание 2" in passing must not get outscored into
        # financial_notes if its own markers score higher.
        text = (
            "Договор банковского займа. Заёмщик обязуется перед Кредитором. "
            "Финансовые ковенанты изложены в Приложении. "
            "См. также Примечание 1 и Примечание 2 к настоящему договору."
        )
        kind, score = classify_kind(text)
        self.assertEqual(kind, "credit_agreement")

    def test_english_credit_agreement_by_structural_phrases(self) -> None:
        # Real confirmed private-dataset gap: scenario J4's credit
        # agreement is fully English ("Borrower"/"Lender" terminology,
        # title "CREDIT AGREEMENT") — no Russian/Kazakh marker and no
        # exact "loan agreement" phrase either, previously scored 0.
        text = (
            "CONFIDENTIAL · EXECUTION COPY\nCREDIT AGREEMENT\n"
            "Senior Secured Credit Facility\n\n"
            "THIS CREDIT AGREEMENT (this \"Agreement\") is made and entered "
            "into as of 1 January 2025, by and between Altai Metals Holding "
            "B.V. (the \"Borrower\") and Halyk Bank of Kazakhstan JSC, as "
            "lender (the \"Lender\")."
        )
        kind, score = classify_kind(text)
        self.assertEqual(kind, "credit_agreement")
        self.assertGreaterEqual(score, 1)

    def test_english_credit_agreement_phrases_do_not_override_a_referencing_financial_notes_doc(
        self,
    ) -> None:
        # Regression guard: a financial_notes document that merely
        # *references* "the credit agreement" / "the Borrower" as terms
        # (real shape, J4's own financial_notes document) must not be
        # misclassified — this is why the marker phrases added for the
        # English credit_agreement gap are structural opening-formula text
        # ("this credit agreement", "senior secured credit facility"), not
        # the bare words that a reference would also contain.
        text = (
            "Примечания к финансовой отчётности\n\n"
            "Раскрытия для агрегирования ковенантов. Как определено в "
            "credit agreement, Borrower обязан раскрывать все внебалансовые "
            "обязательства."
        )
        kind, score = classify_kind(text)
        self.assertEqual(kind, "financial_notes")

    def test_other_kind_title_synonyms(self) -> None:
        cases = {
            "credit_agreement": "Кредитный договор между Банком и Заёмщиком",
            "kyc_dossier": "Идентификация клиента. Досье по проверке контрагента.",
            "audit_report": "Аудиторское заключение независимого аудитора",
            "treasury_memo": "Меморандум казначейства по итогам квартала",
        }
        for expected_kind, phrase in cases.items():
            with self.subTest(kind=expected_kind):
                kind, score = classify_kind(phrase)
                self.assertEqual(kind, expected_kind)
                self.assertGreaterEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
