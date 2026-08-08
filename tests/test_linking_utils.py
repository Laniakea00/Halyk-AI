"""Offline tests for Block 3a's deterministic (non-LLM) pieces: compound
description splitting, fuzzy counterparty matching, date/period parsing,
reclassification linking, and related-party threshold resolution. No API
calls — these are the parts of Block 3a that don't touch an LLM at all.
"""

from __future__ import annotations

import unittest
from datetime import date

from covenant_agent.linking.categories import _split_compound, is_related_party_text, match_category_by_text
from covenant_agent.linking.dates import date_in_range, parse_period
from covenant_agent.linking.fuzzy_match import match_counterparty, normalize_counterparty
from covenant_agent.linking.reclassification_linking import link_reclassifications
from covenant_agent.linking.related_parties import resolve_related_parties
from covenant_agent.models import CategorySpec, Transaction
from covenant_agent.schemas import AuditExtractionResult, AuditReclassification, KycExtractionResult, RelatedPartyDisclosure


def _txn(txn_id, amount, counterparty, date="2025-06-01", currency="USD") -> Transaction:
    return Transaction(
        txn_id=txn_id,
        date=date,
        account_id="ACC-0001",
        scenario_id="X1",
        counterparty=counterparty,
        description="test",
        amount=amount,
        currency=currency,
    )


class SplitCompoundTest(unittest.TestCase):
    def test_no_connector_returns_single_part(self) -> None:
        self.assertEqual(_split_compound("совокупные капитальные затраты"), ["совокупные капитальные затраты"])

    def test_two_way_netting_splits_in_two(self) -> None:
        parts = _split_compound("Выручка за вычетом Операционных расходов")
        self.assertEqual(len(parts), 2)

    def test_four_way_chain_splits_into_five_parts(self) -> None:
        # Real text confirmed on the public dataset (P3's 6.1 denominator).
        text = (
            "выручка минус операционные расходы, минус расходы на амортизацию, "
            "минус расходы на налоги, минус расходы на проценты"
        )
        parts = _split_compound(text)
        self.assertEqual(len(parts), 5)

    def test_mixed_minus_and_plus_chain_splits(self) -> None:
        text = "Выручка минус Операционные расходы плюс разовые статьи"
        parts = _split_compound(text)
        self.assertEqual(len(parts), 3)

    def test_runaway_splitting_is_capped(self) -> None:
        # Pathological input shouldn't loop forever or explode part count.
        text = " и ".join(f"item{i}" for i in range(20))
        parts = _split_compound(text)
        self.assertLessEqual(len(parts), 6)


class RelatedPartyTextTest(unittest.TestCase):
    def test_detects_related_party_language(self) -> None:
        self.assertTrue(is_related_party_text("платежи в пользу связанных сторон"))
        self.assertTrue(is_related_party_text("payments to related parties"))
        self.assertTrue(is_related_party_text("аффилированные лица"))

    def test_does_not_flag_unrelated_text(self) -> None:
        self.assertFalse(is_related_party_text("операционные расходы"))
        self.assertFalse(is_related_party_text(None))


class MatchCategoryByTextTest(unittest.TestCase):
    def test_matches_despite_grammatical_case_difference(self) -> None:
        # Confirmed real bug: "Процентные расходы" (nominative) must match
        # a denominator described with "Процентным расходам" (dative).
        specs = [
            CategorySpec(key="6.1_numerator", covenant_key="6.1", role="numerator", description="EBITDA revenue"),
            CategorySpec(
                key="6.1_denominator",
                covenant_key="6.1",
                role="denominator",
                description="Процентным расходам за период, определяемых на основании отчётности",
            ),
        ]
        result = match_category_by_text("Процентные расходы", specs)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].key, "6.1_denominator")

    def test_generic_stopwords_dont_cause_false_match(self) -> None:
        # "расходы" (expenses) alone is too generic to distinguish
        # "payroll expenses" from "interest expenses" — the specific word
        # ("процентные"/interest) must be what decides it.
        specs = [
            CategorySpec(key="6.2_component_0", covenant_key="6.2", role="component", description="расходы на оплату труда"),
        ]
        result = match_category_by_text("Процентные расходы", specs)
        self.assertIsNone(result)


class FuzzyMatchTest(unittest.TestCase):
    def test_normalize_strips_punctuation_and_location_suffix(self) -> None:
        self.assertEqual(normalize_counterparty("Ertis Capital, LLP"), normalize_counterparty("Ertis Capital LLP"))
        self.assertEqual(
            normalize_counterparty("Aktau Holdings LLP"), normalize_counterparty("Aktau Holdings L.L.P.")
        )
        self.assertEqual(
            normalize_counterparty("Ashford Property Co (Taraz site)"), normalize_counterparty("Ashford Property Co")
        )

    def test_match_counterparty_prefers_exact_over_fuzzy(self) -> None:
        candidates = ["Ertis Capital LLP", "Ertis Capital Group LLP"]
        result = match_counterparty("Ertis Capital, LLP", candidates)
        self.assertEqual(result.candidate, "Ertis Capital LLP")
        self.assertTrue(result.is_exact)

    def test_no_match_below_threshold(self) -> None:
        result = match_counterparty("Completely Different Entity", ["Ertis Capital LLP"])
        self.assertIsNone(result)


class DateParsingTest(unittest.TestCase):
    def test_quarter_parses_to_correct_range(self) -> None:
        period = parse_period("Q4 2025")
        self.assertEqual(period, (date(2025, 10, 1), date(2025, 12, 31)))

    def test_none_input_returns_none(self) -> None:
        self.assertIsNone(parse_period(None))

    def test_date_in_range(self) -> None:
        period = (date(2025, 10, 1), date(2025, 12, 31))
        self.assertTrue(date_in_range("2025-11-15", period))
        self.assertFalse(date_in_range("2025-05-01", period))


class LinkReclassificationsTest(unittest.TestCase):
    def test_unambiguous_link_by_counterparty_and_amount(self) -> None:
        txns = [_txn("TXN-1", -592296.10, "Irtysh Advisory Bureau")]
        audit = AuditExtractionResult(
            report_reference="AR-1",
            is_final_position=True,
            reclassifications=[
                AuditReclassification(
                    counterparty_name="Irtysh Advisory Bureau",
                    amount=592296.10,
                    transaction_date_or_period=None,
                    original_category="Консультационные услуги",
                    reclassified_category="Процентные расходы",
                    reasoning="test",
                    source_quote="quote",
                )
            ],
        )
        linked, unmatched = link_reclassifications("X1", (("doc1", audit),), txns)
        self.assertEqual(len(unmatched), 0)
        self.assertIn("TXN-1", linked)
        self.assertFalse(linked["TXN-1"].was_ambiguous)
        self.assertEqual(linked["TXN-1"].reclassified_category, "Процентные расходы")

    def test_no_matching_amount_is_unmatched(self) -> None:
        txns = [_txn("TXN-1", -100.0, "Some Vendor")]
        audit = AuditExtractionResult(
            report_reference="AR-1",
            is_final_position=True,
            reclassifications=[
                AuditReclassification(
                    counterparty_name="Some Vendor",
                    amount=999999.0,
                    transaction_date_or_period=None,
                    original_category="A",
                    reclassified_category="B",
                    reasoning="test",
                    source_quote="quote",
                )
            ],
        )
        linked, unmatched = link_reclassifications("X1", (("doc1", audit),), txns)
        self.assertEqual(len(linked), 0)
        self.assertEqual(len(unmatched), 1)

    def test_ambiguous_multiple_candidates_flagged(self) -> None:
        txns = [
            _txn("TXN-1", -100.0, "Some Vendor", date="2025-01-01"),
            _txn("TXN-2", -100.0, "Some Vendor", date="2025-06-01"),
        ]
        audit = AuditExtractionResult(
            report_reference="AR-1",
            is_final_position=True,
            reclassifications=[
                AuditReclassification(
                    counterparty_name="Some Vendor",
                    amount=100.0,
                    transaction_date_or_period=None,
                    original_category="A",
                    reclassified_category="B",
                    reasoning="test",
                    source_quote="quote",
                )
            ],
        )
        linked, unmatched = link_reclassifications("X1", (("doc1", audit),), txns)
        self.assertEqual(len(linked), 1)
        chosen = next(iter(linked.values()))
        self.assertTrue(chosen.was_ambiguous)
        self.assertEqual(chosen.txn_id, "TXN-1")  # earliest date, deterministic tie-break


class ResolveRelatedPartiesTest(unittest.TestCase):
    def test_above_threshold_is_related(self) -> None:
        kyc = KycExtractionResult(
            related_party_threshold_pct=20.0,
            related_party_threshold_description=None,
            disclosures=[
                RelatedPartyDisclosure(
                    counterparty_name="Ertis Capital, LLP",
                    ownership_or_voting_pct=31.4,
                    relationship_description="ownership",
                    explicitly_labeled_related_party=False,
                    source_quote="quote",
                )
            ],
        )
        matches = resolve_related_parties("X1", kyc, ["Ertis Capital LLP"])
        self.assertIn("Ertis Capital LLP", matches)
        self.assertTrue(matches["Ertis Capital LLP"].is_related)

    def test_below_threshold_is_not_related(self) -> None:
        kyc = KycExtractionResult(
            related_party_threshold_pct=20.0,
            related_party_threshold_description=None,
            disclosures=[
                RelatedPartyDisclosure(
                    counterparty_name="Irtysh Advisory Bureau",
                    ownership_or_voting_pct=18.6,
                    relationship_description="ownership",
                    explicitly_labeled_related_party=False,
                    source_quote="quote",
                )
            ],
        )
        matches = resolve_related_parties("X1", kyc, ["Irtysh Advisory Bureau"])
        self.assertIn("Irtysh Advisory Bureau", matches)
        self.assertFalse(matches["Irtysh Advisory Bureau"].is_related)

    def test_no_kyc_returns_empty(self) -> None:
        matches = resolve_related_parties("X1", None, ["Anyone LLP"])
        self.assertEqual(matches, {})


if __name__ == "__main__":
    unittest.main()
