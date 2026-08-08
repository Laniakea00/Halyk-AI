"""Offline tests for Block 3a's deterministic (non-LLM) pieces: compound
description splitting, fuzzy counterparty matching, date/period parsing,
reclassification linking, and related-party threshold resolution. No API
calls — these are the parts of Block 3a that don't touch an LLM at all.
"""

from __future__ import annotations

import unittest
from datetime import date

from covenant_agent.linking.categories import (
    _split_compound,
    _strip_defined_as_prefix,
    is_related_party_text,
    match_category_by_text,
)
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

    def test_elided_noun_borrows_trailing_noun_from_second_part(self) -> None:
        # Real text confirmed on the public dataset (P10's 6.1 denominator):
        # "Арендных и Коммунальных расходов" splits grammatically correctly
        # on "и", but the first part ("Арендных") is a bare adjective
        # missing the noun Russian elides after the first of two adjectives
        # sharing one trailing noun.
        parts = _split_compound("сумме Арендных и Коммунальных расходов")
        self.assertEqual(parts, ["Арендных расходов", "Коммунальных расходов"])

    def test_elided_noun_pattern_confirmed_on_a_second_real_case(self) -> None:
        # P2's 6.1 denominator: "операционных и капитальных затрат".
        parts = _split_compound("сумма операционных и капитальных затрат")
        self.assertEqual(parts, ["операционных затрат", "капитальных затрат"])

    def test_noun_borrowing_does_not_fire_when_first_part_already_has_a_noun(self) -> None:
        # P7's 6.1 numerator: "Налогов" is already a complete standalone
        # noun (not an adjective missing one) — must be left untouched.
        parts = _split_compound("сумма Налогов и Коммунальных расходов")
        self.assertEqual(parts, ["Налогов", "Коммунальных расходов"])

    def test_noun_borrowing_does_not_fire_on_multi_word_first_part(self) -> None:
        # P1's 6.1 denominator: "операционных расходов" already has its own
        # noun (two words) — must not be touched even though it starts with
        # an adjective.
        parts = _split_compound("сумме операционных расходов и арендных платежей")
        self.assertEqual(parts, ["операционных расходов", "арендных платежей"])

    def test_ebitda_defined_as_prefix_is_stripped_after_split(self) -> None:
        # Real text confirmed on the public dataset (P5's 6.1 denominator):
        # the netting split correctly separates the two concepts, but the
        # first part retained "EBITDA Заёмщика, рассчитываемая... как"
        # framing, which found zero transaction matches on every run.
        text = (
            "EBITDA Заёмщика, рассчитываемая по его собственной отчётности как "
            "Выручка за вычетом Операционных расходов"
        )
        parts = _split_compound(text)
        self.assertEqual(parts, ["Выручка", "Операционных расходов"])


class StripDefinedAsPrefixTest(unittest.TestCase):
    def test_strips_ebitda_preamble_down_to_definition(self) -> None:
        text = "EBITDA Заёмщика, рассчитываемая по его собственной отчётности как Выручка"
        self.assertEqual(_strip_defined_as_prefix(text), "Выручка")

    def test_leaves_text_without_defined_as_construct_unchanged(self) -> None:
        self.assertEqual(_strip_defined_as_prefix("Операционных расходов"), "Операционных расходов")

    def test_leaves_text_without_aggregate_metric_prefix_unchanged(self) -> None:
        # "как" appears, but not after a known aggregate-metric name — don't touch it.
        text = "оплата услуг, классифицируемых как консультационные"
        self.assertEqual(_strip_defined_as_prefix(text), text)


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
                    txn_id=None,
                    action="recategorize",
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
                    txn_id=None,
                    action="recategorize",
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
                    txn_id=None,
                    action="recategorize",
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

    def test_direct_txn_id_bypasses_counterparty_amount_join(self) -> None:
        # Confirmed on the public dataset: B4/P1's financial_notes findings
        # name the txn_id directly ("Операция TXN-B4-0026") rather than
        # counterparty+amount — that should link with full confidence,
        # no fuzzy join needed at all.
        txns = [_txn("TXN-B4-0026", 979403.89, "Turkistan Petroleum Traders LLP")]
        audit = AuditExtractionResult(
            report_reference=None,
            is_final_position=True,
            reclassifications=[
                AuditReclassification(
                    txn_id="TXN-B4-0026",
                    action="exclude_from_period",
                    counterparty_name=None,
                    amount=None,
                    transaction_date_or_period="2025-11-20",
                    original_category=None,
                    reclassified_category=None,
                    reasoning="Title/risk transfer only in January 2026",
                    source_quote="quote",
                )
            ],
        )
        linked, unmatched = link_reclassifications("X1", (("doc1", audit),), txns)
        self.assertEqual(len(unmatched), 0)
        self.assertIn("TXN-B4-0026", linked)
        self.assertEqual(linked["TXN-B4-0026"].action, "exclude_from_period")
        self.assertEqual(linked["TXN-B4-0026"].match_confidence, 1.0)
        self.assertFalse(linked["TXN-B4-0026"].was_ambiguous)

    def test_unknown_txn_id_is_unmatched(self) -> None:
        txns = [_txn("TXN-1", -100.0, "Some Vendor")]
        audit = AuditExtractionResult(
            report_reference=None,
            is_final_position=True,
            reclassifications=[
                AuditReclassification(
                    txn_id="TXN-DOES-NOT-EXIST",
                    action="exclude_from_period",
                    counterparty_name=None,
                    amount=None,
                    transaction_date_or_period=None,
                    original_category=None,
                    reclassified_category=None,
                    reasoning="test",
                    source_quote="quote",
                )
            ],
        )
        linked, unmatched = link_reclassifications("X1", (("doc1", audit),), txns)
        self.assertEqual(len(linked), 0)
        self.assertEqual(len(unmatched), 1)

    def test_no_change_finding_is_informational_only_not_linked(self) -> None:
        # Confirmed on the public dataset: P10's (7.2) — a reclassification
        # was *considered and explicitly rejected*. Must never appear in
        # the linked dict (which downstream code treats as "this
        # transaction was reclassified").
        txns = [_txn("TXN-P10-0012", -118447.52, "Some Vendor")]
        audit = AuditExtractionResult(
            report_reference=None,
            is_final_position=True,
            reclassifications=[
                AuditReclassification(
                    txn_id="TXN-P10-0012",
                    action="no_change",
                    counterparty_name=None,
                    amount=None,
                    transaction_date_or_period=None,
                    original_category=None,
                    reclassified_category=None,
                    reasoning="Reclassification considered and rejected",
                    source_quote="quote",
                )
            ],
        )
        linked, unmatched = link_reclassifications("X1", (("doc1", audit),), txns)
        self.assertEqual(linked, {})
        self.assertEqual(unmatched, [])


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
