"""Offline tests for Block 3b/3c (calculation/formulas.py, calculation/evidence.py).

Pure synthetic data, no LLM calls, no OPENAI_API_KEY needed — these test the
deterministic arithmetic/logic layer in isolation from categorization
non-determinism, which is exactly the split the pipeline is designed
around (see calculation/formulas.py's module docstring: this is the one
layer with no LLM call at all). Real-data validation of the categorization
step itself is a separate, manual concern (scripts/run_calculation.py
against the public dataset).
"""

from __future__ import annotations

import unittest

from covenant_agent.calculation.evidence import find_evidence
from covenant_agent.calculation.formulas import (
    InsufficientDataError,
    _quarter_number,
    _quarter_start_from_end,
    compare_to_threshold,
    compute_metric,
    in_period,
)
from covenant_agent.models import (
    CategorySpec,
    LinkedReclassification,
    LinkedScenarioData,
    RelatedPartyMatch,
    Transaction,
)
from covenant_agent.schemas import CovenantClause, OtherFact


def _txn(txn_id, amount, counterparty="Some Vendor", date="2025-06-01", currency="USD") -> Transaction:
    return Transaction(
        txn_id=txn_id,
        date=date,
        account_id="ACC-0001",
        scenario_id="X1",
        counterparty=counterparty,
        description="test transaction",
        amount=amount,
        currency=currency,
    )


def _clause(
    covenant_key="6.1",
    metric_type="ratio",
    numerator_description="revenue",
    denominator_description="operating expenses",
    formula_description="revenue over operating expenses",
    threshold_value=1.0,
    threshold_unit="ratio",
    direction="min",
    period_start="2025-01-01",
    period_end="2025-12-31",
    components=None,
) -> CovenantClause:
    return CovenantClause(
        covenant_key=covenant_key,
        metric_name="Test Metric",
        metric_type=metric_type,
        formula_description=formula_description,
        numerator_description=numerator_description,
        denominator_description=denominator_description,
        components=components or [],
        threshold_value=threshold_value,
        threshold_unit=threshold_unit,
        direction=direction,
        period_start=period_start,
        period_end=period_end,
        carve_outs=[],
        aggregation_note=None,
        source_quote="quote",
    )


def _linked(
    transactions,
    category_specs=None,
    txn_category=None,
    reclassifications=None,
    related_parties=None,
    other_facts=None,
) -> LinkedScenarioData:
    return LinkedScenarioData(
        scenario_id="X1",
        transactions=transactions,
        category_specs=category_specs or [],
        txn_category=txn_category or {},
        reclassifications=reclassifications or {},
        unmatched_reclassifications=[],
        related_parties=related_parties or {},
        other_facts=tuple(other_facts or ()),
    )


NUM_SPEC = CategorySpec(key="6.1_numerator", covenant_key="6.1", role="numerator", description="revenue")
DEN_SPEC = CategorySpec(
    key="6.1_denominator", covenant_key="6.1", role="denominator", description="operating expenses"
)


class ComputeMetricRatioTest(unittest.TestCase):
    def test_normal_ratio_computes_correctly(self) -> None:
        txns = [_txn("T1", 1000.0), _txn("T2", -400.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, DEN_SPEC],
            txn_category={"T1": "6.1_numerator", "T2": "6.1_denominator"},
        )
        actual = compute_metric(_clause(), linked)
        self.assertAlmostEqual(actual, 2.5)  # 1000 / 400

    def test_zero_denominator_transactions_raises_insufficient_data(self) -> None:
        # Numerator has data, denominator category matched nothing at all —
        # this is the exact P3/P4/P6/P9 failure mode from the public
        # dataset: the old epsilon-fallback turned this into a
        # multi-hundred-million-dollar fabricated ratio instead of
        # surfacing "we don't know".
        txns = [_txn("T1", 1000.0)]
        linked = _linked(
            txns, category_specs=[NUM_SPEC, DEN_SPEC], txn_category={"T1": "6.1_numerator"}
        )
        with self.assertRaises(InsufficientDataError):
            compute_metric(_clause(), linked)

    def test_zero_numerator_with_real_denominator_does_not_raise(self) -> None:
        # Denominator has real data; numerator being genuinely zero (no
        # matching transactions) is a legitimate "actual=0.0", not a
        # categorization failure — only the denominator side is checked.
        txns = [_txn("T2", -400.0)]
        linked = _linked(
            txns, category_specs=[NUM_SPEC, DEN_SPEC], txn_category={"T2": "6.1_denominator"}
        )
        actual = compute_metric(_clause(), linked)
        self.assertAlmostEqual(actual, 0.0)

    def test_related_party_denominator_zero_does_not_raise(self) -> None:
        # Related-party sides are exempt from the insufficient-data check —
        # "zero related-party payments this period" is a normal, plausibly
        # true business fact, not a sign the classifier found nothing.
        clause = _clause(
            numerator_description="revenue",
            denominator_description="платежи в пользу связанных сторон",
        )
        txns = [_txn("T1", 1000.0)]
        linked = _linked(
            txns, category_specs=[NUM_SPEC], txn_category={"T1": "6.1_numerator"}
        )
        actual = compute_metric(clause, linked)  # should not raise
        self.assertGreaterEqual(actual, 0.0)

    def test_unrestricted_subsidiary_numerator_zero_does_not_raise(self) -> None:
        # Task B (lightweight fix): P9's 6.1 numerator names a
        # counterparty-identity question ("Неограниченным дочерним
        # организациям") with no disclosure anywhere in the public
        # dataset — a zero here is a normal business fact, exempted from
        # InsufficientDataError the same way related-party sides are.
        clause = _clause(
            numerator_description=(
                "совокупная стоимость капитальных активов, переданных "
                "Неограниченным дочерним организациям"
            ),
            denominator_description="совокупные капитальные затраты",
        )
        txns = [_txn("T1", -1000.0)]
        linked = _linked(
            txns, category_specs=[DEN_SPEC], txn_category={"T1": "6.1_denominator"}
        )
        actual = compute_metric(clause, linked)  # should not raise
        self.assertEqual(actual, 0.0)

    def test_netted_numerator_subtracts_correctly_via_natural_sign(self) -> None:
        # revenue (positive) net of opex (already-negative outflow) should
        # subtract via plain addition, not double-subtract.
        plus_spec = CategorySpec(key="6.1_numerator_plus", covenant_key="6.1", role="numerator", description="revenue")
        minus_spec = CategorySpec(key="6.1_numerator_minus", covenant_key="6.1", role="numerator", description="opex")
        txns = [_txn("REV", 1000.0), _txn("OPEX", -300.0), _txn("DEN", -100.0)]
        linked = _linked(
            txns,
            category_specs=[plus_spec, minus_spec, DEN_SPEC],
            txn_category={"REV": "6.1_numerator_plus", "OPEX": "6.1_numerator_minus", "DEN": "6.1_denominator"},
        )
        actual = compute_metric(_clause(), linked)
        self.assertAlmostEqual(actual, 7.0)  # (1000 - 300) / 100

    def test_period_filter_excludes_out_of_period_transactions(self) -> None:
        clause = _clause(period_start="2025-10-01", period_end="2025-12-31")
        in_period_txn = _txn("Q4", 1000.0, date="2025-11-01")
        out_of_period_txn = _txn("Q1", 5000.0, date="2025-02-01")
        den_txn = _txn("DEN", -200.0, date="2025-11-15")
        linked = _linked(
            [in_period_txn, out_of_period_txn, den_txn],
            category_specs=[NUM_SPEC, DEN_SPEC],
            txn_category={"Q4": "6.1_numerator", "Q1": "6.1_numerator", "DEN": "6.1_denominator"},
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 5.0)  # only Q4's 1000, not Q1's 5000, / 200

    def test_non_usd_transactions_excluded(self) -> None:
        txns = [_txn("USD_TXN", 1000.0, currency="USD"), _txn("EUR_TXN", 5000.0, currency="EUR")]
        den = [_txn("DEN", -100.0, currency="USD")]
        linked = _linked(
            txns + den,
            category_specs=[NUM_SPEC, DEN_SPEC],
            txn_category={"USD_TXN": "6.1_numerator", "EUR_TXN": "6.1_numerator", "DEN": "6.1_denominator"},
        )
        actual = compute_metric(_clause(), linked)
        self.assertAlmostEqual(actual, 10.0)  # only the USD 1000, not the EUR 5000


class QuarterPeriodInferenceTest(unittest.TestCase):
    """Diagnosed on the public dataset: B4's covenant 6.1 measures revenue
    for "четвёртый финансовый квартал периода, оканчивающегося 2025-12-31"
    — the source text never states an explicit numeric start date, so 2a
    correctly extracts period_start=None (nothing more literal to grab),
    and without inference in_period() had no lower bound at all, silently
    summing the whole year into a Q4-only test. This is the code-side,
    deterministic fix — no LLM call involved.
    """

    def test_quarter_number_recognizes_all_four_ordinal_stems(self) -> None:
        cases = [
            ("Выручка за первый квартал", 1),
            ("Выручка за второй финансовый квартал", 2),
            ("третьего квартала периода", 3),
            ("четвёртый финансовый квартал периода, оканчивающегося 2025-12-31", 4),
            ("четвертый квартал", 4),  # е/ё spelling variant
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(_quarter_number(text), expected)

    def test_quarter_number_recognizes_digit_forms(self) -> None:
        self.assertEqual(_quarter_number("Revenue for Q4 2025"), 4)
        self.assertEqual(_quarter_number("за 4-й квартал"), 4)

    def test_quarter_number_none_when_no_quarter_mentioned(self) -> None:
        self.assertIsNone(_quarter_number("Выручка за отчётный период"))

    def test_quarter_start_matches_calendar_quarter_end(self) -> None:
        self.assertEqual(_quarter_start_from_end("2025-12-31", 4), "2025-10-01")
        self.assertEqual(_quarter_start_from_end("2025-03-31", 1), "2025-01-01")

    def test_quarter_start_none_when_end_month_disagrees_with_ordinal(self) -> None:
        # Text says Q4 but period_end is June — inconsistent, don't guess.
        self.assertIsNone(_quarter_start_from_end("2025-06-30", 4))

    def test_in_period_infers_q4_start_and_excludes_earlier_transactions(self) -> None:
        clause = _clause(
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description=(
                "Выручка за четвёртый финансовый квартал периода, оканчивающегося 2025-12-31"
            ),
            period_start=None,
            period_end="2025-12-31",
        )
        q3_txn = _txn("Q3", 1000.0, date="2025-09-30")
        q4_txn = _txn("Q4", 1000.0, date="2025-10-01")
        self.assertFalse(in_period(q3_txn, clause))
        self.assertTrue(in_period(q4_txn, clause))

    def test_in_period_does_not_override_an_explicit_period_start(self) -> None:
        clause = _clause(
            formula_description="Выручка за четвёртый квартал",
            period_start="2025-01-01",  # 2a already gave an explicit (different) bound
            period_end="2025-12-31",
        )
        early_txn = _txn("EARLY", 1000.0, date="2025-02-01")
        self.assertTrue(in_period(early_txn, clause))  # explicit period_start wins, not inferred Oct 1

    def test_in_period_unchanged_when_no_quarter_language_present(self) -> None:
        # The other 11 public-dataset scenarios: no quarter wording anywhere
        # near their period fields — behavior must stay exactly as before
        # (no lower bound when period_start is None and nothing to infer).
        clause = _clause(
            formula_description="Совокупные операционные расходы за отчётный период",
            period_start=None,
            period_end="2025-12-31",
        )
        old_txn = _txn("OLD", 1000.0, date="2025-01-05")
        self.assertTrue(in_period(old_txn, clause))


class ComputeMetricOtherTypesTest(unittest.TestCase):
    def test_max_single_component_picks_the_larger(self) -> None:
        comp0 = CategorySpec(key="6.2_component_0", covenant_key="6.2", role="component", description="payroll")
        comp1 = CategorySpec(key="6.2_component_1", covenant_key="6.2", role="component", description="utilities")
        txns = [_txn("PAY", -500.0), _txn("UTIL", -300.0)]
        linked = _linked(
            txns,
            category_specs=[comp0, comp1],
            txn_category={"PAY": "6.2_component_0", "UTIL": "6.2_component_1"},
        )
        clause = _clause(covenant_key="6.2", metric_type="max_single_component")
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 500.0)

    def test_aggregate_amount_zero_matches_raises_insufficient_data(self) -> None:
        # Extended from the ratio-only fix: 0.0 for aggregate_amount reads
        # as a perfectly plausible "no capex this year" answer instead of
        # an obviously-fabricated number, which is exactly why it's
        # dangerous — same underlying bug as the epsilon fallback, just
        # quieter. A category with real transactions genuinely summing to
        # 0.0 (e.g. equal charge/reversal) is a separate, narrower case not
        # covered by this test.
        clause = _clause(covenant_key="6.3", metric_type="aggregate_amount", formula_description="capex")
        linked = _linked([], category_specs=[])
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)

    def test_aggregate_amount_related_party_zero_does_not_raise(self) -> None:
        # Mirrors the ratio-side exemption: a related-party amount that's
        # genuinely zero is a normal business fact, not a categorization miss.
        clause = _clause(
            covenant_key="6.3",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="платежи в пользу связанных сторон",
        )
        linked = _linked([], category_specs=[])
        actual = compute_metric(clause, linked)
        self.assertEqual(actual, 0.0)

    def test_max_single_component_all_zero_raises_insufficient_data(self) -> None:
        comp0 = CategorySpec(key="6.2_component_0", covenant_key="6.2", role="component", description="payroll")
        comp1 = CategorySpec(key="6.2_component_1", covenant_key="6.2", role="component", description="utilities")
        linked = _linked([], category_specs=[comp0, comp1])
        clause = _clause(covenant_key="6.2", metric_type="max_single_component")
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)

    def test_max_single_component_partial_data_does_not_raise(self) -> None:
        # Only one of two components has data — genuinely ambiguous, not
        # the clear-cut "found nothing at all" case; left alone by design.
        comp0 = CategorySpec(key="6.2_component_0", covenant_key="6.2", role="component", description="payroll")
        comp1 = CategorySpec(key="6.2_component_1", covenant_key="6.2", role="component", description="utilities")
        txns = [_txn("PAY", -500.0)]
        linked = _linked(txns, category_specs=[comp0, comp1], txn_category={"PAY": "6.2_component_0"})
        clause = _clause(covenant_key="6.2", metric_type="max_single_component")
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 500.0)


def _fact(fact_description, value, unit="usd") -> OtherFact:
    return OtherFact(
        fact_description=fact_description, value=value, unit=unit, period=None, source_quote="quote"
    )


class OtherFactsWiringTest(unittest.TestCase):
    """Confirmed necessary on the public dataset: P8's 6.1 ("Совокупные
    обязательства по персоналу") is genuinely payroll transactions *plus*
    an off-ledger severance-program obligation disclosed in a note, never a
    ledger transaction. compute_metric must add a matching other_fact's
    value on top of the transaction-derived sum for the same role.
    """

    def test_matching_other_fact_adds_to_aggregate_amount_sum(self) -> None:
        spec = CategorySpec(
            key="6.1_amount",
            covenant_key="6.1",
            role="amount",
            description=(
                "Совокупные обязательства по персоналу означают сумму расходов на оплату "
                "труда и обязательства по программе выходных пособий, сокращения или "
                "удержания персонала"
            ),
        )
        txns = [_txn("PAY", -2418663.27)]
        linked = _linked(
            txns,
            category_specs=[spec],
            txn_category={"PAY": "6.1_amount"},
            other_facts=[_fact("обязательство по программе выходных пособий", 918447.52)],
        )
        clause = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="Совокупные обязательства по персоналу",
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 2418663.27 + 918447.52)

    def test_unmatched_other_fact_does_not_contribute(self) -> None:
        spec = CategorySpec(key="6.1_amount", covenant_key="6.1", role="amount", description="Капитальные затраты")
        txns = [_txn("PAY", -500.0)]
        linked = _linked(
            txns,
            category_specs=[spec],
            txn_category={"PAY": "6.1_amount"},
            other_facts=[_fact("Совершенно не связанный операционный показатель", 999999.0)],
        )
        clause = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="Капитальные затраты",
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 500.0)

    def test_other_fact_alone_satisfies_insufficient_data_check(self) -> None:
        # Zero transactions matched, but a real off-ledger fact exists —
        # must not raise InsufficientDataError.
        spec = CategorySpec(
            key="6.1_amount",
            covenant_key="6.1",
            role="amount",
            description="Обязательства по программе выходных пособий персонала",
        )
        linked = _linked(
            [],
            category_specs=[spec],
            other_facts=[_fact("обязательства по программе выходных пособий", 918447.52)],
        )
        clause = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="Обязательства",
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 918447.52)

    def test_fact_matching_two_covenants_only_counts_toward_the_better_match(self) -> None:
        # spec_61's description shares every stem with the fact text
        # (score 1.0); spec_62's shares half of them (score 0.5, still
        # clears CATEGORY_MATCH_THRESHOLD=0.35 on its own) — the fact must
        # go to 6.1 only, not both.
        fact_text = (
            "обязательство по программе выходных пособий сокращения удержания персонала"
        )
        spec_61 = CategorySpec(
            key="6.1_amount",
            covenant_key="6.1",
            role="amount",
            description=(
                "Совокупные обязательства по персоналу означают сумму расходов на оплату "
                "труда и обязательства по программе выходных пособий, сокращения или "
                "удержания персонала"
            ),
        )
        spec_62 = CategorySpec(
            key="6.2_amount",
            covenant_key="6.2",
            role="amount",
            description="программе обязательства персонала",
        )
        txns = [_txn("PAY61", -500.0), _txn("PAY62", -300.0)]
        linked = _linked(
            txns,
            category_specs=[spec_61, spec_62],
            txn_category={"PAY61": "6.1_amount", "PAY62": "6.2_amount"},
            other_facts=[_fact(fact_text, 918447.52)],
        )
        clause_61 = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="Совокупные обязательства по персоналу",
        )
        clause_62 = _clause(
            covenant_key="6.2",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="программе обязательства персонала",
        )
        actual_61 = compute_metric(clause_61, linked)
        actual_62 = compute_metric(clause_62, linked)
        self.assertAlmostEqual(actual_61, 500.0 + 918447.52)  # fact counted here
        self.assertAlmostEqual(actual_62, 300.0)  # not double-counted here


class MaxSingleComponentOtherFactsTest(unittest.TestCase):
    def test_fact_adds_to_its_own_matching_component_only(self) -> None:
        comp0 = CategorySpec(
            key="6.2_component_0", covenant_key="6.2", role="component", description="Расходы на оплату труда"
        )
        comp1 = CategorySpec(
            key="6.2_component_1", covenant_key="6.2", role="component", description="Коммунальные расходы"
        )
        txns = [_txn("PAY", -500.0), _txn("UTIL", -300.0)]
        linked = _linked(
            txns,
            category_specs=[comp0, comp1],
            txn_category={"PAY": "6.2_component_0", "UTIL": "6.2_component_1"},
            other_facts=[_fact("дополнительные расходы на оплату труда персонала", 1000.0)],
        )
        clause = _clause(covenant_key="6.2", metric_type="max_single_component")
        actual = compute_metric(clause, linked)
        # payroll component (500 + 1000 fact) now exceeds utilities (300).
        self.assertAlmostEqual(actual, 1500.0)

    def test_fact_alone_satisfies_insufficient_data_for_max_single_component(self) -> None:
        comp0 = CategorySpec(key="6.2_component_0", covenant_key="6.2", role="component", description="Расходы на оплату труда")
        comp1 = CategorySpec(key="6.2_component_1", covenant_key="6.2", role="component", description="Коммунальные расходы")
        linked = _linked(
            [],
            category_specs=[comp0, comp1],
            other_facts=[_fact("расходы на оплату труда персонала", 1000.0)],
        )
        clause = _clause(covenant_key="6.2", metric_type="max_single_component")
        actual = compute_metric(clause, linked)  # must not raise
        self.assertAlmostEqual(actual, 1000.0)


class ExcludeFromPeriodReclassificationTest(unittest.TestCase):
    """Confirmed necessary on the public dataset: B4's 6.1 has an auditor
    finding that TXN-B4-0026 (an "advance" cargo settlement) must be
    excluded from the 2025 covenant period entirely, regardless of its
    category — not a recategorization, a full exclusion.
    """

    def _reclass(self, txn_id: str, action: str) -> LinkedReclassification:
        return LinkedReclassification(
            txn_id=txn_id,
            action=action,
            original_category=None,
            reclassified_category=None,
            reasoning="test",
            source_doc_id="doc1",
            match_confidence=1.0,
            was_ambiguous=False,
        )

    def test_excluded_transaction_does_not_count_toward_the_sum(self) -> None:
        amount_spec = CategorySpec(key="6.1_amount", covenant_key="6.1", role="amount", description="revenue")
        txns = [_txn("REAL", 3084375.68), _txn("ADVANCE", 979403.89)]
        linked = _linked(
            txns,
            category_specs=[amount_spec],
            txn_category={"REAL": "6.1_amount", "ADVANCE": "6.1_amount"},
            reclassifications={"ADVANCE": self._reclass("ADVANCE", "exclude_from_period")},
        )
        clause = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="revenue",
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 3084375.68)  # ADVANCE excluded entirely

    def test_reverting_the_exclusion_includes_it_again(self) -> None:
        amount_spec = CategorySpec(key="6.1_amount", covenant_key="6.1", role="amount", description="revenue")
        txns = [_txn("REAL", 3084375.68), _txn("ADVANCE", 979403.89)]
        linked = _linked(
            txns,
            category_specs=[amount_spec],
            txn_category={"REAL": "6.1_amount", "ADVANCE": "6.1_amount"},
            reclassifications={"ADVANCE": self._reclass("ADVANCE", "exclude_from_period")},
        )
        clause = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="revenue",
        )
        actual = compute_metric(clause, linked, revert_txn_id="ADVANCE")
        self.assertAlmostEqual(actual, 3084375.68 + 979403.89)

    def test_exclude_from_period_also_applies_to_related_party_sum(self) -> None:
        related = RelatedPartyMatch(
            ledger_counterparty="Some Vendor",
            kyc_name="Some Vendor",
            ownership_pct=50.0,
            threshold_pct=20.0,
            is_related=True,
            basis="test",
        )
        txns = [_txn("RP1", -1000.0, counterparty="Some Vendor")]
        linked = _linked(
            txns,
            related_parties={"Some Vendor": related},
            reclassifications={"RP1": self._reclass("RP1", "exclude_from_period")},
        )
        clause = _clause(
            covenant_key="6.3",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="платежи в пользу связанных сторон",
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 0.0)


class CompareToThresholdTest(unittest.TestCase):
    def test_max_direction(self) -> None:
        self.assertEqual(compare_to_threshold(0.5, 1.0, "max"), "COMPLIANT")
        self.assertEqual(compare_to_threshold(1.5, 1.0, "max"), "BREACH")

    def test_min_direction(self) -> None:
        self.assertEqual(compare_to_threshold(1.5, 1.0, "min"), "COMPLIANT")
        self.assertEqual(compare_to_threshold(0.5, 1.0, "min"), "BREACH")


class FindEvidenceTest(unittest.TestCase):
    def test_ordinary_category_swing_is_not_reported_as_evidence(self) -> None:
        # Mirrors the real P1 case: an ordinary, never-reclassified
        # transaction that happens to be large enough to flip the verdict
        # on its own must NOT become evidence_txn_id — the case's own
        # rules explicitly disqualify "just happens to tip the sum".
        txns = [_txn("BIG_OPEX", -1000.0), _txn("REV", 1050.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, DEN_SPEC],
            txn_category={"REV": "6.1_numerator", "BIG_OPEX": "6.1_denominator"},
        )
        clause = _clause(direction="min", threshold_value=1.0)  # 1050/1000 = 1.05 -> COMPLIANT
        result = find_evidence(clause, linked)
        self.assertEqual(result.status, "COMPLIANT")
        self.assertIsNone(result.evidence_txn_id)

    def test_reclassification_reversal_is_evidence(self) -> None:
        # Mirrors B1's real 6.1: a transaction the auditor reclassified
        # into the denominator category IS legitimate evidence, because
        # undoing the reclassification (not merely excluding the row) is
        # the counterfactual, per the case's own definition.
        reclass = LinkedReclassification(
            txn_id="RECLASS",
            action="recategorize",
            original_category="consulting",
            reclassified_category="operating expenses",
            reasoning="test",
            source_doc_id="doc1",
            match_confidence=1.0,
            was_ambiguous=False,
        )
        txns = [_txn("REV", 1000.0), _txn("RECLASS", -400.0), _txn("OTHER_DEN", -50.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, DEN_SPEC],
            # Base category for RECLASS is irrelevant/unclassified; its
            # *effective* category comes from the reclassification match
            # against "operating expenses" via match_category_by_text.
            txn_category={"REV": "6.1_numerator", "OTHER_DEN": "6.1_denominator"},
            reclassifications={"RECLASS": reclass},
        )
        # With reclassification applied: denominator = 400 + 50 = 450, ratio = 1000/450 = 2.22 -> COMPLIANT (min 1.0)
        # Reverting RECLASS: denominator = 50 only, ratio = 1000/50 = 20 -> still COMPLIANT actually;
        # use a tighter threshold so reverting flips it to BREACH instead.
        clause = _clause(direction="min", threshold_value=3.0)
        # with reclass: 1000/450=2.22 -> BREACH (below 3.0)
        # without reclass (reverted): 1000/50=20 -> COMPLIANT
        result = find_evidence(clause, linked)
        self.assertEqual(result.status, "BREACH")
        self.assertEqual(result.evidence_txn_id, "RECLASS")

    def test_related_party_exclusion_is_evidence(self) -> None:
        clause = _clause(
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description="платежи в пользу связанных сторон",
            direction="max",
            threshold_value=100.0,
        )
        match = RelatedPartyMatch(
            ledger_counterparty="Related Co",
            kyc_name="Related Co",
            ownership_pct=25.0,
            threshold_pct=20.0,
            is_related=True,
            basis="25.0% >= threshold 20.0%",
        )
        txns = [_txn("PAY", -150.0, counterparty="Related Co")]
        linked = _linked(txns, related_parties={"Related Co": match})
        result = find_evidence(clause, linked)
        self.assertEqual(result.status, "BREACH")
        self.assertEqual(result.evidence_txn_id, "PAY")

    def test_insufficient_data_in_counterfactual_is_skipped_not_crashed(self) -> None:
        # Reverting the only reclassified transaction would empty the
        # denominator entirely (0 matches) — this must be treated as "not
        # a valid counterfactual" and skipped, not raise out of find_evidence.
        reclass = LinkedReclassification(
            txn_id="RECLASS",
            action="recategorize",
            original_category="consulting",
            reclassified_category="operating expenses",
            reasoning="test",
            source_doc_id="doc1",
            match_confidence=1.0,
            was_ambiguous=False,
        )
        txns = [_txn("REV", 1000.0), _txn("RECLASS", -400.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, DEN_SPEC],
            txn_category={"REV": "6.1_numerator"},
            reclassifications={"RECLASS": reclass},
        )
        clause = _clause(direction="min", threshold_value=1.0)
        # Should not raise, even though reverting RECLASS would leave the
        # denominator with 0 transactions.
        result = find_evidence(clause, linked)
        self.assertEqual(result.status, "COMPLIANT")  # 1000/400 = 2.5


if __name__ == "__main__":
    unittest.main()
