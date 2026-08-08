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
    _derive_capex_from_nbv_roll_forward,
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


def _txn(
    txn_id, amount, counterparty="Some Vendor", date="2025-06-01", currency="USD", description="test transaction"
) -> Transaction:
    return Transaction(
        txn_id=txn_id,
        date=date,
        account_id="ACC-0001",
        scenario_id="X1",
        counterparty=counterparty,
        description=description,
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
    net_against_description=None,
) -> CovenantClause:
    return CovenantClause(
        covenant_key=covenant_key,
        metric_name="Test Metric",
        metric_type=metric_type,
        formula_description=formula_description,
        numerator_description=numerator_description,
        denominator_description=denominator_description,
        components=components or [],
        net_against_description=net_against_description,
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

    def test_zero_numerator_now_raises_insufficient_data(self) -> None:
        # Fix 2 (offline post-fix review): reversed from the original
        # "denominator-only" design after P4 6.1 confirmed the same "0
        # matched transactions" ambiguity applies to the numerator side —
        # a real business "this is genuinely zero" and "the classifier
        # found nothing" look identical here too, and silently returning
        # 0.0 let a total numerator miss masquerade as an ordinary,
        # plausible ratio result instead of surfacing as insufficient data.
        txns = [_txn("T2", -400.0)]
        linked = _linked(
            txns, category_specs=[NUM_SPEC, DEN_SPEC], txn_category={"T2": "6.1_denominator"}
        )
        with self.assertRaises(InsufficientDataError):
            compute_metric(_clause(), linked)

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

    def test_related_party_numerator_zero_does_not_raise(self) -> None:
        # Symmetric to the denominator case above, now that Fix 2 checks
        # both sides — a related-party numerator being genuinely zero must
        # stay exempt too, not just when it happens to be the denominator.
        clause = _clause(
            numerator_description="платежи в пользу связанных сторон",
            denominator_description="operating expenses",
        )
        txns = [_txn("T1", -400.0)]
        linked = _linked(
            txns, category_specs=[DEN_SPEC], txn_category={"T1": "6.1_denominator"}
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


class NetAgainstMaxSingleComponentTest(unittest.TestCase):
    """Prompt fix B (offline post-fix review): "Выручка за вычетом
    наибольшей из величин Расходов на оплату труда и Налогов" — real P10
    6.2 numbers throughout, confirmed against the actual ground_truth.json
    value (6,000,763.63).
    """

    NET_AGAINST_SPEC = CategorySpec(key="6.2_net_against", covenant_key="6.2", role="net_against", description="Выручка")
    COMP0 = CategorySpec(key="6.2_component_0", covenant_key="6.2", role="component", description="Расходов на оплату труда")
    COMP1 = CategorySpec(key="6.2_component_1", covenant_key="6.2", role="component", description="Налогов")

    def test_nets_revenue_against_the_larger_component(self) -> None:
        txns = [
            _txn("REV", 7204882.16, description="revenue"),
            _txn("PAY", -1204118.53, description="Terminal staff payroll disbursement 2025"),
            _txn("TAX", -882447.19, description="Corporate income tax instalment 2025"),
        ]
        linked = _linked(
            txns,
            category_specs=[self.NET_AGAINST_SPEC, self.COMP0, self.COMP1],
            txn_category={"REV": "6.2_net_against", "PAY": "6.2_component_0", "TAX": "6.2_component_1"},
        )
        clause = _clause(
            covenant_key="6.2",
            metric_type="max_single_component",
            components=["Расходов на оплату труда", "Налогов"],
            net_against_description="Выручка",
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 6000763.63, places=2)  # real ground_truth.json value for P10 6.2

    def test_without_net_against_description_behaves_as_before(self) -> None:
        # No net_against_description set — must fall back to the bare
        # max(components) behavior that predates this fix, unchanged.
        txns = [_txn("PAY", -1204118.53), _txn("TAX", -882447.19)]
        linked = _linked(
            txns,
            category_specs=[self.COMP0, self.COMP1],
            txn_category={"PAY": "6.2_component_0", "TAX": "6.2_component_1"},
        )
        clause = _clause(
            covenant_key="6.2", metric_type="max_single_component", components=["Расходов на оплату труда", "Налогов"]
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 1204118.53)

    def test_net_against_zero_matches_raises_insufficient_data(self) -> None:
        txns = [_txn("PAY", -1204118.53), _txn("TAX", -882447.19)]
        linked = _linked(
            txns,
            category_specs=[self.NET_AGAINST_SPEC, self.COMP0, self.COMP1],
            txn_category={"PAY": "6.2_component_0", "TAX": "6.2_component_1"},  # nothing classified as revenue
        )
        clause = _clause(
            covenant_key="6.2",
            metric_type="max_single_component",
            components=["Расходов на оплату труда", "Налогов"],
            net_against_description="Выручка",
        )
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)


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


class CapexRollForwardDerivationTest(unittest.TestCase):
    """P5 6.1's numerator ("совокупные капитальные затраты Группы") has no
    ledger transactions and no directly-stated capex figure anywhere in its
    Group-parent financial notes — only a PP&E roll-forward note (opening
    NBV, the year's depreciation charge, closing NBV). Real wording and
    real values taken from the actual P5 audit-fact extraction log.
    """

    NUM_SPEC = CategorySpec(
        key="6.1_numerator",
        covenant_key="6.1",
        role="numerator",
        description=(
            "совокупные капитальные затраты Группы (по консолидированной отчётности "
            "конечной материнской компании Группы, включая затраты всех участников Группы)"
        ),
    )
    DEN_SPEC = CategorySpec(key="6.1_denominator", covenant_key="6.1", role="denominator", description="EBITDA")
    # aggregate_amount's own _resolve_side_sum call always uses role="amount"
    # (see compute_metric's aggregate_amount/other branch) — a distinct spec
    # from NUM_SPEC (role="numerator"), used only by the aggregate_amount
    # tests below.
    AMOUNT_SPEC = CategorySpec(key="6.1_amount", covenant_key="6.1", role="amount", description=NUM_SPEC.description)

    def test_derivation_success_is_logged(self) -> None:
        # Fix 6 (offline post-fix review): the deriver had no success log,
        # making it impossible to confirm from a live run's logs whether
        # it fired at all — confirmed as a real observability gap while
        # forensically reviewing fullrun_a/fullrun_b.
        facts = (
            _fact("Net book value at the beginning of the year", 148028989.69),
            _fact("Depreciation charge for the year", 15826229.43),
            _fact("Net book value at the end of the year", 154050122.81),
        )
        with self.assertLogs("covenant_agent.calculation.formulas", level="INFO") as cm:
            derived = _derive_capex_from_nbv_roll_forward(facts)
        self.assertIsNotNone(derived)
        self.assertTrue(any("Derived capex" in message for message in cm.output))

    def test_derives_capex_from_nbv_roll_forward_when_no_ready_figure(self) -> None:
        facts = [
            _fact("Net book value at the beginning of the year", 148028989.69),
            _fact("Depreciation charge for the year", 15826229.43),
            _fact("Net book value at the end of the year", 154050122.81),
        ]
        txns = [_txn("EBITDA", 1000.0)]
        linked = _linked(
            txns,
            category_specs=[self.NUM_SPEC, self.DEN_SPEC],
            txn_category={"EBITDA": "6.1_denominator"},
            other_facts=facts,
        )
        clause = _clause(
            covenant_key="6.1",
            numerator_description=self.NUM_SPEC.description,
            denominator_description="EBITDA",
        )
        actual = compute_metric(clause, linked)  # must not raise InsufficientDataError
        self.assertAlmostEqual(actual, 21847362.55 / 1000.0, places=2)

    def test_partial_roll_forward_is_not_derived(self) -> None:
        # Depreciation charge is missing — must not guess at two-thirds of
        # a roll-forward, so the numerator stays empty (InsufficientDataError).
        facts = [
            _fact("Net book value at the beginning of the year", 148028989.69),
            _fact("Net book value at the end of the year", 154050122.81),
        ]
        txns = [_txn("EBITDA", 1000.0)]
        linked = _linked(
            txns,
            category_specs=[self.NUM_SPEC, self.DEN_SPEC],
            txn_category={"EBITDA": "6.1_denominator"},
            other_facts=facts,
        )
        clause = _clause(
            covenant_key="6.1",
            numerator_description=self.NUM_SPEC.description,
            denominator_description="EBITDA",
        )
        # Check the aggregate_amount shape directly too (same underlying
        # side-resolution code path, cleaner to assert on in isolation).
        agg_clause = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description=self.AMOUNT_SPEC.description,
        )
        agg_linked = _linked(
            [], category_specs=[self.AMOUNT_SPEC], other_facts=facts
        )
        with self.assertRaises(InsufficientDataError):
            compute_metric(agg_clause, agg_linked)
        # Fix 2 (offline post-fix review): the ratio call now *also* raises
        # — a numerator that stayed empty (partial roll-forward correctly
        # not derived) is InsufficientDataError too, not a silent 0.0,
        # since numerator zero-counts are checked the same as denominator's.
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)

    def test_ready_capex_figure_takes_precedence_over_derivation(self) -> None:
        # A direct disclosure must win outright — not be added on top of a
        # derived figure computed from the same underlying roll-forward.
        facts = [
            _fact("Net book value at the beginning of the year", 148028989.69),
            _fact("Depreciation charge for the year", 15826229.43),
            _fact("Net book value at the end of the year", 154050122.81),
            _fact("Капитальные затраты Группы за период", 9000000.0),
        ]
        linked = _linked(
            [],
            category_specs=[self.AMOUNT_SPEC],
            other_facts=facts,
        )
        clause = _clause(
            covenant_key="6.1",
            metric_type="aggregate_amount",
            numerator_description=None,
            denominator_description=None,
            formula_description=self.AMOUNT_SPEC.description,
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 9000000.0)


class SiblingCategoryBorrowTest(unittest.TestCase):
    """Confirmed necessary on the public dataset: P5's covenant 6.1
    denominator ("Выручка") and covenant 6.2 ("совокупный объём
    поступлений по статье «Выручка»...") describe the same real-world
    revenue transactions — the categorizer assigns each transaction to
    exactly one category, so 6.2's fuller wording can "win" every real
    revenue transaction, leaving 6.1's own category with zero matches even
    though the answer sits right there under 6.2. This tests the
    code-level (no LLM) borrow that recovers it.
    """

    def test_borrows_sibling_covenants_transactions_when_own_category_is_empty(self) -> None:
        den_61 = CategorySpec(key="6.1_denominator", covenant_key="6.1", role="denominator", description="Выручка")
        amount_62 = CategorySpec(
            key="6.2_amount",
            covenant_key="6.2",
            role="amount",
            description="совокупный объём поступлений по статье «Выручка», понимаемых как суммы, отнесённые к данной статье",
        )
        txns = [_txn("NUMTXN", 2000.0), _txn("REV", 1000.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, den_61, amount_62],
            txn_category={"NUMTXN": "6.1_numerator", "REV": "6.2_amount"},  # classifier picked 6.2, not 6.1
        )
        clause = _clause(
            covenant_key="6.1", numerator_description="revenue", denominator_description="Выручка"
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 2.0)  # 2000 / 1000(borrowed) — not InsufficientDataError

    def test_netted_role_borrows_each_part_from_a_different_sibling(self) -> None:
        # Fix 5 (offline post-fix review, confirmed on P5 6.1): a *netted*
        # denominator (revenue net of opex, split into two specs) used to
        # borrow using the combined text once, finding only a revenue
        # sibling and completely missing the opex deduction. Each part
        # must now be able to borrow from a *different* sibling covenant.
        rev_part = CategorySpec(key="6.1_denominator_part0", covenant_key="6.1", role="denominator", description="Выручка")
        opex_part = CategorySpec(
            key="6.1_denominator_part1", covenant_key="6.1", role="denominator", description="Операционных расходов"
        )
        sibling_revenue = CategorySpec(
            key="6.2_amount",
            covenant_key="6.2",
            role="amount",
            description="совокупный объём поступлений по статье «Выручка», понимаемых как суммы, отнесённые к данной статье",
        )
        sibling_opex = CategorySpec(
            key="6.3_amount", covenant_key="6.3", role="amount", description="Совокупные операционные расходы Заёмщика"
        )
        txns = [_txn("NUMTXN", 2000.0), _txn("REV", 1000.0), _txn("OPEX", -300.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, rev_part, opex_part, sibling_revenue, sibling_opex],
            txn_category={"NUMTXN": "6.1_numerator", "REV": "6.2_amount", "OPEX": "6.3_amount"},
        )
        clause = _clause(
            covenant_key="6.1", numerator_description="revenue", denominator_description="EBITDA"
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 2000.0 / 700.0)  # denominator = 1000(revenue) - 300(opex) = 700

    def test_does_not_borrow_when_own_category_already_has_data(self) -> None:
        # No override/double-count: if 6.1's own denominator already found
        # something, the sibling must never be consulted at all.
        den_61 = CategorySpec(key="6.1_denominator", covenant_key="6.1", role="denominator", description="Выручка")
        amount_62 = CategorySpec(
            key="6.2_amount",
            covenant_key="6.2",
            role="amount",
            description="совокупный объём поступлений по статье «Выручка»",
        )
        txns = [_txn("NUMTXN", 1000.0), _txn("OWN", 500.0), _txn("SIBLING", 9999.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, den_61, amount_62],
            txn_category={"NUMTXN": "6.1_numerator", "OWN": "6.1_denominator", "SIBLING": "6.2_amount"},
        )
        clause = _clause(
            covenant_key="6.1", numerator_description="revenue", denominator_description="Выручка"
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 2.0)  # 1000 / 500 — sibling's 9999 never touched

    def test_does_not_borrow_from_a_different_role_of_the_same_covenant(self) -> None:
        # The denominator ("Операционные расходы") has zero matches; the
        # only other category in scope is the *numerator's own* spec
        # (same covenant, different role) — it must not be treated as a
        # valid donor even though it's the only thing around, so this must
        # still raise InsufficientDataError rather than silently borrowing
        # across roles within the same covenant.
        num_61 = CategorySpec(key="6.1_numerator", covenant_key="6.1", role="numerator", description="revenue")
        den_61 = CategorySpec(key="6.1_denominator", covenant_key="6.1", role="denominator", description="Операционные расходы")
        txns = [_txn("REV", 1000.0)]
        linked = _linked(
            txns,
            category_specs=[num_61, den_61],
            txn_category={"REV": "6.1_numerator"},  # only the numerator's own category has any data
        )
        clause = _clause(
            covenant_key="6.1", numerator_description="revenue", denominator_description="Операционные расходы"
        )
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)  # denominator empty; numerator is not a valid donor

    def test_no_matching_sibling_still_raises_insufficient_data(self) -> None:
        den_61 = CategorySpec(key="6.1_denominator", covenant_key="6.1", role="denominator", description="Выручка")
        unrelated_62 = CategorySpec(
            key="6.2_amount", covenant_key="6.2", role="amount", description="Капитальные затраты на оборудование"
        )
        txns = [_txn("CAPEX", -500.0)]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, den_61, unrelated_62],
            txn_category={"CAPEX": "6.2_amount"},
        )
        clause = _clause(
            covenant_key="6.1", numerator_description="revenue", denominator_description="Выручка"
        )
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)

    def test_max_single_component_borrows_per_component(self) -> None:
        comp0 = CategorySpec(key="6.1_component_0", covenant_key="6.1", role="component", description="Расходы на оплату труда")
        sibling = CategorySpec(
            key="6.2_amount", covenant_key="6.2", role="amount", description="совокупные расходы на оплату труда персонала"
        )
        txns = [_txn("PAY", -750.0)]
        linked = _linked(
            txns,
            category_specs=[comp0, sibling],
            txn_category={"PAY": "6.2_amount"},
        )
        clause = _clause(covenant_key="6.1", metric_type="max_single_component")
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 750.0)


class RevenueRescueTest(unittest.TestCase):
    """Idea 1 (offline audit): deterministic rescue for P8 6.3's confirmed
    failure shape — a real revenue transaction the LLM categorizer misses
    non-deterministically. Transaction wording below is taken directly from
    the real public-dataset P8 ledger (descriptions, not amounts scaled for
    readability), to validate the filter against the real decoy shapes it
    has to survive, not synthetic ones.
    """

    DEN_SPEC = CategorySpec(key="6.1_denominator", covenant_key="6.1", role="denominator", description="Выручка")

    def test_rescues_the_unique_clean_candidate_among_real_decoys(self) -> None:
        txns = [
            _txn("NUM", 2000.0, description="numerator transaction"),
            _txn(
                "P8-0015",
                7884663.19,
                counterparty="KazMunayGas Exploration JSC",
                description="Drilling services sales settlement 2025",
            ),
            _txn("P8-0038", 3505589.65, description="Rent deposit returned — Astana branch, January 2025"),
            _txn("P8-0023", 1670416.50, description="VAT refund received — December"),
            _txn("P8-0020", 1885336.06, description="Interest income on term deposit — Shymkent depot 2025"),
            _txn("P8-0013", 3031930.94, description="Marketing overbilling refund — period 04"),
        ]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, self.DEN_SPEC],
            txn_category={"NUM": "6.1_numerator"},  # everything else is unclassified
        )
        clause = _clause(covenant_key="6.1", numerator_description="revenue", denominator_description="Выручка")
        actual = compute_metric(clause, linked)  # must not raise
        self.assertAlmostEqual(actual, 2000.0 / 7884663.19)

    def test_does_not_rescue_when_two_candidates_are_ambiguous(self) -> None:
        txns = [
            _txn("NUM", 2000.0, description="numerator transaction"),
            _txn("A", 500.0, description="Equipment sales settlement"),
            _txn("B", 600.0, description="Product sales settlement"),
        ]
        linked = _linked(txns, category_specs=[NUM_SPEC, self.DEN_SPEC], txn_category={"NUM": "6.1_numerator"})
        clause = _clause(covenant_key="6.1", numerator_description="revenue", denominator_description="Выручка")
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)

    def test_does_not_rescue_when_every_candidate_is_a_decoy(self) -> None:
        txns = [
            _txn("NUM", 2000.0, description="numerator transaction"),
            _txn("A", 500.0, description="Marketing overbilling refund"),
            _txn("B", 600.0, description="Insurance broker rebate"),
        ]
        linked = _linked(txns, category_specs=[NUM_SPEC, self.DEN_SPEC], txn_category={"NUM": "6.1_numerator"})
        clause = _clause(covenant_key="6.1", numerator_description="revenue", denominator_description="Выручка")
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)

    def test_does_not_poach_a_transaction_already_classified_elsewhere(self) -> None:
        other_spec = CategorySpec(key="6.2_amount", covenant_key="6.2", role="amount", description="Прочие доходы")
        txns = [
            _txn("NUM", 2000.0, description="numerator transaction"),
            _txn("P8-0015", 7884663.19, description="Drilling services sales settlement 2025"),
        ]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, self.DEN_SPEC, other_spec],
            txn_category={"NUM": "6.1_numerator", "P8-0015": "6.2_amount"},  # already confidently classified
        )
        clause = _clause(covenant_key="6.1", numerator_description="revenue", denominator_description="Выручка")
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)

    def test_does_not_fire_for_a_non_revenue_role(self) -> None:
        opex_spec = CategorySpec(
            key="6.1_denominator", covenant_key="6.1", role="denominator", description="Операционные расходы"
        )
        txns = [
            _txn("NUM", 2000.0, description="numerator transaction"),
            _txn("P8-0015", 7884663.19, description="Drilling services sales settlement 2025"),
        ]
        linked = _linked(txns, category_specs=[NUM_SPEC, opex_spec], txn_category={"NUM": "6.1_numerator"})
        clause = _clause(
            covenant_key="6.1", numerator_description="revenue", denominator_description="Операционные расходы"
        )
        with self.assertRaises(InsufficientDataError):
            compute_metric(clause, linked)


class AddbackReclassificationTest(unittest.TestCase):
    """Idea 3 (offline audit): action="addback" — P4 6.1's "Скорректированная
    EBITDA" pattern (revenue minus opex, plus one-time items the auditors
    approve adding back). Unvalidated against a live extraction (no cached
    document has ever populated this field, since it didn't exist in the
    schema before) — synthetic tests only, per the plan agreed before
    implementation.
    """

    OPEX_SPEC = CategorySpec(
        key="6.1_denominator",
        covenant_key="6.1",
        role="denominator",
        description="Операционные расходы, включая реструктуризацию бизнес процессов",
    )

    @staticmethod
    def _addback(txn_id, original_category) -> LinkedReclassification:
        return LinkedReclassification(
            txn_id=txn_id,
            action="addback",
            original_category=original_category,
            reclassified_category=None,
            reasoning="one-time item, auditor-approved addback",
            source_doc_id="doc1",
            match_confidence=1.0,
            was_ambiguous=False,
        )

    def test_addback_cancels_a_one_time_cost_already_in_the_expense_sum(self) -> None:
        txns = [
            _txn("NUM", 1000.0, description="revenue"),
            _txn("ONEOFF", -500.0, description="Restructuring cost"),
            _txn("NORMAL", -300.0, description="normal opex"),
        ]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, self.OPEX_SPEC],
            txn_category={"NUM": "6.1_numerator", "ONEOFF": "6.1_denominator", "NORMAL": "6.1_denominator"},
            reclassifications={"ONEOFF": self._addback("ONEOFF", "Реструктуризация бизнес процессов")},
        )
        clause = _clause(
            covenant_key="6.1",
            numerator_description="revenue",
            denominator_description=self.OPEX_SPEC.description,
        )
        actual = compute_metric(clause, linked)
        # -500 (oneoff) + -300 (normal) + 500 (addback cancels oneoff) = -300 -> abs 300.
        self.assertAlmostEqual(actual, 1000.0 / 300.0)

    def test_without_the_addback_the_one_time_cost_would_count_normally(self) -> None:
        # Same ledger, no reclassification linked — sanity check that the
        # *previous* test's difference really is the addback's doing.
        txns = [
            _txn("NUM", 1000.0, description="revenue"),
            _txn("ONEOFF", -500.0, description="Restructuring cost"),
            _txn("NORMAL", -300.0, description="normal opex"),
        ]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, self.OPEX_SPEC],
            txn_category={"NUM": "6.1_numerator", "ONEOFF": "6.1_denominator", "NORMAL": "6.1_denominator"},
        )
        clause = _clause(
            covenant_key="6.1",
            numerator_description="revenue",
            denominator_description=self.OPEX_SPEC.description,
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 1000.0 / 800.0)

    def test_unmatched_addback_label_does_not_contribute(self) -> None:
        txns = [
            _txn("NUM", 1000.0, description="revenue"),
            _txn("ONEOFF", -500.0, description="Restructuring cost"),
        ]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, self.OPEX_SPEC],
            txn_category={"NUM": "6.1_numerator", "ONEOFF": "6.1_denominator"},
            reclassifications={"ONEOFF": self._addback("ONEOFF", "Совершенно не связанная формулировка")},
        )
        clause = _clause(
            covenant_key="6.1",
            numerator_description="revenue",
            denominator_description=self.OPEX_SPEC.description,
        )
        actual = compute_metric(clause, linked)
        self.assertAlmostEqual(actual, 1000.0 / 500.0)  # addback never matched, cost counts normally

    def test_reverted_addback_is_not_applied(self) -> None:
        # revert_txn_id simulates "what if this auditor finding didn't
        # happen" — same counterfactual mechanism exclude_from_period uses.
        txns = [
            _txn("NUM", 1000.0, description="revenue"),
            _txn("ONEOFF", -500.0, description="Restructuring cost"),
        ]
        linked = _linked(
            txns,
            category_specs=[NUM_SPEC, self.OPEX_SPEC],
            txn_category={"NUM": "6.1_numerator", "ONEOFF": "6.1_denominator"},
            reclassifications={"ONEOFF": self._addback("ONEOFF", "Реструктуризация бизнес процессов")},
        )
        clause = _clause(
            covenant_key="6.1",
            numerator_description="revenue",
            denominator_description=self.OPEX_SPEC.description,
        )
        actual = compute_metric(clause, linked, revert_txn_id="ONEOFF")
        self.assertAlmostEqual(actual, 1000.0 / 500.0)  # addback reverted, cost counts normally


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
