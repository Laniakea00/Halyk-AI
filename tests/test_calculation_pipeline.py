"""Offline tests for calculation/pipeline.py's calculate_covenant.

Idea 2 (offline audit): CovenantClause.carve_outs is extracted by 2a for
real covenants across the public dataset (confirmed via cache/llm_logs on
P2/P3/P4/P7/P9/P10) but has zero readers anywhere in formulas.py/
evidence.py — every carve-out condition is silently dropped at calculation
time. This does not change `actual`/`status`; it only makes the drop
visible via a WARNING log, so it stops being silent.
"""

from __future__ import annotations

import unittest

from covenant_agent.calculation.pipeline import calculate_covenant
from covenant_agent.models import LinkedScenarioData, ScenarioFacts
from covenant_agent.schemas import CovenantClause, CovenantExtractionResult


def _clause(carve_outs, metric_type="aggregate_amount") -> CovenantClause:
    return CovenantClause(
        covenant_key="6.1",
        metric_name="Test Metric",
        metric_type=metric_type,
        formula_description="выплаты в пользу связанных сторон",
        numerator_description=None,
        denominator_description=None,
        components=[],
        net_against_description=None,
        threshold_value=100.0,
        threshold_unit="usd",
        direction="max",
        period_start="2025-01-01",
        period_end="2025-12-31",
        carve_outs=carve_outs,
        aggregation_note=None,
        source_quote="quote",
    )


def _facts(clause: CovenantClause) -> ScenarioFacts:
    return ScenarioFacts(
        scenario_id="X1",
        covenants=CovenantExtractionResult(covenants=[clause]),
        kyc=None,
    )


def _linked() -> LinkedScenarioData:
    return LinkedScenarioData(
        scenario_id="X1",
        transactions=[],
        category_specs=[],
        txn_category={},
        reclassifications={},
        unmatched_reclassifications=[],
        related_parties={},
    )


class CarveOutsVisibilityTest(unittest.TestCase):
    def test_logs_a_warning_when_carve_outs_are_present(self) -> None:
        clause = _clause(["Тест применяется только при условии, что поступления превышают $4,000,000.00."])
        with self.assertLogs("covenant_agent.calculation.pipeline", level="WARNING") as cm:
            calculate_covenant("X1", "6.1", _facts(clause), _linked())
        self.assertTrue(any("carve-out" in message for message in cm.output))

    def test_does_not_log_when_carve_outs_are_empty(self) -> None:
        clause = _clause([])
        with self.assertNoLogs("covenant_agent.calculation.pipeline", level="WARNING"):
            calculate_covenant("X1", "6.1", _facts(clause), _linked())

    def test_carve_outs_do_not_change_the_computed_result(self) -> None:
        # Visibility only — the same clause with vs. without carve_outs
        # must produce an identical CovenantResult.
        without = calculate_covenant("X1", "6.1", _facts(_clause([])), _linked())
        with_carveout = calculate_covenant("X1", "6.1", _facts(_clause(["some condition"])), _linked())
        self.assertEqual(without.status, with_carveout.status)
        self.assertEqual(without.actual, with_carveout.actual)


class MetricTypeVisibilityTest(unittest.TestCase):
    """Structural-surprises audit: metric_type='other' shares formulas.py's
    generic aggregate_amount catch-all path, not a truly general handler —
    must be visible in the calculation report, not just debug logs.
    """

    def test_metric_type_is_carried_into_the_result_on_success(self) -> None:
        clause = _clause([], metric_type="aggregate_amount")
        result = calculate_covenant("X1", "6.1", _facts(clause), _linked())
        self.assertEqual(result.metric_type, "aggregate_amount")

    def test_other_metric_type_logs_a_distinct_warning(self) -> None:
        clause = _clause([], metric_type="other")
        with self.assertLogs("covenant_agent.calculation.pipeline", level="WARNING") as cm:
            result = calculate_covenant("X1", "6.1", _facts(clause), _linked())
        self.assertEqual(result.metric_type, "other")
        self.assertTrue(any("metric_type='other'" in message for message in cm.output))

    def test_aggregate_amount_does_not_trigger_the_other_type_warning(self) -> None:
        clause = _clause([], metric_type="aggregate_amount")
        with self.assertNoLogs("covenant_agent.calculation.pipeline", level="WARNING"):
            calculate_covenant("X1", "6.1", _facts(clause), _linked())

    def test_metric_type_is_carried_into_fallback_results_too(self) -> None:
        clause = _clause([], metric_type="ratio")
        result = calculate_covenant("X1", "6.1", _facts(clause), linked=None)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.metric_type, "ratio")

    def test_metric_type_is_none_when_no_clause_was_ever_resolved(self) -> None:
        result = calculate_covenant("X1", "6.1", facts=None, linked=_linked())
        self.assertTrue(result.used_fallback)
        self.assertIsNone(result.metric_type)


if __name__ == "__main__":
    unittest.main()
