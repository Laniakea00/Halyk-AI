"""Offline tests for linking/categories.py's derive_category_specs /
_specs_for_clause — category-spec generation from 2a's CovenantClause
output. Covers the three category-generation fixes from the 9-cell
post-fix forensic review (all confirmed against real fullrun_a/fullrun_b
logs, not hypothetical):

- max_single_component descriptions must not leak the covenant's own
  metric_name into a component's stem-matching text (P10 6.3).
- a bare, undefined "EBITDA" numerator/denominator must expand to the
  dataset's own standard Revenue-minus-OpEx convention (P3/P7).
- _split_compound must not mistake a descriptive "и включающие..."
  continuation clause for a genuine additive/list construction (P5 6.1).
"""

from __future__ import annotations

import unittest

from covenant_agent.linking.categories import _split_compound, derive_category_specs
from covenant_agent.schemas import CovenantClause, CovenantExtractionResult


def _clause(
    covenant_key="6.1",
    metric_name="Test Metric",
    metric_type="ratio",
    numerator_description=None,
    denominator_description=None,
    formula_description="test formula",
    components=None,
    net_against_description=None,
) -> CovenantClause:
    return CovenantClause(
        covenant_key=covenant_key,
        metric_name=metric_name,
        metric_type=metric_type,
        formula_description=formula_description,
        numerator_description=numerator_description,
        denominator_description=denominator_description,
        components=components or [],
        net_against_description=net_against_description,
        threshold_value=1.0,
        threshold_unit="ratio",
        direction="min",
        period_start="2025-01-01",
        period_end="2025-12-31",
        carve_outs=[],
        aggregation_note=None,
        source_quote="quote",
    )


class MaxSingleComponentDescriptionTest(unittest.TestCase):
    def test_component_description_does_not_include_metric_name(self) -> None:
        # Real P10 6.2 shape: metric_name itself contains "выручка", which
        # must never leak into a payroll/tax component's own description.
        clause = _clause(
            covenant_key="6.2",
            metric_name="Минимальная выручка за вычетом наибольшей статьи накладных расходов",
            metric_type="max_single_component",
            components=["Расходов на оплату труда", "Налогов"],
        )
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        for spec in specs:
            self.assertNotIn("выручка", spec.description.lower())
        self.assertEqual({s.description for s in specs}, {"Расходов на оплату труда", "Налогов"})

    def test_component_label_still_preserved(self) -> None:
        clause = _clause(metric_type="max_single_component", components=["Аренда", "Коммунальные услуги"])
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        self.assertEqual([s.component_label for s in specs], ["Аренда", "Коммунальные услуги"])


class NetAgainstDescriptionSpecTest(unittest.TestCase):
    """Prompt fix B (offline post-fix review, confirmed on P10 6.2): a
    max_single_component clause can measure some other amount minus the
    largest component ("Выручка за вычетом наибольшей из величин..."),
    not the largest component alone — the schema had no way to represent
    the "other amount" before this, so formulas.py's max_single_component
    branch could only ever return the bare max().
    """

    def test_net_against_description_produces_an_extra_spec(self) -> None:
        clause = _clause(
            metric_type="max_single_component",
            components=["Расходов на оплату труда", "Налогов"],
            net_against_description="Выручка",
        )
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        net_specs = [s for s in specs if s.role == "net_against"]
        self.assertEqual(len(net_specs), 1)
        self.assertEqual(net_specs[0].description, "Выручка")
        # The two components must still be generated normally alongside it.
        self.assertEqual(len([s for s in specs if s.role == "component"]), 2)

    def test_no_net_against_description_means_no_extra_spec(self) -> None:
        clause = _clause(
            metric_type="max_single_component",
            components=["Расходов на оплату труда", "Налогов"],
        )
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        self.assertEqual([s for s in specs if s.role == "net_against"], [])


class BareEbitdaAliasTest(unittest.TestCase):
    def test_bare_ebitda_denominator_expands_to_revenue_minus_opex(self) -> None:
        # Real P3/P7 shape: denominator_description is literally "EBITDA",
        # never defined anywhere else in either source document.
        clause = _clause(denominator_description="EBITDA")
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        den_specs = [s for s in specs if s.role == "denominator"]
        self.assertEqual(len(den_specs), 2)
        descriptions = {s.description for s in den_specs}
        self.assertIn("Выручка", descriptions)
        self.assertIn("Операционных расходов", descriptions)

    def test_bare_ebitda_with_zaemshchika_suffix_also_expands(self) -> None:
        clause = _clause(denominator_description="EBITDA Заёмщика")
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        den_specs = [s for s in specs if s.role == "denominator"]
        self.assertEqual(len(den_specs), 2)

    def test_defined_ebitda_is_not_double_expanded(self) -> None:
        # P5's own real shape: EBITDA WITH an inline "как <X>" definition
        # already splits correctly via the existing netting logic — must
        # not also trigger the bare-alias expansion on top of that.
        clause = _clause(
            denominator_description=(
                "EBITDA Заёмщика, рассчитываемая по его собственной отчётности как "
                "Выручка за вычетом Операционных расходов"
            )
        )
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        den_specs = [s for s in specs if s.role == "denominator"]
        self.assertEqual(len(den_specs), 2)
        descriptions = {s.description for s in den_specs}
        self.assertIn("Выручка", descriptions)

    def test_numerator_bare_ebitda_also_expands(self) -> None:
        clause = _clause(numerator_description="EBITDA", denominator_description="Процентные расходы")
        specs = derive_category_specs(CovenantExtractionResult(covenants=[clause]))
        num_specs = [s for s in specs if s.role == "numerator"]
        self.assertEqual(len(num_specs), 2)


class SplitCompoundNettingVsDescriptiveTest(unittest.TestCase):
    def test_real_p5_text_is_not_split_on_descriptive_i(self) -> None:
        # Confirmed real bug: this "и включающие" is a defining
        # continuation clause, not a second summable concept — splitting
        # it let an unrelated opex transaction into the capex numerator.
        text = (
            "совокупные капитальные затраты Группы, определяемые по консолидированной "
            "отчётности конечной материнской компании Группы и включающие затраты всех "
            "участников Группы"
        )
        parts = _split_compound(text)
        self.assertEqual(len(parts), 1)

    def test_genuine_additive_list_still_splits(self) -> None:
        # Must not overcorrect — a real two-concept list ("rent AND
        # utilities") must still split, per the pre-existing P10 6.1 case.
        parts = _split_compound("Арендных и Коммунальных расходов")
        self.assertEqual(len(parts), 2)

    def test_genuine_netting_still_splits(self) -> None:
        parts = _split_compound("Выручка за вычетом Операционных расходов")
        self.assertEqual(len(parts), 2)


if __name__ == "__main__":
    unittest.main()
