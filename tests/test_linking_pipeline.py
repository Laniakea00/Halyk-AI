"""Offline tests for Block 3a orchestration (linking/pipeline.py) batch
resilience — covers code-review findings #1/#2 for the linking side: a
categorization-call failure must degrade one scenario gracefully instead
of raising, and one scenario's unexpected exception must not abort the
rest of the batch. No API calls (categorize_transactions is mocked).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from covenant_agent.linking.pipeline import (
    _addback_amounts,
    _apply_amount_corrections,
    _collect_other_facts,
    _log_non_usd_transactions,
    link_all_scenarios,
    link_scenario,
)
from covenant_agent.linking.transaction_categorization import UNCLASSIFIED
from covenant_agent.models import IngestionResult, LinkedReclassification, ScenarioBundle, ScenarioFacts, Transaction
from covenant_agent.schemas import AuditExtractionResult, OtherFact, TransactionAmountCorrection


def _txn(
    txn_id: str, account_id: str, amount: float | None = -100.0, currency: str = "USD"
) -> Transaction:
    return Transaction(
        txn_id=txn_id,
        date="2025-06-01",
        account_id=account_id,
        scenario_id="X",
        counterparty="Some Vendor",
        description="test",
        amount=amount,
        currency=currency,
    )


def _facts() -> ScenarioFacts:
    return ScenarioFacts(scenario_id="X", covenants=None, kyc=None)


def _correction_audit(txn_id: str, corrected_amount: float) -> AuditExtractionResult:
    return AuditExtractionResult(
        report_reference=None,
        is_final_position=True,
        reclassifications=[],
        transaction_amount_corrections=[
            TransactionAmountCorrection(
                txn_id=txn_id,
                corrected_amount=corrected_amount,
                reasoning="сумма не отражена в выгрузке реестра",
                source_quote="quote",
            )
        ],
    )


class ApplyAmountCorrectionsTest(unittest.TestCase):
    """Confirmed necessary on the public dataset: P8's TXN-P8-0031 and P7's
    TXN-P7-0033 both have amount=None in the raw ledger, with the true
    value disclosed in a financial_notes/treasury_memo document.
    """

    def test_dirty_row_amount_is_patched(self) -> None:
        txns = [_txn("TXN-P8-0031", "ACC-1", amount=None)]
        audit_reports = (("doc1", _correction_audit("TXN-P8-0031", 884204.16)),)
        patched = _apply_amount_corrections("P8", txns, audit_reports)
        self.assertEqual(patched[0].amount, 884204.16)
        # Every other field is untouched.
        self.assertEqual(patched[0].txn_id, "TXN-P8-0031")
        self.assertEqual(patched[0].counterparty, "Some Vendor")

    def test_transactions_without_a_correction_are_untouched(self) -> None:
        txns = [_txn("TXN-1", "ACC-1", amount=-500.0)]
        audit_reports = (("doc1", _correction_audit("TXN-DIFFERENT", 1.0)),)
        patched = _apply_amount_corrections("X", txns, audit_reports)
        self.assertEqual(patched[0].amount, -500.0)

    def test_no_corrections_returns_the_same_list_unchanged(self) -> None:
        txns = [_txn("TXN-1", "ACC-1")]
        patched = _apply_amount_corrections("X", txns, ())
        self.assertEqual(patched, txns)

    def test_correction_for_unknown_txn_id_is_ignored_not_crashed(self) -> None:
        txns = [_txn("TXN-1", "ACC-1")]
        audit_reports = (("doc1", _correction_audit("TXN-DOES-NOT-EXIST", 1.0)),)
        patched = _apply_amount_corrections("X", txns, audit_reports)
        self.assertEqual(len(patched), 1)
        self.assertEqual(patched[0].amount, -100.0)

    def test_conflicting_corrections_keep_the_first(self) -> None:
        txns = [_txn("TXN-1", "ACC-1", amount=None)]
        audit_reports = (
            ("doc1", _correction_audit("TXN-1", 100.0)),
            ("doc2", _correction_audit("TXN-1", 200.0)),
        )
        patched = _apply_amount_corrections("X", txns, audit_reports)
        self.assertEqual(patched[0].amount, 100.0)


class LinkScenarioCategorizationFailureTest(unittest.TestCase):
    def test_categorization_failure_degrades_to_all_unclassified(self) -> None:
        txns = [_txn("T1", "ACC-1")]
        with patch(
            "covenant_agent.linking.pipeline.categorize_transactions",
            side_effect=RuntimeError("simulated LLM call failure"),
        ):
            linked = link_scenario("X", "ACC-1", txns, _facts())

        self.assertEqual(linked.txn_category, {"T1": UNCLASSIFIED})
        self.assertEqual(len(linked.transactions), 1)  # transactions themselves aren't lost


class LinkAllScenariosBatchResilienceTest(unittest.TestCase):
    def _ingestion(self, scenario_ids: list[str]) -> IngestionResult:
        return IngestionResult(
            ledger=[_txn(f"T-{sid}", f"ACC-{sid}") for sid in scenario_ids],
            template={},
            scenarios={
                sid: ScenarioBundle(scenario_id=sid, account_id=f"ACC-{sid}", current_documents={})
                for sid in scenario_ids
            },
            unmatched_documents=(),
        )

    def test_one_scenario_raising_does_not_abort_the_batch(self) -> None:
        ingestion = self._ingestion(["S1", "S2", "S3"])
        facts_by_scenario = {sid: _facts() for sid in ("S1", "S2", "S3")}

        def fake_link(scenario_id, account_id, ledger, facts, *, log_dir=None):
            if scenario_id == "S2":
                raise RuntimeError("simulated unexpected failure in link_scenario")
            return f"linked-{scenario_id}"

        with patch("covenant_agent.linking.pipeline.link_scenario", side_effect=fake_link):
            linked, status = link_all_scenarios(ingestion, facts_by_scenario)

        # All three scenarios present — S2 degraded (empty), not missing.
        self.assertEqual(set(linked.keys()), {"S1", "S2", "S3"})
        self.assertEqual(linked["S1"], "linked-S1")
        self.assertEqual(linked["S3"], "linked-S3")
        self.assertEqual(linked["S2"].transactions, [])  # degraded, empty LinkedScenarioData

        self.assertEqual(status["S1"], "ok")
        self.assertEqual(status["S3"], "ok")
        self.assertTrue(status["S2"].startswith("FAILED"))
        self.assertIn("simulated unexpected failure", status["S2"])

    def test_missing_facts_entry_is_treated_as_a_scenario_failure_not_a_crash(self) -> None:
        # facts_by_scenario deliberately missing "S2" (e.g. extraction
        # dropped it somehow) — must not crash the whole batch with a
        # raw KeyError.
        ingestion = self._ingestion(["S1", "S2"])
        facts_by_scenario = {"S1": _facts()}  # S2 missing on purpose

        with patch(
            "covenant_agent.linking.pipeline.link_scenario", return_value="linked-ok"
        ):
            linked, status = link_all_scenarios(ingestion, facts_by_scenario)

        self.assertEqual(status["S1"], "ok")
        self.assertTrue(status["S2"].startswith("FAILED"))
        self.assertIn("S2", linked)  # still present, degraded


class LogNonUsdTransactionsTest(unittest.TestCase):
    """No conversion, no silent drop — a single scenario-level summary
    warning so a non-USD transaction's exclusion from every covenant sum
    (formulas.py never converts, see README's FX scope note) is visible,
    not silent.
    """

    def test_logs_a_warning_when_non_usd_transactions_present(self) -> None:
        txns = [_txn("T1", "ACC-1", amount=-100.0, currency="EUR")]
        with self.assertLogs("covenant_agent.linking.pipeline", level="WARNING") as cm:
            _log_non_usd_transactions("X", txns)
        self.assertTrue(any("T1" in msg and "EUR" in msg for msg in cm.output))

    def test_no_log_when_everything_is_usd(self) -> None:
        txns = [_txn("T1", "ACC-1", amount=-100.0, currency="USD")]
        with self.assertNoLogs("covenant_agent.linking.pipeline", level="WARNING"):
            _log_non_usd_transactions("X", txns)


def _addback(txn_id: str) -> LinkedReclassification:
    return LinkedReclassification(
        txn_id=txn_id,
        action="addback",
        original_category="Разовые расходы на реструктуризацию",
        reclassified_category=None,
        reasoning="one-time item, auditor-approved addback",
        source_doc_id="doc1",
        match_confidence=1.0,
        was_ambiguous=False,
    )


class AddbackAmountsTest(unittest.TestCase):
    def test_returns_only_addback_transactions_amounts(self) -> None:
        recategorize = LinkedReclassification(
            txn_id="T2",
            action="recategorize",
            original_category="a",
            reclassified_category="b",
            reasoning="r",
            source_doc_id="doc1",
            match_confidence=1.0,
            was_ambiguous=False,
        )
        reclassifications = {"T1": _addback("T1"), "T2": recategorize}
        txns = [_txn("T1", "ACC-1", amount=-500.0), _txn("T2", "ACC-1", amount=-300.0)]
        self.assertEqual(_addback_amounts(reclassifications, txns), [500.0])

    def test_empty_when_no_addback_reclassifications(self) -> None:
        recategorize = LinkedReclassification(
            txn_id="T2",
            action="recategorize",
            original_category="a",
            reclassified_category="b",
            reasoning="r",
            source_doc_id="doc1",
            match_confidence=1.0,
            was_ambiguous=False,
        )
        self.assertEqual(_addback_amounts({"T2": recategorize}, [_txn("T2", "ACC-1", amount=-300.0)]), [])


class CollectOtherFactsDoubleCountGuardTest(unittest.TestCase):
    """Fix #5 (post-fix forensic review): action="addback" and other_facts
    come from the SAME extraction call over the SAME source text — nothing
    in the schema stops the same one-time item from being reported in
    both, which would double-count it (once via addback's net-logic
    cancellation, again as a standalone other_fact). Not yet observed
    live (no cached extraction has ever populated addback), but the
    schema allows it, so this is covered synthetically.
    """

    def _audit(self, *facts: OtherFact) -> tuple:
        return (
            (
                "doc1",
                AuditExtractionResult(
                    report_reference=None,
                    is_final_position=True,
                    reclassifications=[],
                    other_facts=list(facts),
                ),
            ),
        )

    def test_fact_matching_an_addback_amount_is_dropped(self) -> None:
        fact = OtherFact(
            fact_description="Разовые расходы на реструктуризацию",
            value=500.0,
            unit="usd",
            period=None,
            source_quote="quote",
        )
        with self.assertLogs("covenant_agent.linking.pipeline", level="WARNING") as cm:
            result = _collect_other_facts("X", self._audit(fact), addback_amounts=[500.0])
        self.assertEqual(result, ())
        self.assertTrue(any("double-counting" in msg for msg in cm.output))

    def test_fact_not_matching_any_addback_amount_is_kept(self) -> None:
        fact = OtherFact(
            fact_description="Обязательство по программе выходных пособий",
            value=918447.52,
            unit="usd",
            period=None,
            source_quote="quote",
        )
        result = _collect_other_facts("X", self._audit(fact), addback_amounts=[500.0])
        self.assertEqual(result, (fact,))

    def test_no_addback_amounts_keeps_all_facts_unchanged(self) -> None:
        # Regression guard: the pre-Fix-#5 behavior (no addback at all)
        # must be completely unaffected.
        fact = OtherFact(
            fact_description="anything", value=42.0, unit="usd", period=None, source_quote="quote"
        )
        result = _collect_other_facts("X", self._audit(fact), addback_amounts=[])
        self.assertEqual(result, (fact,))


if __name__ == "__main__":
    unittest.main()
