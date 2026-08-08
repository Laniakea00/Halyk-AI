"""Offline tests for Block 3a orchestration (linking/pipeline.py) batch
resilience — covers code-review findings #1/#2 for the linking side: a
categorization-call failure must degrade one scenario gracefully instead
of raising, and one scenario's unexpected exception must not abort the
rest of the batch. No API calls (categorize_transactions is mocked).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from covenant_agent.linking.pipeline import link_all_scenarios, link_scenario
from covenant_agent.linking.transaction_categorization import UNCLASSIFIED
from covenant_agent.models import IngestionResult, ScenarioBundle, ScenarioFacts, Transaction


def _txn(txn_id: str, account_id: str) -> Transaction:
    return Transaction(
        txn_id=txn_id,
        date="2025-06-01",
        account_id=account_id,
        scenario_id="X",
        counterparty="Some Vendor",
        description="test",
        amount=-100.0,
        currency="USD",
    )


def _facts() -> ScenarioFacts:
    return ScenarioFacts(scenario_id="X", covenants=None, kyc=None)


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


if __name__ == "__main__":
    unittest.main()
