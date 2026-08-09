"""Regression tests for Block 1, run against the real public dataset.

These intentionally exercise the actual pipeline end-to-end (not mocks)
against agentic-bank-public/ — every case here was a real failure mode
found while validating on the real data, not a hypothetical:

* decoy sub-accounts (ACC-7801-08 etc.) must never resolve to the parent
  scenario account (ACC-7801)
* the 2024 credit agreements, explicitly watermarked as superseded, must
  never be picked as "current"
* draft/interim audit workpapers ("ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ ... НЕ ЯВЛЯЕТСЯ
  ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ АУДИТОРА") must not become "current" just because
  no competing final document happens to exist for that scenario
* a ledger row with a missing amount must survive ingestion as
  amount=None rather than crashing or being silently dropped

Run with: python -m unittest tests.test_resolution -v
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from covenant_agent.config import REPO_ROOT
from covenant_agent.resolution.pipeline import run_ingestion

# Hardcoded, not DEFAULT_DATA_DIR: this suite's assertions are specific to
# the public dataset's own scenario ids/doc ids (P1, P5, P6, ...) — it must
# keep validating against public data even after DEFAULT_DATA_DIR was
# repointed at agentic-bank-hidden/ for the final run.
DATA_DIR = REPO_ROOT / "agentic-bank-public"


@unittest.skipUnless(DATA_DIR.exists(), f"public dataset not found at {DATA_DIR}")
class ResolutionAgainstRealDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Isolated cache dir so this test never depends on (or pollutes)
        # the repo's real cache/ — but still benefits from pdftotext
        # caching across the test run itself.
        cls._cache_dir = Path(tempfile.mkdtemp(prefix="covenant_agent_test_cache_"))
        cls.result = run_ingestion(DATA_DIR, cls._cache_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._cache_dir, ignore_errors=True)

    def test_scenario_accounts_derived_from_ledger(self) -> None:
        expected = {
            "P1": "ACC-7801",
            "P6": "ACC-7806",
            "B1": "ACC-7201",
            "B4": "ACC-7204",
        }
        for scenario_id, account_id in expected.items():
            self.assertEqual(self.result.scenarios[scenario_id].account_id, account_id)

    def test_decoy_subaccounts_never_match_parent_scenario(self) -> None:
        p1 = self.result.scenarios["P1"]
        current_ids = {
            d.parsed.doc_id
            for docs in p1.current_documents.values()
            for d in docs
        }
        # These three documents only ever mention ACC-7801-08 / -02 / -05
        # (a different legal entity's sub-account), never bare ACC-7801.
        decoy_doc_ids = {"7965d4e85cb1", "8af2d7044491", "2d44bdf2437c"}
        self.assertTrue(decoy_doc_ids.isdisjoint(current_ids))
        all_ids = {d.parsed.doc_id for d in p1.all_matched_documents}
        self.assertTrue(decoy_doc_ids.isdisjoint(all_ids))

    def test_p5_group_parent_report_linked_via_segment_reference(self) -> None:
        # P5's covenant 6.1 references Group-consolidated capex "по
        # консолидированной отчётности конечной материнской компании
        # Группы" — the actual consolidated report (Sarybel Energy Holding
        # JSC, in English) never mentions ACC-7805, so accounts.py's
        # primary ACC-token match can never place it. segment_linking.py's
        # secondary pass must find it via its Note 6 "conducted through
        # Ekibastuz Power Services JSC" segment reference instead.
        p5 = self.result.scenarios["P5"]
        group_docs = p5.current_documents.get("group_financials", ())
        self.assertEqual({d.parsed.doc_id for d in group_docs}, {"a5cc1400b640"})

    def test_segment_linking_does_not_false_positive_on_other_scenarios(self) -> None:
        # The confirmed universal noise source ("Kazakhstan JSC", a
        # fragment of every document's bank letterhead) and the Sarybel
        # report's own name (not Group-affiliated with any *other*
        # scenario by anything but a superficial "sounds corporate" read)
        # must not cause a spurious link anywhere else.
        for scenario_id, bundle in self.result.scenarios.items():
            if scenario_id == "P5":
                continue
            with self.subTest(scenario_id=scenario_id):
                self.assertNotIn("group_financials", bundle.current_documents)

    def test_superseded_2024_agreement_excluded_from_current(self) -> None:
        p1 = self.result.scenarios["P1"]
        current_agreement = p1.current_documents["credit_agreement"]
        self.assertEqual(len(current_agreement), 1)
        self.assertEqual(current_agreement[0].parsed.doc_id, "8d878af064f2")

        superseded_ids = {d.parsed.doc_id for d in p1.superseded_documents}
        self.assertIn("6dd84ab9ef0e", superseded_ids)

    def test_lone_draft_audit_report_not_treated_as_current(self) -> None:
        # P3, P6, P8, P9's only audit-shaped document in the public set is
        # an interim workpaper explicitly marked non-final, with no
        # competing final report to lose a tie-break against. It must
        # still end up excluded, not accepted by default.
        for scenario_id in ("P3", "P6", "P8", "P9"):
            bundle = self.result.scenarios[scenario_id]
            self.assertNotIn(
                "audit_report",
                bundle.current_documents,
                msg=f"{scenario_id} should have no current audit_report (draft-only)",
            )

    def test_b1_audit_tie_resolves_to_the_final_report(self) -> None:
        b1 = self.result.scenarios["B1"]
        current_audit = b1.current_documents["audit_report"]
        self.assertEqual(len(current_audit), 1)
        self.assertEqual(current_audit[0].parsed.doc_id, "448b59e12768")

    def test_every_scenario_has_exactly_one_current_credit_agreement(self) -> None:
        for scenario_id, bundle in self.result.scenarios.items():
            docs = bundle.current_documents.get("credit_agreement", ())
            self.assertEqual(
                len(docs), 1, msg=f"{scenario_id} has {len(docs)} current credit agreements"
            )

    def test_dirty_ledger_rows_kept_with_none_amount(self) -> None:
        by_id = {t.txn_id: t for t in self.result.ledger}
        self.assertIn("TXN-P7-0033", by_id)
        self.assertIsNone(by_id["TXN-P7-0033"].amount)
        self.assertIn("TXN-P8-0031", by_id)
        self.assertIsNone(by_id["TXN-P8-0031"].amount)

    def test_all_pdfs_produced_non_trivial_text(self) -> None:
        # Sanity check that pdftotext extraction is working at all, allowing
        # for the one known near-empty PDF found during validation.
        empty_docs = [
            d.parsed.doc_id
            for d in self.result.unmatched_documents
            if d.parsed.file_type == "pdf" and d.parsed.char_count < 20
        ]
        self.assertLessEqual(len(empty_docs), 1)


if __name__ == "__main__":
    unittest.main()
