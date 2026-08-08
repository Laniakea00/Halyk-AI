"""Offline tests for Block 2 orchestration (extract_scenario_facts).

These mock out the actual LLM calls (extract_covenants / extract_kyc_facts /
extract_audit_facts / extract_other_facts) so they run without
OPENAI_API_KEY or network access, and check the *wiring*: which document
gets routed to which extractor, and that a missing document kind degrades
to None/empty rather than raising.

Real-API coverage (does the model actually extract correctly from real
credit agreements) is a separate, manual concern — see scripts/run_extraction.py.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from covenant_agent.extraction.pipeline import extract_all_facts, extract_scenario_facts
from covenant_agent.models import (
    DocumentMetadata,
    IngestionResult,
    ParsedDocument,
    ResolvedDocument,
    ScenarioBundle,
    ScenarioFacts,
)
from covenant_agent.schemas import (
    AuditExtractionResult,
    CovenantClause,
    CovenantExtractionResult,
    KycExtractionResult,
    OtherFactsExtractionResult,
)


def _doc(doc_id: str, text: str = "some text") -> ResolvedDocument:
    parsed = ParsedDocument(
        doc_id=doc_id,
        source_path=Path(f"/fake/{doc_id}.pdf"),
        file_type="pdf",
        text=text,
        char_count=len(text),
        from_cache=False,
    )
    metadata = DocumentMetadata(
        doc_id=doc_id,
        account_tokens=(),
        matched_scenario_accounts=(),
        company_names=(),
        kind="other",
        kind_score=0,
        is_superseded=False,
        supersede_reasons=(),
        revision=None,
        dates_found=(),
        latest_date=None,
    )
    return ResolvedDocument(parsed=parsed, metadata=metadata)


class ExtractScenarioFactsTest(unittest.TestCase):
    def test_full_bundle_routes_each_kind_to_its_extractor(self) -> None:
        bundle = ScenarioBundle(
            scenario_id="X1",
            account_id="ACC-0001",
            current_documents={
                "credit_agreement": (_doc("agreement"),),
                "kyc_dossier": (_doc("kyc"),),
                "audit_report": (_doc("audit"),),
                "treasury_memo": (_doc("treasury"),),
            },
        )

        fake_covenants = CovenantExtractionResult(
            covenants=[
                CovenantClause(
                    covenant_key="6.1",
                    metric_name="Test Metric",
                    metric_type="ratio",
                    formula_description="a / b",
                    numerator_description="a",
                    denominator_description="b",
                    components=[],
                    threshold_value=1.0,
                    threshold_unit="ratio",
                    direction="max",
                    period_start=None,
                    period_end=None,
                    carve_outs=[],
                    aggregation_note=None,
                    source_quote="quote",
                )
            ]
        )
        fake_kyc = KycExtractionResult(
            related_party_threshold_pct=20.0,
            related_party_threshold_description=None,
            disclosures=[],
        )
        fake_audit = AuditExtractionResult(
            report_reference="AR-1", is_final_position=True, reclassifications=[]
        )
        fake_other = OtherFactsExtractionResult(facts=[])

        with (
            patch(
                "covenant_agent.extraction.pipeline.extract_covenants",
                return_value=fake_covenants,
            ) as m_cov,
            patch(
                "covenant_agent.extraction.pipeline.extract_kyc_facts", return_value=fake_kyc
            ) as m_kyc,
            patch(
                "covenant_agent.extraction.pipeline.extract_audit_facts",
                return_value=fake_audit,
            ) as m_audit,
            patch(
                "covenant_agent.extraction.pipeline.extract_other_facts",
                return_value=fake_other,
            ) as m_other,
        ):
            facts = extract_scenario_facts(bundle, ["6.1"])

        m_cov.assert_called_once()
        m_kyc.assert_called_once()
        m_audit.assert_called_once()
        m_other.assert_called_once()

        self.assertIs(facts.covenants, fake_covenants)
        self.assertIs(facts.kyc, fake_kyc)
        self.assertEqual(facts.audit_reports, (("audit", fake_audit),))
        self.assertEqual(facts.other_facts, (("treasury", fake_other),))

    def test_missing_kinds_degrade_to_none_and_empty_without_raising(self) -> None:
        bundle = ScenarioBundle(
            scenario_id="X2",
            account_id="ACC-0002",
            current_documents={"credit_agreement": (_doc("agreement"),)},
        )
        fake_covenants = CovenantExtractionResult(covenants=[])

        with patch(
            "covenant_agent.extraction.pipeline.extract_covenants", return_value=fake_covenants
        ):
            facts = extract_scenario_facts(bundle, ["6.1"])

        self.assertIs(facts.covenants, fake_covenants)
        self.assertIsNone(facts.kyc)
        self.assertEqual(facts.audit_reports, ())
        self.assertEqual(facts.other_facts, ())

    def test_no_credit_agreement_yields_none_covenants_without_raising(self) -> None:
        bundle = ScenarioBundle(scenario_id="X3", account_id="ACC-0003", current_documents={})
        facts = extract_scenario_facts(bundle, ["6.1"])
        self.assertIsNone(facts.covenants)
        self.assertIsNone(facts.kyc)


def _ingestion_with_scenarios(scenario_ids: list[str]) -> IngestionResult:
    return IngestionResult(
        ledger=[],
        template={"answers": {sid: {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}} for sid in scenario_ids}},
        scenarios={sid: ScenarioBundle(scenario_id=sid, account_id=f"ACC-{sid}", current_documents={}) for sid in scenario_ids},
        unmatched_documents=(),
    )


class ExtractAllFactsBatchResilienceTest(unittest.TestCase):
    """Covers code-review findings #1/#2: one scenario's unexpected
    exception must not abort the batch, and progress must be saved after
    every scenario, not once at the end.
    """

    def test_one_scenario_raising_does_not_abort_the_batch(self) -> None:
        ingestion = _ingestion_with_scenarios(["S1", "S2", "S3"])
        ok_facts = ScenarioFacts(scenario_id="ok", covenants=None, kyc=None)

        def fake_extract(bundle, keys, *, log_dir=None):
            if bundle.scenario_id == "S2":
                raise RuntimeError("simulated unexpected failure (not ExtractionError)")
            return ok_facts

        with patch(
            "covenant_agent.extraction.pipeline.extract_scenario_facts", side_effect=fake_extract
        ):
            facts, status = extract_all_facts(ingestion)

        # All three scenarios present in the result — S2 degraded, not missing.
        self.assertEqual(set(facts.keys()), {"S1", "S2", "S3"})
        self.assertIs(facts["S1"], ok_facts)
        self.assertIs(facts["S3"], ok_facts)
        self.assertIsInstance(facts["S2"], ScenarioFacts)
        self.assertIsNone(facts["S2"].covenants)

        self.assertEqual(status["S1"], "ok")
        self.assertEqual(status["S3"], "ok")
        self.assertTrue(status["S2"].startswith("FAILED"))
        self.assertIn("simulated unexpected failure", status["S2"])

    def test_progress_is_saved_after_every_scenario_not_once_at_the_end(self) -> None:
        ingestion = _ingestion_with_scenarios(["S1", "S2", "S3"])
        ok_facts = ScenarioFacts(scenario_id="ok", covenants=None, kyc=None)

        # save_scenario_facts is called with the *live* facts dict, mutated
        # in place across the loop — a mock's call_args_list captures the
        # reference, not a snapshot, so every recorded call would otherwise
        # alias the same fully-mutated dict by the time we inspect it. The
        # side_effect below records a real snapshot (via dict(...)) at the
        # moment of each call, which is the actual thing under test: that
        # save fires progressively, not once at the end.
        snapshot_sizes = []

        def fake_save(facts_by_scenario, path):
            snapshot_sizes.append(len(dict(facts_by_scenario)))

        with TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "scenario_facts.json"
            with (
                patch(
                    "covenant_agent.extraction.pipeline.extract_scenario_facts", return_value=ok_facts
                ),
                patch(
                    "covenant_agent.extraction.pipeline.save_scenario_facts", side_effect=fake_save
                ) as m_save,
            ):
                extract_all_facts(ingestion, save_path=save_path)

        # Saved once per scenario (3 scenarios -> 3 calls), not once at the end.
        self.assertEqual(m_save.call_count, 3)
        # Each call's dict grows: 1, then 2, then 3 entries.
        self.assertEqual(snapshot_sizes, [1, 2, 3])

    def test_crash_on_last_scenario_still_leaves_earlier_ones_saved(self) -> None:
        # Directly models the real incident: scenario N/N fails, scenarios
        # 1..N-1's real results must already be on disk, not lost.
        ingestion = _ingestion_with_scenarios(["S1", "S2", "S3"])
        ok_facts = ScenarioFacts(scenario_id="ok", covenants=None, kyc=None)

        def fake_extract(bundle, keys, *, log_dir=None):
            if bundle.scenario_id == "S3":
                raise RuntimeError("boom")
            return ok_facts

        saved_snapshots = []

        def fake_save(facts_by_scenario, path):
            saved_snapshots.append(set(facts_by_scenario.keys()))

        with TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "scenario_facts.json"
            with (
                patch("covenant_agent.extraction.pipeline.extract_scenario_facts", side_effect=fake_extract),
                patch("covenant_agent.extraction.pipeline.save_scenario_facts", side_effect=fake_save),
            ):
                facts, status = extract_all_facts(ingestion, save_path=save_path)

        # By the time S3 "crashed", S1 and S2 had already been persisted
        # (their snapshot appears before S3 is added).
        self.assertIn({"S1"}, saved_snapshots)
        self.assertIn({"S1", "S2"}, saved_snapshots)
        self.assertIn({"S1", "S2", "S3"}, saved_snapshots)  # S3 still gets an entry (degraded), just recorded as failed
        self.assertEqual(status["S3"][:6], "FAILED")


if __name__ == "__main__":
    unittest.main()
