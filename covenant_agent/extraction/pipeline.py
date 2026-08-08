"""Block 2 orchestration: ScenarioBundle (Block 1) -> ScenarioFacts.

One function per scenario ties together 2a (covenant extraction) and 2b
(fact extraction), and a top-level function runs it over every scenario in
an IngestionResult. A failed or missing extraction for one scenario/document
never aborts the run for the others — see the module-level docstring in
models.py: every ScenarioFacts field is designed to be safely
None/empty when something upstream didn't resolve, and Block 3 must be able
to compute *something* defensible from that rather than the whole pipeline
dying on one bad document.

`extract_all_facts` takes that one step further, at the batch level: it
catches *any* exception per scenario (not just the four inner ExtractionError
catches below, which only cover the "expected" LLM-call-failure shape) and
saves progress after every scenario, not once at the end. Both are direct
fixes for a real failure mode, not speculative hardening — see the
code-review note in the project README: a bare `openai.APIConnectionError`
once killed a run at scenario 9/12 with zero of the first 8 scenarios'
(real, paid-for) results saved anywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

from covenant_agent.extraction.cache import save_run_status, save_scenario_facts, status_path_for
from covenant_agent.extraction.covenant_extraction import extract_covenants
from covenant_agent.extraction.fact_extraction import (
    extract_audit_facts,
    extract_kyc_facts,
    extract_other_facts,
)
from covenant_agent.ingestion.template import required_covenant_keys
from covenant_agent.llm_client import ExtractionError
from covenant_agent.models import IngestionResult, ScenarioBundle, ScenarioFacts

logger = logging.getLogger(__name__)

# Kinds handled by their own dedicated extractor (2b) rather than the
# generic fallback. Anything in current_documents under any other kind key
# goes through extract_other_facts instead.
#
# financial_notes and treasury_memo route through extract_audit_facts
# (the *same* function and schema as audit_report), not the generic
# fallback — confirmed necessary on the public dataset: both document
# kinds routinely carry the same three disclosure shapes a standalone
# audit_report does (reclassifications, transaction-amount corrections,
# off-ledger facts), in the same "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ"
# section shape. See _AUDIT_FACT_KINDS below and schemas.py's
# AuditExtractionResult docstring.
_AUDIT_FACT_KINDS = ("audit_report", "financial_notes", "treasury_memo")
_DEDICATED_KINDS = {"credit_agreement", "kyc_dossier", *_AUDIT_FACT_KINDS}

STATUS_OK = "ok"


def extract_scenario_facts(
    bundle: ScenarioBundle,
    covenant_keys: list[str],
    *,
    log_dir: Path | None = None,
) -> ScenarioFacts:
    scenario_id = bundle.scenario_id

    covenants = None
    agreement_docs = bundle.current_documents.get("credit_agreement", ())
    if not agreement_docs:
        logger.error(
            "Scenario %s: no current credit_agreement — cannot extract covenant rules.",
            scenario_id,
        )
    else:
        if len(agreement_docs) > 1:
            logger.warning(
                "Scenario %s: %d current credit agreements (unresolved version tie) — "
                "using %s for covenant extraction.",
                scenario_id,
                len(agreement_docs),
                agreement_docs[0].parsed.doc_id,
            )
        try:
            covenants = extract_covenants(
                scenario_id, agreement_docs[0].parsed.text, covenant_keys, log_dir=log_dir
            )
        except ExtractionError:
            logger.exception("Scenario %s: covenant extraction call failed", scenario_id)

    kyc = None
    kyc_docs = bundle.current_documents.get("kyc_dossier", ())
    if kyc_docs:
        if len(kyc_docs) > 1:
            logger.warning(
                "Scenario %s: %d current KYC dossiers (unresolved version tie) — using %s.",
                scenario_id,
                len(kyc_docs),
                kyc_docs[0].parsed.doc_id,
            )
        try:
            kyc = extract_kyc_facts(scenario_id, kyc_docs[0].parsed.text, log_dir=log_dir)
        except ExtractionError:
            logger.exception("Scenario %s: KYC extraction call failed", scenario_id)
    else:
        logger.info(
            "Scenario %s: no current KYC dossier — related-party facts will be empty.",
            scenario_id,
        )

    audit_reports: list[tuple[str, object]] = []
    for kind in _AUDIT_FACT_KINDS:
        for doc in bundle.current_documents.get(kind, ()):
            try:
                result = extract_audit_facts(
                    scenario_id, doc.parsed.text, log_dir=log_dir, doc_id=doc.parsed.doc_id
                )
                audit_reports.append((doc.parsed.doc_id, result))
            except ExtractionError:
                logger.exception(
                    "Scenario %s: audit-fact extraction call failed for doc %s (kind=%s)",
                    scenario_id,
                    doc.parsed.doc_id,
                    kind,
                )

    other_facts: list[tuple[str, object]] = []
    for kind, docs in bundle.current_documents.items():
        if kind in _DEDICATED_KINDS:
            continue
        for doc in docs:
            try:
                result = extract_other_facts(
                    scenario_id, kind, doc.parsed.text, log_dir=log_dir, doc_id=doc.parsed.doc_id
                )
                other_facts.append((doc.parsed.doc_id, result))
            except ExtractionError:
                logger.exception(
                    "Scenario %s: other-facts extraction call failed for doc %s (kind=%s)",
                    scenario_id,
                    doc.parsed.doc_id,
                    kind,
                )

    return ScenarioFacts(
        scenario_id=scenario_id,
        covenants=covenants,
        kyc=kyc,
        audit_reports=tuple(audit_reports),
        other_facts=tuple(other_facts),
    )


def extract_all_facts(
    result: IngestionResult,
    *,
    log_dir: Path | None = None,
    save_path: Path | None = None,
) -> tuple[dict[str, ScenarioFacts], dict[str, str]]:
    """Run extract_scenario_facts for every scenario in `result`.

    Returns (facts_by_scenario, status_by_scenario) — the latter maps
    scenario_id -> "ok" or an error message, so a caller can print/log
    exactly which scenarios need a `--scenario` re-run without re-deriving
    that from log output.

    If `save_path` is given, the accumulated facts (and a sidecar status
    file — see cache.status_path_for) are written to disk after *every*
    scenario, not once at the end — the whole reason this exists. Each
    scenario is wrapped in a blanket `except Exception`, deliberately wider
    than extract_scenario_facts's own internal ExtractionError-only
    catches: those cover the "expected" shape of an LLM call failing
    cleanly, this one covers everything else (an unanticipated SDK
    exception, a bug in our own code) so it can never again take the whole
    batch down with it — see this module's docstring.
    """
    facts: dict[str, ScenarioFacts] = {}
    status: dict[str, str] = {}
    status_path = status_path_for(save_path) if save_path is not None else None

    for scenario_id, bundle in result.scenarios.items():
        keys = required_covenant_keys(result.template, scenario_id)
        logger.info("Extracting facts for scenario %s (covenants: %s)...", scenario_id, keys)
        try:
            facts[scenario_id] = extract_scenario_facts(bundle, keys, log_dir=log_dir)
            status[scenario_id] = STATUS_OK
        except Exception as exc:  # noqa: BLE001 - intentional: see docstring
            logger.exception(
                "Scenario %s: extraction failed entirely (uncaught exception) — recording "
                "empty facts for this scenario and continuing with the rest of the batch.",
                scenario_id,
            )
            facts[scenario_id] = ScenarioFacts(scenario_id=scenario_id, covenants=None, kyc=None)
            status[scenario_id] = f"FAILED: {exc!r}"

        if save_path is not None:
            save_scenario_facts(facts, save_path)
        if status_path is not None:
            save_run_status(status, status_path)

    failed = [sid for sid, s in status.items() if s != STATUS_OK]
    if failed:
        logger.warning(
            "Extraction batch finished with %d/%d scenario(s) failed: %s — re-run these "
            "individually with --scenario once the underlying issue is fixed.",
            len(failed),
            len(status),
            failed,
        )
    else:
        logger.info("Extraction batch finished: %d/%d scenario(s) ok.", len(status), len(status))

    return facts, status
