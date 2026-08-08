"""Block 1 orchestration: documents/ + ledger + template -> IngestionResult.

Nothing in here is scenario-specific. The set of "known accounts" it
resolves documents against is derived entirely from the template and the
ledger at run time (see ingestion/ledger.py:derive_scenario_accounts) —
pointing this at the private dataset on Aug 9 requires no code changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from covenant_agent.config import (
    DOCUMENTS_DIRNAME,
    LEDGER_FILENAME,
    TEMPLATE_FILENAME,
)
from covenant_agent.ingestion.documents import load_documents
from covenant_agent.ingestion.ledger import derive_scenario_accounts, load_ledger
from covenant_agent.ingestion.template import load_template, required_scenario_ids
from covenant_agent.models import (
    DocumentMetadata,
    IngestionResult,
    ParsedDocument,
    ResolvedDocument,
    ScenarioBundle,
)
from covenant_agent.resolution.accounts import (
    extract_account_tokens,
    extract_company_names,
    match_known_accounts,
)
from covenant_agent.resolution.classify import KIND_MARKERS, classify_kind
from covenant_agent.resolution.versioning import (
    detect_supersede,
    extract_dates,
    extract_revision,
    resolve_current_documents,
)

logger = logging.getLogger(__name__)


def _build_metadata(doc: ParsedDocument, known_account_ids: set[str]) -> DocumentMetadata:
    tokens = extract_account_tokens(doc.text)
    matched = match_known_accounts(tokens, known_account_ids)
    kind, kind_score = classify_kind(doc.text)
    is_superseded, reasons = detect_supersede(doc.text)
    dates = extract_dates(doc.text)
    return DocumentMetadata(
        doc_id=doc.doc_id,
        account_tokens=tokens,
        matched_scenario_accounts=matched,
        company_names=extract_company_names(doc.text),
        kind=kind,
        kind_score=kind_score,
        is_superseded=is_superseded,
        supersede_reasons=reasons,
        revision=extract_revision(doc.text),
        dates_found=dates,
        latest_date=dates[-1] if dates else None,
    )


def run_ingestion(data_dir: Path, cache_dir: Path) -> IngestionResult:
    template = load_template(data_dir / TEMPLATE_FILENAME)
    scenario_ids = required_scenario_ids(template)
    logger.info("Template requires %d scenarios: %s", len(scenario_ids), scenario_ids)

    transactions = load_ledger(data_dir / LEDGER_FILENAME)
    scenario_accounts = derive_scenario_accounts(transactions, scenario_ids)
    account_to_scenario = {acc: sid for sid, acc in scenario_accounts.items()}
    known_account_ids = set(account_to_scenario)
    logger.info("Resolved %d scenario accounts from the ledger.", len(scenario_accounts))

    parsed_docs = load_documents(data_dir / DOCUMENTS_DIRNAME, cache_dir)
    logger.info("Loaded %d documents from %s.", len(parsed_docs), DOCUMENTS_DIRNAME)

    resolved_docs = [
        ResolvedDocument(parsed=doc, metadata=_build_metadata(doc, known_account_ids))
        for doc in parsed_docs
    ]

    # Fan out each document to every scenario it exactly matches (normally
    # one; a document referencing two scenarios' accounts is unusual but not
    # treated as an error — both scenarios legitimately get to see it).
    docs_by_scenario: dict[str, list[ResolvedDocument]] = {sid: [] for sid in scenario_ids}
    unmatched: list[ResolvedDocument] = []
    for rdoc in resolved_docs:
        target_scenarios = {
            account_to_scenario[acc] for acc in rdoc.metadata.matched_scenario_accounts
        }
        if not target_scenarios:
            unmatched.append(rdoc)
            continue
        for sid in target_scenarios:
            docs_by_scenario[sid].append(rdoc)

    scenarios: dict[str, ScenarioBundle] = {}
    for sid in scenario_ids:
        scenario_docs = docs_by_scenario[sid]
        current_by_kind: dict[str, tuple[ResolvedDocument, ...]] = {}
        all_superseded: list[ResolvedDocument] = []

        kinds_present = {d.metadata.kind for d in scenario_docs if d.metadata.kind != "other"}
        for kind in kinds_present:
            docs_of_kind = [d for d in scenario_docs if d.metadata.kind == kind]
            current, superseded = resolve_current_documents(
                docs_of_kind, scenario_id=sid, kind=kind
            )
            if current:
                current_by_kind[kind] = tuple(current)
            all_superseded.extend(superseded)

        scenarios[sid] = ScenarioBundle(
            scenario_id=sid,
            account_id=scenario_accounts[sid],
            current_documents=current_by_kind,
            superseded_documents=tuple(all_superseded),
            all_matched_documents=tuple(scenario_docs),
        )

        missing_kinds = set(KIND_MARKERS) - set(current_by_kind)
        if missing_kinds:
            logger.warning(
                "Scenario %s has no current document for kind(s): %s",
                sid,
                sorted(missing_kinds),
            )

    return IngestionResult(
        ledger=transactions,
        template=template,
        scenarios=scenarios,
        unmatched_documents=tuple(unmatched),
    )
