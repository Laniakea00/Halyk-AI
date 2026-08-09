"""Block 1 orchestration: documents/ + ledger + template -> IngestionResult.

Nothing in here is scenario-specific. The set of "known accounts" it
resolves documents against is derived entirely from the template and the
ledger at run time (see ingestion/ledger.py:derive_scenario_accounts) —
pointing this at the private dataset on Aug 9 requires no code changes.
"""

from __future__ import annotations

import logging
from dataclasses import replace
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
from covenant_agent.resolution.segment_linking import references_borrower_as_segment
from covenant_agent.resolution.versioning import (
    detect_supersede,
    extract_dates,
    extract_revision,
    resolve_current_documents,
)

logger = logging.getLogger(__name__)

# Fraction of required scenarios missing a current credit_agreement above
# which run_ingestion refuses to continue — see AccountLinkingRiskError.
ACCOUNT_LINKING_RISK_THRESHOLD = 0.5


class AccountLinkingRiskError(RuntimeError):
    """Raised when a majority of required scenarios have no current
    credit_agreement document at all.

    Organizers have explicitly confirmed the credit agreement is always
    present for every scenario in the private dataset (unlike KYC, which
    is not guaranteed) — so a majority of scenarios missing one is not a
    plausible "these borrowers happen to lack a credit agreement"
    outcome. It is the strongest available signal that Block 1's
    document-to-scenario linking failed structurally, most plausibly
    because `resolution/accounts.py`'s `ACCOUNT_TOKEN_RE` hardcoded
    "ACC-" prefix doesn't match this dataset's actual account_id
    convention (a real, identified risk — see README's structural-
    surprises audit) — every downstream block would otherwise run to
    completion and quietly produce a full submission of
    fallback-COMPLIANT cells for every single covenant, discovered only
    by a human reading a suspiciously uniform result.

    Deliberately raised from `run_ingestion` itself, not just the CLI
    scripts, so every entrypoint (extraction/linking/calculation) that
    calls it as a library function stops immediately too — there is no
    point spending API budget on Block 2/3 against a dataset that
    structurally can't be linked to any account.
    """


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

    # Secondary pass: only for documents the primary ACC-token match (above)
    # left in `unmatched` — see segment_linking.py's module docstring for
    # why this is a separate, narrower mechanism and not a change to
    # accounts.py itself. A linked document's kind is reassigned away from
    # "other" (to "group_financials") purely so it survives the
    # `!= "other"` filter below and reaches Block 2's generic fact
    # extractor — classify.py's shared KIND_MARKERS taxonomy is untouched.
    still_unmatched: list[ResolvedDocument] = []
    for rdoc in unmatched:
        linked_scenarios = set()
        for sid in scenario_ids:
            borrower_names = {
                name
                for doc in docs_by_scenario[sid]
                if doc.metadata.kind == "credit_agreement"
                for name in doc.metadata.company_names
            }
            if any(
                references_borrower_as_segment(rdoc.parsed.text, name) for name in borrower_names
            ):
                linked_scenarios.add(sid)
        if not linked_scenarios:
            still_unmatched.append(rdoc)
            continue
        linked_doc = ResolvedDocument(
            parsed=rdoc.parsed,
            metadata=replace(rdoc.metadata, kind="group_financials", kind_score=1),
        )
        for sid in linked_scenarios:
            logger.warning(
                "Scenario %s: document %s linked via segment-reference match (not an "
                "ACC-token match) — verify manually: %s",
                sid,
                rdoc.parsed.doc_id,
                rdoc.parsed.source_path,
            )
            docs_by_scenario[sid].append(linked_doc)
    unmatched = still_unmatched

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

    scenarios_missing_credit_agreement = sorted(
        sid for sid, bundle in scenarios.items() if "credit_agreement" not in bundle.current_documents
    )
    if scenario_ids and len(scenarios_missing_credit_agreement) / len(scenario_ids) >= ACCOUNT_LINKING_RISK_THRESHOLD:
        raise AccountLinkingRiskError(
            f"{len(scenarios_missing_credit_agreement)}/{len(scenario_ids)} required scenarios have "
            f"NO current credit_agreement document — organizers confirmed this document kind is "
            f"always present, so this many missing at once means document-to-scenario linking "
            f"almost certainly failed structurally, not that these borrowers genuinely lack a "
            f"credit agreement. Most likely cause: this dataset's account_id format doesn't match "
            f"resolution/accounts.py's ACCOUNT_TOKEN_RE (hardcoded \"ACC-<digits>\" prefix) — "
            f"before re-running, manually check a real account_id value from the ledger and from a "
            f"document's own text and confirm they still look like \"ACC-1234\". "
            f"Affected scenarios: {scenarios_missing_credit_agreement}"
        )

    return IngestionResult(
        ledger=transactions,
        template=template,
        scenarios=scenarios,
        unmatched_documents=tuple(unmatched),
    )
