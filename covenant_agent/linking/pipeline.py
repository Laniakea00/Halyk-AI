"""Block 3a orchestration: ScenarioFacts (Block 2) + the full ledger -> LinkedScenarioData.

Ties together the three linking concerns, each independently testable:
transaction categorization (transaction_categorization.py, LLM), auditor
reclassification joins (reclassification_linking.py, code), and
related-party resolution (related_parties.py, code). None of Block 3b's
arithmetic lives here — this module's only job is producing a fully-linked
view of one scenario's data for calculation/formulas.py to consume.

Two layers of failure isolation, same reasoning as extraction/pipeline.py:
`link_scenario` catches a categorization-call failure specifically and
degrades to "everything unclassified" (calculation's InsufficientDataError
then does the right thing per-covenant); `link_all_scenarios` catches
*anything else* per scenario so one scenario's unexpected failure can't
abort the other 11 — confirmed as a real, not hypothetical, gap: this
module's categorization call previously had no try/except at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

from covenant_agent.linking.categories import derive_category_specs
from covenant_agent.linking.reclassification_linking import link_reclassifications
from covenant_agent.linking.related_parties import resolve_related_parties
from covenant_agent.linking.transaction_categorization import UNCLASSIFIED, categorize_transactions
from covenant_agent.models import IngestionResult, LinkedScenarioData, ScenarioFacts, Transaction

logger = logging.getLogger(__name__)

STATUS_OK = "ok"


def link_scenario(
    scenario_id: str,
    account_id: str,
    ledger: list[Transaction],
    facts: ScenarioFacts,
    *,
    log_dir: Path | None = None,
) -> LinkedScenarioData:
    transactions = [t for t in ledger if t.account_id == account_id]
    logger.info(
        "Scenario %s: %d ledger transactions on account %s.",
        scenario_id,
        len(transactions),
        account_id,
    )

    if facts.covenants is not None:
        category_specs = derive_category_specs(facts.covenants)
    else:
        category_specs = []
        logger.error(
            "Scenario %s: no covenant clauses were extracted — transaction categorization "
            "has no vocabulary to work with. Calculation will have to use its fallback path "
            "for every covenant here.",
            scenario_id,
        )

    try:
        txn_category = categorize_transactions(
            scenario_id, transactions, category_specs, log_dir=log_dir
        )
    except Exception:
        logger.exception(
            "Scenario %s: transaction categorization call failed entirely — treating every "
            "transaction as unclassified for this run. Calculation will raise "
            "InsufficientDataError (and fall back) for any covenant that needed real category "
            "data here, rather than silently computing from nothing.",
            scenario_id,
        )
        txn_category = {txn.txn_id: UNCLASSIFIED for txn in transactions}

    reclassifications, unmatched = link_reclassifications(scenario_id, facts.audit_reports, transactions)
    if unmatched:
        logger.warning(
            "Scenario %s: %d auditor reclassification(s) could not be linked to any ledger "
            "transaction: %s",
            scenario_id,
            len(unmatched),
            [(u.counterparty_name, u.amount, u.reason) for u in unmatched],
        )
    ambiguous_count = sum(1 for r in reclassifications.values() if r.was_ambiguous)
    if ambiguous_count:
        logger.warning(
            "Scenario %s: %d reclassification(s) linked ambiguously (multiple candidate "
            "transactions) — see prior warnings for details.",
            scenario_id,
            ambiguous_count,
        )

    distinct_counterparties = sorted({t.counterparty for t in transactions})
    related_parties = resolve_related_parties(scenario_id, facts.kyc, distinct_counterparties)

    return LinkedScenarioData(
        scenario_id=scenario_id,
        transactions=transactions,
        category_specs=category_specs,
        txn_category=txn_category,
        reclassifications=reclassifications,
        unmatched_reclassifications=unmatched,
        related_parties=related_parties,
    )


def link_all_scenarios(
    ingestion: IngestionResult,
    facts_by_scenario: dict[str, ScenarioFacts],
    *,
    log_dir: Path | None = None,
) -> tuple[dict[str, LinkedScenarioData], dict[str, str]]:
    """Run link_scenario for every scenario in `ingestion`.

    Returns (linked_by_scenario, status_by_scenario) — the latter maps
    scenario_id -> "ok" or an error message, mirroring
    extraction.pipeline.extract_all_facts's contract so callers can print
    one combined "what needs a re-run" summary across both blocks.
    """
    linked: dict[str, LinkedScenarioData] = {}
    status: dict[str, str] = {}

    for scenario_id, bundle in ingestion.scenarios.items():
        try:
            facts = facts_by_scenario[scenario_id]
            linked[scenario_id] = link_scenario(
                scenario_id,
                bundle.account_id,
                ingestion.ledger,
                facts,
                log_dir=log_dir,
            )
            status[scenario_id] = STATUS_OK
        except Exception as exc:  # noqa: BLE001 - intentional: see module docstring
            logger.exception(
                "Scenario %s: linking failed entirely (uncaught exception) — recording an "
                "empty LinkedScenarioData and continuing with the rest of the batch.",
                scenario_id,
            )
            linked[scenario_id] = LinkedScenarioData(
                scenario_id=scenario_id,
                transactions=[],
                category_specs=[],
                txn_category={},
                reclassifications={},
                unmatched_reclassifications=[],
                related_parties={},
            )
            status[scenario_id] = f"FAILED: {exc!r}"

    failed = [sid for sid, s in status.items() if s != STATUS_OK]
    if failed:
        logger.warning(
            "Linking batch finished with %d/%d scenario(s) failed: %s — re-run these "
            "individually with --scenario once the underlying issue is fixed.",
            len(failed),
            len(status),
            failed,
        )
    else:
        logger.info("Linking batch finished: %d/%d scenario(s) ok.", len(status), len(status))

    return linked, status
