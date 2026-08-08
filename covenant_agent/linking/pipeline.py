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
from dataclasses import replace
from pathlib import Path

from covenant_agent.linking.categories import derive_category_specs
from covenant_agent.linking.reclassification_linking import link_reclassifications
from covenant_agent.linking.related_parties import resolve_related_parties
from covenant_agent.linking.transaction_categorization import UNCLASSIFIED, categorize_transactions
from covenant_agent.models import IngestionResult, LinkedScenarioData, ScenarioFacts, Transaction
from covenant_agent.schemas import AuditExtractionResult, OtherFact

logger = logging.getLogger(__name__)

STATUS_OK = "ok"


def _apply_amount_corrections(
    scenario_id: str,
    transactions: list[Transaction],
    audit_reports: "tuple[tuple[str, AuditExtractionResult], ...]",
) -> list[Transaction]:
    """Patch Transaction.amount for every txn_id a financial_notes/audit_report/
    treasury_memo disclosure names a corrected true amount for — a dirty
    ledger row (amount=None, or an export glitch) gets the auditor's own
    stated true value applied *before* categorization/calculation ever
    see it, rather than being carried as a fact no downstream layer reads.
    Confirmed necessary on the public dataset: P8's TXN-P8-0031 and P7's
    TXN-P7-0033 both have amount=None in the raw ledger and a disclosed
    true amount in a supporting document (see schemas.py's
    TransactionAmountCorrection).

    Applied once per txn_id (first disclosure wins if more than one
    document names the same txn_id, logged either way) — logs every patch
    explicitly, at WARNING level, so it's never a silent substitution.
    """
    corrections: dict[str, tuple[float, str, str]] = {}  # txn_id -> (amount, doc_id, reasoning)
    for doc_id, audit in audit_reports:
        for item in audit.transaction_amount_corrections:
            if item.txn_id in corrections:
                logger.warning(
                    "Scenario %s: txn %s already has a correction from %s — a second "
                    "correction from %s also names it; keeping the first.",
                    scenario_id,
                    item.txn_id,
                    corrections[item.txn_id][1],
                    doc_id,
                )
                continue
            corrections[item.txn_id] = (item.corrected_amount, doc_id, item.reasoning)

    if not corrections:
        return transactions

    unapplied = set(corrections)
    patched: list[Transaction] = []
    for txn in transactions:
        correction = corrections.get(txn.txn_id)
        if correction is None:
            patched.append(txn)
            continue
        corrected_amount, doc_id, reasoning = correction
        logger.warning(
            "Scenario %s: txn %s amount patched from %r to %.2f per disclosure in %s "
            "(%s) — using the patched value for categorization and calculation.",
            scenario_id,
            txn.txn_id,
            txn.amount,
            corrected_amount,
            doc_id,
            reasoning,
        )
        patched.append(replace(txn, amount=corrected_amount))
        unapplied.discard(txn.txn_id)

    for missing_txn_id in unapplied:
        logger.warning(
            "Scenario %s: a disclosure names txn_id %r for an amount correction, but no "
            "transaction with that ID exists in this scenario's ledger — ignoring.",
            scenario_id,
            missing_txn_id,
        )

    return patched


def _log_non_usd_transactions(scenario_id: str, transactions: list[Transaction]) -> None:
    """One scenario-level summary warning for every non-USD transaction —
    calculation/formulas.py silently excludes these from every sum (no FX
    conversion is implemented; see README's FX scope note), which is the
    right call for 11 of 12 public-dataset scenarios that have no
    disclosed rate at all, but "silently" is the operative risk word.
    Logged once here, at link time, rather than from inside
    _category_signed_sum/_related_party_sum — those run many times per
    covenant (once per counterfactual in evidence.py), so logging there
    would repeat the same warning dozens of times for the same
    transaction; this is deliberately a single, scenario-level summary
    instead. Never converts 1:1 or otherwise — exclusion is the existing,
    intentional behavior; this only makes it visible.
    """
    non_usd = [t for t in transactions if t.currency != "USD"]
    if not non_usd:
        return
    logger.warning(
        "Scenario %s: %d transaction(s) in a non-USD currency will be excluded from every "
        "covenant sum (no FX conversion implemented — see README's FX scope note): %s",
        scenario_id,
        len(non_usd),
        [(t.txn_id, t.currency, t.amount) for t in non_usd],
    )


def _collect_other_facts(
    audit_reports: "tuple[tuple[str, AuditExtractionResult], ...]",
) -> tuple[OtherFact, ...]:
    """Every OtherFact across every audit_report/financial_notes/treasury_memo
    disclosure for this scenario, flattened — see formulas.py's
    _resolve_side_sum for how these get matched to a covenant side.
    """
    return tuple(fact for _doc_id, audit in audit_reports for fact in audit.other_facts)


def link_scenario(
    scenario_id: str,
    account_id: str,
    ledger: list[Transaction],
    facts: ScenarioFacts,
    *,
    log_dir: Path | None = None,
) -> LinkedScenarioData:
    transactions = [t for t in ledger if t.account_id == account_id]
    transactions = _apply_amount_corrections(scenario_id, transactions, facts.audit_reports)
    logger.info(
        "Scenario %s: %d ledger transactions on account %s.",
        scenario_id,
        len(transactions),
        account_id,
    )
    _log_non_usd_transactions(scenario_id, transactions)

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
        other_facts=_collect_other_facts(facts.audit_reports),
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
