"""Block 3c: the counterfactual evidence test.

Implements the case's own definition of `evidence_txn_id` literally — but
the case is explicit about a trap here that a naive reading misses:

    "Транзакция, которая лишь вносит вклад в сумму, доказательством не
    является: ни самая крупная строка в проверяемой категории, ни
    последняя перед закрытием периода, ни та, что случайно вывела
    накопленную сумму за порог."

I.e. "removing transaction X flips the verdict" is NOT sufficient — in any
multi-transaction sum sitting close to its threshold, *some* ordinary line
item will mechanically do that by coincidence of magnitude, and the case
explicitly disqualifies exactly that. Confirmed on the public dataset: a
first version of this module tested every category-contributing
transaction for exclusion and found one for P1's 6.1 (an ordinary,
never-reclassified opex line) whose removal happens to flip BREACH to
COMPLIANT — but the ground truth key for that cell is null. Evidence has to
come from a transaction whose *treatment itself* is the determining fact,
not from its size:

  - a transaction an auditor reclassified (reverting the reclassification
    is the counterfactual) — confirmed as the real mechanism behind B1's
    6.1 evidence (TXN-B1-0020, Irtysh Advisory Bureau).
  - a transaction to a counterparty whose related-party status is itself
    what's being determined (excluding it from the related-party pool is
    the counterfactual) — a KYC/threshold determination, not a magnitude
    coincidence, so it's a legitimate "inclusion/exclusion" case under the
    same rule.

Ordinary category-classified transactions with no such story (the general
"this row happens to be revenue/opex/capex" case) are deliberately *not*
tested for exclusion here — only for computing the real `actual` in
formulas.py. That split is what keeps this module from manufacturing false
evidence out of ordinary sum arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from covenant_agent.calculation.formulas import (
    InsufficientDataError,
    compare_to_threshold,
    compute_metric,
    in_period,
)
from covenant_agent.linking.categories import is_related_party_text
from covenant_agent.models import LinkedScenarioData
from covenant_agent.schemas import CovenantClause

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceResult:
    actual: float
    status: str
    evidence_txn_id: str | None
    notes: tuple[str, ...]


def _related_party_candidates(linked: LinkedScenarioData, clause: CovenantClause) -> set[str]:
    return {
        txn.txn_id
        for txn in linked.transactions
        if txn.amount is not None
        and txn.amount < 0
        and txn.currency == "USD"
        and in_period(txn, clause)
        and (match := linked.related_parties.get(txn.counterparty)) is not None
        and match.is_related
    }


def _candidate_transactions(
    clause: CovenantClause, linked: LinkedScenarioData
) -> tuple[set[str], set[str]]:
    """(exclusion_candidates, revert_candidates).

    Deliberately narrow — see module docstring. `exclusion_candidates` is
    related-party transactions only, when this covenant routes through
    that dimension at all (a related-party inclusion/exclusion call is a
    treatment decision, not a magnitude coincidence). `revert_candidates`
    is every transaction this covenant's categories touch that was also
    reclassified by an auditor (a reclassification reversal is the other
    "treatment, not magnitude" case the rule allows).
    """
    routes_through_related_party = any(
        is_related_party_text(text)
        for text in (clause.numerator_description, clause.denominator_description, clause.formula_description)
    )
    exclusion = _related_party_candidates(linked, clause) if routes_through_related_party else set()

    # Every reclassified transaction, not pre-filtered to this covenant:
    # reverting a reclassification that this covenant's categories never
    # touched simply can't change compute_metric()'s result, so it costs
    # nothing to include and can never manufacture a false swing.
    revert = set(linked.reclassifications.keys())

    return exclusion, revert


def find_evidence(clause: CovenantClause, linked: LinkedScenarioData) -> EvidenceResult:
    baseline_actual = compute_metric(clause, linked)
    baseline_status = compare_to_threshold(baseline_actual, clause.threshold_value, clause.direction)

    exclusion_candidates, revert_candidates = _candidate_transactions(clause, linked)

    swings: list[str] = []
    for txn_id in sorted(exclusion_candidates):
        try:
            alt_actual = compute_metric(clause, linked, excluded_txn_id=txn_id)
        except InsufficientDataError:
            # Excluding this one candidate happened to zero out a category
            # that already had barely any data — not a meaningful
            # counterfactual either way, so it's neither a swing nor a
            # crash; just not evidence.
            continue
        alt_status = compare_to_threshold(alt_actual, clause.threshold_value, clause.direction)
        if alt_status != baseline_status:
            swings.append(txn_id)
    for txn_id in sorted(revert_candidates):
        if txn_id in swings:
            continue
        try:
            alt_actual = compute_metric(clause, linked, revert_txn_id=txn_id)
        except InsufficientDataError:
            continue
        alt_status = compare_to_threshold(alt_actual, clause.threshold_value, clause.direction)
        if alt_status != baseline_status:
            swings.append(txn_id)

    notes: list[str] = []
    evidence_txn_id: str | None

    if not swings:
        evidence_txn_id = None
        notes.append(
            f"no related-party-exclusion or reclassification-reversal candidate flips "
            f"{baseline_status} — evidence=null (checked {len(exclusion_candidates)} "
            f"related-party candidate(s), {len(revert_candidates)} reclassification "
            f"candidate(s); ordinary category sums are never tested for evidence — see "
            f"module docstring)"
        )
    elif len(swings) == 1:
        evidence_txn_id = swings[0]
    else:
        by_id = {t.txn_id: t for t in linked.transactions}
        reclassified_swings = [t for t in swings if t in linked.reclassifications]
        pool = reclassified_swings or swings
        evidence_txn_id = max(pool, key=lambda tid: abs(by_id[tid].amount or 0.0))
        notes.append(
            f"{len(swings)} candidate transactions each independently flip the verdict "
            f"({swings}) — not a single determining transaction per the case's own "
            f"definition. Tie-broken to {evidence_txn_id} "
            f"({'a reclassified transaction' if reclassified_swings else 'largest magnitude'}); "
            f"treat this evidence_txn_id as a best-effort pick, not a confident single answer."
        )
        logger.warning(
            "Covenant %s: %d transactions independently flip status — picked %s. %s",
            clause.covenant_key,
            len(swings),
            evidence_txn_id,
            notes[-1],
        )

    # Organizer clarification (confirmed 2026-08-09): evidence_txn_id names
    # the transaction *causing the breach* — it is null for every COMPLIANT
    # cell, full stop, regardless of "accepts any value" language for
    # ratio/aggregate tests. Confirmed as a real, live bug, not a
    # hypothetical: P10 6.1 (status=COMPLIANT) returned
    # evidence_txn_id='TXN-P10-0017' in both of today's confirmed runs — a
    # related-party-exclusion swing was found (excluding that transaction
    # would flip the verdict to BREACH), which is genuinely useful
    # diagnostic information, but is a "what's keeping this compliant"
    # fact, not "what's causing a breach" — the two are different
    # questions, and only the second one is ever a valid evidence_txn_id
    # per the case's own definition. The swing-detection above still runs
    # unconditionally (its notes/logging are useful regardless of
    # direction); only the final evidence_txn_id is gated here.
    if baseline_status != "BREACH" and evidence_txn_id is not None:
        notes.append(
            f"evidence candidate {evidence_txn_id!r} found but suppressed — evidence_txn_id is "
            f"only ever reported for BREACH; {baseline_status} always returns null regardless of "
            f"what would flip it."
        )
        evidence_txn_id = None

    return EvidenceResult(
        actual=baseline_actual,
        status=baseline_status,
        evidence_txn_id=evidence_txn_id,
        notes=tuple(notes),
    )
