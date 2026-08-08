"""Block 3a: join auditor reclassification findings to ledger transactions.

Audit reports cite a counterparty and a dollar amount, rarely a txn_id and
rarely even a date (confirmed on the public dataset's one real example —
Ekibastuz Energy's report names "Irtysh Advisory Bureau" and
"$592,296.10", nothing else). So the join key is (counterparty, amount),
with date used only when the report happens to give one, and counterparty
matched via the same normalize-then-fuzzy logic as related_parties.py
(same underlying data quirk: KYC/audit text and ledger text don't always
spell an entity's name identically).

Every unmatched and every ambiguous reclassification is returned alongside
the confident links, not swallowed — the caller (linking/pipeline.py) logs
both loudly. An ambiguous link still resolves to *a* transaction (guessing
none at all would drop a real fact the calculation layer needs), but it's
tagged `was_ambiguous=True` end to end so nothing downstream mistakes it
for a confident match.
"""

from __future__ import annotations

import logging

from covenant_agent.linking.dates import date_in_range, parse_period
from covenant_agent.linking.fuzzy_match import match_counterparty
from covenant_agent.models import LinkedReclassification, Transaction, UnmatchedReclassification
from covenant_agent.schemas import AuditExtractionResult, AuditReclassification

logger = logging.getLogger(__name__)

AMOUNT_ABS_TOLERANCE = 0.01
AMOUNT_REL_TOLERANCE = 0.001


def _amounts_match(a: float, b: float) -> bool:
    return abs(a - b) <= max(AMOUNT_ABS_TOLERANCE, AMOUNT_REL_TOLERANCE * max(abs(a), abs(b)))


def link_reclassifications(
    scenario_id: str,
    audit_reports: "tuple[tuple[str, AuditExtractionResult], ...]",
    transactions: list[Transaction],
) -> tuple[dict[str, LinkedReclassification], list[UnmatchedReclassification]]:
    """Returns (txn_id -> LinkedReclassification, [UnmatchedReclassification, ...])."""
    distinct_counterparties = sorted({t.counterparty for t in transactions})
    by_counterparty: dict[str, list[Transaction]] = {}
    by_txn_id: dict[str, Transaction] = {}
    for t in transactions:
        by_counterparty.setdefault(t.counterparty, []).append(t)
        by_txn_id[t.txn_id] = t

    linked: dict[str, LinkedReclassification] = {}
    unmatched: list[UnmatchedReclassification] = []

    for doc_id, audit in audit_reports:
        for item in audit.reclassifications:
            if item.action == "no_change":
                # Purely informational ("considered, rejected" / "no
                # reclassification needed") — must never become a linked
                # reclassification, since that would (via
                # effective_category's revert_txn_id logic) make an
                # explicit non-event look like a real one.
                logger.info(
                    "Scenario %s: reclassification in %s is action=no_change — informational "
                    "only, not linked: %s",
                    scenario_id,
                    doc_id,
                    item.reasoning,
                )
                continue
            result = _link_one(
                scenario_id, doc_id, item, distinct_counterparties, by_counterparty, by_txn_id
            )
            if isinstance(result, UnmatchedReclassification):
                unmatched.append(result)
            else:
                if result.txn_id in linked:
                    logger.warning(
                        "Scenario %s: txn %s already linked to a reclassification from %s — "
                        "a second reclassification from %s also targets it; keeping the first, "
                        "dropping the second.",
                        scenario_id,
                        result.txn_id,
                        linked[result.txn_id].source_doc_id,
                        doc_id,
                    )
                    continue
                linked[result.txn_id] = result

    return linked, unmatched


def _link_one(
    scenario_id: str,
    doc_id: str,
    item: AuditReclassification,
    distinct_counterparties: list[str],
    by_counterparty: dict[str, list[Transaction]],
    by_txn_id: dict[str, Transaction],
) -> LinkedReclassification | UnmatchedReclassification:
    # A finding that names its txn_id directly (confirmed on the public
    # dataset: B4/P1's "исключена из ковенантного периода" findings) skips
    # the counterparty+amount fuzzy join entirely — an exact ID beats a
    # fuzzy guess whenever the source text actually gives one.
    if item.txn_id is not None:
        txn = by_txn_id.get(item.txn_id)
        if txn is None:
            logger.warning(
                "Scenario %s: reclassification in %s names txn_id %r, which isn't in this "
                "scenario's ledger — cannot link.",
                scenario_id,
                doc_id,
                item.txn_id,
            )
            return UnmatchedReclassification(
                counterparty_name=item.counterparty_name,
                amount=item.amount,
                source_doc_id=doc_id,
                reason=f"named txn_id {item.txn_id!r} not found in this scenario's ledger",
            )
        return LinkedReclassification(
            txn_id=txn.txn_id,
            action=item.action,
            original_category=item.original_category,
            reclassified_category=item.reclassified_category,
            reasoning=item.reasoning,
            source_doc_id=doc_id,
            match_confidence=1.0,
            was_ambiguous=False,
        )

    if not item.counterparty_name or item.amount is None:
        logger.warning(
            "Scenario %s: reclassification in %s has no txn_id, counterparty, and/or amount "
            "to join on (counterparty=%r, amount=%r) — cannot link to a specific transaction.",
            scenario_id,
            doc_id,
            item.counterparty_name,
            item.amount,
        )
        return UnmatchedReclassification(
            counterparty_name=item.counterparty_name,
            amount=item.amount,
            source_doc_id=doc_id,
            reason="missing counterparty and/or amount in the source report",
        )

    cp_match = match_counterparty(item.counterparty_name, distinct_counterparties)
    if cp_match is None:
        logger.warning(
            "Scenario %s: reclassification in %s cites counterparty %r, which doesn't "
            "resemble any counterparty in this scenario's ledger — no link possible.",
            scenario_id,
            doc_id,
            item.counterparty_name,
        )
        return UnmatchedReclassification(
            counterparty_name=item.counterparty_name,
            amount=item.amount,
            source_doc_id=doc_id,
            reason="no resembling counterparty found in the ledger",
        )
    if not cp_match.is_exact:
        logger.warning(
            "Scenario %s: fuzzy-matched reclassification counterparty %r to ledger "
            "counterparty %r (similarity %.2f).",
            scenario_id,
            item.counterparty_name,
            cp_match.candidate,
            cp_match.score,
        )

    candidates = [
        t
        for t in by_counterparty.get(cp_match.candidate, [])
        if t.amount is not None and _amounts_match(abs(t.amount), item.amount)
    ]
    if not candidates:
        logger.warning(
            "Scenario %s: reclassification in %s — counterparty %r matched, but no "
            "transaction of theirs has amount ≈ %.2f.",
            scenario_id,
            doc_id,
            cp_match.candidate,
            item.amount,
        )
        return UnmatchedReclassification(
            counterparty_name=item.counterparty_name,
            amount=item.amount,
            source_doc_id=doc_id,
            reason=f"counterparty matched ({cp_match.candidate!r}) but no amount≈{item.amount:.2f} found",
        )

    period = parse_period(item.transaction_date_or_period)
    if period is not None and len(candidates) > 1:
        date_filtered = [t for t in candidates if date_in_range(t.date, period)]
        if date_filtered:
            candidates = date_filtered
        else:
            logger.warning(
                "Scenario %s: reclassification in %s — date/period %r didn't match any of "
                "%d counterparty+amount candidates; ignoring the date filter rather than "
                "discarding otherwise-valid candidates.",
                scenario_id,
                doc_id,
                item.transaction_date_or_period,
                len(candidates),
            )

    was_ambiguous = len(candidates) > 1
    if was_ambiguous:
        candidates = sorted(candidates, key=lambda t: (t.date, t.txn_id))
        logger.warning(
            "Scenario %s: reclassification in %s is AMBIGUOUS — %d candidate transactions "
            "match counterparty %r and amount≈%.2f: %s. Picking %s (earliest date, then "
            "lowest txn_id) deterministically; this is a visible fallback, not a confident "
            "link.",
            scenario_id,
            doc_id,
            len(candidates),
            cp_match.candidate,
            item.amount,
            [t.txn_id for t in candidates],
            candidates[0].txn_id,
        )

    chosen = candidates[0]
    return LinkedReclassification(
        txn_id=chosen.txn_id,
        action=item.action,
        original_category=item.original_category,
        reclassified_category=item.reclassified_category,
        reasoning=item.reasoning,
        source_doc_id=doc_id,
        match_confidence=(cp_match.score if not was_ambiguous else cp_match.score * 0.5),
        was_ambiguous=was_ambiguous,
    )
