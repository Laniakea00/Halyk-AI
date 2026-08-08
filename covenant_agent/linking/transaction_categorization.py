"""Classify a scenario's ledger transactions against its own dynamic
category vocabulary (see categories.py) via one Structured Outputs call.

This uses TRANSACTION_CATEGORIZATION_MODEL, which sits in the same "gets
promoted to the top tier in the final profile" group as covenant
extraction, not the cheaper fact-extraction group — a departure from Block
2's original two-tier split (covenant clause extraction = critical,
KYC/audit fact extraction = standard). A miscategorized transaction here
corrupts a sum that feeds directly into `actual` and potentially `status`
for whichever covenant owns that category, exactly the failure mode that
tier exists to guard against; it isn't a "supporting fact" the way a KYC
disclosure is. See llm_client.py's model-profile docstring for the
dev/final split and how to promote just this one call to a top-tier model
for a stability check without touching the other two call types.

Tried and deliberately reverted: running this call 3x and taking a
majority vote per transaction. The theory was that temperature=0 still
isn't bit-identical across calls (confirmed: B1's status flipped between
two otherwise-identical runs), so voting should damp the noise out. In
practice it made things worse on this dataset — measured, not assumed. The
prompt below deliberately instructs "when in doubt, choose unclassified"
(the case's decoy transactions punish over-inclusion far more than
under-inclusion), which means the model's rare true positives — a single
transaction that's the *only* real match for a narrow category — are
exactly the calls most likely to be missed on any given pass. Voting with
a 2-of-3 majority requirement doesn't average that out; it amplifies it,
since a real match needs to survive the model's conservative bias twice
independently to win, while "unclassified" only needs the model to hedge
twice. A single well-prompted call, accepted as-is, scored higher in
practice than best-of-3 majority voting.
"""

from __future__ import annotations

import logging
from pathlib import Path

from covenant_agent.linking.categories import CategorySpec
from covenant_agent.llm_client import TRANSACTION_CATEGORIZATION_MODEL, extract_structured
from covenant_agent.models import Transaction
from covenant_agent.schemas import TransactionClassificationResult

logger = logging.getLogger(__name__)

UNCLASSIFIED = "unclassified"

SYSTEM_PROMPT = """\
You are a precise financial-transaction classifier for a bank's covenant-compliance pipeline.
You are given a list of category keys (each with a definition drawn from that borrower's own \
loan covenants) and a list of ledger transactions. Classify every transaction into exactly one \
category key, or 'unclassified' if none genuinely apply.

Rules:
- Be CONSERVATIVE and PRECISE. This ledger is deliberately full of transactions that sound like \
general business costs — marketing, insurance, tax payments, rent, refunds, professional fees — \
but are NOT the specific accounting concept a given category refers to, even though they are all \
"expenses" in a loose, everyday sense. A category like "operating expenses" or "payroll costs" \
means the narrow, specific accounting line item it names, not "any cost the business incurred". \
Most transactions in this ledger will genuinely match NONE of the given categories — that is the \
expected, correct outcome for most rows, not a sign you're being too strict.
- Prefer a transaction's description to name or closely paraphrase the category's own concept \
(e.g. a description containing "operating ... expenses" is real evidence for an "operating \
expenses" category; a description about marketing, insurance, or tax is evidence AGAINST it, \
even though both are technically costs the business incurred). A description can still count as \
a match without sharing the exact words when it names a well-established equivalent concept — \
e.g. "purchase of equipment/machinery" clearly is capital expenditure, "loan interest payment" \
clearly is interest expense — but that bar is for unambiguous, textbook equivalences, not for \
"this is also a cost of doing business".
- Only classify a transaction into a category when its description is a clear, specific match to \
that category's stated definition. When in doubt between a specific category and \
'unclassified', choose 'unclassified' — a missed classification is a smaller error than a wrong \
one, because it leaves the transaction available for a human/downstream check instead of \
silently corrupting a sum.
- A transaction's amount sign is a supporting clue (negative = money leaving the account, \
positive = money received) — useful for sanity-checking (e.g. a "revenue" category should \
usually hold positive amounts), but the description is the primary signal. The same trap \
applies on the positive-amount side as on the negative one: this ledger is deliberately full \
of positive-amount transactions that are NOT revenue even though money is coming in — refunds, \
rebates, tax credits/overpayments returned, deposits returned, lease incentives, insurance \
payouts, marketing co-op funding received, overbilling corrections, and advances/prepayments \
received. A "revenue" category means genuine sales/operating income from the borrower's core \
business, not any inflow. If a positive transaction's description names a refund, rebate, \
credit, return, incentive, or reimbursement of something the business already spent or already \
holds, that is evidence AGAINST it being revenue, regardless of how large or well-timed the \
amount is. The same applies to an advance, prepayment, or deposit received against a future or \
not-yet-completed sale ("аванс", "предоплата", "advance", "prepayment") — cash received in \
advance of the underlying sale is not yet recognized revenue, especially when a covenant's own \
definition of revenue is stated on an accrual/recognition basis ("выручка, признанная в этом \
периоде") rather than a cash basis; treat such a transaction as evidence AGAINST the revenue \
category even though money did come in.
- Never invent a category key that wasn't given to you.
- Never decide or imply anything about covenant compliance — you are only labeling what each \
transaction is, not evaluating any limit.
- Classify every transaction you are given, using its exact txn_id, exactly once each.
- Before marking a transaction 'unclassified', re-check it specifically against every category \
that could plausibly apply — a genuine, unambiguous match is sometimes missed simply because a \
transaction sits among many others in the same batch, not because the description was actually \
unclear. If a description is a plain, textbook instance of a category's own concept, classify \
it — do not default to 'unclassified' out of general caution once a real match is that clear.

Worked examples of unambiguous matches that must NOT be left unclassified (these are the exact \
shape of match to accept, not a request to match these literal strings):
- "Plant operating and maintenance expenses" against a category meaning operating expenses — \
a plain, direct instance of the category's own concept. Classify it.
- "Purchase of blast freezer equipment" / "Purchase of switchyard equipment" against a category \
meaning capital expenditures — "purchase of ... equipment" is a textbook capex description. \
Classify it.
- "Drilling services sales settlement" against a category meaning revenue — a genuine sales \
settlement description, no refund/rebate/return/advance language present. Classify it.
These are exactly as unambiguous as the marketing/insurance/tax/refund decoys are exactly NOT \
matches — apply the same confidence in both directions, not just toward caution.
"""


def _format_categories(specs: list[CategorySpec]) -> str:
    return "\n".join(f"- {spec.key}: {spec.description}" for spec in specs)


def _format_transactions(transactions: list[Transaction]) -> str:
    lines = []
    for txn in transactions:
        amount = "unknown" if txn.amount is None else f"{txn.amount:.2f}"
        lines.append(
            f"{txn.txn_id} | date={txn.date} | counterparty={txn.counterparty} | "
            f"amount={amount} {txn.currency} | description={txn.description}"
        )
    return "\n".join(lines)


def categorize_transactions(
    scenario_id: str,
    transactions: list[Transaction],
    category_specs: list[CategorySpec],
    *,
    log_dir: Path | None = None,
) -> dict[str, str]:
    """Return txn_id -> category key (or UNCLASSIFIED) for every given transaction.

    Skips the API call entirely (returns everything UNCLASSIFIED) if there
    are no category specs to classify against — a scenario whose covenants
    are all related-party tests has nothing for this classifier to do.
    """
    if not category_specs:
        logger.info(
            "Scenario %s: no description-based categories needed (all covenants are "
            "related-party tests) — skipping transaction categorization call.",
            scenario_id,
        )
        return {txn.txn_id: UNCLASSIFIED for txn in transactions}

    if not transactions:
        return {}

    known_keys = {spec.key for spec in category_specs}
    input_text = (
        f"Borrower scenario: {scenario_id}\n\n"
        f"Category keys and definitions:\n{_format_categories(category_specs)}\n\n"
        f"Transactions to classify ({len(transactions)} total):\n"
        f"{_format_transactions(transactions)}"
    )

    result, _raw = extract_structured(
        instructions=SYSTEM_PROMPT,
        input_text=input_text,
        response_model=TransactionClassificationResult,
        config=TRANSACTION_CATEGORIZATION_MODEL,
        log_dir=log_dir,
        log_tag=f"txn_categorize_{scenario_id}",
    )

    txn_ids = {txn.txn_id for txn in transactions}
    classification: dict[str, str] = {}
    for item in result.classifications:
        if item.txn_id not in txn_ids:
            logger.warning(
                "Scenario %s: categorization returned unknown txn_id %r — ignoring.",
                scenario_id,
                item.txn_id,
            )
            continue
        category = item.category if item.category in known_keys else UNCLASSIFIED
        if category != item.category:
            logger.warning(
                "Scenario %s: txn %s classified into unknown category %r — treating as "
                "unclassified.",
                scenario_id,
                item.txn_id,
                item.category,
            )
        if item.txn_id in classification:
            logger.warning(
                "Scenario %s: txn %s classified twice (%r then %r) — keeping the first.",
                scenario_id,
                item.txn_id,
                classification[item.txn_id],
                category,
            )
            continue
        classification[item.txn_id] = category

    missing = txn_ids - classification.keys()
    if missing:
        logger.warning(
            "Scenario %s: %d transaction(s) missing from categorization response, "
            "defaulting to unclassified: %s",
            scenario_id,
            len(missing),
            sorted(missing),
        )
        for txn_id in missing:
            classification[txn_id] = UNCLASSIFIED

    return classification
