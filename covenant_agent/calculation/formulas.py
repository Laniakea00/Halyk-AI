"""Block 3b: turn a CovenantClause + LinkedScenarioData into an actual number.

This is the one module in the whole pipeline where BREACH/COMPLIANT and the
`actual` figure get decided — deliberately pure code, no LLM call anywhere
in this file. Every function here also accepts `excluded_txn_id` /
`revert_txn_id` so the exact same code path computes both the real value
and every counterfactual the evidence test (evidence.py) needs — there is
no separate "evidence formula", just this formula run twice.

Branching is strictly on `metric_type` (never on `covenant_key` — see
categories.py's module docstring for why: the same clause number means a
different formula on almost every borrower in the public dataset).
"""

from __future__ import annotations

import logging
import re

from covenant_agent.linking.categories import (
    borrow_matching_text,
    is_related_party_text,
    is_unrestricted_subsidiary_text,
    match_category_by_text,
)
from covenant_agent.linking.transaction_categorization import UNCLASSIFIED
from covenant_agent.models import CategorySpec, LinkedScenarioData, Transaction
from covenant_agent.schemas import CovenantClause, OtherFact

logger = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    """Raised when a description-based category sum matched zero transactions,
    for any metric_type — a ratio's denominator, an aggregate_amount/other's
    whole value, or every component of a max_single_component test at once.

    Originally added for ratio denominators only, superseding a "divide by
    a tiny epsilon" fallback that turned a categorization miss into a
    fabricated multi-hundred-million-dollar ratio (confirmed on the public
    dataset: P3/P4/P6/P9's denominators — all EBITDA- or opex-style
    category sums, not related-party sums — occasionally come back with 0
    matched transactions). Extended to aggregate_amount/max_single_component/
    other on the same reasoning, once code review flagged that those three
    metric_types had the *same* underlying bug in a quieter form: 0 matched
    transactions there silently produced `actual=0.0`, which — unlike the
    ratio case — reads as a perfectly ordinary, plausible answer ("no capex
    this year" ⇒ COMPLIANT) rather than an obviously-fabricated number, so
    it's arguably more dangerous precisely because it doesn't look wrong.

    Zero matched transactions for a category like "EBITDA", "operating
    expenses", or a named overhead line is not a real business fact (no
    operating company has literally zero operating expenses); it means the
    classifier found nothing this pass, which is a "we don't know" state,
    not a "the value is zero" or "the value is huge" one. This exception is
    caught by calculation/pipeline.py and converted into the same explicit,
    logged fallback used for every other "insufficient information" case
    (missing credit agreement, failed extraction, ...) — never a fabricated
    or falsely-reassuring number, and never a silent divide or silent zero.

    Deliberately NOT raised when the zero-count side is related-party
    sourced: "zero related-party payments this period" is a completely
    ordinary, plausibly-true business fact (see _resolve_side_sum), not a
    categorization failure — only description-based category sums (which
    a real business can never legitimately have zero of, for the concepts
    these covenants measure) trigger this.
    """


def _needs_data_check(description_text: str | None) -> bool:
    """True if a zero-count side of this shape is suspicious enough to flag.

    See InsufficientDataError's docstring — related-party sides get a pass
    because a real zero is a normal outcome there. "Unrestricted
    Subsidiary"-style sides (P9's 6.1) get the same pass, same reasoning —
    see is_unrestricted_subsidiary_text's docstring for why this is a
    deliberately narrow, code-only exemption and not a new resolution
    mechanism.
    """
    return not (is_related_party_text(description_text) or is_unrestricted_subsidiary_text(description_text))


# Numerical safety net for _safe_ratio only, once InsufficientDataError has
# already ruled out "denominator matched zero transactions" — see that
# exception's docstring for why this is no longer the primary defense
# against a fabricated ratio.
DIVISION_EPSILON = 0.01


_QUARTER_ORDINAL_RE = re.compile(
    r"\b(перв\w*|втор\w*|трет\w*|четверт\w*|четвёрт\w*)\s+(?:финансов\w*\s+)?квартал",
    re.IGNORECASE,
)
_QUARTER_ORDINAL_STEMS = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4, "четвёрт": 4}
_QUARTER_DIGIT_RE = re.compile(r"\bQ([1-4])\b|\b([1-4])[-\s]?(?:-?[йыого]{1,3})?\s+квартал", re.IGNORECASE)
_QUARTER_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}


def _quarter_number(text: str) -> int | None:
    """Which quarter (1-4) `text` names, by ordinal word or digit — or None.

    Deliberately narrow: only recognizes an explicit quarter reference right
    next to the word "квартал" (optionally "N-й финансовый квартал"). No
    attempt to parse fiscal-year conventions, non-calendar quarters, or
    English "first quarter" phrasing — those haven't shown up in this
    dataset, and guessing wrong here silently mis-bounds a period, which is
    worse than not inferring at all.
    """
    if not text:
        return None
    lowered = text.lower()
    match = _QUARTER_ORDINAL_RE.search(lowered)
    if match:
        stem = match.group(1)
        for prefix, number in _QUARTER_ORDINAL_STEMS.items():
            if stem.startswith(prefix):
                return number
    match = _QUARTER_DIGIT_RE.search(lowered)
    if match:
        return int(match.group(1) or match.group(2))
    return None


def _quarter_start_from_end(period_end: str, quarter_number: int) -> str | None:
    """Derive a calendar quarter's first day from its stated last day + ordinal.

    Returns None (no inference) rather than guessing if `period_end`'s month
    doesn't match the named quarter's expected end month (Q1->Mar, Q2->Jun,
    Q3->Sep, Q4->Dec) — a mismatch means either a non-calendar fiscal year or
    the text/date disagreeing, and either way a wrong silent guess is worse
    than leaving period_start unset (in_period then just has no lower bound,
    the pre-existing behavior).
    """
    try:
        year_s, month_s, _day_s = period_end.split("-")
        year, month = int(year_s), int(month_s)
    except ValueError:
        return None
    if _QUARTER_END_MONTH.get(quarter_number) != month:
        return None
    return f"{year:04d}-{month - 2:02d}-01"


def _effective_period_start(clause: CovenantClause) -> str | None:
    """`clause.period_start` if 2a extracted one; otherwise a deterministic,
    code-side inference from "N-й квартал, оканчивающийся <date>" phrasing —
    never an LLM guess. See _quarter_number/_quarter_start_from_end.

    Confirmed necessary on the public dataset: B4's covenant 6.1 measures
    revenue for "четвёртый финансовый квартал" (Q4) only, not the full year
    — the source text never states an explicit numeric start date (only
    "квартал, оканчивающийся 2025-12-31"), so 2a correctly reports
    period_start=None (there's nothing more literal to extract), and without
    this inference in_period() had no lower bound at all, silently summing
    the whole year's revenue into a Q4-only test.
    """
    if clause.period_start is not None:
        return clause.period_start
    if clause.period_end is None:
        return None
    text = " ".join(
        filter(
            None,
            [
                clause.metric_name,
                clause.formula_description,
                clause.numerator_description,
                clause.denominator_description,
                clause.aggregation_note,
            ],
        )
    )
    quarter_number = _quarter_number(text)
    if quarter_number is None:
        return None
    return _quarter_start_from_end(clause.period_end, quarter_number)


def in_period(txn: Transaction, clause: CovenantClause) -> bool:
    """True unless `clause` states or implies a period bound `txn.date` is outside of.

    ISO YYYY-MM-DD strings compare correctly lexicographically, so no date
    parsing is needed for the comparison itself — see _effective_period_start
    for where a missing period_start gets inferred, if it safely can be.
    """
    period_start = _effective_period_start(clause)
    if period_start is not None and txn.date < period_start:
        return False
    if clause.period_end is not None and txn.date > clause.period_end:
        return False
    return True


def effective_category(
    txn: Transaction,
    linked: LinkedScenarioData,
    covenant_specs: list[CategorySpec],
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
) -> str | None:
    """Which category key `txn` counts under, in the given counterfactual world.

    - `txn.txn_id == excluded_txn_id`: the transaction is fully excluded
      (simulates "this payment didn't happen") — returns None.
    - `txn.txn_id` has a linked reclassification AND isn't `revert_txn_id`:
      resolved via the reclassification's own `action` —
      "exclude_from_period" excludes the transaction entirely (returns
      None) regardless of its category, matching an auditor's finding that
      a specific transaction shouldn't count this period at all (e.g.
      revenue recognized in a different fiscal year than the transaction
      date implies); "recategorize" resolves via the reclassification's
      target text, matched against this covenant's category descriptions
      (match_category_by_text) — the normal "reclassification applied"
      path.
    - `txn.txn_id == revert_txn_id`: the reclassification is deliberately
      *not* applied (regardless of its action) — simulates "what if the
      auditor's finding is undone for this one transaction", per the
      case's own evidence definition, which names reclassification as one
      of the operations whose reversal can flip a verdict.
    - Otherwise: the transaction's base (description-only) category from
      Block 3a's classifier.
    """
    if txn.txn_id == excluded_txn_id:
        return None

    reclass = linked.reclassifications.get(txn.txn_id)
    if reclass is not None and txn.txn_id != revert_txn_id:
        if reclass.action == "exclude_from_period":
            return None
        match = match_category_by_text(reclass.reclassified_category, covenant_specs)
        if match is not None:
            return match[0].key

    return linked.txn_category.get(txn.txn_id)


def _category_signed_sum(
    category_key: str,
    linked: LinkedScenarioData,
    covenant_specs: list[CategorySpec],
    clause: CovenantClause,
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
) -> tuple[float, int]:
    """Raw signed sum of transactions in `category_key` (ledger sign preserved:
    negative = outflow), plus how many transactions contributed. Callers
    decide when/whether to take abs() of the sum — the netting case
    (_resolve_side_sum) needs the sign preserved until *after* combining
    plus/minus specs; a lone non-netted category wants abs() of this
    directly. The count is what lets compute_metric tell "genuinely zero"
    apart from "categorization found nothing" — see InsufficientDataError.
    """
    total = 0.0
    count = 0
    for txn in linked.transactions:
        if txn.amount is None or txn.currency != "USD" or not in_period(txn, clause):
            continue
        category = effective_category(
            txn, linked, covenant_specs, excluded_txn_id=excluded_txn_id, revert_txn_id=revert_txn_id
        )
        if category == category_key:
            total += txn.amount
            count += 1
    return total, count


def _related_party_sum(
    linked: LinkedScenarioData,
    clause: CovenantClause,
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
) -> float:
    total = 0.0
    for txn in linked.transactions:
        if (
            txn.txn_id == excluded_txn_id
            or txn.amount is None
            or txn.currency != "USD"
            or not in_period(txn, clause)
        ):
            continue
        reclass = linked.reclassifications.get(txn.txn_id)
        if (
            reclass is not None
            and reclass.action == "exclude_from_period"
            and txn.txn_id != revert_txn_id
        ):
            continue
        if txn.amount >= 0:
            continue  # only outflows count as "payments to" a related party
        match = linked.related_parties.get(txn.counterparty)
        if match is not None and match.is_related:
            total += txn.amount
    return abs(total)


def _resolve_side_sum(
    clause: CovenantClause,
    role: str,
    description_text: str | None,
    linked: LinkedScenarioData,
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
) -> tuple[float, int]:
    """(magnitude, contributing_txn_count) for one ratio side / the aggregate
    amount, role-scoped. Related-party sides report count=1 when their sum
    is nonzero and 0 when it's genuinely zero — see _needs_data_check /
    InsufficientDataError: a real zero there is a normal outcome, so it's
    never treated as "insufficient data" regardless of what count says
    (compute_metric only applies the check when _needs_data_check is True).

    A role normally maps to exactly one CategorySpec, but categories.py
    splits a "net of" description (e.g. EBITDA = revenue net of opex) into
    two or more specs sharing the role. They're summed with their *raw
    ledger sign*, no artificial +1/-1 flip: the ledger already records
    inflows positive and outflows negative (per the case's own
    convention), so "revenue net of operating expenses" =
    signed_sum(revenue) [positive] + signed_sum(opex) [already negative]
    falls out correctly from plain addition. Deliberately confirmed by
    testing the opposite (multiplying the "minus" side by -1) against B1's
    real numbers first: it roughly *doubled* operating expenses onto the
    numerator instead of subtracting them (double-negating an
    already-negative figure), producing an absurdly high coverage ratio. A
    non-netted role has one spec, so this is just that spec's signed total
    either way.

    Off-ledger facts (linked.other_facts — a document-level figure with no
    corresponding ledger transaction at all, e.g. an accrued severance
    obligation disclosed in a note) are added on top when one matches this
    role's own category text — see _other_facts_magnitude for the
    scenario-wide, at-most-once-per-fact matching rule. Confirmed
    necessary on the public dataset: P8's 6.1 ("Совокупные обязательства
    по персоналу") is genuinely the sum of paid payroll transactions
    *plus* a severance-program obligation that will never appear as a
    ledger row. A matched fact counts toward `count` too, so a covenant
    whose ledger side is empty but has a real off-ledger figure doesn't
    spuriously raise InsufficientDataError.

    A fact's value is combined as a *magnitude*, added after abs(net), not
    mixed into the signed running total — OtherFact.value has no ledger
    sign convention at all (it's whatever plain number the document
    states, e.g. "$918,447.52", never a signed ledger amount), so summing
    it directly into `net` before the final abs() would partially cancel
    against a negative (outflow-dominated) transaction total instead of
    adding to it. Confirmed as a real bug during testing: a $2.4M payroll
    outflow (net=-2,418,663.27) plus a $918K fact summed to $1.5M instead
    of $3.3M until this was split out.
    """
    if is_related_party_text(description_text):
        total = _related_party_sum(
            linked, clause, excluded_txn_id=excluded_txn_id, revert_txn_id=revert_txn_id
        )
        return total, (1 if total != 0 else 0)

    covenant_specs = [s for s in linked.category_specs if s.covenant_key == clause.covenant_key]
    role_specs = [s for s in covenant_specs if s.role == role]
    net = 0.0
    count = 0
    for spec in role_specs:
        spec_total, spec_count = _category_signed_sum(
            spec.key,
            linked,
            covenant_specs,
            clause,
            excluded_txn_id=excluded_txn_id,
            revert_txn_id=revert_txn_id,
        )
        net += spec_total
        count += spec_count

    if count == 0:
        borrowed_total, borrowed_count = _borrow_sibling_category_sum(
            role_specs,
            description_text,
            linked,
            clause,
            excluded_txn_id=excluded_txn_id,
            revert_txn_id=revert_txn_id,
            log_context=f"covenant {clause.covenant_key} {role}",
        )
        net += borrowed_total
        count += borrowed_count

    if count == 0:
        rescued_total, rescued_count = _rescue_unclassified_revenue_transaction(
            description_text,
            linked,
            clause,
            excluded_txn_id=excluded_txn_id,
            revert_txn_id=revert_txn_id,
            log_context=f"covenant {clause.covenant_key} {role}",
        )
        net += rescued_total
        count += rescued_count

    # Addback is folded into `net` *before* abs(), not added to the
    # abs()'d magnitude the way other_facts is — see _addback_magnitude's
    # docstring. The flagged transaction is already counted once by the
    # per-spec loop above (with its real ledger sign, e.g. -500 for a
    # one-time cost); adding +abs(that same amount) here cancels it back
    # to a zero net contribution, which is what "adjusted" means. Adding
    # it after abs() would double the value instead of removing it.
    addback_delta, addback_count = _addback_magnitude(
        role_specs,
        linked.category_specs,
        linked,
        excluded_txn_id=excluded_txn_id,
        revert_txn_id=revert_txn_id,
        log_context=f"covenant {clause.covenant_key} {role}",
    )
    net += addback_delta
    count += addback_count

    fact_magnitude, fact_count = _other_facts_magnitude(
        role_specs,
        linked.category_specs,
        _effective_other_facts(linked),
        log_context=f"covenant {clause.covenant_key} {role}",
    )
    return abs(net) + fact_magnitude, count + fact_count


def _borrow_sibling_category_sum(
    role_specs: list[CategorySpec],
    description_text: str | None,
    linked: LinkedScenarioData,
    clause: CovenantClause,
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
    log_context: str,
) -> tuple[float, int]:
    """When `role_specs` (this covenant/role's own categories) found zero
    matching transactions, look scenario-wide for a *different* covenant's
    category describing the same real-world concept and, if one clears
    the match threshold, borrow its already-categorized transactions
    instead of surfacing InsufficientDataError.

    Confirmed necessary on the public dataset: P5's covenant 6.1
    denominator ("Выручка", net of opex) and P5's covenant 6.2 amount (a
    fuller-worded standalone revenue test) describe the exact same
    real-world revenue transactions for the same borrower — the
    transaction classifier assigns each transaction to exactly one
    category key (see transaction_categorization.py's "classify into
    exactly one category" instruction), so when two categories from
    different covenants genuinely describe the same concept, whichever is
    worded more fully/specifically tends to "win" the classification,
    leaving the other with zero matches even though the true answer sits
    right there under a sibling covenant's category.

    Deliberately a code-level, calculation-layer rule — never a second LLM
    call, never a change to the categorization schema/prompt (no new
    context-window cost, no change to what gets prompt-cached). Only ever
    activates when the primary path found *nothing at all* (`count == 0`
    in the caller), so it can never override or double-count against a
    real, already-successful match — this is the same "try a clearer rule
    before giving up" step InsufficientDataError's fallback used to skip
    straight past. Every borrow is logged at WARNING, clearly marked as a
    degraded match, not a confident primary one.

    Fix 5 (offline post-fix review, confirmed on P5 6.1): when `role_specs`
    has more than one spec — a netted role, e.g. "revenue net of opex"
    split into two parts by categories.py's compound-splitting — a single
    borrow attempt using the *combined* `description_text` only ever finds
    ONE sibling category (typically whichever concept scores highest,
    usually revenue) and completely misses that the other part (opex)
    also needs its own deduction. Confirmed live: P5 6.1's EBITDA
    denominator borrowed only a revenue sibling, silently dropping the
    opex side entirely and inflating EBITDA. For a multi-spec role, this
    borrows each part *separately* against its own focused description
    and sums the results, so revenue and opex can each find their own
    (possibly different) sibling donor.
    """
    # Fix #3 (post-fix forensic review): own_role_keys is always excluded
    # from the candidate pool, regardless of which covenant(s) end up in
    # scope for this attempt — matters once same-covenant candidates are
    # allowed below (Adjusted-EBITDA/Revenue shape), where a role's own
    # (already-confirmed-empty) sibling spec would otherwise self-match at
    # score 1.0 and win over the genuinely different role we want.
    own_role_keys = {s.key for s in role_specs}
    if len(role_specs) > 1:
        total = 0.0
        count = 0
        borrowed_keys: set[str] = set(own_role_keys)
        for spec in role_specs:
            part_total, part_count, borrowed_key = _borrow_one_sibling(
                spec.description,
                linked,
                clause,
                excluded_txn_id=excluded_txn_id,
                revert_txn_id=revert_txn_id,
                log_context=f"{log_context} ({spec.component_label or spec.key})",
                exclude_keys=borrowed_keys,
            )
            if borrowed_key is not None:
                borrowed_keys.add(borrowed_key)
            total += part_total
            count += part_count
        return total, count
    total, count, _borrowed_key = _borrow_one_sibling(
        description_text,
        linked,
        clause,
        excluded_txn_id=excluded_txn_id,
        revert_txn_id=revert_txn_id,
        log_context=log_context,
        exclude_keys=own_role_keys,
    )
    return total, count


def _borrow_one_sibling(
    description_text: str | None,
    linked: LinkedScenarioData,
    clause: CovenantClause,
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
    log_context: str,
    exclude_keys: "set[str] | frozenset[str]",
) -> tuple[float, int, str | None]:
    """One sibling-covenant borrow attempt for a single, focused piece of
    text — see _borrow_sibling_category_sum for the multi-part fan-out.
    `exclude_keys` keeps a multi-part borrow from crediting the same
    sibling category to two different parts of the same role (e.g. both
    "revenue" and "opex" text coincidentally matching the same sibling) —
    and, as of Fix #3, always includes the searching role's own spec keys
    too (see caller), which matters now that same-covenant candidates are
    allowed in the narrow case below.

    Returns (total, count, borrowed_spec_key_or_None).
    """
    if not description_text:
        return 0.0, 0, None
    # Fix #1 (post-fix forensic review): use a short, targeted phrase for
    # the match instead of a possibly-long, undecomposed description_text
    # (aggregate_amount/other's formula_description in particular) — see
    # borrow_matching_text's own docstring for why the long text alone
    # dilutes the stem-overlap score to near-zero even when it names the
    # right concept. No-op for already-short text (ratio numerator/
    # denominator, component labels).
    match_text = borrow_matching_text(description_text, clause.metric_name)
    # Restricted to a genuinely *different* covenant — EXCEPT for one
    # narrow, justified case (Fix #3, confirmed on P4 6.1): a ratio shaped
    # "Adjusted EBITDA / Revenue" defines its numerator partly in terms of
    # the exact same revenue figure the denominator needs on its own
    # (Adjusted EBITDA = Revenue - OpEx + addback) — the categorizer can
    # only assign the one real revenue transaction to ONE of the two, so
    # the loser is a *correct*, not spurious, same-covenant donor. This is
    # NOT a general same-covenant allowance: both the searching role's own
    # text AND the candidate spec's own text must independently read as
    # revenue-shaped (is_revenue_text), or the covenant-match restriction
    # applies exactly as before. own_role_keys (folded into exclude_keys
    # by the caller) still blocks a role from "matching" its own
    # already-confirmed-empty sibling spec.
    same_covenant_revenue_donor = is_revenue_text(description_text)
    sibling_pool = [
        s
        for s in linked.category_specs
        if s.key not in exclude_keys
        and (
            s.covenant_key != clause.covenant_key
            or (same_covenant_revenue_donor and is_revenue_text(s.description))
        )
    ]
    match = match_category_by_text(match_text, sibling_pool)
    if match is None:
        return 0.0, 0, None
    sibling_spec, score = match
    sibling_covenant_specs = [s for s in linked.category_specs if s.covenant_key == sibling_spec.covenant_key]
    total, count = _category_signed_sum(
        sibling_spec.key,
        linked,
        sibling_covenant_specs,
        clause,
        excluded_txn_id=excluded_txn_id,
        revert_txn_id=revert_txn_id,
    )
    if count > 0:
        logger.warning(
            "%s: own category found zero transactions — borrowed sibling covenant %s's "
            "category %r (match score %.2f) instead, which found %d transaction(s) for "
            "the same period. Degraded, code-level fallback — not a confident primary "
            "match; verify this borrow is semantically correct if this cell looks wrong.",
            log_context,
            sibling_spec.covenant_key,
            sibling_spec.key,
            score,
            count,
        )
        return total, count, sibling_spec.key
    return total, count, None


# Idea 1 (offline audit): a deterministic, last-resort rescue for the exact
# non-determinism shape confirmed on P8 6.3 — a genuinely revenue-shaped
# role whose own category AND every sibling covenant's category matched
# zero transactions, purely because transaction_categorization.py's LLM
# call missed the one real revenue transaction on a given pass (confirmed
# via cache/step2/run1-5's identical-input reruns: 0/5 classified
# TXN-P8-0015 as revenue, vs 8/10 on other historical runs — genuine model
# non-determinism, not a linking bug).
_REVENUE_ROLE_RE = re.compile(r"выручк\w*|revenue|доход\w*\s+от\s+реализац\w*|табыс", re.IGNORECASE)
_REVENUE_SIGNAL_RE = re.compile(r"\bsales?\b|продаж\w*|реализац\w*|settlement", re.IGNORECASE)
_REVENUE_DECOY_RE = re.compile(
    r"refund|rebate|\breturn\w*|recovery|credit\s+note|advance|prepay\w*"
    r"|возврат\w*|рекредит\w*|аванс\w*|предоплат\w*|скидк\w*",
    re.IGNORECASE,
)


def is_revenue_text(text: str | None) -> bool:
    if not text:
        return False
    return bool(_REVENUE_ROLE_RE.search(text))


def _rescue_unclassified_revenue_transaction(
    description_text: str | None,
    linked: LinkedScenarioData,
    clause: CovenantClause,
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
    log_context: str,
) -> tuple[float, int]:
    """Last-resort rescue: when a revenue-shaped role has zero matches from
    both its own category and sibling-borrow, scan currently-unclassified
    transactions for exactly one that (a) has a positive USD amount, (b)
    names a sale/settlement concept, (c) carries none of the known decoy
    keywords (refund/rebate/return/recovery/advance — the same taxonomy
    transaction_categorization.py's own system prompt already documents as
    "positive but not revenue"). Never poaches a transaction that already
    has a confident category from *any* covenant.

    Deliberately requires *exactly one* clean candidate — 0 or 2+ means the
    signal is ambiguous, so this returns nothing and lets
    InsufficientDataError surface rather than guessing. Validated against
    the real public dataset: among P8's 55 real ledger transactions, this
    filter uniquely isolates TXN-P8-0015 ("Drilling services sales
    settlement", KazMunayGas Exploration JSC) with no other candidate.
    """
    if not is_revenue_text(description_text):
        return 0.0, 0

    candidates = []
    for txn in linked.transactions:
        if txn.txn_id == excluded_txn_id:
            continue
        if txn.amount is None or txn.amount <= 0 or txn.currency != "USD":
            continue
        if not in_period(txn, clause):
            continue
        reclass = linked.reclassifications.get(txn.txn_id)
        if reclass is not None and txn.txn_id != revert_txn_id:
            continue  # already has an auditor finding attached — not this rescue's business
        if linked.txn_category.get(txn.txn_id, UNCLASSIFIED) != UNCLASSIFIED:
            continue
        if _REVENUE_DECOY_RE.search(txn.description):
            continue
        if not _REVENUE_SIGNAL_RE.search(txn.description):
            continue
        candidates.append(txn)

    if len(candidates) != 1:
        return 0.0, 0

    txn = candidates[0]
    logger.warning(
        "%s: own category and sibling-borrow both matched zero transactions for a "
        "revenue-shaped role — deterministic rescue found exactly one unclassified "
        "transaction by keyword signal: %s (%.2f %s, %r). Last-resort, code-only "
        "heuristic, not an LLM re-classification — verify this cell if it looks wrong.",
        log_context,
        txn.txn_id,
        txn.amount,
        txn.currency,
        txn.description,
    )
    return txn.amount, 1


def _other_facts_magnitude(
    specs: list[CategorySpec],
    all_specs: list[CategorySpec],
    other_facts: tuple[OtherFact, ...],
    *,
    log_context: str,
) -> tuple[float, int]:
    """Sum of other_facts whose single *scenario-wide best-matching* category
    is among `specs` — not just whichever facts happen to clear the match
    threshold against `specs` alone, which is what a naive per-call re-match
    would do and would let the same fact double-count into two different
    covenants (or two different roles/components of the same covenant) if
    its wording happens to overlap both.

    `match_category_by_text` already picks the single highest-scoring spec
    out of whatever list it's given — passing the *whole scenario's*
    `all_specs` here (not just `specs`) makes that pick a genuine, unique,
    scenario-wide winner, then this only credits the fact to `specs` if
    that winner is a member of it. A fact that would have cleared
    threshold against `specs` on its own merits, but scores higher
    elsewhere in the scenario, is logged (not silently dropped) so a
    private-dataset ambiguity is visible.
    """
    magnitude = 0.0
    count = 0
    for fact in other_facts:
        if fact.value is None:
            continue
        best = match_category_by_text(fact.fact_description, all_specs)
        if best is None:
            continue
        if best[0] in specs:
            magnitude += fact.value
            count += 1
            continue
        local = match_category_by_text(fact.fact_description, specs)
        if local is not None:
            logger.warning(
                "%s: other_fact %r also matches this side locally, but its "
                "scenario-wide best match is %r — counted there only, not "
                "here, to avoid double-counting the same fact into two "
                "covenants/roles.",
                log_context,
                fact.fact_description,
                best[0].key,
            )
    return magnitude, count


_NBV_START_RE = re.compile(
    r"(net\s+book\s+value|балансов\w*\s+стоимост\w*).{0,40}(beginning|start|начал\w*)"
    r"|(beginning|start|начал\w*).{0,40}(net\s+book\s+value|балансов\w*\s+стоимост\w*)",
    re.IGNORECASE,
)
_NBV_END_RE = re.compile(
    r"(net\s+book\s+value|балансов\w*\s+стоимост\w*).{0,40}(end|clos\w*|конец|конц\w*)"
    r"|(end|clos\w*|конец|конц\w*).{0,40}(net\s+book\s+value|балансов\w*\s+стоимост\w*)",
    re.IGNORECASE,
)
_DEPRECIATION_CHARGE_RE = re.compile(
    r"depreciation\s+(charge|expense)|амортизацион\w*\s+отчислен\w*|начислен\w*\s+амортизац\w*",
    re.IGNORECASE,
)
_READY_CAPEX_RE = re.compile(
    r"capital\s+expenditur\w*|\bcapex\b|капитальн\w*\s+затрат\w*|капитальн\w*\s+вложен\w*",
    re.IGNORECASE,
)


def _derive_capex_from_nbv_roll_forward(other_facts: tuple[OtherFact, ...]) -> OtherFact | None:
    """capex = NBV_end - NBV_start + depreciation, when a document discloses
    a PP&E roll-forward note (opening/closing net book value + the year's
    depreciation charge) instead of a single ready-made capex figure.

    Confirmed necessary on the public dataset: P5's Group-parent financial
    notes disclose exactly this roll-forward ("Net book value at the
    beginning/end of the year", "Depreciation charge for the year") for its
    PP&E note, but never states a "capital expenditure"/"капитальные
    затраты" figure directly anywhere in the document — the extractor
    faithfully pulls out the three raw line items as separate other_facts,
    but nothing downstream previously combined them, so a covenant whose
    numerator needs Group capex saw none of its off-ledger facts match.

    Deliberately code-only arithmetic, not an LLM computation — the three
    inputs are still LLM-extracted facts (each independently sourced from
    its own source_quote), this only does the addition. Skipped entirely
    if a fact already looks like a ready-made capex figure (a direct
    disclosure is always preferred over a derived one), or if any of the
    three roll-forward components is missing — a partial roll-forward
    (e.g. depreciation without both NBV endpoints) isn't safe to guess at.
    """
    if any(fact.value is not None and _READY_CAPEX_RE.search(fact.fact_description) for fact in other_facts):
        return None

    nbv_start = next(
        (f for f in other_facts if f.value is not None and _NBV_START_RE.search(f.fact_description)), None
    )
    nbv_end = next((f for f in other_facts if f.value is not None and _NBV_END_RE.search(f.fact_description)), None)
    depreciation = next(
        (f for f in other_facts if f.value is not None and _DEPRECIATION_CHARGE_RE.search(f.fact_description)), None
    )
    if nbv_start is None or nbv_end is None or depreciation is None:
        return None

    capex = nbv_end.value - nbv_start.value + depreciation.value
    logger.info(
        "Derived capex from PP&E roll-forward: %.2f (NBV end %.2f - NBV start %.2f + "
        "depreciation %.2f) — no ready-made capex figure was stated directly.",
        capex,
        nbv_end.value,
        nbv_start.value,
        depreciation.value,
    )
    return OtherFact(
        # Deliberately short — match_category_by_text scores on stem
        # *overlap ratio* (matched / len(target_stems)), so padding this
        # with an explanation of the arithmetic (as an earlier version did)
        # dilutes the ratio and can drop a genuine match below
        # CATEGORY_MATCH_THRESHOLD. The provenance lives in source_quote.
        fact_description="Капитальные затраты Группы",
        value=capex,
        unit=nbv_end.unit,
        period=nbv_end.period,
        source_quote=f"{nbv_start.source_quote} | {nbv_end.source_quote} | {depreciation.source_quote}",
    )


def _effective_other_facts(linked: LinkedScenarioData) -> tuple[OtherFact, ...]:
    """linked.other_facts, plus a code-derived capex fact when the raw
    facts contain a PP&E roll-forward but no ready capex figure — see
    _derive_capex_from_nbv_roll_forward. Computed fresh (cheap regex scan
    over a handful of facts) rather than cached on LinkedScenarioData, so
    this has no effect on any other reader of linked.other_facts.
    """
    derived = _derive_capex_from_nbv_roll_forward(linked.other_facts)
    if derived is None:
        return linked.other_facts
    return linked.other_facts + (derived,)


def _addback_magnitude(
    specs: list[CategorySpec],
    all_specs: list[CategorySpec],
    linked: LinkedScenarioData,
    *,
    excluded_txn_id: str | None,
    revert_txn_id: str | None,
    log_context: str,
) -> tuple[float, int]:
    """Idea 3 (offline audit): (signed delta, count) for `action="addback"`
    reclassifications whose scenario-wide best-matching category (by the
    auditor's own original_category label) is among `specs` — same
    single-best-match rule as _other_facts_magnitude, so one addback can't
    double-count into two covenants/roles.

    The caller must add the returned delta into its *signed* running total
    BEFORE taking abs() — NOT onto the abs()'d magnitude the way
    other_facts is added (other_facts has no ledger sign convention at
    all; an addback's magnitude is always +abs(txn.amount), the mirror of
    a real, already-negative ledger outflow this same transaction
    contributed elsewhere in the same sum). Confirmed as a real,
    currently-unrepresented pattern on the public dataset: P4 6.1 defines
    "Скорректированная EBITDA" as revenue minus operating expenses "с
    прибавлением разовых статей, признанных аудиторами... подлежащими
    обратному добавлению" (plus one-time items the auditors approve
    adding back) — before this, AuditReclassification had no action that
    could represent "add this back", so even a perfectly-extracted
    finding of this shape had nowhere to go.

    Deliberately does NOT touch effective_category — the flagged
    transaction still counts normally in whatever category it naturally
    falls under (e.g. as a real cost in an *unadjusted* expense covenant).
    Adding +abs(txn.amount) back into the same signed `net` the per-spec
    loop already put -abs(txn.amount) into cancels that one item back to
    zero net effect — which is exactly what "adjusted" means — without
    needing new per-covenant-scoped exclusion machinery.

    Unvalidated against a real extraction: action="addback" didn't exist
    in the schema before this, so no cached document has ever populated
    it — this is forward-looking infrastructure, covered by synthetic
    tests, not a fix confirmed against a live P4 run.
    """
    magnitude = 0.0
    count = 0
    by_txn_id = {t.txn_id: t for t in linked.transactions}
    for reclass in linked.reclassifications.values():
        if reclass.action != "addback":
            continue
        if reclass.txn_id in (excluded_txn_id, revert_txn_id):
            continue
        txn = by_txn_id.get(reclass.txn_id)
        if txn is None or txn.amount is None:
            continue
        label = reclass.original_category or reclass.reasoning
        best = match_category_by_text(label, all_specs)
        if best is None:
            continue
        if best[0] in specs:
            magnitude += abs(txn.amount)
            count += 1
            continue
        local = match_category_by_text(label, specs)
        if local is not None:
            logger.warning(
                "%s: addback reclassification %r also matches this side locally, but its "
                "scenario-wide best match is %r — counted there only, not here, to avoid "
                "double-counting the same addback into two covenants/roles.",
                log_context,
                label,
                best[0].key,
            )
    return magnitude, count


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Numerical safety net only — by the time this runs, compute_metric has
    already ruled out "denominator matched zero transactions" (that raises
    InsufficientDataError instead). This just guards the residual edge case
    of real, counted transactions happening to net to ~$0 (e.g. equal and
    opposite amounts), which is rare but not a categorization failure.
    """
    if denominator > DIVISION_EPSILON:
        return numerator / denominator
    if numerator <= DIVISION_EPSILON:
        return 0.0
    return numerator / DIVISION_EPSILON


def compute_metric(
    clause: CovenantClause,
    linked: LinkedScenarioData,
    *,
    excluded_txn_id: str | None = None,
    revert_txn_id: str | None = None,
) -> float:
    """The `actual` value for `clause` under ledger data with the given
    counterfactual applied (both None = the real, reportable value).

    Raises InsufficientDataError whenever a description-based category sum
    that should never legitimately be zero matched zero transactions — a
    ratio's denominator, an aggregate_amount/other's whole value, or every
    component of a max_single_component test at once. See that exception's
    docstring for the reasoning and for what's deliberately exempt
    (related-party-sourced sides, and max_single_component's partial-miss
    case). Callers (calculation/pipeline.py for the baseline, evidence.py
    for counterfactuals) are responsible for catching it.
    """
    if clause.metric_type == "ratio":
        numerator, num_count = _resolve_side_sum(
            clause,
            "numerator",
            clause.numerator_description,
            linked,
            excluded_txn_id=excluded_txn_id,
            revert_txn_id=revert_txn_id,
        )
        denominator, den_count = _resolve_side_sum(
            clause,
            "denominator",
            clause.denominator_description,
            linked,
            excluded_txn_id=excluded_txn_id,
            revert_txn_id=revert_txn_id,
        )
        # Fix 2 (offline post-fix review, confirmed on P4 6.1): this used
        # to check only the denominator. A numerator that matches zero
        # transactions is the *same* "we don't know", not "genuinely zero",
        # signal InsufficientDataError already exists to catch — but
        # silently returning numerator=0.0 reads as a plausible, ordinary
        # ratio result (e.g. a shortfall against a "min" threshold) instead
        # of the obviously-fabricated numbers a bad denominator produces,
        # so it's easy to miss. Confirmed as a real, live gap: P4 6.1's
        # "Скорректированная EBITDA" numerator matched 0 transactions on
        # both of its sub-specs, sibling-borrow and rescue both came up
        # empty too, and the ratio silently computed 0/revenue = 0.0 —
        # BREACH against a min threshold, with no fallback signal at all.
        if num_count == 0 and _needs_data_check(clause.numerator_description):
            raise InsufficientDataError(
                f"covenant {clause.covenant_key}: numerator "
                f"({clause.numerator_description!r}) matched 0 transactions"
            )
        if den_count == 0 and _needs_data_check(clause.denominator_description):
            raise InsufficientDataError(
                f"covenant {clause.covenant_key}: denominator "
                f"({clause.denominator_description!r}) matched 0 transactions"
            )
        return _safe_ratio(numerator, denominator)

    if clause.metric_type == "max_single_component":
        covenant_specs = [s for s in linked.category_specs if s.covenant_key == clause.covenant_key]
        component_specs = [s for s in covenant_specs if s.role == "component"] or covenant_specs
        component_results = []
        for spec in component_specs:
            total, count = _category_signed_sum(
                spec.key,
                linked,
                covenant_specs,
                clause,
                excluded_txn_id=excluded_txn_id,
                revert_txn_id=revert_txn_id,
            )
            if count == 0:
                borrowed_total, borrowed_count = _borrow_sibling_category_sum(
                    [spec],
                    spec.description,
                    linked,
                    clause,
                    excluded_txn_id=excluded_txn_id,
                    revert_txn_id=revert_txn_id,
                    log_context=f"covenant {clause.covenant_key} component {spec.component_label or spec.key}",
                )
                total += borrowed_total
                count += borrowed_count
            # addback folds into the signed `total` before abs(), same
            # reasoning as _resolve_side_sum — it cancels a cost already
            # counted by the per-spec sum above, it doesn't add a new one.
            addback_delta, addback_count = _addback_magnitude(
                [spec],
                linked.category_specs,
                linked,
                excluded_txn_id=excluded_txn_id,
                revert_txn_id=revert_txn_id,
                log_context=f"covenant {clause.covenant_key} component {spec.component_label or spec.key}",
            )
            total += addback_delta
            count += addback_count
            # Each component is matched against other_facts *individually*
            # (a single-spec list, not the whole role) — components must
            # stay separate for max() to mean anything; summing a fact into
            # every component would be wrong even before the
            # cross-covenant double-count risk _other_facts_magnitude
            # otherwise guards against.
            fact_magnitude, fact_count = _other_facts_magnitude(
                [spec],
                linked.category_specs,
                _effective_other_facts(linked),
                log_context=f"covenant {clause.covenant_key} component {spec.component_label or spec.key}",
            )
            component_results.append((abs(total) + fact_magnitude, count + fact_count))
        # Zero transactions/facts across *every* component at once means
        # the classifier found nothing for this test at all — same "we
        # don't know" signal as a ratio's empty denominator, not "every
        # component is genuinely zero" (implausible for real overhead
        # lines). A partial finding (some components populated, others
        # not) is left alone — genuinely ambiguous, not the clear-cut
        # total-miss case this guards against.
        if component_results and sum(count for _total, count in component_results) == 0:
            raise InsufficientDataError(
                f"covenant {clause.covenant_key}: max_single_component matched 0 "
                f"transactions across all {len(component_specs)} component(s)"
            )
        sums = [total for total, _count in component_results]
        largest = max(sums) if sums else 0.0

        # Prompt fix B (offline post-fix review, confirmed on P10 6.2):
        # "Выручка за вычетом наибольшей из величин..." measures some
        # OTHER amount minus the largest component — not the largest
        # component alone. Reuses _resolve_side_sum so this side gets the
        # exact same safety nets (sibling-borrow, rescue, other_facts,
        # addback) every ratio side already gets, not a stripped-down copy.
        if clause.net_against_description:
            net_amount, net_count = _resolve_side_sum(
                clause,
                "net_against",
                clause.net_against_description,
                linked,
                excluded_txn_id=excluded_txn_id,
                revert_txn_id=revert_txn_id,
            )
            if net_count == 0 and _needs_data_check(clause.net_against_description):
                raise InsufficientDataError(
                    f"covenant {clause.covenant_key}: net_against "
                    f"({clause.net_against_description!r}) matched 0 transactions"
                )
            return net_amount - largest

        return largest

    # aggregate_amount and other (best-effort fallback — see categories.py).
    magnitude, count = _resolve_side_sum(
        clause,
        "amount",
        clause.formula_description,
        linked,
        excluded_txn_id=excluded_txn_id,
        revert_txn_id=revert_txn_id,
    )
    if count == 0 and _needs_data_check(clause.formula_description):
        raise InsufficientDataError(
            f"covenant {clause.covenant_key}: amount ({clause.formula_description!r}) "
            f"matched 0 transactions"
        )
    return magnitude


def compare_to_threshold(actual: float, threshold: float, direction: str) -> str:
    if direction == "max":
        return "COMPLIANT" if actual <= threshold else "BREACH"
    return "COMPLIANT" if actual >= threshold else "BREACH"
