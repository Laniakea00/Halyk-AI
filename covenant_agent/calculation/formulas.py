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

from covenant_agent.linking.categories import is_related_party_text, match_category_by_text
from covenant_agent.models import CategorySpec, LinkedScenarioData, Transaction
from covenant_agent.schemas import CovenantClause


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
    because a real zero is a normal outcome there.
    """
    return not is_related_party_text(description_text)


# Numerical safety net for _safe_ratio only, once InsufficientDataError has
# already ruled out "denominator matched zero transactions" — see that
# exception's docstring for why this is no longer the primary defense
# against a fabricated ratio.
DIVISION_EPSILON = 0.01


def in_period(txn: Transaction, clause: CovenantClause) -> bool:
    """True unless `clause` states a period bound `txn.date` falls outside of.

    Confirmed necessary on the public dataset: B4's covenant 6.1 measures
    revenue for "четвёртый финансовый квартал" (Q4) only, not the full
    year — 2a correctly extracts period_start=2025-10-01/period_end=2025-12-31
    for it, but nothing downstream applied that bound before this function
    existed, silently summing the whole year's revenue into a
    Q4-only test. ISO YYYY-MM-DD strings compare correctly lexicographically,
    so no date parsing is needed here.
    """
    if clause.period_start is not None and txn.date < clause.period_start:
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
      resolved via the reclassification's own target text, matched against
      this covenant's category descriptions (match_category_by_text) —
      this is the normal, "reclassification applied" path.
    - `txn.txn_id == revert_txn_id`: the reclassification is deliberately
      *not* applied — simulates "what if the auditor's reclassification is
      undone for this one transaction", per the case's own evidence
      definition, which names reclassification as one of the operations
      whose reversal can flip a verdict.
    - Otherwise: the transaction's base (description-only) category from
      Block 3a's classifier.
    """
    if txn.txn_id == excluded_txn_id:
        return None

    reclass = linked.reclassifications.get(txn.txn_id)
    if reclass is not None and txn.txn_id != revert_txn_id:
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
    linked: LinkedScenarioData, clause: CovenantClause, *, excluded_txn_id: str | None
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
    """
    if is_related_party_text(description_text):
        total = _related_party_sum(linked, clause, excluded_txn_id=excluded_txn_id)
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
    return abs(net), count


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
        numerator, _num_count = _resolve_side_sum(
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
        if den_count == 0 and _needs_data_check(clause.denominator_description):
            raise InsufficientDataError(
                f"covenant {clause.covenant_key}: denominator "
                f"({clause.denominator_description!r}) matched 0 transactions"
            )
        return _safe_ratio(numerator, denominator)

    if clause.metric_type == "max_single_component":
        covenant_specs = [s for s in linked.category_specs if s.covenant_key == clause.covenant_key]
        component_specs = [s for s in covenant_specs if s.role == "component"] or covenant_specs
        component_results = [
            _category_signed_sum(
                spec.key,
                linked,
                covenant_specs,
                clause,
                excluded_txn_id=excluded_txn_id,
                revert_txn_id=revert_txn_id,
            )
            for spec in component_specs
        ]
        # Zero transactions across *every* component at once means the
        # classifier found nothing for this test at all — same "we don't
        # know" signal as a ratio's empty denominator, not "every
        # component is genuinely zero" (implausible for real overhead
        # lines). A partial finding (some components populated, others
        # not) is left alone — genuinely ambiguous, not the clear-cut
        # total-miss case this guards against.
        if component_results and sum(count for _sum, count in component_results) == 0:
            raise InsufficientDataError(
                f"covenant {clause.covenant_key}: max_single_component matched 0 "
                f"transactions across all {len(component_specs)} component(s)"
            )
        sums = [abs(total) for total, _count in component_results]
        return max(sums) if sums else 0.0

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
