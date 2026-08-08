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

import re

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
    role's own category text, via the same match_category_by_text stem-
    overlap machinery already used to resolve a reclassification's target
    category — confirmed necessary on the public dataset: P8's 6.1
    ("Совокупные обязательства по персоналу") is genuinely the sum of paid
    payroll transactions *plus* a severance-program obligation that will
    never appear as a ledger row. A matched fact counts toward `count` too,
    so a covenant whose ledger side is empty but has a real off-ledger
    figure doesn't spuriously raise InsufficientDataError.

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

    fact_magnitude = 0.0
    for fact in linked.other_facts:
        if fact.value is None:
            continue
        match = match_category_by_text(fact.fact_description, role_specs)
        if match is not None:
            fact_magnitude += fact.value
            count += 1

    return abs(net) + fact_magnitude, count


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
