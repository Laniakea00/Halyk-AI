"""Derive a per-scenario, per-covenant transaction-category vocabulary.

The master ledger has no category column by design (the case brief says so
explicitly) — every covenant that sums or ratios transaction amounts by
concept ("capital expenditure", "operating expense", "revenue", "interest
expense", ...) needs those concepts assigned to transactions first, and
nothing upstream does that yet. This module is the fix: for each covenant
clause, it derives the *category keys a transaction classifier should use*,
directly from that clause's own extracted numerator/denominator/component
descriptions — never a fixed, hardcoded taxonomy, so a private-dataset
covenant phrased around some other concept still gets a vocabulary that
fits it.

One category dimension is deliberately excluded here: related-party
payments. Whether a transaction counts toward a related-party test is a
question of *who the counterparty is* (an identity/ownership fact from KYC),
not what the transaction's own description says — so it's resolved in related_parties.py
by comparing ownership percentages to the document's own threshold, in
code, never by asking the transaction-description classifier to guess it.
`is_related_party_text()` is how a covenant side gets routed to that
dimension instead of getting a category key here.
"""

from __future__ import annotations

import re

from covenant_agent.models import CategorySpec
from covenant_agent.schemas import CovenantClause, CovenantExtractionResult

_RELATED_PARTY_RE = re.compile(
    r"связанн|аффилиров|related.?part|affiliat", re.IGNORECASE
)
_PUNCTUATION_RE = re.compile(r"[.,;:()\"'«»]")
_WHITESPACE_RE = re.compile(r"\s+")

# How confident a match_category_by_text() result has to be before it's
# trusted. Deliberately permissive: reclassification text ("Процентные
# расходы") is short, category descriptions are long clause paraphrases —
# a clean substring containment is common and should score comfortably
# above this; free-floating word overlap between unrelated concepts should
# not.
CATEGORY_MATCH_THRESHOLD = 0.35


def is_related_party_text(text: str | None) -> bool:
    """True if `text` is about related/affiliated parties, by keyword.

    Same style as Block 1's doc-kind classifier: cheap, keyword-based, and
    deliberately over-inclusive rather than under — a false positive here
    just means a covenant side gets resolved by the related-party dimension
    when it maybe didn't need to (harmless if there happen to be no related
    parties involved); a false negative would silently misclassify a real
    related-party test as an ordinary expense category, which is worse.
    """
    if not text:
        return False
    return bool(_RELATED_PARTY_RE.search(text))


_UNRESTRICTED_SUBSIDIARY_RE = re.compile(
    r"неограниченн\w*\s+дочерн\w*|unrestricted\s+subsidiar\w*", re.IGNORECASE
)


def is_unrestricted_subsidiary_text(text: str | None) -> bool:
    """True if `text` is about "Unrestricted Subsidiary"-style designation.

    Confirmed on the public dataset: P9's covenant 6.1 numerator
    ("активов, переданных Неограниченным дочерним организациям") names a
    counterparty-identity question — which entities carry this
    designation — but checking all of P9's own documents (KYC dossier,
    financial_notes, credit agreement, full text) directly found *no*
    disclosure anywhere naming which counterparty, if any, is designated
    an Unrestricted Subsidiary; the term only appears inside the credit
    agreement's own definition of the concept. A genuine zero for this
    side (no such transfer disclosed) is therefore a normal business fact,
    not a categorization miss — same reasoning as is_related_party_text's
    InsufficientDataError exemption (see formulas.py's
    _needs_data_check), reused here rather than building a parallel
    related-party-style resolution mechanism with zero real examples to
    validate it against (see README's Task B design note for the full
    tradeoff). Deliberately narrow — requires "неограниченн" AND "дочерн"
    together (or the English equivalent), not a bare "дочерн" check, which
    would also fire on ordinary subsidiary mentions that have nothing to
    do with this designation and don't deserve the same exemption. This
    does NOT route the side away from ordinary transaction categorization
    the way is_related_party_text does (see derive_category_specs) — only
    exempts a resulting zero from being flagged as suspicious.
    """
    if not text:
        return False
    return bool(_UNRESTRICTED_SUBSIDIARY_RE.search(text))


def derive_category_specs(covenants: CovenantExtractionResult) -> list[CategorySpec]:
    """One CategorySpec per (covenant, role) that needs description-based
    transaction classification — i.e. everything except related-party
    sides, which route to related_parties.py instead.
    """
    specs: list[CategorySpec] = []
    for clause in covenants.covenants:
        specs.extend(_specs_for_clause(clause))
    return specs


def _specs_for_clause(clause: CovenantClause) -> list[CategorySpec]:
    if clause.metric_type == "ratio":
        specs = []
        if clause.numerator_description and not is_related_party_text(clause.numerator_description):
            specs.extend(
                _make_side_specs(
                    clause.covenant_key, "numerator", clause.numerator_description, allow_split=True
                )
            )
        if clause.denominator_description and not is_related_party_text(
            clause.denominator_description
        ):
            specs.extend(
                _make_side_specs(
                    clause.covenant_key, "denominator", clause.denominator_description, allow_split=True
                )
            )
        return specs

    if clause.metric_type == "max_single_component":
        if clause.components:
            return [
                CategorySpec(
                    key=f"{clause.covenant_key}_component_{i}",
                    covenant_key=clause.covenant_key,
                    role="component",
                    description=f"{clause.metric_name}: {label}",
                    component_label=label,
                )
                for i, label in enumerate(clause.components)
            ]
        # Extraction gap (2a returned max_single_component with no
        # components listed): fall back to a single aggregate bucket so
        # calculation still has *something* to sum, rather than silently
        # computing 0 for every candidate. Logged by the caller, not here —
        # this module has no logger of its own by design (pure derivation).
        return [
            CategorySpec(
                key=f"{clause.covenant_key}_amount",
                covenant_key=clause.covenant_key,
                role="amount",
                description=clause.formula_description,
            )
        ]

    # aggregate_amount and other (best-effort: treat like aggregate_amount).
    # allow_split=False: formula_description is a full paraphrase, not a
    # short focused field — see _make_side_specs docstring for why
    # splitting it is unsafe (confirmed regression on P2's 6.2).
    if is_related_party_text(clause.formula_description):
        return []
    return _make_side_specs(clause.covenant_key, "amount", clause.formula_description, allow_split=False)


# Two shapes of compound description, both confirmed on the public
# dataset, both fixed the same way: split into one clean sub-description
# per underlying concept, so the transaction classifier is never asked to
# match one blob covering two different things.
#
# 1. Netting/subtraction — B1's 6.1 numerator: "EBITDA (Выручка за вычетом
#    Операционных расходов)" = revenue MINUS operating expenses.
# 2. Additive — P1's 6.1 denominator: "сумма операционных расходов и
#    арендных платежей" = operating expenses PLUS rent (two sibling
#    categories being totaled, not one netted against the other).
#
# Both reduce to "create one CategorySpec per part, same role" — no sign
# bookkeeping needed in either case: formulas.py sums every spec sharing a
# role using transactions' *raw ledger sign*, and outflows are already
# negative in this dataset's convention, so revenue (positive) + opex
# (already negative) subtracts correctly on its own, while opex (negative)
# + rent (negative) simply combines into a larger negative total either way.
_NETTING_RE = re.compile(
    r"(?P<a>.+?)\s*(?:за вычетом|уменьшенн\w*\s+на|минус|net of|less|excluding|без учёта)\s*"
    r"(?P<b>.+)",
    re.IGNORECASE,
)
_ADDITIVE_RE = re.compile(
    r"(?:сумм[а-я]*\s+)?(?P<a>.+?)\s+(?:и|and|плюс|plus)\s+(?P<b>.+)",
    re.IGNORECASE,
)


_MAX_COMPOUND_PARTS = 6  # safety cap against pathological/runaway splitting


def _split_compound(text: str) -> list[str]:
    """Split a compound description into parts, handling *chains*, not just pairs.

    Confirmed necessary on the public dataset: P3's 6.1 denominator is
    "EBITDA ... (выручка минус операционные расходы, минус расходы на
    амортизацию, минус расходы на налоги, минус расходы на проценты)" — a
    four-way subtraction chain, not a single "X net of Y" pair. A
    one-shot split (match once, return two parts) turns this into one
    clean part ("выручка") and one *still-compound* blob covering the
    other three concepts glued together — exactly the "one unfocused
    bucket" problem this function exists to avoid, just pushed one level
    deeper. Splitting instead runs to a fixed point: keep re-applying the
    connector patterns to every part until nothing splits further (or the
    safety cap is hit).

    Returns [text] unchanged if it doesn't match either shape at all —
    most covenant descriptions are already a single clean concept. Callers
    decide *whether* to call this at all based on which field `text` came
    from (see _make_side_specs's `allow_split`) — splitting is only safe
    on numerator_description/denominator_description, which are meant to
    be short, focused phrases per their own schema description.
    formula_description is a full paraphrase, often several sentences of
    carve-outs and qualifiers, and must never be split: confirmed as a
    real bug on P2's 6.2, where formula_description's *second sentence*,
    deep in a reclassification carve-out, happened to contain "за
    вычетом" — splitting on it produced one giant, unfocused blob (nearly
    the whole paragraph) and one tiny fragment, regressing a covenant that
    classified correctly as a single bucket before this function existed.
    """
    parts = [text]
    for _ in range(_MAX_COMPOUND_PARTS):
        next_parts: list[str] = []
        split_any = False
        for part in parts:
            split = None
            for pattern in (_NETTING_RE, _ADDITIVE_RE):
                match = pattern.search(part)
                if not match:
                    continue
                a = match.group("a").strip(" (),.")
                b = match.group("b").strip(" ().,")
                if a and b:
                    split = (_borrow_trailing_noun(a, b), b)
                    break
            if split:
                next_parts.extend(split)
                split_any = True
            else:
                next_parts.append(part)
        parts = next_parts
        if not split_any or len(parts) >= _MAX_COMPOUND_PARTS:
            break
    return [_strip_defined_as_prefix(p) for p in parts]


# Confirmed necessary on the public dataset: P10's 6.1 denominator "сумме
# Арендных и Коммунальных расходов" (rental and utility expenses) splits
# grammatically correctly on "и" into "Арендных" / "Коммунальных расходов"
# — but "Арендных" ("rental", a bare adjective) is missing the noun
# "расходов" ("expenses") that Russian grammar elides after the first of
# two adjectives sharing one trailing noun. A category worded as a bare
# adjective with nothing to agree with is meaningless to the transaction
# classifier (there is no ledger line ever literally described as just
# "rental"). Same pattern confirmed on P2's 6.1 denominator ("операционных
# и капитальных затрат" -> "операционных" / "капитальных затрат").
_ADJECTIVE_SUFFIX_RE = re.compile(
    r"(?:ых|их|ого|его|ому|ему|ый|ий|ая|яя|ое|ее|ой|ей|ые|ие)$", re.IGNORECASE
)


def _borrow_trailing_noun(a: str, b: str) -> str:
    """If `a` is a single bare adjective, append `b`'s trailing word (the
    noun presumed shared/elided across both sides of the "и") to it.

    Deliberately narrow: only fires when `a` is exactly one word (a phrase
    like "Операционных расходов" already has its own noun and is left
    alone) and that word has a Russian adjective ending. `b` must have at
    least two words to have a noun worth borrowing at all. Doesn't attempt
    full morphological agreement (case/gender matching) — just concatenates
    the words, which is enough for the transaction classifier's purposes
    (a semantically complete phrase to match against, not grammatically
    perfect prose).
    """
    a_words = a.split()
    b_words = b.split()
    if len(a_words) != 1 or len(b_words) < 2:
        return a
    if not _ADJECTIVE_SUFFIX_RE.search(a_words[0]):
        return a
    return f"{a} {b_words[-1]}"


# Confirmed necessary on the public dataset: P5's 6.1 denominator is "EBITDA
# Заёмщика, рассчитываемая по его собственной отчётности как Выручка за
# вычетом Операционных расходов" — _split_compound correctly nets this into
# two parts on "за вычетом", but the first part retains the full "EBITDA
# Заёмщика, рассчитываемая... как" preamble, and a category worded that way
# found zero transaction matches on every run tested, while a plainly-worded
# "Выручка" category elsewhere in the same scenario (6.2_amount) matched
# fine — the preamble, not the underlying concept, appears to be what
# confuses the classifier. EBITDA is never a literal transaction-level
# concept (nothing in a ledger is ever labeled "EBITDA"); once a category is
# phrased as "EBITDA ... as X", X is the actual matchable concept and the
# EBITDA framing is redundant by that point — the split already captured
# EBITDA as "revenue net of opex" via the two parts themselves, so nothing
# is lost by dropping the label wording here.
_DEFINED_AS_RE = re.compile(
    r"^\s*(?:EBITDA|EBIT|чистая прибыль|валовая прибыль)\b.*?\bкак\s+(?P<definition>.+?)\s*$",
    re.IGNORECASE,
)


def _strip_defined_as_prefix(text: str) -> str:
    """Drop a leading "EBITDA ... как <X>" framing down to just <X>.

    Deliberately narrow (only fires when the part starts with a known
    aggregate/derived-metric name and contains "как") — see the comment
    above for why. Returns `text` unchanged if the pattern doesn't match.
    """
    match = _DEFINED_AS_RE.match(text)
    return match.group("definition") if match else text


def _make_side_specs(
    covenant_key: str, role: str, description: str, *, allow_split: bool
) -> list[CategorySpec]:
    """One CategorySpec per part of `description` (see _split_compound), sharing `role`.

    `allow_split` must be True only for numerator_description/
    denominator_description callers — see _split_compound's docstring.
    """
    parts = _split_compound(description) if allow_split else [description]
    if len(parts) == 1:
        return [
            CategorySpec(
                key=f"{covenant_key}_{role}",
                covenant_key=covenant_key,
                role=role,
                description=parts[0],
            )
        ]
    return [
        CategorySpec(
            key=f"{covenant_key}_{role}_part{i}",
            covenant_key=covenant_key,
            role=role,
            description=part,
        )
        for i, part in enumerate(parts)
    ]


def _normalize(text: str) -> str:
    text = _PUNCTUATION_RE.sub(" ", text.lower())
    return _WHITESPACE_RE.sub(" ", text).strip()


_MIN_STEM_WORD_LEN = 4
_STEM_CHARS = 5

# Generic financial-document vocabulary that shows up in nearly *every*
# covenant/category description — "expenses", "costs", "payments",
# "period", "aggregate/total" — and therefore has no discriminating power
# for telling categories apart. Confirmed as a real bug on B1: the
# reclassification target "Процентные расходы" (interest EXPENSES) scored
# a false-positive 0.5 overlap against the unrelated "payroll expenses"
# component purely because both share the stem for "расходы" (expenses) —
# the one word that actually mattered ("процентные"/interest) was ignored
# because the *other* word happened to clear the threshold on its own.
# Stems here are dropped before the overlap score is computed, not just
# down-weighted, so a match must be earned on the words that actually
# distinguish one category from another.
_STOPWORD_STEMS = {
    "расхо",  # расходы/расход(ов/ам/ами) — expenses
    "затра",  # затраты/затрат(ам) — costs
    "плате",  # платеж(и/ам/ей) — payment(s)
    "перио",  # период(а/е/ом) — period
    "совок",  # совокупный/-ая/-ые/-ого/-ых — aggregate/total
    "сумма", "сумму", "суммы", "суммо",  # сумма and its case forms
    "стать", "статы",  # статья/статьи — line item
}


def _stems(text: str) -> set[str]:
    """Crude, language-agnostic stemming: first _STEM_CHARS of each
    significant word, minus generic financial stopwords (see
    _STOPWORD_STEMS). Russian is heavily case-inflected — "процентные"
    (nominative) vs "процентных"/"процентным" (genitive/dative) share no
    exact substring but share the stem "проце" — so plain substring or
    whole-string SequenceMatcher comparisons silently fail on exactly the
    kind of short free-text-vs-clause-paraphrase matching this function
    exists for. Confirmed on the real B1 case (see module docstring
    example below): without stemming, "Процентные расходы" scored *higher*
    against an unrelated EBITDA description than against its actual match,
    purely on raw character overlap. Words shorter than _MIN_STEM_WORD_LEN
    are dropped (prepositions/conjunctions) so they can't pad the score.
    """
    return {
        word[:_STEM_CHARS]
        for word in _normalize(text).split(" ")
        if len(word) >= _MIN_STEM_WORD_LEN and word[:_STEM_CHARS] not in _STOPWORD_STEMS
    }


def match_category_by_text(text: str, specs: list[CategorySpec]) -> tuple[CategorySpec, float] | None:
    """Best CategorySpec whose description text matches `text`, or None.

    Used to resolve what an auditor's *free-text* reclassified_category
    ("Процентные расходы") means in terms of *this scenario's* category
    keys — confirmed on the real B1 case: covenant 6.1's own
    denominator_description independently mentions "...в состав Процентных
    расходов..." (a different grammatical case of the same words), which
    stem-overlap resolves correctly where substring/SequenceMatcher did not.

    Returns (spec, score) so the caller can log low-confidence matches;
    None if nothing clears CATEGORY_MATCH_THRESHOLD, or if `text` has no
    stems left after dropping short words (nothing meaningful to match on).
    """
    if not text or not specs:
        return None

    target_stems = _stems(text)
    if not target_stems:
        return None

    best: tuple[CategorySpec, float] | None = None
    for spec in specs:
        candidate_stems = _stems(spec.description)
        if not candidate_stems:
            continue
        overlap = len(target_stems & candidate_stems) / len(target_stems)
        if best is None or overlap > best[1]:
            best = (spec, overlap)

    if best is not None and best[1] >= CATEGORY_MATCH_THRESHOLD:
        return best
    return None
