"""Data contract for the ingestion / resolution stage.

Design decisions (so later layers — and future-me — don't have to guess):

* Plain ``@dataclass(frozen=True)`` instead of Pydantic. We don't need
  network/JSON-boundary validation at this layer — inputs are local files we
  fully control the reading of. Pydantic turned out to be exactly the right
  tool one boundary over: ``schemas.py`` uses it for every LLM Structured
  Output (Block 2), where the SDK itself enforces the schema on a boundary
  we don't control. ``ScenarioFacts`` below is the seam between the two —
  a frozen dataclass (like everything in this file) whose fields hold
  pydantic result objects from ``schemas.py``. Frozen dataclasses give us
  "can't be mutated after resolution" for free, which is the property that
  actually matters here: once a ScenarioBundle or ScenarioFacts is built,
  downstream layers must be able to trust it without wondering if some
  earlier step is still writing to it.

* Every model carries enough provenance (doc_id, source_path, raw signals
  that drove a decision) to answer "why did the pipeline think this was the
  current credit agreement?" without re-parsing anything. That provenance is
  what layer 7 (Explanation/Evidence) will read from directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedDocument:
    """Raw text extracted from one file under documents/, plus cache bookkeeping."""

    doc_id: str  # filename stem, e.g. "8d878af064f2" — opaque, not meaningful
    source_path: Path
    file_type: str  # "pdf" | "csv" | "txt" | other suffix, lowercase, no dot
    text: str
    char_count: int
    from_cache: bool  # True if this run read cached text instead of re-parsing


@dataclass(frozen=True)
class Transaction:
    """One row of master_ledger_2025.csv."""

    txn_id: str
    date: str  # kept as the raw ISO string (YYYY-MM-DD); parsed to date only where needed
    account_id: str
    scenario_id: str  # derived from txn_id's middle segment, not stored in the CSV
    counterparty: str
    description: str
    # Sign preserved as given: negative = outflow, positive = inflow.
    # None means the ledger's amount cell was empty/unparseable for this
    # row — a handful of rows in the public dataset are genuinely dirty
    # this way (see ingestion/ledger.py). We keep the row rather than
    # dropping it: the txn_id/counterparty/description are still real
    # facts, and recovering the true amount from a supporting document is
    # exactly the kind of "hidden fact" this challenge is testing for.
    amount: float | None
    currency: str


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentMetadata:
    """Signals extracted from a document's text that drive scenario/version resolution.

    This is deliberately a bag of *signals*, not a verdict — the verdict
    (which document is "current" for a given scenario+kind) is computed once
    in resolution/versioning.py and recorded on ScenarioBundle, so there is
    exactly one place that owns that decision.
    """

    doc_id: str
    account_tokens: tuple[str, ...]  # every ACC-<digits>[-<digits>] token found, verbatim
    matched_scenario_accounts: tuple[str, ...]  # exact matches against known accounts only
    company_names: tuple[str, ...]  # secondary, human-readable cue — never used for resolution
    kind: str  # "credit_agreement" | "kyc_dossier" | "audit_report" | "treasury_memo" | "other"
    kind_score: int  # keyword-hit count behind the kind decision; 0 = pure default/"other"
    is_superseded: bool
    supersede_reasons: tuple[str, ...]  # matched marker phrases, for the audit trail
    revision: int | None  # e.g. 3 for "v3" / "редакция 3", None if not found
    dates_found: tuple[str, ...]  # ISO-normalized (YYYY-MM-DD), de-duplicated, sorted
    latest_date: str | None


@dataclass(frozen=True)
class ResolvedDocument:
    """A document paired with the metadata computed about it."""

    parsed: ParsedDocument
    metadata: DocumentMetadata


@dataclass(frozen=True)
class ScenarioBundle:
    """Everything resolved for one borrower (scenario_id), ready for fact extraction."""

    scenario_id: str  # e.g. "P1", "B4" — matches submission_template.json keys
    account_id: str  # e.g. "ACC-7801" — derived from the ledger, never hardcoded

    # Current (non-superseded) documents, grouped by kind. A tuple, not a
    # single document: "exactly one current version" is only a meaningful
    # constraint for kinds where the dataset actually maintains version
    # history (credit_agreement, kyc_dossier — see versioning.py) and
    # ends up length-1 there because the superseded copies get filtered
    # into superseded_documents below. For kinds like audit_report or
    # treasury_memo, multiple genuinely-distinct documents legitimately
    # coexist (e.g. two audit reports about two different transactions) —
    # forcing a single pick there would silently discard real facts, so we
    # don't. A kind is simply absent from this dict if no matching document
    # was found — callers must handle that explicitly (a missing
    # credit_agreement is a real failure mode worth surfacing loudly).
    current_documents: dict[str, tuple[ResolvedDocument, ...]] = field(default_factory=dict)

    # Older/draft versions we deliberately excluded, kept for traceability
    # (e.g. so a human — or the Explanation layer — can show "we used the
    # 2025 agreement, not the 2024 one, because the 2024 copy is stamped
    # superseded").
    superseded_documents: tuple[ResolvedDocument, ...] = ()

    # Every document that matched this scenario's account, current or not.
    # len(all_matched_documents) >= len(current_documents) + len(superseded_documents)
    # is a useful sanity invariant (equality can fail if the same kind had an
    # unresolved tie — see versioning.py).
    all_matched_documents: tuple[ResolvedDocument, ...] = ()


@dataclass(frozen=True)
class IngestionResult:
    """Top-level output of Block 1 — the input contract for Block 2 (fact extraction)."""

    ledger: list[Transaction]
    template: dict  # raw parsed submission_template.json (scenario_id -> covenant -> null cells)
    scenarios: dict[str, ScenarioBundle]  # keyed by scenario_id
    unmatched_documents: tuple[ResolvedDocument, ...]  # matched no known scenario account at all


# ---------------------------------------------------------------------------
# Block 2 output (fact extraction) — see covenant_agent/schemas.py for the
# pydantic models referenced below, and covenant_agent/extraction/ for how
# they get produced.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioFacts:
    """Everything Block 2 extracted for one borrower — the input to Block 3
    (Linking + Decision/Calculation).

    Every field is Optional/empty-collection-safe by design: a missing
    credit agreement, missing KYC dossier, or zero audit reports are all
    real, observed states in the public dataset (see README "Findings"),
    and Block 3 must be able to compute *something* defensible for every
    covenant regardless — an empty submission cell scores the same as a
    wrong one.
    """

    scenario_id: str

    # None only if this scenario had no current credit_agreement document
    # at all (Block 1 failure) or the extraction call itself failed — both
    # are real failure modes that must propagate as "we have nothing here",
    # never as a silently empty CovenantExtractionResult.
    covenants: "CovenantExtractionResult | None"

    # None if the scenario has no current KYC dossier (observed for real in
    # the public dataset — see README). Block 3's related-party logic must
    # treat this as "no disclosures found", not "zero related parties
    # confirmed" — those are different claims.
    kyc: "KycExtractionResult | None"

    # (doc_id, result) pairs rather than a bare list: provenance matters
    # here specifically because Block 1 may occasionally leave more than
    # one current audit_report for a scenario (unresolved version tie), and
    # the Explanation layer needs to say which document a given
    # reclassification came from.
    audit_reports: "tuple[tuple[str, AuditExtractionResult], ...]" = ()

    # Same shape, covering every other current document matched to this
    # scenario that isn't a credit agreement, KYC dossier, or audit report
    # (treasury memos today; whatever else the private dataset has).
    other_facts: "tuple[tuple[str, OtherFactsExtractionResult], ...]" = ()


# Imported at the bottom, not the top: these are pydantic models used only
# as type annotations above (all quoted as strings so this stays a
# forward-reference until resolved). Keeping the import local to this
# section makes the dependency direction explicit — models.py depends on
# schemas.py, never the reverse.
from covenant_agent.schemas import (  # noqa: E402
    AuditExtractionResult,
    CovenantExtractionResult,
    KycExtractionResult,
    OtherFactsExtractionResult,
)


# ---------------------------------------------------------------------------
# Block 3 (linking) output contract.
#
# These four small dataclasses (CategorySpec, RelatedPartyMatch,
# LinkedReclassification, UnmatchedReclassification) are defined *here*,
# not in the covenant_agent/linking/*.py modules that construct them —
# deliberately, to keep the dependency graph one-directional. Every linking
# submodule imports Transaction (and these types) from models.py; if
# LinkedScenarioData instead referenced types defined inside e.g.
# reclassification_linking.py, this file would have to import from that
# module too, and that module already imports *this* file — a cycle. All
# pipeline-stage output contracts living in one leaf module (this one) is
# the same rule Block 1/2 already followed for IngestionResult/ScenarioFacts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategorySpec:
    """One transaction-classification bucket, derived from a single covenant
    clause's own numerator/denominator/component text — see
    linking/categories.py:derive_category_specs. Never a fixed taxonomy.

    A role (numerator/denominator/amount) can map to *more than one* spec:
    these descriptions are sometimes themselves a net figure — confirmed on
    the public dataset, B1's covenant 6.1 numerator is "EBITDA (Выручка за
    вычетом Операционных расходов)", i.e. revenue net of operating
    expenses, not one homogeneous bucket. categories.py splits a detected
    "X net of Y" description into two specs sharing the role instead of
    asking the transaction classifier to somehow match one compound
    concept. calculation/formulas.py sums every spec sharing a role using
    each transaction's *raw ledger sign* (no artificial sign flip needed —
    outflows are already negative in the source data, so summing revenue
    [positive] with operating expenses [already negative] nets them
    correctly on its own). A description with no netting language yields
    exactly one spec per role, so untouched covenants compute exactly as
    a single bucket would.
    """

    key: str  # stable slug, e.g. "6.1_numerator", "6.2_amount", "6.2_component_0"
    covenant_key: str
    role: str  # "numerator" | "denominator" | "amount" | "component"
    description: str  # human-readable text fed to the transaction classifier prompt
    component_label: str | None = None  # original component name, role=="component" only


@dataclass(frozen=True)
class RelatedPartyMatch:
    """A KYC-disclosed entity matched to a ledger counterparty, with the
    related-party comparison already done in code — see
    linking/related_parties.py:resolve_related_parties.
    """

    ledger_counterparty: str  # exact string as it appears in the ledger
    kyc_name: str  # the KYC disclosure's own name for this entity
    ownership_pct: float | None
    threshold_pct: float | None
    is_related: bool
    basis: str  # human-readable reason, for logging/evidence


@dataclass(frozen=True)
class LinkedReclassification:
    """An auditor reclassification joined to a specific ledger transaction —
    see linking/reclassification_linking.py:link_reclassifications.
    """

    txn_id: str
    original_category: str
    reclassified_category: str
    reasoning: str
    source_doc_id: str
    match_confidence: float  # 1.0 = exact counterparty + exact amount; lower if fuzzy/ambiguous
    was_ambiguous: bool


@dataclass(frozen=True)
class UnmatchedReclassification:
    """An auditor reclassification that could NOT be joined to any ledger
    transaction — kept (not discarded) so it stays visible in debug output.
    """

    counterparty_name: str | None
    amount: float | None
    source_doc_id: str
    reason: str


@dataclass(frozen=True)
class LinkedScenarioData:
    """Block 3a's output for one scenario — the input to Block 3b/3c
    (calculation/formulas.py, calculation/evidence.py).

    `txn_category` covers every transaction with a description-based
    category assignment (or the literal "unclassified"); transactions
    absent from it were never classifiable (e.g. no category vocabulary was
    needed for this scenario's covenants at all). `reclassifications` only
    covers the subset of txn_ids an audit finding was actually linked to —
    absence means "never reclassified", not "explicitly confirmed
    unchanged". `related_parties` is keyed by the *ledger's own* spelling of
    a counterparty name (post fuzzy-match), so calculation code can look up
    `related_parties.get(txn.counterparty)` directly.
    """

    scenario_id: str
    transactions: list[Transaction]  # this scenario's exact-account transactions only
    category_specs: list[CategorySpec]
    txn_category: dict[str, str]
    reclassifications: dict[str, LinkedReclassification]
    unmatched_reclassifications: list[UnmatchedReclassification]
    related_parties: dict[str, RelatedPartyMatch]


# ---------------------------------------------------------------------------
# Block 3b/3c (calculation) output contract — one CovenantResult per
# submission cell. This is the direct precursor to the actual
# submission.json cell (Block 5 just needs status/actual/evidence_txn_id,
# in that priority order per the case's own scoring weights); the other
# fields exist so a human — or the Explanation layer — can see *why*
# without re-running the calculation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CovenantResult:
    covenant_key: str
    status: str  # "COMPLIANT" | "BREACH"
    actual: float
    evidence_txn_id: str | None
    used_fallback: bool  # True only when there was no covenant rule to evaluate at all
    fallback_reason: str | None
    calculation_notes: tuple[str, ...] = ()  # human-readable trace, for debugging/explanation
