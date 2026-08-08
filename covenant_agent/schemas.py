"""Pydantic schemas for every LLM Structured Output call in the pipeline.

These are the *only* place the LLM's output shape is defined. Every field
has a description because that description is literally what the model
sees as field-level instruction under Structured Outputs — it is prompt
text, not documentation.

None of these schemas contain a compliance verdict, a computed boolean, or
anything requiring judgment beyond "what does this text say". That split is
intentional and mirrors the hackathon's core rule: the LLM extracts and
structures, code decides. Concretely:

* CovenantClause captures the *rule* (formula, threshold, direction), never
  whether it's met — Block 3 (Decision/Calculation) does that arithmetic
  against the ledger.
* RelatedPartyDisclosure captures the ownership *percentage* and the
  document's own stated *threshold*, never a computed "is this a related
  party" boolean — Block 3 does that comparison. The one boolean it does
  carry, `explicitly_labeled_related_party`, is a literal reading of
  whether the source text itself already categorizes the entity that way
  (e.g. listed under a "related parties" heading) — that's extraction of an
  explicit textual fact, not a judgment the model is making up.
* AuditReclassification captures what changed and why, never which
  covenant it affects or whether that flips a verdict — Block 3's linking
  step joins this back to ledger transactions by counterparty/amount/date
  (these reports don't reliably cite txn_id directly) and Block 3's
  calculation step figures out the consequence.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 2a: Covenant clause extraction
# ---------------------------------------------------------------------------


class CovenantClause(BaseModel):
    covenant_key: str = Field(
        description="The clause number exactly as labelled in the source text, e.g. '6.1'."
    )
    metric_name: str = Field(
        description="The name of the test as given in the source text, in the source "
        "document's own language, e.g. 'Maximum Capital Intensity Ratio' or its "
        "native-language equivalent."
    )
    metric_type: Literal["ratio", "aggregate_amount", "max_single_component", "other"] = Field(
        description="Shape of the test: 'ratio' = one quantity divided by another, compared "
        "to a multiple (e.g. 0.42x); 'aggregate_amount' = a summed dollar total compared to a "
        "limit; 'max_single_component' = compliance is judged on whichever named component is "
        "largest, not their sum (read the clause carefully for this — it is easy to miss); "
        "'other' = any shape not covered above, explained in formula_description."
    )
    formula_description: str = Field(
        description="Plain-language description, in the source document's own language, of "
        "exactly what is measured — close to a faithful paraphrase of the clause, not a "
        "translation and not a summary that drops detail."
    )
    numerator_description: Optional[str] = Field(
        description="For ratio tests: what the numerator is, in the source language. "
        "Null for non-ratio tests."
    )
    denominator_description: Optional[str] = Field(
        description="For ratio tests: what the denominator is, in the source language. "
        "Null for non-ratio tests."
    )
    components: list[str] = Field(
        description="For metric_type='max_single_component' only: the name of each named "
        "component being compared, each a short label in the source language (e.g. "
        "['расходы на оплату труда', 'расходы на коммунальные услуги']). One entry per "
        "component named in the clause — do not merge them. Empty list for every other "
        "metric_type."
    )
    threshold_value: float = Field(
        description="The numeric limit stated in the clause. Always positive, even if the "
        "clause concerns an expense or outflow."
    )
    threshold_unit: Literal["ratio", "usd"] = Field(
        description="'ratio' if threshold_value is a multiple (e.g. 0.42 meaning 0.42x); "
        "'usd' if it is a dollar amount."
    )
    direction: Literal["max", "min"] = Field(
        description="'max' if the measured value must not exceed threshold_value; 'min' if "
        "it must not fall below threshold_value."
    )
    period_start: Optional[str] = Field(
        description="ISO date (YYYY-MM-DD) the measurement period starts, if stated in or "
        "near the clause. Null if not stated."
    )
    period_end: Optional[str] = Field(
        description="ISO date (YYYY-MM-DD) the measurement period ends, if stated in or near "
        "the clause. Null if not stated."
    )
    carve_outs: list[str] = Field(
        description="Any exceptions, qualifiers, or conditions under which exceeding the "
        "threshold is still permitted, stated in or directly referenced by the clause. "
        "Each item should be a self-contained statement. Empty list if there are none."
    )
    aggregation_note: Optional[str] = Field(
        description="Any instruction on HOW to combine multiple components into the measured "
        "value, beyond simple summation (e.g. 'compliance is judged by whichever single line "
        "item is largest, not their sum'). Null if the clause is a plain sum/ratio with "
        "nothing unusual about how components combine."
    )
    source_quote: str = Field(
        description="The clause's own text, copied as exactly as you can from the source "
        "(minor PDF-extraction spacing artifacts aside)."
    )


class CovenantExtractionResult(BaseModel):
    covenants: list[CovenantClause]


# ---------------------------------------------------------------------------
# 2b: KYC / related-party fact extraction
# ---------------------------------------------------------------------------


class RelatedPartyDisclosure(BaseModel):
    counterparty_name: str = Field(description="Name of the counterparty/entity as written.")
    ownership_or_voting_pct: Optional[float] = Field(
        description="The ownership or voting-rights percentage disclosed for this entity, as "
        "a plain number (e.g. 18.6 for 18.6%). Null if the document discloses a relationship "
        "without stating a percentage."
    )
    relationship_description: str = Field(
        description="How the relationship is described in the source text (e.g. direct "
        "ownership, indirect/beneficial ownership, common control, family relationship to a "
        "director), in the source document's own language."
    )
    explicitly_labeled_related_party: bool = Field(
        description="True only if the source text itself explicitly categorizes or lists "
        "this specific entity as a related party (e.g. under a 'related parties' heading or "
        "an explicit statement to that effect) — a literal reading of the document's own "
        "labeling, independent of any percentage threshold. False if the document merely "
        "lists an ownership stake without itself calling it a related party."
    )
    source_quote: str = Field(description="The text this disclosure was extracted from.")


class KycExtractionResult(BaseModel):
    related_party_threshold_pct: Optional[float] = Field(
        description="The ownership/voting-rights percentage this document states as the "
        "definition of a related party (e.g. 20.0 for '20% or more'). Null if the document "
        "does not state a numeric threshold."
    )
    related_party_threshold_description: Optional[str] = Field(
        description="If the document's related-party definition is not a bare percentage "
        "(e.g. it references an accounting standard, control, or a qualitative test instead "
        "or in addition), describe that definition here in the source language. Null if "
        "related_party_threshold_pct fully captures it or no definition is given."
    )
    disclosures: list[RelatedPartyDisclosure]


# ---------------------------------------------------------------------------
# 2b: Generic supporting-fact extraction (treasury memos, financial notes,
# or any other document type not covered by the schemas below but
# potentially relevant to a covenant calculation). Defined before
# AuditExtractionResult, which now embeds this same OtherFact shape — see
# that section's docstring for why one document can carry all three fact
# shapes at once.
# ---------------------------------------------------------------------------


class OtherFact(BaseModel):
    fact_description: str = Field(
        description="A short, self-contained description of the fact, in the source "
        "language, e.g. 'Совокупная выручка за 2025 год по аудированной отчётности'."
    )
    value: Optional[float] = Field(
        description="The numeric value of this fact, if it has one. Null for purely "
        "qualitative facts."
    )
    unit: Optional[str] = Field(
        description="Unit of value, e.g. 'usd', 'eur', 'ratio', 'percent', 'count'. "
        "Null if value is null."
    )
    period: Optional[str] = Field(
        description="The period this fact applies to, if stated (ISO dates if exact, "
        "otherwise as stated). Null if not applicable/stated."
    )
    source_quote: str = Field(description="The text this fact was extracted from.")


class OtherFactsExtractionResult(BaseModel):
    facts: list[OtherFact] = Field(
        description="Any facts in this document that could plausibly feed into a loan "
        "covenant calculation (financial figures, operational figures tied to a covenant-like "
        "concept, explicit statements about classification or treatment of specific amounts). "
        "Empty list if this document has nothing covenant-relevant in it — that is a valid "
        "and expected answer for purely administrative or unrelated documents."
    )


# ---------------------------------------------------------------------------
# 2b: Auditor / financial-notes disclosure extraction.
#
# Confirmed on the public dataset: a borrower's own "Примечания к
# финансовой отчётности" (Notes to Financial Statements) document — issued
# by the same audit firm, same "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ" section
# shape as a standalone audit_report — routinely mixes three different
# fact shapes in one document: ordinary reclassifications (ratio/aggregate
# category reassignment), a specific transaction's *true amount* when the
# ledger export dropped it (a dirty-row correction, joined by txn_id, not
# by category), and free-standing figures with no ledger transaction at all
# (an off-ledger accrual, a settlement-implied FX rate). One extraction
# call now covers all three rather than routing to three different
# extractors, since a real document doesn't self-sort into single-purpose
# files.
# ---------------------------------------------------------------------------


class AuditReclassification(BaseModel):
    txn_id: Optional[str] = Field(
        description="The specific transaction ID this finding names, if the source text "
        "states one directly (e.g. 'Операция TXN-B4-0026'). Null if the finding only "
        "identifies the transaction by counterparty and amount (the more common case for "
        "standalone audit reports) — Block 3 joins on those instead."
    )
    counterparty_name: Optional[str] = Field(
        description="Counterparty named in this reclassification finding, if any is stated. "
        "Null if the finding does not name a specific counterparty."
    )
    amount: Optional[float] = Field(
        description="The dollar amount cited for this reclassification, as a positive number, "
        "if the report states one. Null if no specific amount is cited."
    )
    transaction_date_or_period: Optional[str] = Field(
        description="Any date or period stated for the reclassified item (ISO date if exact, "
        "otherwise as stated, e.g. 'Q1 2025'). Null if none stated."
    )
    action: Literal["recategorize", "exclude_from_period", "no_change"] = Field(
        description="'recategorize' if the item was moved to a different category — "
        "original_category and reclassified_category are both given. "
        "'exclude_from_period' if the finding says this specific transaction should be "
        "excluded entirely from the covenant period regardless of its category (e.g. "
        "revenue/risk recognized in a different fiscal period than the transaction date "
        "suggests — 'исключена из ковенантного периода', 'относится к периоду ...'). "
        "'no_change' if the finding says a reclassification was considered or requested but "
        "was explicitly REJECTED, or that no reclassification was needed — this is purely "
        "informational and must never change any calculation."
    )
    original_category: Optional[str] = Field(
        description="The category the item was originally recorded under, in the source "
        "language. Null unless action is 'recategorize'."
    )
    reclassified_category: Optional[str] = Field(
        description="The category the auditor reassigned the item to, in the source "
        "language. Null unless action is 'recategorize'."
    )
    reasoning: str = Field(
        description="The auditor's stated reason for this finding, in the source language."
    )
    source_quote: str = Field(description="The finding's own text, as exactly as possible.")


class TransactionAmountCorrection(BaseModel):
    txn_id: str = Field(
        description="The exact transaction ID this correction applies to (e.g. "
        "'TXN-P8-0031'), always stated directly in these findings — never inferred."
    )
    corrected_amount: float = Field(
        description="The true amount for this transaction, signed exactly as the ledger "
        "convention (negative = outflow, positive = inflow) — read the finding's own words "
        "carefully for direction ('расход' = outflow = negative)."
    )
    reasoning: str = Field(
        description="Why a correction is needed, in the source language (e.g. 'сумма не "
        "отражена в выгрузке реестра')."
    )
    source_quote: str = Field(description="The finding's own text, as exactly as possible.")


class AuditExtractionResult(BaseModel):
    report_reference: Optional[str] = Field(
        description="The report's own reference/engagement number, if stated (e.g. "
        "'AR-2025-0634'). Null if none is given."
    )
    is_final_position: bool = Field(
        description="True if the report presents itself as the auditor's final/authoritative "
        "position for covenant purposes. False if the text itself indicates this is a draft, "
        "interim, or otherwise non-final workpaper (read any disclaimers near the top "
        "carefully — this distinction is explicitly called out in some of these reports)."
    )
    reclassifications: list[AuditReclassification]
    transaction_amount_corrections: list[TransactionAmountCorrection] = Field(
        default_factory=list,
        description="Findings that state a specific transaction's ledger amount was dropped "
        "or wrong, with its true value given directly (e.g. 'сумма не отражена в выгрузке "
        "реестра; фактическая сумма операции составляет $X'). Empty list if none.",
    )
    other_facts: list[OtherFact] = Field(
        default_factory=list,
        description="Any other covenant-relevant fact in this document that isn't a "
        "reclassification or a transaction-amount correction — an off-ledger obligation "
        "disclosed in a note, a settlement-implied FX rate, an operational figure. Same "
        "shape and same bar as OtherFactsExtractionResult. Empty list if none.",
    )


# ---------------------------------------------------------------------------
# Block 3a: transaction categorization (linking layer)
#
# The ledger has no category column (by the case's own design), so every
# covenant that sums transactions by concept — capital expenditure,
# operating expense, revenue, interest expense, a named overhead line —
# needs each transaction's description read and matched against that
# concept first. This is the one Structured Output call in Block 3 that
# still touches an LLM at all; everything after it (which category wins a
# ratio, which threshold a percentage is compared against, whether status
# is BREACH or COMPLIANT) is plain arithmetic in calculation/formulas.py.
#
# Deliberately NOT here: related-party determination. That's a question of
# who a counterparty is (an ownership fact from KYC, compared to a
# threshold), not what a transaction's description says — see
# linking/related_parties.py and linking/categories.py:is_related_party_text.
# ---------------------------------------------------------------------------


class TransactionClassification(BaseModel):
    txn_id: str = Field(description="The transaction's own txn_id, copied exactly as given.")
    category: str = Field(
        description="Exactly one of the category keys provided in the prompt whose "
        "description this transaction's own description matches — or the literal string "
        "'unclassified' if it matches none of them. Never invent a category key that "
        "wasn't provided."
    )
    reasoning: str = Field(
        description="One short sentence on why this transaction matches (or doesn't match "
        "any) category, referencing its description."
    )


class TransactionClassificationResult(BaseModel):
    classifications: list[TransactionClassification] = Field(
        description="Exactly one entry per transaction given in the prompt, same txn_ids, "
        "no more and no fewer."
    )
