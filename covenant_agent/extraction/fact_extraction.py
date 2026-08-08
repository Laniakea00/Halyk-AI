"""Block 2b: extract supporting facts (KYC, auditor reclassifications, and
anything else covenant-relevant) from a borrower's other current documents.

Three narrow extractors, one per document kind we know how to interpret
specifically (KYC dossier, audit reclassification report), plus a generic
fallback for anything else that matched a scenario but isn't one of those
two (treasury memos today; possibly financial statements or other kinds in
the private dataset). All three use FACT_EXTRACTION_MODEL — wrong here means
a missing/incomplete fact for Block 3 to notice and degrade around, not a
silently wrong number, so the cost/accuracy tradeoff is different from
covenant extraction (and, per llm_client.py's model-profile docstring, this
call type stays on the mini tier even in the "final" profile).

None of these functions decide anything. See covenant_agent/schemas.py for
exactly where the extraction/decision line is drawn in each schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

from covenant_agent.llm_client import FACT_EXTRACTION_MODEL, extract_structured
from covenant_agent.schemas import (
    AuditExtractionResult,
    KycExtractionResult,
    OtherFactsExtractionResult,
)

logger = logging.getLogger(__name__)

_SHARED_RULES = """\
- Never decide or imply anything about loan covenant compliance. You are not given any \
covenant thresholds or transaction totals.
- Preserve the source document's own language in every free-text field. Do not translate.
- Numbers must come only from what the text actually states; do not infer, round, or \
estimate a value that is not written down.
- The text may have been extracted from a PDF; stylized headers can show irregular \
letter-spacing (e.g. "Д О ГО В О Р") as a rendering artifact — read through it, it does not \
change the meaning.
- Some documents in this dataset are explicitly marked as drafts, interim workpapers, or \
otherwise not-yet-authoritative (watch for disclaimers near the top). Extract what such a \
document says, but flag its status accurately wherever the schema asks you to — do not treat \
a self-declared draft as if it were silent about its own status.
"""

KYC_SYSTEM_PROMPT = f"""\
You are a precise compliance-document reader for a bank's covenant-compliance pipeline. \
Your job is to read a KYC ("know your customer") / related-party dossier and extract every \
ownership or relationship disclosure it contains, plus how the document itself defines a \
"related party".

{_SHARED_RULES}
- Extract every counterparty/entity disclosure you find, even ones far below any threshold — \
Block 3 of this pipeline does the threshold comparison, not you.
- Do not compute or assert whether any entity "is" a related party by comparing its \
percentage to the threshold yourself — only report explicitly_labeled_related_party as True \
when the document's own text already categorizes that entity that way.
"""

AUDIT_SYSTEM_PROMPT = f"""\
You are a precise compliance-document reader for a bank's covenant-compliance pipeline. \
Your job is to read an auditor's agreed-upon-procedures / transaction-reclassification report \
and extract every reclassification finding it contains.

{_SHARED_RULES}
- Pay close attention to whether this specific report presents itself as the auditor's final \
position or as a draft/interim workpaper superseded by a later final report — set \
is_final_position accordingly based on the document's own words, not on your own guess.
- Extract every reclassification finding, regardless of its dollar size.
"""

OTHER_SYSTEM_PROMPT = f"""\
You are a precise document reader for a bank's covenant-compliance pipeline. \
Your job is to scan a supporting document (not a credit agreement, not a KYC dossier, not an \
audit reclassification report — something else, e.g. a treasury memo or financial summary) \
for any fact that could plausibly feed into a loan covenant calculation: financial figures, \
operational figures, or explicit statements about how a specific amount should be classified \
or treated.

{_SHARED_RULES}
- It is normal and expected for a document to contain nothing covenant-relevant — an empty \
facts list is a valid, correct answer. Do not stretch to invent relevance.
- Do not extract routine administrative content (staffing notes, unrelated project status, \
generic policy text) as if it were a financial fact.
"""


def extract_kyc_facts(
    scenario_id: str, doc_text: str, *, log_dir: Path | None = None
) -> KycExtractionResult:
    input_text = (
        f"Borrower scenario id: {scenario_id}\n\n"
        f"--- KYC / RELATED-PARTY DOCUMENT TEXT ---\n{doc_text}\n--- END OF TEXT ---"
    )
    result, _raw = extract_structured(
        instructions=KYC_SYSTEM_PROMPT,
        input_text=input_text,
        response_model=KycExtractionResult,
        config=FACT_EXTRACTION_MODEL,
        log_dir=log_dir,
        log_tag=f"kyc_{scenario_id}",
    )
    return result


def extract_audit_facts(
    scenario_id: str, doc_text: str, *, log_dir: Path | None = None, doc_id: str = ""
) -> AuditExtractionResult:
    input_text = (
        f"Borrower scenario id: {scenario_id}\n\n"
        f"--- AUDIT RECLASSIFICATION REPORT TEXT ---\n{doc_text}\n--- END OF TEXT ---"
    )
    result, _raw = extract_structured(
        instructions=AUDIT_SYSTEM_PROMPT,
        input_text=input_text,
        response_model=AuditExtractionResult,
        config=FACT_EXTRACTION_MODEL,
        log_dir=log_dir,
        log_tag=f"audit_{scenario_id}_{doc_id}" if doc_id else f"audit_{scenario_id}",
    )
    if not result.is_final_position:
        logger.warning(
            "Scenario %s: audit report %s self-identifies as NOT final — Block 1's version "
            "resolution should already have excluded lone drafts like this from "
            "current_documents. Seeing one here means either a genuine multi-document case "
            "or a resolution gap worth checking.",
            scenario_id,
            doc_id or "(unknown doc)",
        )
    return result


def extract_other_facts(
    scenario_id: str, doc_kind: str, doc_text: str, *, log_dir: Path | None = None, doc_id: str = ""
) -> OtherFactsExtractionResult:
    input_text = (
        f"Borrower scenario id: {scenario_id}\n"
        f"Document kind (as classified by our pipeline): {doc_kind}\n\n"
        f"--- DOCUMENT TEXT ---\n{doc_text}\n--- END OF TEXT ---"
    )
    result, _raw = extract_structured(
        instructions=OTHER_SYSTEM_PROMPT,
        input_text=input_text,
        response_model=OtherFactsExtractionResult,
        config=FACT_EXTRACTION_MODEL,
        log_dir=log_dir,
        log_tag=f"other_{scenario_id}_{doc_id}" if doc_id else f"other_{scenario_id}",
    )
    return result
