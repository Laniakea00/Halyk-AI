"""Block 2a: extract each required covenant's rule from a borrower's
current credit agreement.

One Structured Outputs call per scenario, over the full agreement text —
not per clause — so the model can resolve cross-references to terms
defined elsewhere in the same document (e.g. a ratio's denominator defined
once in the Terms article and reused in three separate covenants).

The model never sees the ledger and is explicitly told not to judge
compliance — see SYSTEM_PROMPT. It also never sees a fixed mapping from
clause number to formula type, because there isn't one: confirmed on the
public dataset that clause 6.1 alone covers at least a capital-intensity
ratio (P1), an interest-coverage ratio (B1), and a plain dollar ceiling
(P8) across different borrowers. The prompt says so explicitly so this
generalizes to whatever the private dataset's clause numbers turn out to
mean.
"""

from __future__ import annotations

import logging
from pathlib import Path

from covenant_agent.llm_client import COVENANT_EXTRACTION_MODEL, extract_structured
from covenant_agent.schemas import CovenantExtractionResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise legal/financial document reader for a bank's covenant-compliance pipeline.
Your only job is to read a loan agreement's financial covenants article and convert each \
requested clause into a structured description of what it measures and what its limit is.

Rules:
- Extract exactly the clause numbers you are given in the request — no more, no fewer. If a \
requested clause is genuinely not present in the text, omit it from your output rather than \
inventing one; do not include clauses you were not asked for.
- Never decide whether the covenant is breached or compliant. You are not given any \
transaction data and must not guess or imply a compliance outcome anywhere in your output.
- Do not assume clauses with the same or similar numbers mean the same thing across different \
agreements — each borrower's agreement defines its own formula from scratch. Read every \
clause fresh, on its own terms.
- Read carefully for unusual aggregation rules — e.g. a clause that says compliance is judged \
by whichever single named component is largest, not their sum. Capture this in \
aggregation_note; do not silently normalize it into a plain sum. When metric_type is \
'max_single_component', list every named component in the `components` field, each as its own \
short entry — a downstream system will need to categorize transactions against these exact \
component names, so do not merge or paraphrase two components into one.
- numerator_description/denominator_description/formula_description feed a downstream system \
that matches them against transaction descriptions — a bare defined-term name is useless to it, \
and so is an abstract restatement that doesn't decompose into things a transaction can actually \
be. If a numerator, denominator, or amount is a capitalized defined term or a well-known \
financial acronym (e.g. "EBITDA", "EBIT", "Net Debt") rather than a plain description, resolve \
it: (1) if the document defines that term elsewhere (commonly in an early "Термины"/"Definitions" \
article), search the full text you were given for that definition and write it out in full \
instead of the bare term; (2) if the document uses the term without ever defining it, write out \
its standard meaning IN DECOMPOSED, SUMMABLE FORM — e.g. write EBITDA as "revenue minus operating \
expenses", not as "earnings before interest, taxes, depreciation and amortization": the first \
form names two concrete things a transaction classifier can match against (revenue lines, \
operating-expense lines); the second is a restatement of the acronym that doesn't. Write this in \
the source document's own language either way. The field must always end up describing concrete, \
summable components — never just naming or re-deriving an accounting acronym in the abstract.
- Preserve the source document's own language in every free-text field. Do not translate.
- Numbers, dates, and thresholds must come only from what the text actually states; do not \
infer, round, or estimate a value that is not written down.
- The text may have been extracted from a PDF; stylized headers can show irregular \
letter-spacing (e.g. "Д О ГО В О Р") as a rendering artifact — read through it, it does not \
change the meaning of the text.
"""


def extract_covenants(
    scenario_id: str,
    agreement_text: str,
    required_covenant_keys: list[str],
    *,
    log_dir: Path | None = None,
) -> CovenantExtractionResult:
    """Extract exactly `required_covenant_keys` from `agreement_text`.

    Raises covenant_agent.llm_client.ExtractionError if the call itself
    fails. Does NOT raise if the model's returned key set doesn't exactly
    match what was requested — that's logged as a warning so the caller
    (and, later, the submission assembler) can see and handle a missing
    clause explicitly rather than the pipeline crashing mid-run.
    """
    input_text = (
        f"Borrower scenario id: {scenario_id}\n"
        f"Extract exactly these covenant clause numbers: "
        f"{', '.join(required_covenant_keys)}\n\n"
        f"--- CREDIT AGREEMENT TEXT ---\n{agreement_text}\n--- END OF TEXT ---"
    )
    result, _raw = extract_structured(
        instructions=SYSTEM_PROMPT,
        input_text=input_text,
        response_model=CovenantExtractionResult,
        config=COVENANT_EXTRACTION_MODEL,
        log_dir=log_dir,
        log_tag=f"covenant_{scenario_id}",
    )

    found_keys = {c.covenant_key for c in result.covenants}
    required = set(required_covenant_keys)
    missing = required - found_keys
    extra = found_keys - required
    if missing:
        logger.warning(
            "Scenario %s: covenant extraction is missing required key(s) %s",
            scenario_id,
            sorted(missing),
        )
    if extra:
        logger.warning(
            "Scenario %s: covenant extraction returned unrequested key(s) %s — ignoring them",
            scenario_id,
            sorted(extra),
        )

    return result
