"""Lightweight, keyword-scored document-kind classifier.

We deliberately do *not* try to hard-classify every document into a fixed
taxonomy with high precision — the private dataset on Aug 9 will have
different companies and probably different phrasing. Instead we key off
phrases that describe a document's *function* (what kind of instrument or
report it is), not its subject company — those are far more likely to
recur across a same-template, different-borrowers dataset than any
company-specific text would be.

Every kind list is intentionally short and easy to extend if the private
dataset shows a new phrasing; scoring is "how many distinct marker phrases
matched", and ties/zero-score default to "other" rather than guessing.
"other" is a safe default: it just means the fact-extraction layer (Block 2)
won't treat this document as a source of covenant-relevant facts, which is
the correct behavior for genuine noise (HR templates, weekly ops updates,
access logs, etc.) and a safe failure mode for anything we simply failed to
recognize.
"""

from __future__ import annotations

import re

# kind -> marker phrases (checked case-insensitively as substrings).
# Phrases are function-describing, not company-specific, by design.
KIND_MARKERS: dict[str, tuple[str, ...]] = {
    # Listed first, and with two markers, so it reliably outscores
    # credit_agreement/audit_report on documents that share vocabulary with
    # both ("Заёмщик", "аудитор") — confirmed necessary on the public
    # dataset: every scenario's own "Примечания к финансовой отчётности"
    # (Notes to Financial Statements — issued by an audit firm, discussing
    # the borrower's accounting policy, FX settlement notes, dirty-ledger
    # corrections, off-ledger obligations) scored a tie between
    # credit_agreement (on "Заёмщик") and audit_report (on "аудитор") once
    # accounts.py's letter-spaced-header fix let these documents match a
    # scenario at all — ties previously resolved to credit_agreement by
    # dict order, silently creating a second "current" credit agreement.
    # This is a distinct document kind, not a duplicate of either.
    "financial_notes": (
        "примечания к финансовой отчётности",
        "notes to financial statements",
        "дополнение о соблюдении ковенантов",
        # H4 (red-team): Kazakh coverage gap — every marker above was
        # Russian/English only, so a Kazakh-language notes/statements
        # document scored 0 against every kind and fell through to "other",
        # invisible to every downstream layer. Not confirmed as a live
        # occurrence on the public dataset (all 200 documents are
        # Russian/English) — a defensive gap closed ahead of the private
        # dataset, same reasoning as the H1-H3/M1/M4 hardening pass.
        "қаржылық есептілік",  # financial statements/reporting
        "табыс",  # income/revenue
        # Template-variation hardening (2026-08-09): organizers confirmed
        # the private dataset uses multiple financial-report templates —
        # a title synonym here is the exact same failure shape as
        # Sarybel's pre-fix letter-spacing artifact (whole document kind
        # scores 0, falls through to "other", invisible downstream).
        "пояснения к финансовой отчётности",
        "финансовые примечания",
        "пояснительная записка к отчётности",
    ),
    "credit_agreement": (
        "договор банковского займа",
        "loan agreement",
        "финансовые ковенанты",
        "заёмщик",
        "кредитор",
        "несиелік келісім",  # credit/loan agreement (kk)
        "шарт",  # agreement/contract (kk)
        "міндеттемелер",  # obligations/liabilities (kk)
        # Template-variation hardening (2026-08-09) — see financial_notes.
        "кредитный договор",
        "договор о предоставлении кредита",
        # Confirmed live gap (2026-08-09): scenario J4's credit agreement is
        # a fully English-language document ("Borrower"/"Lender", title
        # "CREDIT AGREEMENT") with zero Russian/Kazakh markers and no exact
        # "loan agreement" phrase either — scored 0 and fell through to
        # "other", leaving J4 with no current credit_agreement document at
        # all. These two structural opening-formula/title phrases are
        # deliberately specific (not "credit agreement"/"borrower" alone,
        # which were checked and confirmed to false-positive on J4's own
        # financial_notes document — a disclosure that legitimately
        # *references* "the credit agreement" and "the Borrower" as terms
        # without being one) — confirmed to hit only the two real J4
        # credit_agreement documents and nowhere else in either corpus.
        "this credit agreement",
        "senior secured credit facility",
    ),
    "kyc_dossier": (
        "знай своего клиент",
        "know your customer",
        "надлежащая проверка клиента",
        "kyc",
        "связанных сторон",
        # Template-variation hardening (2026-08-09) — see financial_notes.
        "идентификация клиента",
        "проверка контрагента",
    ),
    "audit_report": (
        "отчёт о выполнении согласованных процедур",
        "агреед-упон procedures",
        "registered auditors",
        "аудитор",
        "переклассифиц",
        # Template-variation hardening (2026-08-09) — see financial_notes.
        "аудиторское заключение",
        "независимый аудитор",
    ),
    "treasury_memo": (
        "служебная записка казначейства",
        "казначейство группы",
        "операционное досье на конец периода",
        # Template-variation hardening (2026-08-09) — see financial_notes.
        "меморандум казначейства",
        "казначейская записка",
    ),
}

MIN_SCORE_TO_CLASSIFY = 1

# Template-variation hardening (2026-08-09): "Примечание N"/"Note N"
# numbering is a far more stable structural convention across different
# financial-report templates than the exact title phrase — confirmed on
# the public dataset: all 12 real financial_notes documents have 6-9
# distinct numbered notes; grepped the other ~190 documents in the corpus
# for the same pattern and found zero false positives (the one 2+ hit
# outside financial_notes is Sarybel's own Group-parent consolidated
# statements, which is correctly re-routed to "group_financials" by
# segment_linking.py regardless of what classify_kind gives it here — see
# resolution/pipeline.py's secondary linking pass, which keys off "no
# account-token match at all", not off `kind`). Scoped to financial_notes
# only — not extended to other kinds without the same real-data check.
_NUMBERED_NOTE_RE = re.compile(r"\b(?:примечание|note)\s*№?\s*(\d+)", re.IGNORECASE)
_NUMBERED_NOTE_MIN_DISTINCT = 2


def _has_numbered_notes_structure(text: str) -> bool:
    return len({m for m in _NUMBERED_NOTE_RE.findall(text)}) >= _NUMBERED_NOTE_MIN_DISTINCT


def classify_kind(text: str) -> tuple[str, int]:
    """Return (kind, score). kind is "other" when nothing scores >= threshold."""
    lowered = text.lower()
    best_kind = "other"
    best_score = 0
    for kind, markers in KIND_MARKERS.items():
        score = sum(1 for marker in markers if marker in lowered)
        if kind == "financial_notes" and _has_numbered_notes_structure(text):
            score += 1
        if score > best_score:
            best_kind, best_score = kind, score
    if best_score < MIN_SCORE_TO_CLASSIFY:
        return "other", 0
    return best_kind, best_score
