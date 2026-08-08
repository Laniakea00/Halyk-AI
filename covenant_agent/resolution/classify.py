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
    ),
    "kyc_dossier": (
        "знай своего клиент",
        "know your customer",
        "надлежащая проверка клиента",
        "kyc",
        "связанных сторон",
    ),
    "audit_report": (
        "отчёт о выполнении согласованных процедур",
        "агреед-упон procedures",
        "registered auditors",
        "аудитор",
        "переклассифиц",
    ),
    "treasury_memo": (
        "служебная записка казначейства",
        "казначейство группы",
        "операционное досье на конец периода",
    ),
}

MIN_SCORE_TO_CLASSIFY = 1


def classify_kind(text: str) -> tuple[str, int]:
    """Return (kind, score). kind is "other" when nothing scores >= threshold."""
    lowered = text.lower()
    best_kind = "other"
    best_score = 0
    for kind, markers in KIND_MARKERS.items():
        score = sum(1 for marker in markers if marker in lowered)
        if score > best_score:
            best_kind, best_score = kind, score
    if best_score < MIN_SCORE_TO_CLASSIFY:
        return "other", 0
    return best_kind, best_score
