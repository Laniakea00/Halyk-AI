"""Extract ACC-<digits>[-<digits>] tokens from text and resolve them against
known scenario accounts via *exact* token match — never substring/prefix.

This is the fix for the decoy-subaccount trap confirmed in the public
dataset: an internal ops report for Aktau Port Services JSC (ACC-7801)
mentions an unrelated "вспомогательный счёт ACC-7801-08" tied to a
*different* legal entity (Aktau Port Services AG, not JSC). A naive
substring search for "ACC-7801" matches that text too, because "ACC-7801-08"
contains "ACC-7801" as a substring.

Fix: the token regex is greedy over the optional sub-account suffix, so it
captures "ACC-7801-08" as one whole token rather than truncating it to
"ACC-7801". Resolution then requires the *whole token* to equal a known
account_id verbatim. Verified against the real files: the compliance
procedure doc mentioning "ACC-7801-05" and the two ops-update decoys
mentioning "ACC-7801-08" / "ACC-7801-02" all correctly produce zero exact
matches against "ACC-7801", while the real credit agreements (current and
superseded) correctly do match.
"""

from __future__ import annotations

import re

ACCOUNT_TOKEN_RE = re.compile(r"\bACC-\d+(?:-\d+)?\b")

# PDF extraction of a large stylized title routinely inserts a space between
# every character (the same class of artifact fact_extraction.py's
# SYSTEM_PROMPT already documents for "Д О ГО В О Р") — confirmed on the
# public dataset: all 12 scenarios' own "Примечания к финансовой отчётности"
# (Notes to Financial Statements) documents render their account number this
# way in their title ("АУДИТОРСКОЕ ДЕЛО № A C C - 7 8 0 3 / ..."), which
# ACCOUNT_TOKEN_RE's exact \bACC-\d+\b match cannot see through — these
# documents sat completely unmatched, hiding covenant-relevant disclosures
# (FX settlement rates, dirty-ledger-row true amounts, off-ledger
# obligations) for every scenario at once.
#
# This is a fix to the *input*, not a new or looser matching mechanism: the
# same exact-token regex above still runs, on the same class of official
# account numbers, after this collapses inserted whitespace back out.
# Deliberately narrow — only matches text that already reads "A C C -
# <spaced digits>", so it cannot accidentally collapse unrelated spaced-out
# text elsewhere in a document the way a blanket whitespace-strip would.
_SPACED_ACCOUNT_RE = re.compile(
    r"A\s*C\s*C\s*-\s*\d(?:\s*\d){1,5}(?:\s*-\s*\d(?:\s*\d){0,3})?", re.IGNORECASE
)


def _normalize_spaced_account_tokens(text: str) -> str:
    """Collapse whitespace inside an "A C C - 7 8 0 3"-shaped span to "ACC-7803"."""
    return _SPACED_ACCOUNT_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)

# Company names are a secondary, human-readable signal only — see
# ScenarioBundle / models.py. Never used to decide scenario membership,
# because near-duplicate legal-entity names are a deliberate trap in this
# dataset (e.g. "Shymkent Refinery JSC" vs "Shymkent Refinery Services JSC"
# are two different borrowers, one is P3 the other is B4). Captured only for
# logging / human sanity checks.
COMPANY_NAME_RE = re.compile(
    r"[A-Z][A-Za-z\-]*(?:\s+[A-Z][A-Za-z\-]*){0,4}\s+(?:JSC|LLP|AG|Inc|Corp|LLC|Ltd)\b"
)


def extract_account_tokens(text: str) -> tuple[str, ...]:
    """All ACC-... tokens in text, verbatim, de-duplicated, first-seen order.

    Runs against a locally de-spaced copy of `text` (see
    _normalize_spaced_account_tokens) so a stylized, letter-spaced title
    doesn't hide a real token — nothing else about `text` is touched, and
    every other consumer (company-name extraction, LLM input, ...) still
    sees the original.
    """
    normalized = _normalize_spaced_account_tokens(text)
    return tuple(dict.fromkeys(ACCOUNT_TOKEN_RE.findall(normalized)))


def extract_company_names(text: str) -> tuple[str, ...]:
    """Best-effort company name mentions, for logging/debugging only."""
    return tuple(dict.fromkeys(m.strip() for m in COMPANY_NAME_RE.findall(text)))


def match_known_accounts(
    tokens: tuple[str, ...], known_account_ids: set[str]
) -> tuple[str, ...]:
    """Exact-match only: a token counts iff it equals a known account_id verbatim."""
    return tuple(t for t in tokens if t in known_account_ids)
