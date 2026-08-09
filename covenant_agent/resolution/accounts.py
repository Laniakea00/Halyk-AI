"""Extract <PREFIX>-<digits>[-<digits>] tokens from text and resolve them
against known scenario accounts via *exact* token match — never
substring/prefix.

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

Private dataset (2026-08-09): scenario KC's account_id is "TELE-4471", not
"ACC-<digits>" — confirmed live: the hardcoded "ACC-" literal left KC with
zero matched documents (0/6 real docs, all present in the corpus under their
own "TELE-4471" token) even though `AccountLinkingRiskError`'s >=50%
threshold correctly stayed quiet for a single scoped scenario, so this
would have silently fallen back to fallback-COMPLIANT for all of KC's
cells with no loud signal at all. The prefix is generalized from the
literal "ACC" to any 2-6 uppercase letters; the exact-match-against-known-
account_ids step below is what actually prevents false positives, not the
prefix literal, so this generalization carries the same decoy-subaccount
safety guarantee for any prefix, not just "ACC".
"""

from __future__ import annotations

import re

ACCOUNT_TOKEN_RE = re.compile(r"\b[A-Z]{2,6}-\d+(?:-\d+)?\b")

# PDF extraction of a large stylized title routinely inserts a space between
# every character (the same class of artifact fact_extraction.py's
# SYSTEM_PROMPT already documents for "Д О ГО В О Р") — confirmed on the
# public dataset: all 12 scenarios' own "Примечания к финансовой отчётности"
# (Notes to Financial Statements) documents render their account number this
# way in their title ("АУДИТОРСКОЕ ДЕЛО № A C C - 7 8 0 3 / ..."), which
# ACCOUNT_TOKEN_RE's exact \bACC-\d+\b match cannot see through — these
# documents sat completely unmatched, hiding covenant-relevant disclosures
# (FX settlement rates, dirty-ledger-row true amounts, off-ledger
# obligations) for every scenario at once. Confirmed the same artifact
# recurs on the private dataset's KC/"TELE-4471" scenario ("T E L E - 4 4 7 1
# / 2 0 2 5" in its own financial_notes title).
#
# The letter class is generalized alongside ACCOUNT_TOKEN_RE, but the
# whitespace unit is deliberately a single MANDATORY literal space between
# *every* character (letter-letter, letter-hyphen, hyphen-digit,
# digit-digit) — not "\s*" (any amount, including newlines). A first attempt
# using "\s*" (matching the original hardcoded-"ACC" version's own
# whitespace unit) was confirmed live to be unsafe once genericized: e.g.
# "Halyk Bank ... JSC\n<many spaces>\nACC-7604" and even plain single-space
# prose like "AUDIT FILE REF ACC-7604" both got spuriously matched as one
# span (letters J,S,C,A,C,C or R,E,F,A,C,C) and destructively collapsed to
# "JSCACC-7604" / "REFACC-7604", corrupting an otherwise-clean plain token
# that ACCOUNT_TOKEN_RE would have found on its own. Requiring every
# character to be individually space-separated is what real letter-spaced
# titles look like ("A C C - 7 8 0 3") and what ordinary adjacent
# words/acronyms never look like (their internal letters are contiguous) —
# confirmed against every real document in both the public and private
# corpora: every match this stricter regex produces collapses to an actual
# known account token, no false positives.
_SPACED_ACCOUNT_RE = re.compile(r"\b[A-Z](?: [A-Z]){1,5} - \d(?: \d){1,5}(?: - \d(?: \d){0,3})?\b")


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
