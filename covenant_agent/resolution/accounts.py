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
    """All ACC-... tokens in text, verbatim, de-duplicated, first-seen order."""
    return tuple(dict.fromkeys(ACCOUNT_TOKEN_RE.findall(text)))


def extract_company_names(text: str) -> tuple[str, ...]:
    """Best-effort company name mentions, for logging/debugging only."""
    return tuple(dict.fromkeys(m.strip() for m in COMPANY_NAME_RE.findall(text)))


def match_known_accounts(
    tokens: tuple[str, ...], known_account_ids: set[str]
) -> tuple[str, ...]:
    """Exact-match only: a token counts iff it equals a known account_id verbatim."""
    return tuple(t for t in tokens if t in known_account_ids)
