"""Shared counterparty-name normalization and matching.

Two independent linking problems in this block need the exact same fix:
matching a KYC-disclosed entity name against the ledger's `counterparty`
column (related-party resolution), and matching an audit report's cited
counterparty against the same column (reclassification linking). Both hit
the same real formatting drift, confirmed on the public dataset:

    KYC "Ertis Capital, LLP"      vs  ledger "Ertis Capital LLP"       (comma)
    KYC "Aktau Holdings LLP"      vs  ledger "Aktau Holdings L.L.P."   (periods)

The ledger also routinely appends a location tag to otherwise-identical
names, e.g. "Ashford Property Co (Taraz site)" — stripped here too, since
neither KYC dossiers nor audit reports were observed doing the same.

Matching is exact-after-normalization first; a fuzzy fallback only engages
when nothing normalizes to an exact match, and every fuzzy match is
reported with its similarity score so callers can log it rather than trust
it silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_LOCATION_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
_PUNCTUATION_RE = re.compile(r"[.,]")
_WHITESPACE_RE = re.compile(r"\s+")

# Below this similarity, two names are treated as unrelated rather than a
# low-confidence match — chosen conservatively (two genuinely different
# short company names rarely score this high by accident) but this is a
# heuristic, not a proof; every fuzzy match, not just borderline ones, gets
# logged by the callers so a human can sanity-check it.
FUZZY_MATCH_THRESHOLD = 0.90


def normalize_counterparty(name: str) -> str:
    name = _LOCATION_SUFFIX_RE.sub("", name)
    name = _PUNCTUATION_RE.sub("", name)
    name = _WHITESPACE_RE.sub(" ", name)
    return name.strip().lower()


@dataclass(frozen=True)
class CounterpartyMatch:
    candidate: str  # the original (non-normalized) candidate string that matched
    score: float  # 1.0 = exact after normalization; < 1.0 = fuzzy
    is_exact: bool


def match_counterparty(target: str, candidates: list[str]) -> CounterpartyMatch | None:
    """Best match for `target` among `candidates` (e.g. ledger counterparty names).

    Returns None if nothing clears FUZZY_MATCH_THRESHOLD. Ties on score are
    broken by shortest normalized candidate (the more specific/less padded
    name), then lexicographically, for determinism.
    """
    target_norm = normalize_counterparty(target)
    if not target_norm:
        return None

    exact: list[str] = []
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        candidate_norm = normalize_counterparty(candidate)
        if not candidate_norm:
            continue
        if candidate_norm == target_norm:
            exact.append(candidate)
            continue
        score = SequenceMatcher(None, target_norm, candidate_norm).ratio()
        if score >= FUZZY_MATCH_THRESHOLD:
            scored.append((score, candidate))

    if exact:
        best = sorted(exact, key=lambda c: (len(normalize_counterparty(c)), c))[0]
        return CounterpartyMatch(candidate=best, score=1.0, is_exact=True)

    if scored:
        scored.sort(key=lambda pair: (-pair[0], len(normalize_counterparty(pair[1])), pair[1]))
        best_score, best_candidate = scored[0]
        return CounterpartyMatch(candidate=best_candidate, score=best_score, is_exact=False)

    return None
