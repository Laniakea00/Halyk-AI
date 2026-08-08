"""Secondary, narrower document-to-scenario linking — only for documents
accounts.py's exact ACC-token match already failed to place anywhere.

accounts.py (left untouched — it's the primary, reliable path) matches
documents to scenarios via an exact ACC-<digits> token. That mechanism can
never link a Group-parent-level document (e.g. a consolidated annual
report) to a subsidiary scenario, because such a document is about a
*different* legal entity and will never carry the subsidiary's own bank
account token. Confirmed on the public dataset: P5's covenant 6.1
references "капитальные затраты Группы, определяемые по консолидированной
отчётности конечной материнской компании Группы" (Group capex, per the
ultimate parent's consolidated financial statements) — the actual
consolidated report (Sarybel Energy Holding JSC, in English) sat in
unmatched_documents because it obviously never mentions ACC-7805.

The fix here is narrow and specific, not a general company-name fuzzy
matcher: it looks for an *exact* match of a scenario's own already-verified
borrower name (taken from that scenario's own ACC-token-matched credit
agreement — never guessed from the candidate document's own name) appearing
next to explicit "this document names the borrower as its own
subsidiary/segment" language. Confirmed necessary and sufficient on the real
case: the Sarybel document's Note 6 says "The Group's ... segment is
conducted through Ekibastuz Power Services JSC" — an explicit, in-document
statement naming the borrower by its own verified name. Matching on the
*reporting company's own* name similarity instead (e.g. assuming "Sarybel
Energy Holding JSC" relates to Ekibastuz Power Services JSC because the
names sound like they could be Group-affiliated) is exactly the shortcut
this dataset is known to punish — see accounts.py's docstring on the
near-duplicate-entity trap — and is deliberately not attempted here.
"""

from __future__ import annotations

import re

_SEGMENT_REFERENCE_MARKERS = (
    "conducted through",
    "subsidiary",
    "the group's",
    "сегмент группы",
    "дочерн",  # дочерняя/дочернее (stem)
    "осуществляется через",
)

_PROXIMITY_WINDOW_CHARS = 200

# "Kazakhstan JSC" is a confirmed, universal false-positive source: every
# document in this dataset carries the "Halyk Bank of Kazakhstan JSC"
# letterhead, and accounts.py's best-effort COMPANY_NAME_RE regex extracts
# "Kazakhstan JSC" as a trailing fragment of it in literally every
# scenario's credit agreement — verified across all 12 public scenarios.
# Trusting that fragment as a scenario's "verified borrower name" would
# make this secondary matcher fire on almost any document that happens to
# mention a marker phrase near any Kazakhstan-registered entity's name,
# which is the opposite of the narrow, high-confidence signal this is
# supposed to be. Excluded by exact (case-insensitive) name, not by a
# general heuristic — this is one specific, confirmed noise source, not a
# guess at what else might be noisy.
_KNOWN_NOISE_NAMES = {"kazakhstan jsc"}


def is_trustworthy_borrower_name(name: str) -> bool:
    """False for known-noise extractions (see _KNOWN_NOISE_NAMES) or blanks."""
    normalized = re.sub(r"\s+", " ", name).strip().lower()
    return bool(normalized) and normalized not in _KNOWN_NOISE_NAMES


def references_borrower_as_segment(doc_text: str, borrower_name: str) -> bool:
    """True if `doc_text` names `borrower_name` (exact, whitespace-normalized
    match) within _PROXIMITY_WINDOW_CHARS of an explicit subsidiary/segment
    marker phrase. See module docstring for what this is and isn't for.
    """
    if not is_trustworthy_borrower_name(borrower_name):
        return False
    normalized_text = re.sub(r"\s+", " ", doc_text)
    normalized_name = re.sub(r"\s+", " ", borrower_name).strip()
    for match in re.finditer(re.escape(normalized_name), normalized_text):
        window = normalized_text[
            max(0, match.start() - _PROXIMITY_WINDOW_CHARS) : match.end() + _PROXIMITY_WINDOW_CHARS
        ].lower()
        if any(marker in window for marker in _SEGMENT_REFERENCE_MARKERS):
            return True
    return False
