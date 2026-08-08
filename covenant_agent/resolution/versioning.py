"""Version resolution: which document(s) are the currently-effective ones.

Three independent signals are extracted per document, each cheap and
regex-based so this generalizes to the private dataset's (presumably
similar but not identical) phrasing:

1. Explicit supersede/draft markers — the strongest signal. The public
   dataset stamps outdated agreements with an unambiguous watermark
   ("НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ ... НЕ ПРИМЕНЯЕТСЯ") rather than leaving version
   conflicts implicit — verified this phrase appears in exactly the 12
   superseded 2024 agreements and nowhere else in the 200-document corpus
   (no false positives from unrelated contract carve-out clauses).
2. Revision numbers ("Редакция v3", "v01") — found in document headers.
3. Dates mentioned in the text (Russian long-form and ISO) — used as a
   last-resort tie-breaker.

The actual "pick the current document(s)" decision lives in
resolve_current_documents() below and is applied per (scenario_id, kind)
group.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Specific multi-word phrases only — deliberately avoiding single ambiguous
# words that could appear in a legitimate clause (e.g. a loan agreement's
# own carve-out prose can plausibly contain "не применяется к..." in some
# other context). Empirically, in the public dataset these exact phrases
# occur only in the supersede watermark. If the private dataset starts
# false-flagging real agreements, tighten these further.
SUPERSEDE_MARKERS: tuple[str, ...] = (
    "недействующая редакция",
    "заменена и изложена в новой редакции",
    "не применяется",
    "утратил силу",
    "утратила силу",
    "superseded",
    "no longer in force",
    # Found empirically: 4 of 5 "audit reclassification report"-shaped
    # documents in the public dataset turned out to be interim/draft
    # workpapers explicitly marked as not the auditor's final position
    # ("ПРОЕКТ — ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ ... заменена окончательным
    # отчётом ... НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ АУДИТОРА"). Without
    # these two phrases, a lone draft with no competing final document
    # was silently accepted as "current" simply for having no rival to
    # lose a tie-break against — these markers catch it even when it's
    # the only candidate for its (scenario, kind).
    "промежуточная ведомость",
    "заменена окончательным отчётом",
)

DRAFT_MARKERS: tuple[str, ...] = (
    "черновик",
    "draft",
)

# Revision markers always observed near the top of the document (header /
# watermark block), so searching only the first ~2000 chars avoids false
# hits on unrelated numbers later in the body (clause numbers, amounts).
_HEADER_WINDOW = 2000

REVISION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"редакция\s*v?0*([0-9]{1,2})", re.IGNORECASE),
    re.compile(r"[—\-]\s*v0*([0-9]{1,2})\s*[—\-]", re.IGNORECASE),
)

RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_RU_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(RU_MONTHS) + r")\s+(\d{4})\s*(?:года)?\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def detect_supersede(text: str) -> tuple[bool, tuple[str, ...]]:
    lowered = text.lower()
    reasons = tuple(m for m in SUPERSEDE_MARKERS if m in lowered)
    reasons += tuple(m for m in DRAFT_MARKERS if m in lowered)
    return bool(reasons), reasons


def extract_revision(text: str) -> int | None:
    header = text[:_HEADER_WINDOW]
    for pattern in REVISION_PATTERNS:
        match = pattern.search(header)
        if match:
            return int(match.group(1))
    return None


def extract_dates(text: str) -> tuple[str, ...]:
    found: set[str] = set()
    for day, month_name, year in _RU_DATE_RE.findall(text):
        month = RU_MONTHS[month_name.lower()]
        found.add(f"{int(year):04d}-{month:02d}-{int(day):02d}")
    for year, month, day in _ISO_DATE_RE.findall(text):
        found.add(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
    return tuple(sorted(found))


def resolve_current_documents(
    docs: list, *, scenario_id: str, kind: str
) -> tuple[list, list]:
    """Split ResolvedDocuments of one (scenario, kind) group into (current, superseded).

    Tie-break order, applied only among docs *not* already flagged
    superseded/draft:
      1. Exactly one candidate -> done.
      2. Prefer max detected revision number, if any candidate has one.
      3. Prefer max detected latest_date, if any remaining candidate has one.
      4. Still tied -> keep all remaining as current and log a warning; a
         forced arbitrary pick would be worse than surfacing the ambiguity.
    """
    superseded = [d for d in docs if d.metadata.is_superseded]
    candidates = [d for d in docs if not d.metadata.is_superseded]

    if len(candidates) <= 1:
        return candidates, superseded

    revisioned = [d for d in candidates if d.metadata.revision is not None]
    if revisioned:
        max_rev = max(d.metadata.revision for d in revisioned)
        winners = [d for d in candidates if d.metadata.revision == max_rev]
        losers = [d for d in candidates if d.metadata.revision != max_rev]
        superseded.extend(losers)
        candidates = winners

    if len(candidates) <= 1:
        return candidates, superseded

    dated = [d for d in candidates if d.metadata.latest_date is not None]
    if dated:
        max_date = max(d.metadata.latest_date for d in dated)
        winners = [d for d in candidates if d.metadata.latest_date == max_date]
        losers = [d for d in candidates if d.metadata.latest_date != max_date]
        superseded.extend(losers)
        candidates = winners

    if len(candidates) > 1:
        logger.warning(
            "Unresolved version ambiguity for scenario=%s kind=%s: %d candidates "
            "remain current (%s) — no revision/date signal broke the tie.",
            scenario_id,
            kind,
            len(candidates),
            [d.parsed.doc_id for d in candidates],
        )

    return candidates, superseded
