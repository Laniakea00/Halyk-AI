"""Loose date/period parsing for audit-reclassification linking.

`AuditReclassification.transaction_date_or_period` is free text: an ISO
date if the report is precise, a coarser period ("Q1 2025", a month name)
if it isn't, or null if the report gives no date at all (confirmed on the
public dataset's only real example — see linking/reclassification_linking.py).
Everything here degrades to "couldn't parse, don't use date as a filter"
rather than raising — a date we can't parse should weaken confidence, not
break the join.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_QUARTER_RE = re.compile(r"\bQ([1-4])\s*(\d{4})\b", re.IGNORECASE)
_YEAR_MONTH_RE = re.compile(r"\b(\d{4})-(\d{2})\b")

RU_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
    "ма": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

# How many days on either side of a parsed exact date still counts as a
# plausible match — reclassification reports round to a period more often
# than they cite an exact date, so a same-day requirement would under-match.
DATE_TOLERANCE_DAYS = 14


def _safe_date(year: int, month: int, day: int) -> date | None:
    """date(...) construction that degrades to None on an out-of-range
    component (e.g. month=99, day=45) instead of raising.

    Confirmed necessary: this module's own docstring already promises
    "everything here degrades... rather than raising", but a
    pattern-shaped-yet-calendar-invalid string (OCR noise, a reference
    number that happens to look date-shaped, e.g. "9999-99-99") broke that
    promise for real — date() raised ValueError, uncaught, all the way up
    through reclassification_linking.py's caller.
    """
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_period(text: str | None) -> tuple[date, date] | None:
    """Best-effort parse of a date/period string into an inclusive (start, end) range."""
    if not text:
        return None

    iso = _ISO_RE.search(text)
    if iso:
        d = _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        if d is not None:
            return d - timedelta(days=DATE_TOLERANCE_DAYS), d + timedelta(days=DATE_TOLERANCE_DAYS)

    quarter = _QUARTER_RE.search(text)
    if quarter:
        q, year = int(quarter.group(1)), int(quarter.group(2))
        start = _safe_date(year, (q - 1) * 3 + 1, 1)
        if start is not None:
            return start, _next_month(start, 3) - timedelta(days=1)

    year_month = _YEAR_MONTH_RE.search(text)
    if year_month:
        year, month = int(year_month.group(1)), int(year_month.group(2))
        start = _safe_date(year, month, 1)
        if start is not None:
            return start, _next_month(start, 1) - timedelta(days=1)

    lowered = text.lower()
    for stem, month in RU_MONTHS.items():
        if stem in lowered:
            year_match = re.search(r"\b(20\d{2})\b", text)
            if year_match:
                start = _safe_date(int(year_match.group(1)), month, 1)
                if start is not None:
                    return start, _next_month(start, 1) - timedelta(days=1)

    return None

    return None


def _next_month(d: date, n: int) -> date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return date(year, month, 1)


def date_in_range(txn_date: str, period: tuple[date, date]) -> bool:
    try:
        d = date.fromisoformat(txn_date)
    except ValueError:
        return False
    return period[0] <= d <= period[1]
