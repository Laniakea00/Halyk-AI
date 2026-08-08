"""Load master_ledger_2025.csv and derive scenario_id -> account_id.

The ledger mixes ~350+ accounts in one flat table; only the handful listed
in submission_template.json are ever relevant to scoring. We deliberately
derive the scenario -> account_id mapping from the ledger itself (via the
txn_id's own encoding, `TXN-<scenario_id>-<seq>`) rather than hardcoding
account numbers, so this keeps working unmodified against the private
dataset's different borrowers/accounts.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

from covenant_agent.models import Transaction

logger = logging.getLogger(__name__)

# "TXN-<scenario_id>-<sequence>" — scenario_id is alnum (e.g. "P1", "B4",
# but also plain-numeric noise accounts like "9001"); sequence is digits.
TXN_ID_RE = re.compile(r"^TXN-(?P<scenario>[A-Za-z0-9]+)-(?P<seq>\d+)$")


_THOUSANDS_GROUP_RE = re.compile(r"^-?\d{1,3}(,\d{3})+$")


def _normalize_amount_string(raw: str, txn_id: str) -> str:
    """Best-effort thousands/decimal-separator cleanup before float() ever
    sees the string.

    Confirmed necessary as a defensive measure, not observed as a live bug:
    plain float() cannot parse a comma at all, so a private-dataset ledger
    using thousands-separated numbers (e.g. "1,435,608.42" — a very common
    CSV export convention) would otherwise degrade *every* amount to None,
    not just genuinely malformed ones — a dataset-wide failure that would
    look like "the classifier found nothing" everywhere rather than what
    it actually is. The public dataset's own numbers never use a separator
    at all ("-1435608.42"), so this must never change an already-plain
    number — the three branches below are mutually exclusive and only
    activate when a comma is actually present.
    """
    if "," not in raw:
        return raw
    if "." in raw:
        # Both present: comma=thousands, period=decimal (US convention) —
        # the only shape this case's own materials ever use when a
        # separator appears at all.
        normalized = raw.replace(",", "")
    elif _THOUSANDS_GROUP_RE.match(raw):
        # Comma-only, but shaped like proper 3-digit grouping
        # ("1,435,608") — a whole number with thousands separators, no
        # decimal part.
        normalized = raw.replace(",", "")
    else:
        # A single comma not shaped like thousands-grouping — most likely
        # a European decimal comma ("1435608,42").
        normalized = raw.replace(",", ".")
    if normalized != raw:
        logger.warning(
            "Ledger row %s: amount %r normalized to %r (thousands/decimal separator "
            "cleanup) before parsing — verify this matches the dataset's real number "
            "format if this looks wrong.",
            txn_id,
            raw,
            normalized,
        )
    return normalized


def _parse_amount(raw: str, txn_id: str) -> float | None:
    raw = raw.strip()
    if not raw:
        logger.warning(
            "Ledger row %s has an empty amount — a genuinely dirty row in the "
            "public dataset, not a parsing bug. Keeping the row with amount=None; "
            "downstream layers must decide whether/how to recover the true value "
            "(e.g. from a supporting document) rather than silently treating it as 0.",
            txn_id,
        )
        return None
    normalized = _normalize_amount_string(raw, txn_id)
    try:
        return float(normalized)
    except ValueError:
        logger.warning("Ledger row %s has an unparseable amount %r — keeping as None.", txn_id, raw)
        return None


def load_ledger(ledger_path: Path) -> list[Transaction]:
    transactions: list[Transaction] = []
    # errors="replace": a single non-UTF-8 byte anywhere in the file (e.g.
    # CP1251-encoded Cyrillic, or a stray BOM from a different export tool)
    # must not crash the single most foundational data load in the whole
    # pipeline — documents.py's TEXT_EXTENSIONS path already had this same
    # safety net; the ledger load didn't, confirmed as an inconsistency,
    # not an intentional stricter check.
    with ledger_path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txn_id = row["txn_id"].strip()
            match = TXN_ID_RE.match(txn_id)
            if not match:
                logger.warning("Skipping ledger row with unparseable txn_id: %r", txn_id)
                continue
            transactions.append(
                Transaction(
                    txn_id=txn_id,
                    date=row["date"].strip(),
                    account_id=row["account_id"].strip(),
                    scenario_id=match.group("scenario"),
                    counterparty=row["counterparty"].strip(),
                    description=row["description"].strip(),
                    amount=_parse_amount(row["amount"], txn_id),
                    currency=row["currency"].strip(),
                )
            )
    return transactions


def derive_scenario_accounts(
    transactions: list[Transaction], scenario_ids: list[str]
) -> dict[str, str]:
    """Map each required scenario_id to its single account_id, per the ledger.

    Raises if a required scenario has no rows at all (nothing to compute
    from) or if its rows disagree on account_id (an ambiguity the rest of
    the pipeline cannot silently paper over — better to fail loudly here
    than to quietly answer for the wrong account).
    """
    accounts_by_scenario: dict[str, set[str]] = defaultdict(set)
    for txn in transactions:
        if txn.scenario_id in scenario_ids:
            accounts_by_scenario[txn.scenario_id].add(txn.account_id)

    resolved: dict[str, str] = {}
    problems: list[str] = []
    for scenario_id in scenario_ids:
        accounts = accounts_by_scenario.get(scenario_id, set())
        if not accounts:
            problems.append(f"{scenario_id}: no ledger rows found (0 accounts)")
        elif len(accounts) > 1:
            problems.append(f"{scenario_id}: ambiguous accounts {sorted(accounts)}")
        else:
            resolved[scenario_id] = next(iter(accounts))

    if problems:
        raise ValueError(
            "Could not derive a unique account_id for every required scenario:\n  "
            + "\n  ".join(problems)
        )
    return resolved
