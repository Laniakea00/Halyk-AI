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
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ledger row %s has an unparseable amount %r — keeping as None.", txn_id, raw)
        return None


def load_ledger(ledger_path: Path) -> list[Transaction]:
    transactions: list[Transaction] = []
    with ledger_path.open(encoding="utf-8", newline="") as f:
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
