"""Offline tests for resolution/pipeline.py's run_ingestion — specifically
AccountLinkingRiskError (see README's structural-surprises audit): if the
account_id convention in a new dataset doesn't match ACCOUNT_TOKEN_RE's
hardcoded "ACC-" prefix, document-to-scenario linking fails for every
scenario at once, and every downstream covenant silently falls back to
COMPLIANT — this must stop the run loudly instead. Tiny synthetic
data_dir fixtures, no real dataset, no API calls.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from covenant_agent.resolution.pipeline import AccountLinkingRiskError, run_ingestion

_TEMPLATE = {
    "team": "",
    "contact_email": "",
    "model": "",
    "answers": {
        "X1": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
        "X2": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
    },
}

_LEDGER_HEADER = "txn_id,date,account_id,counterparty,description,amount,currency\n"
_LEDGER_ROWS = (
    "TXN-X1-0001,2025-01-01,ACC-0001,Vendor A,test,-100.00,USD\n"
    "TXN-X2-0001,2025-01-01,ACC-0002,Vendor B,test,-200.00,USD\n"
)

_CREDIT_AGREEMENT_TEXT = (
    "Договор банковского займа. Заёмщик обязуется перед Кредитором. "
    "Финансовые ковенанты. Счёт ACC-{acc}."
)


def _build_data_dir(tmp: str, *, documents: dict[str, str]) -> Path:
    data_dir = Path(tmp)
    (data_dir / "submission_template.json").write_text(json.dumps(_TEMPLATE), encoding="utf-8")
    (data_dir / "master_ledger_2025.csv").write_text(_LEDGER_HEADER + _LEDGER_ROWS, encoding="utf-8")
    docs_dir = data_dir / "documents"
    docs_dir.mkdir()
    for name, text in documents.items():
        (docs_dir / name).write_text(text, encoding="utf-8")
    return data_dir


class AccountLinkingRiskErrorTest(unittest.TestCase):
    def test_raises_when_no_documents_link_to_any_scenario(self) -> None:
        # Simulates the exact failure mode: account_id format changed, so
        # no document's ACC- token (or lack thereof) ever matches a known
        # account — both required scenarios end up with zero documents.
        with TemporaryDirectory() as tmp, TemporaryDirectory() as cache:
            data_dir = _build_data_dir(tmp, documents={})
            with self.assertRaises(AccountLinkingRiskError) as cm:
                run_ingestion(data_dir, Path(cache))
        message = str(cm.exception)
        self.assertIn("2/2", message)
        self.assertIn("ACC-", message)
        self.assertIn("X1", message)
        self.assertIn("X2", message)

    def test_does_not_raise_when_documents_link_normally(self) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as cache:
            data_dir = _build_data_dir(
                tmp,
                documents={
                    "doc1.txt": _CREDIT_AGREEMENT_TEXT.format(acc="0001"),
                    "doc2.txt": _CREDIT_AGREEMENT_TEXT.format(acc="0002"),
                },
            )
            result = run_ingestion(data_dir, Path(cache))  # must not raise
        self.assertIn("credit_agreement", result.scenarios["X1"].current_documents)
        self.assertIn("credit_agreement", result.scenarios["X2"].current_documents)

    def test_does_not_raise_when_only_a_minority_are_missing(self) -> None:
        # Only 1 of a 3-scenario template misses its credit_agreement —
        # below the 50% threshold, a plausible genuine per-scenario gap,
        # not a systemic linking failure signal.
        template = {
            "team": "",
            "contact_email": "",
            "model": "",
            "answers": {
                "X1": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
                "X2": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
                "X3": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
            },
        }
        with TemporaryDirectory() as tmp, TemporaryDirectory() as cache:
            data_dir = Path(tmp)
            (data_dir / "submission_template.json").write_text(json.dumps(template), encoding="utf-8")
            (data_dir / "master_ledger_2025.csv").write_text(
                _LEDGER_HEADER + _LEDGER_ROWS + "TXN-X3-0001,2025-01-01,ACC-0003,Vendor C,test,-300.00,USD\n",
                encoding="utf-8",
            )
            docs_dir = data_dir / "documents"
            docs_dir.mkdir()
            (docs_dir / "doc1.txt").write_text(_CREDIT_AGREEMENT_TEXT.format(acc="0001"), encoding="utf-8")
            (docs_dir / "doc2.txt").write_text(_CREDIT_AGREEMENT_TEXT.format(acc="0002"), encoding="utf-8")
            # X3 deliberately gets no document at all.
            result = run_ingestion(data_dir, Path(cache))  # must not raise
        self.assertNotIn("credit_agreement", result.scenarios["X3"].current_documents)


if __name__ == "__main__":
    unittest.main()
