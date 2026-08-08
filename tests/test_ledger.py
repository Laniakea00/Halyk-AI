"""Offline tests for ingestion/ledger.py — red-team findings H2 (encoding)
and H3 (thousands-separator parsing). No API calls, no real dataset needed.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from covenant_agent.ingestion.ledger import _normalize_amount_string, _parse_amount, load_ledger

CSV_HEADER = "txn_id,date,account_id,scenario_id,counterparty,description,amount,currency\n"


def _write_ledger(tmp: str, body_bytes: bytes) -> Path:
    path = Path(tmp) / "ledger.csv"
    path.write_bytes(CSV_HEADER.encode("utf-8") + body_bytes)
    return path


class NormalizeAmountStringTest(unittest.TestCase):
    """H3: float() alone cannot parse a thousands separator at all — a
    private-dataset ledger using one (a common CSV export convention)
    would otherwise degrade every single amount to None.
    """

    def test_us_style_thousands_and_decimal(self) -> None:
        self.assertEqual(_normalize_amount_string("1,435,608.42", "T1"), "1435608.42")
        self.assertEqual(_normalize_amount_string("-1,435,608.42", "T1"), "-1435608.42")

    def test_comma_only_proper_thousands_grouping(self) -> None:
        self.assertEqual(_normalize_amount_string("1,435,608", "T1"), "1435608")

    def test_single_comma_not_shaped_like_grouping_treated_as_decimal(self) -> None:
        self.assertEqual(_normalize_amount_string("1435608,42", "T1"), "1435608.42")

    def test_plain_number_without_separators_is_unchanged(self) -> None:
        # The public dataset's own confirmed format — must never be touched.
        self.assertEqual(_normalize_amount_string("-1435608.42", "T1"), "-1435608.42")
        self.assertEqual(_normalize_amount_string("884204.16", "T1"), "884204.16")

    def test_parse_amount_end_to_end_with_thousands_separator(self) -> None:
        self.assertEqual(_parse_amount("1,435,608.42", "T1"), 1435608.42)
        self.assertEqual(_parse_amount("-2,418,663.27", "T1"), -2418663.27)

    def test_parse_amount_still_returns_none_for_genuinely_unparseable(self) -> None:
        self.assertIsNone(_parse_amount("not-a-number", "T1"))


class LoadLedgerEncodingTest(unittest.TestCase):
    """H2: a non-UTF-8 byte anywhere in the ledger must not crash the
    single most foundational data load in the pipeline.
    """

    def test_invalid_utf8_bytes_do_not_crash_load(self) -> None:
        # A raw CP1251-style byte sequence that is NOT valid UTF-8,
        # embedded in the counterparty field.
        bad_bytes = b"TXN-P1-0001,2025-01-01,ACC-0001,P1,Vendor\xd0\xdd\xe2\xd0\xdb,test,-100.00,USD\n"
        with TemporaryDirectory() as tmp:
            path = _write_ledger(tmp, bad_bytes)
            transactions = load_ledger(path)  # must not raise UnicodeDecodeError
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0].amount, -100.0)

    def test_clean_utf8_still_parses_normally(self) -> None:
        clean = "TXN-P1-0001,2025-01-01,ACC-0001,P1,Продавец ООО,test,-100.00,USD\n".encode("utf-8")
        with TemporaryDirectory() as tmp:
            path = _write_ledger(tmp, clean)
            transactions = load_ledger(path)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0].counterparty, "Продавец ООО")


if __name__ == "__main__":
    unittest.main()
