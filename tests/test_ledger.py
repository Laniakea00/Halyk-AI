"""Offline tests for ingestion/ledger.py — red-team findings H2 (encoding)
and H3 (thousands-separator parsing). No API calls, no real dataset needed.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from covenant_agent.ingestion.ledger import TXN_ID_RE, _normalize_amount_string, _parse_amount, load_ledger

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


class TxnIdCategorySegmentTest(unittest.TestCase):
    """Private dataset (2026-08-09): scenario KC's rows use an extra alpha
    category segment before the sequence number (e.g. "TXN-KC-CAP-29"),
    which the original "TXN-<scenario>-<digits>" regex rejected outright —
    silently dropping every KC row and crashing derive_scenario_accounts
    with "0 accounts". Confirmed live against the real private ledger.
    """

    def test_category_segmented_ids_parse_with_correct_scenario_and_seq(self) -> None:
        for txn_id, expected_scenario, expected_seq in (
            ("TXN-KC-CAP-29", "KC", "29"),
            ("TXN-KC-FIN-19", "KC", "19"),
            ("TXN-KC-REV-52", "KC", "52"),
            ("TXN-KC-CON-42", "KC", "42"),
            ("TXN-KC-MKT-08", "KC", "08"),
        ):
            with self.subTest(txn_id=txn_id):
                match = TXN_ID_RE.match(txn_id)
                self.assertIsNotNone(match)
                self.assertEqual(match.group("scenario"), expected_scenario)
                self.assertEqual(match.group("seq"), expected_seq)

    def test_plain_scenario_ids_still_parse_unchanged(self) -> None:
        # Regression guard: the public dataset's plain "TXN-<scenario>-<seq>"
        # shape, and the noise-account "TXN-<digits>-<seq>" shape, must never
        # be affected by the new optional category-segment group.
        for txn_id, expected_scenario, expected_seq in (
            ("TXN-J3-0035", "J3", "0035"),
            ("TXN-X1-0065", "X1", "0065"),
            ("TXN-9170-0002", "9170", "0002"),
            ("TXN-P1-0001", "P1", "0001"),
        ):
            with self.subTest(txn_id=txn_id):
                match = TXN_ID_RE.match(txn_id)
                self.assertIsNotNone(match)
                self.assertEqual(match.group("scenario"), expected_scenario)
                self.assertEqual(match.group("seq"), expected_seq)

    def test_load_ledger_end_to_end_with_category_segment(self) -> None:
        body = (
            "TXN-KC-CAP-29,2025-01-09,TELE-4471,KC,Vendor,description,-100.00,USD\n"
        ).encode("utf-8")
        with TemporaryDirectory() as tmp:
            path = _write_ledger(tmp, body)
            transactions = load_ledger(path)
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0].scenario_id, "KC")
        self.assertEqual(transactions[0].amount, -100.0)


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
