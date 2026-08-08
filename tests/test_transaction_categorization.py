"""Offline regression guard for transaction_categorization.py's SYSTEM_PROMPT.

Can't test LLM behavior without a real API call (see scripts/run_calculation.py
for that), but this at least catches an accidental deletion/rewording of the
decoy guidance that was added after diagnosing B4's 6.1: an "advance cargo
sales settlement" transaction was being counted as revenue even though the
covenant's own definition is accrual/recognition-based, not cash-received-based.
"""

from __future__ import annotations

import unittest

from covenant_agent.linking.transaction_categorization import SYSTEM_PROMPT


class DecoyGuidanceTest(unittest.TestCase):
    def test_advance_and_prepayment_decoys_are_named(self) -> None:
        for term in ("advance", "prepayment", "аванс", "предоплата"):
            with self.subTest(term=term):
                self.assertIn(term, SYSTEM_PROMPT.lower())

    def test_accrual_recognition_rationale_is_present(self) -> None:
        self.assertIn("not yet recognized revenue", SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
