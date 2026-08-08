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


class ObviousMatchWorkedExamplesTest(unittest.TestCase):
    """Regression guard for the worked examples added after the 9-cell
    post-fix forensic review: B1 and P2 both missed a textbook-clean
    match (a plain operating-cost description, an equipment purchase)
    despite the prompt's own abstract guidance already covering the
    pattern — added as concrete few-shot examples, a different lever
    than the (already-present) abstract rule.
    """

    def test_operating_expense_example_is_present(self) -> None:
        self.assertIn("plant operating and maintenance expenses", SYSTEM_PROMPT.lower())

    def test_equipment_purchase_example_is_present(self) -> None:
        self.assertIn("purchase of blast freezer equipment", SYSTEM_PROMPT.lower())

    def test_recheck_before_unclassified_instruction_is_present(self) -> None:
        self.assertIn("before marking a transaction 'unclassified'", SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
