"""Offline tests for resolution/accounts.py — the letter-spaced-header
normalization fix for extract_account_tokens.

Confirmed necessary on the public dataset: all 12 scenarios' own "Примечания
к финансовой отчётности" documents render their account number inside a
large stylized title with a space inserted between every character
("A C C - 7 8 0 3"), which the exact ACCOUNT_TOKEN_RE match couldn't see
through before this fix — every one of those documents sat unmatched,
hiding covenant-relevant disclosures for every scenario at once.
"""

from __future__ import annotations

import unittest

from covenant_agent.resolution.accounts import extract_account_tokens


class ExtractAccountTokensSpacedHeaderTest(unittest.TestCase):
    def test_letter_spaced_account_token_is_recovered(self) -> None:
        text = "АУДИТОРСКОЕ ДЕЛО № A C C - 7 8 0 3 / 2 0 2 5 ФИН.ГОД"
        self.assertEqual(extract_account_tokens(text), ("ACC-7803",))

    def test_letter_spaced_subaccount_token_is_recovered(self) -> None:
        text = "вспомогательный счёт A C C - 7 8 0 1 - 0 8"
        self.assertEqual(extract_account_tokens(text), ("ACC-7801-08",))

    def test_normal_unspaced_token_still_works(self) -> None:
        text = "Счёт ACC-7803, обычный текст."
        self.assertEqual(extract_account_tokens(text), ("ACC-7803",))

    def test_does_not_swallow_a_trailing_unrelated_number(self) -> None:
        # The real confirmed case: "A C C - 7 8 0 3 / 2 0 2 5" — the "/2025"
        # year must not be absorbed into the account number.
        text = "№ A C C - 7 8 0 3 / 2 0 2 5 ФИН.ГОД"
        self.assertEqual(extract_account_tokens(text), ("ACC-7803",))

    def test_unrelated_spaced_out_text_is_not_touched(self) -> None:
        # A stylized header for something else entirely (no "ACC") must not
        # be affected — this is deliberately narrow, not a blanket
        # whitespace collapse.
        text = "Д О ГО В О Р банковского займа"
        self.assertEqual(extract_account_tokens(text), ())

    def test_multiple_scenarios_worth_of_documents_each_resolve_correctly(self) -> None:
        cases = {
            "A C C - 7 2 0 1": "ACC-7201",
            "A C C - 7 2 0 4": "ACC-7204",
            "A C C - 7 8 0 1": "ACC-7801",
            "A C C - 7 8 1 0": "ACC-7810",
        }
        for spaced, expected in cases.items():
            with self.subTest(spaced=spaced):
                self.assertEqual(extract_account_tokens(f"Дело № {spaced} финансовый год"), (expected,))


if __name__ == "__main__":
    unittest.main()
