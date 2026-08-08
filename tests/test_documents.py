"""Offline tests for ingestion/documents.py — red-team finding H1: a single
corrupt/unparseable PDF must not take down document loading for the rest
of the corpus. No real PDFs needed — extract_pdf_text is mocked so this
never shells out to pdftotext.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from covenant_agent.ingestion.documents import _restore_lost_spaces, load_documents


class RestoreLostSpacesTest(unittest.TestCase):
    """H5: the mirror-image of the known letter-spacing artifact — pdftotext
    sometimes drops a real space at a token boundary instead of inserting
    spurious ones inside a token.
    """

    def test_digit_followed_by_uppercase_cyrillic_gets_a_space(self) -> None:
        self.assertEqual(_restore_lost_spaces("7803Заемщик"), "7803 Заемщик")

    def test_digit_followed_by_uppercase_latin_gets_a_space(self) -> None:
        self.assertEqual(_restore_lost_spaces("83690.23USD"), "83690.23 USD")

    def test_lowercase_cyrillic_followed_by_uppercase_latin_gets_a_space(self) -> None:
        self.assertEqual(_restore_lost_spaces("заемщикACC"), "заемщик ACC")

    def test_lowercase_latin_followed_by_uppercase_cyrillic_gets_a_space(self) -> None:
        self.assertEqual(_restore_lost_spaces("borrowerЗаемщик"), "borrower Заемщик")

    def test_does_not_touch_account_token_itself(self) -> None:
        self.assertEqual(_restore_lost_spaces("ACC-7803"), "ACC-7803")

    def test_does_not_split_scenario_and_quarter_codes(self) -> None:
        # Letter-then-digit is deliberately left alone — splitting it would
        # corrupt short alphanumeric codes this pipeline depends on.
        for code in ("P5", "B4", "Q1", "Q4"):
            self.assertEqual(_restore_lost_spaces(code), code)

    def test_already_spaced_text_is_unchanged(self) -> None:
        text = "Заёмщик, Kyzylorda Drilling Services JSC, обязуется"
        self.assertEqual(_restore_lost_spaces(text), text)

    def test_acronym_run_followed_by_lowercase_gets_a_space(self) -> None:
        # Mirror direction to the digit/lowercase->uppercase rule: a 2+
        # letter ALL-CAPS acronym glued directly to a lowercase continuation.
        self.assertEqual(_restore_lost_spaces("EBITDAза2023"), "EBITDA за2023")
        self.assertEqual(_restore_lost_spaces("расчёт KZTсуммы"), "расчёт KZT суммы")

    def test_single_capital_word_start_is_not_touched(self) -> None:
        # Ordinary capitalization ("Кредит", "Заёмщик") must never get a
        # space forced into the middle of it — only a 2+ letter acronym run
        # is treated as a merge artifact.
        self.assertEqual(_restore_lost_spaces("Кредит"), "Кредит")
        self.assertEqual(_restore_lost_spaces("Заёмщик"), "Заёмщик")


class LoadDocumentsCorruptPdfTest(unittest.TestCase):
    def test_one_corrupt_pdf_is_skipped_not_fatal(self) -> None:
        with TemporaryDirectory() as docs_dir, TemporaryDirectory() as cache_dir:
            docs_path = Path(docs_dir)
            # Content doesn't matter — extract_pdf_text is mocked below,
            # these just need to exist so load_documents' directory scan
            # has something to iterate over.
            (docs_path / "good1.pdf").write_bytes(b"%PDF-fake")
            (docs_path / "corrupt.pdf").write_bytes(b"%PDF-fake")
            (docs_path / "good2.pdf").write_bytes(b"%PDF-fake")

            def fake_extract(path, cache_dir):
                if path.name == "corrupt.pdf":
                    raise RuntimeError("pdftotext failed on corrupt.pdf: simulated corruption")
                return f"real text from {path.name}", False

            with patch(
                "covenant_agent.ingestion.documents.extract_pdf_text", side_effect=fake_extract
            ):
                parsed = load_documents(docs_path, Path(cache_dir))

        # Both good documents survive; the corrupt one is skipped, not fatal.
        names = {d.doc_id for d in parsed}
        self.assertEqual(names, {"good1", "good2"})
        self.assertEqual(len(parsed), 2)

    def test_all_pdfs_corrupt_still_returns_empty_list_not_raises(self) -> None:
        with TemporaryDirectory() as docs_dir, TemporaryDirectory() as cache_dir:
            docs_path = Path(docs_dir)
            (docs_path / "corrupt.pdf").write_bytes(b"%PDF-fake")

            with patch(
                "covenant_agent.ingestion.documents.extract_pdf_text",
                side_effect=RuntimeError("simulated total failure"),
            ):
                parsed = load_documents(docs_path, Path(cache_dir))

        self.assertEqual(parsed, [])

    def test_load_documents_restores_lost_spaces_in_pdf_text(self) -> None:
        with TemporaryDirectory() as docs_dir, TemporaryDirectory() as cache_dir:
            docs_path = Path(docs_dir)
            (docs_path / "doc.pdf").write_bytes(b"%PDF-fake")

            with patch(
                "covenant_agent.ingestion.documents.extract_pdf_text",
                return_value=("Счёт ACC-7803Заемщик получил $100.00USD", False),
            ):
                parsed = load_documents(docs_path, Path(cache_dir))

        self.assertEqual(parsed[0].text, "Счёт ACC-7803 Заемщик получил $100.00 USD")


if __name__ == "__main__":
    unittest.main()
