"""Offline tests for resolution/versioning.py's supersede-watermark detection.

Private dataset (2026-08-09): "не применяется" was removed from
SUPERSEDE_MARKERS after a confirmed live false positive — springing-covenant
carve-out prose routinely reads "...ограничение ... не применяется" as
ordinary contract text, not a supersede watermark. This wrongly demoted the
genuinely current 2025 credit agreement to "superseded" across 6 real
scenarios (B2, H2, J6, X1, X2, X3), leaving each with zero current
credit_agreement documents.
"""

from __future__ import annotations

import unittest

from covenant_agent.resolution.versioning import detect_supersede


class DetectSupersedeTest(unittest.TestCase):
    def test_springing_covenant_carve_out_is_not_flagged_as_superseded(self) -> None:
        # Real confirmed private-dataset text (B2's genuinely current 2025
        # credit agreement).
        text = (
            "Пункт 6.1 Капитальные затраты. Пока Коэффициент долговой "
            "нагрузки не превышает 3.00x, указанное ограничение "
            "Капитальных затрат не применяется.\n\n"
            "Пункт 6.2 Минимальная выручка по категории. Mangystau "
            "Industrial JSC обязуется..."
        )
        is_superseded, reasons = detect_supersede(text)
        self.assertFalse(is_superseded)
        self.assertEqual(reasons, ())

    def test_real_supersede_watermark_still_detected(self) -> None:
        text = "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ (2024 г.). Заменена и изложена в новой редакции действующим Договором текущего периода. НЕ ПРИМЕНЯЕТСЯ."
        is_superseded, reasons = detect_supersede(text)
        self.assertTrue(is_superseded)
        self.assertIn("недействующая редакция", reasons)
        self.assertIn("заменена и изложена в новой редакции", reasons)

    def test_no_markers_at_all_is_not_superseded(self) -> None:
        is_superseded, reasons = detect_supersede("Обычный текст договора без пометок.")
        self.assertFalse(is_superseded)
        self.assertEqual(reasons, ())

    def test_draft_workpaper_markers_still_detected(self) -> None:
        text = "ПРОЕКТ — ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ, заменена окончательным отчётом. НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ АУДИТОРА."
        is_superseded, reasons = detect_supersede(text)
        self.assertTrue(is_superseded)
        self.assertIn("промежуточная ведомость", reasons)
        self.assertIn("заменена окончательным отчётом", reasons)

    def test_overdraft_disclosure_is_not_flagged_as_draft(self) -> None:
        # Real confirmed private-dataset text (G3's only financial_notes
        # document) — the plain substring check for "draft" matched inside
        # "overdraft", wrongly flagging this document as a draft workpaper
        # with no competing final document to lose a tie-break against.
        text = (
            "Раскрытия для агрегирования ковенантов\n\n(8.1) Для целей "
            "агрегирования по ковенантам an approved overdraft facility "
            "в размере $312,480.55 раскрывается и не отражается отдельной "
            "операцией в бухгалтерском учёте."
        )
        is_superseded, reasons = detect_supersede(text)
        self.assertFalse(is_superseded)
        self.assertEqual(reasons, ())

    def test_actual_draft_watermark_still_detected_with_word_boundary(self) -> None:
        text = "DRAFT — INTERIM FIELDWORK SCHEDULE. Not the auditor's final position."
        is_superseded, reasons = detect_supersede(text)
        self.assertTrue(is_superseded)
        self.assertIn("draft", reasons)


if __name__ == "__main__":
    unittest.main()
