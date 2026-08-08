"""Load every file under documents/ into a ParsedDocument, dispatched by type.

The case description says the documents/ folder is a bulk export of mixed
working files with opaque hashed filenames — and in practice it also
contains a handful of files that are not documents at all (a Windows
thumbnail cache, an unrelated server-access-log CSV). We don't special-case
those by name; we just skip anything we can't read as text, loudly, so a
genuinely relevant file never disappears silently.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from covenant_agent.ingestion.pdf_text import extract_pdf_text
from covenant_agent.models import ParsedDocument

logger = logging.getLogger(__name__)

# H5 (red-team): a digit or lowercase letter immediately followed by an
# uppercase letter (either script) — the mirror-image failure to the
# already-known letter-*spacing* artifact fixed in resolution/accounts.py
# (that one inserts spurious spaces INSIDE a stylized header token; this one
# is pdftotext instead losing a real space AT a token boundary — an account
# number, currency code, or capitalized company name abutting plain body
# text with no whitespace in the underlying PDF content stream).
#
# Deliberately narrow: only acts on a digit/lowercase -> uppercase
# transition, never on a letter directly followed by a digit. Splitting
# letter-then-digit indiscriminately would corrupt this dataset's own short
# alphanumeric codes that must stay glued — scenario ids ("P5", "B4"),
# quarter references ("Q1"-"Q4") — so that direction is left alone rather
# than guessed at.
_MERGED_TOKEN_RE = re.compile(r"([0-9а-яёa-z])([A-ZА-ЯЁ])")

# The mirror direction: a multi-letter ALL-CAPS run (an acronym — "EBITDA",
# "KZT", "ACC") immediately followed by a lowercase letter, e.g.
# "EBITDAза2023" -> "EBITDA за2023". Requires 2+ uppercase letters, not 1 —
# a single capital immediately followed by lowercase is just an ordinary
# capitalized word («Кредит», «Заёмщик») and must never get a space forced
# into the middle of it.
_ACRONYM_MERGED_RE = re.compile(r"([A-ZА-ЯЁ]{2,})([а-яёa-z])")


def _restore_lost_spaces(text: str) -> str:
    text = _MERGED_TOKEN_RE.sub(r"\1 \2", text)
    return _ACRONYM_MERGED_RE.sub(r"\1 \2", text)

# Extensions we know how to read as plain text. Deliberately small and
# explicit — an unfamiliar extension should be skipped-and-logged, not
# guessed at.
TEXT_EXTENSIONS = {".txt", ".csv", ".md", ".json"}

# Below this many characters, treat a PDF extraction as suspect (most likely
# a scanned/image-only page pdftotext couldn't read) rather than a real
# empty document.
MIN_PLAUSIBLE_CHARS = 20


def load_documents(documents_dir: Path, cache_dir: Path) -> list[ParsedDocument]:
    parsed: list[ParsedDocument] = []
    pdf_cache_dir = cache_dir / "pdf_text"

    for path in sorted(documents_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            try:
                text, from_cache = extract_pdf_text(path, pdf_cache_dir)
            except Exception:
                # A single corrupt/unparseable PDF (pdftotext exits
                # non-zero — a genuinely broken file, not the "scanned,
                # no text layer" case handled below via char_count) must
                # not take down ingestion for the other ~200 documents.
                # Confirmed as a real gap: this loop previously had no
                # per-file isolation at all, unlike every other batch loop
                # in the pipeline (extraction, linking) fixed earlier.
                logger.exception(
                    "Failed to extract text from %s — skipping this document, "
                    "continuing with the rest of the corpus.",
                    path.name,
                )
                continue
            text = _restore_lost_spaces(text)
            file_type = "pdf"
        elif suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            from_cache = False
            file_type = suffix.lstrip(".")
        else:
            logger.warning("Skipping unrecognized file in documents/: %s", path.name)
            continue

        doc = ParsedDocument(
            doc_id=path.stem,
            source_path=path,
            file_type=file_type,
            text=text,
            char_count=len(text),
            from_cache=from_cache,
        )
        if doc.char_count < MIN_PLAUSIBLE_CHARS:
            logger.warning(
                "Near-empty extraction for %s (%d chars) — possible scanned PDF "
                "with no text layer; this document will effectively be invisible "
                "to every downstream layer.",
                path.name,
                doc.char_count,
            )
        parsed.append(doc)

    return parsed
