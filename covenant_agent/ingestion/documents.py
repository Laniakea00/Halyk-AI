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
from pathlib import Path

from covenant_agent.ingestion.pdf_text import extract_pdf_text
from covenant_agent.models import ParsedDocument

logger = logging.getLogger(__name__)

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
            text, from_cache = extract_pdf_text(path, pdf_cache_dir)
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
