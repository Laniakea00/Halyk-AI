"""PDF -> text extraction via the system `pdftotext` binary, disk-cached.

Caching is keyed by a hash of the source file's *bytes*, not its filename or
mtime: if an input PDF ever changes, the hash changes and we simply
re-parse. No staleness tracking to get wrong, no manual cache-busting.

Why shell out to `pdftotext -layout` instead of a Python PDF library: it was
verified against all 200 public PDFs to extract clean text with layout
preserved (columns, clause numbering), and poppler is a single `brew
install poppler` away with no Python binding fragility. If the private
dataset turns out to include scanned/image-only PDFs, `char_count` on the
resulting ParsedDocument will be near-zero and downstream code can flag it —
see documents.py — rather than silently losing that document.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from covenant_agent.config import PDFTOTEXT_ARGS

PDFTOTEXT_BIN = shutil.which("pdftotext")


class PdfTotextNotFound(RuntimeError):
    pass


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:20]


def extract_pdf_text(path: Path, cache_dir: Path) -> tuple[str, bool]:
    """Return (text, from_cache) for a single PDF, using/populating the disk cache."""
    if PDFTOTEXT_BIN is None:
        raise PdfTotextNotFound(
            "pdftotext not found on PATH. Install poppler (e.g. `brew install poppler` "
            "on macOS, `apt-get install poppler-utils` on Debian/Ubuntu)."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_content_hash(path)}.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace"), True

    result = subprocess.run(
        [PDFTOTEXT_BIN, *PDFTOTEXT_ARGS, str(path), "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed on {path.name}: {result.stderr.strip()}")

    text = result.stdout
    cache_path.write_text(text, encoding="utf-8")
    return text, False
