"""Central paths and constants for the pipeline.

Nothing scenario-specific lives here (no hardcoded account IDs, borrower
names, or covenant numbers) — the private dataset dropped on Aug 9 will
have different borrowers, and the only things allowed to be fixed ahead of
time are *file locations* and *generic structural patterns* (regexes,
keyword lists). Anything scenario-specific must be *derived* at runtime
from the ledger and the submission template (see resolution/pipeline.py).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default dataset location. Overridable via CLI flags in scripts/*.py so the
# same code runs unmodified against the private dataset on the day.
DEFAULT_DATA_DIR = REPO_ROOT / "agentic-bank-hidden"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"

LEDGER_FILENAME = "master_ledger_2025.csv"
TEMPLATE_FILENAME = "submission_template.json"
DOCUMENTS_DIRNAME = "documents"

# pdftotext with -layout preserves column/table alignment far better than
# plain extraction, which matters for reading clause numbers and tables in
# these agreements. Confirmed all 200 public PDFs carry a real text layer
# (no OCR needed); ingestion still records char_count so an empty/near-empty
# extraction (e.g. a scanned PDF in the private set) is visible downstream
# instead of silently producing an unusable document.
PDFTOTEXT_ARGS = ("-layout",)
