#!/usr/bin/env python3
"""CLI entrypoint for Block 1 (Ingestion + Entity/Version resolution).

Usage:
    python scripts/run_ingestion.py [--data-dir DIR] [--cache-dir DIR] [--report PATH] [-v]

Prints a per-scenario summary to stdout and, if --report is given, writes a
full JSON debug report (every resolved document's metadata, minus raw text)
for inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from covenant_agent.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR  # noqa: E402
from covenant_agent.resolution.pipeline import run_ingestion  # noqa: E402


def _doc_summary(rdoc) -> dict:
    return {
        "doc_id": rdoc.parsed.doc_id,
        "file_type": rdoc.parsed.file_type,
        "kind": rdoc.metadata.kind,
        "kind_score": rdoc.metadata.kind_score,
        "is_superseded": rdoc.metadata.is_superseded,
        "supersede_reasons": rdoc.metadata.supersede_reasons,
        "revision": rdoc.metadata.revision,
        "latest_date": rdoc.metadata.latest_date,
        "matched_scenario_accounts": rdoc.metadata.matched_scenario_accounts,
        "company_names": rdoc.metadata.company_names,
        "char_count": rdoc.parsed.char_count,
        "text_preview": rdoc.parsed.text[:160].replace("\n", " "),
    }


def build_report(result) -> dict:
    return {
        "scenario_count": len(result.scenarios),
        "ledger_rows": len(result.ledger),
        "unmatched_document_count": len(result.unmatched_documents),
        "scenarios": {
            sid: {
                "account_id": bundle.account_id,
                "current_documents": {
                    kind: [_doc_summary(d) for d in docs]
                    for kind, docs in bundle.current_documents.items()
                },
                "superseded_documents": [_doc_summary(d) for d in bundle.superseded_documents],
                "all_matched_document_count": len(bundle.all_matched_documents),
            }
            for sid, bundle in result.scenarios.items()
        },
        "unmatched_documents": [_doc_summary(d) for d in result.unmatched_documents],
    }


def print_summary(result) -> None:
    print(f"\n{'=' * 70}")
    print(
        f"Ledger rows: {len(result.ledger)}   Scenarios: {len(result.scenarios)}   "
        f"Unmatched documents: {len(result.unmatched_documents)}"
    )
    print(f"{'=' * 70}")
    for sid, bundle in sorted(result.scenarios.items()):
        print(
            f"\n[{sid}]  account={bundle.account_id}  "
            f"matched_docs={len(bundle.all_matched_documents)}  "
            f"superseded={len(bundle.superseded_documents)}"
        )
        if not bundle.current_documents:
            print("    !! NO current documents resolved for this scenario !!")
        for kind, docs in sorted(bundle.current_documents.items()):
            doc_ids = ", ".join(d.parsed.doc_id for d in docs)
            flag = "  <-- multiple, unresolved tie" if len(docs) > 1 else ""
            print(f"    {kind:18s} -> {doc_ids}{flag}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--report", type=Path, default=None, help="Optional path to write a JSON debug report"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = run_ingestion(args.data_dir, args.cache_dir)
    print_summary(result)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(build_report(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Debug report written to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
