#!/usr/bin/env python3
"""Assemble the final submission file from run_calculation.py's --report output.

Usage:
    python scripts/run_calculation.py --report cache/calculation_results.json ...
    python scripts/build_submission.py \\
        --results cache/calculation_results.json \\
        --team "Our Team" --contact-email "team@example.com" --model "gpt-5.4 / gpt-5.4-mini" \\
        --output submission.json

Wraps run_calculation.py's plain {scenario_id: {covenant_key: {...}}}
results into the exact structure submission_template.json requires
(top-level team/contact_email/model/answers), cross-checked against the
dataset's own template so every required cell is guaranteed present in
the output. Any required cell --results didn't have an answer for is
printed loudly and filled with null (a missing cell scores exactly like a
wrong one — never silently drop one). The written file is re-loaded
through the same structural validation the rest of the pipeline trusts
before this script reports success, so a bug in the assembly itself is
caught here, not at submission time.

Offline — reads --results and --template from disk, makes no API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from covenant_agent.config import DEFAULT_DATA_DIR, TEMPLATE_FILENAME  # noqa: E402
from covenant_agent.ingestion.template import load_template  # noqa: E402
from covenant_agent.submission import build_submission  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True, help="run_calculation.py --report output")
    parser.add_argument("--template", type=Path, default=DEFAULT_DATA_DIR / TEMPLATE_FILENAME)
    parser.add_argument("--team", type=str, required=True)
    parser.add_argument("--contact-email", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template = load_template(args.template)
    with args.results.open(encoding="utf-8") as f:
        results = json.load(f)

    submission, missing_cells, extra_cells = build_submission(
        template, results, team=args.team, contact_email=args.contact_email, model=args.model
    )

    if extra_cells:
        print(f"WARNING: {len(extra_cells)} cell(s) in --results are not required by the template (dropped):")
        for cell in extra_cells:
            print(f"  {cell}")
    if missing_cells:
        print(f"WARNING: {len(missing_cells)} required cell(s) missing from --results, filled with null:")
        for cell in missing_cells:
            print(f"  {cell}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, indent=2, ensure_ascii=False), encoding="utf-8")

    # Round-trip validation: catches a bug in this script's own assembly
    # logic here, not at submission time.
    load_template(args.output)

    total_cells = sum(len(covenants) for covenants in submission["answers"].values())
    print(
        f"\nWrote {args.output} — {len(submission['answers'])} scenario(s), {total_cells} cell(s) total. "
        f"Structurally valid (re-loaded and checked)."
    )
    if missing_cells:
        print(
            f"\nNOTE: {len(missing_cells)} cell(s) above are null — these score 0, exactly like a wrong "
            f"answer. Re-run calculation for the missing scenario(s)/covenant(s) before submitting."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
