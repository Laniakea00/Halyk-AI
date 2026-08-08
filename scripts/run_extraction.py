#!/usr/bin/env python3
"""CLI entrypoint for Block 2 (Covenant clause extraction + Fact extraction).

Runs Block 1 (ingestion/resolution) first, then Block 2 on top of it.
Requires OPENAI_API_KEY (env var or .env file in the repo root).

Usage:
    python scripts/run_extraction.py [--scenario P1] [--data-dir DIR] [--report PATH] [-v]

--scenario limits the run to one scenario_id — useful for fast iteration
and for spot-checking a specific borrower's covenant formulas without
paying for (and waiting on) all 12.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from covenant_agent.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR  # noqa: E402
from covenant_agent.extraction.cache import load_scenario_facts, save_scenario_facts  # noqa: E402
from covenant_agent.extraction.pipeline import extract_all_facts, extract_scenario_facts  # noqa: E402
from covenant_agent.ingestion.template import required_covenant_keys  # noqa: E402
from covenant_agent.llm_client import print_usage_summary  # noqa: E402
from covenant_agent.models import ScenarioFacts  # noqa: E402
from covenant_agent.resolution.pipeline import run_ingestion  # noqa: E402


def print_summary(facts: ScenarioFacts) -> None:
    print(f"\n[{facts.scenario_id}]")
    if facts.covenants is None:
        print("  covenants: NONE EXTRACTED (missing credit agreement or call failed)")
    else:
        for c in facts.covenants.covenants:
            print(
                f"  {c.covenant_key}  {c.metric_name!r}  "
                f"[{c.metric_type}] {c.direction} {c.threshold_value} {c.threshold_unit}"
            )
            print(f"        formula: {c.formula_description}")
            if c.carve_outs:
                print(f"        carve_outs: {c.carve_outs}")
            if c.aggregation_note:
                print(f"        aggregation_note: {c.aggregation_note}")

    if facts.kyc is None:
        print("  kyc: none found")
    else:
        thr = facts.kyc.related_party_threshold_pct
        print(f"  kyc: threshold={thr}%  disclosures={len(facts.kyc.disclosures)}")
        for d in facts.kyc.disclosures:
            flag = " [LABELED RELATED PARTY]" if d.explicitly_labeled_related_party else ""
            print(f"        {d.counterparty_name}: {d.ownership_or_voting_pct}%{flag}")

    for doc_id, audit in facts.audit_reports:
        print(f"  audit_report ({doc_id}, final={audit.is_final_position}):")
        for r in audit.reclassifications:
            print(
                f"        {r.counterparty_name} ${r.amount}: "
                f"{r.original_category!r} -> {r.reclassified_category!r}"
            )

    for doc_id, other in facts.other_facts:
        if other.facts:
            print(f"  other_facts ({doc_id}): {len(other.facts)} fact(s)")
            for f in other.facts:
                print(f"        {f.fact_description}: {f.value} {f.unit}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--scenario", type=str, default=None, help="Limit to one scenario_id")
    parser.add_argument(
        "--report", type=Path, default=None, help="Path to write reloadable JSON output"
    )
    parser.add_argument(
        "--load-cache",
        type=Path,
        default=None,
        help="Load ScenarioFacts from a previous --report file instead of calling the LLM again",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ingestion = run_ingestion(args.data_dir, args.cache_dir)
    log_dir = args.cache_dir / "llm_logs"
    status: dict[str, str] | None = None

    if args.load_cache:
        all_facts = load_scenario_facts(args.load_cache)
        if args.scenario:
            all_facts = {args.scenario: all_facts[args.scenario]}
    elif args.scenario:
        if args.scenario not in ingestion.scenarios:
            print(f"Unknown scenario_id {args.scenario!r}. Known: {sorted(ingestion.scenarios)}")
            return 1
        bundle = ingestion.scenarios[args.scenario]
        keys = required_covenant_keys(ingestion.template, args.scenario)
        all_facts = {args.scenario: extract_scenario_facts(bundle, keys, log_dir=log_dir)}
    else:
        # save_path is passed straight through so each scenario's result is
        # written to disk as soon as it's done, not once at the very end —
        # see extraction/pipeline.py's docstring for why that matters.
        all_facts, status = extract_all_facts(ingestion, log_dir=log_dir, save_path=args.report)

    for facts in all_facts.values():
        print_summary(facts)

    if args.report and status is None:
        # Single-scenario or --load-cache path: extract_all_facts didn't
        # already write this for us.
        save_scenario_facts(all_facts, args.report)
        print(f"\nReport written to {args.report}")
    elif args.report:
        print(f"\nReport (written incrementally throughout the run) at {args.report}")

    if status:
        ok = [sid for sid, s in status.items() if s == "ok"]
        failed = {sid: s for sid, s in status.items() if s != "ok"}
        print(f"\nStatus: {len(ok)}/{len(status)} scenario(s) ok.")
        if failed:
            print(f"FAILED ({len(failed)}): {list(failed)}")
            for sid, reason in failed.items():
                print(f"  {sid}: {reason}")
            print("Re-run just these with: --scenario <id>")

    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
