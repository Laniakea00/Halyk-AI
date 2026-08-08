#!/usr/bin/env python3
"""CLI entrypoint for Block 3a (Linking).

Usage:
    python scripts/run_linking.py [--scenario B1] [--facts-cache cache/scenario_facts.json] [-v]

Runs Block 1 (ingestion), then Block 2 (extraction) unless --facts-cache
already exists (in which case it's loaded instead of calling the LLM
again — see extraction/cache.py), then Block 3a (linking), and prints a
per-scenario summary: transaction category counts, linked/unmatched
reclassifications, and resolved related parties.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from covenant_agent.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR  # noqa: E402
from covenant_agent.extraction.cache import load_scenario_facts  # noqa: E402
from covenant_agent.extraction.pipeline import extract_all_facts  # noqa: E402
from covenant_agent.linking.pipeline import link_all_scenarios, link_scenario  # noqa: E402
from covenant_agent.models import LinkedScenarioData  # noqa: E402
from covenant_agent.resolution.pipeline import run_ingestion  # noqa: E402


def print_linking_summary(linked: LinkedScenarioData) -> None:
    print(f"\n[{linked.scenario_id}] {len(linked.transactions)} transactions")

    counts = Counter(linked.txn_category.values())
    if counts:
        print("  category counts:")
        for category, n in sorted(counts.items()):
            print(f"    {category}: {n}")

    if linked.reclassifications:
        print("  reclassifications linked:")
        for txn_id, r in linked.reclassifications.items():
            flag = "  [AMBIGUOUS]" if r.was_ambiguous else ""
            print(
                f"    {txn_id}: {r.original_category!r} -> {r.reclassified_category!r} "
                f"(source={r.source_doc_id}, confidence={r.match_confidence:.2f}){flag}"
            )
    if linked.unmatched_reclassifications:
        print("  UNMATCHED reclassifications:")
        for u in linked.unmatched_reclassifications:
            print(f"    {u.counterparty_name!r} ${u.amount}: {u.reason}")

    if linked.related_parties:
        print("  related parties:")
        for counterparty, m in linked.related_parties.items():
            verdict = "RELATED" if m.is_related else "not related"
            print(f"    {counterparty!r} (KYC: {m.kyc_name!r}) -> {verdict}  [{m.basis}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--facts-cache",
        type=Path,
        default=None,
        help="Load/save ScenarioFacts here instead of always calling the LLM",
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ingestion = run_ingestion(args.data_dir, args.cache_dir)
    log_dir = args.cache_dir / "llm_logs"

    if args.facts_cache and args.facts_cache.exists():
        all_facts = load_scenario_facts(args.facts_cache)
    else:
        # save_path -> written incrementally, once per scenario — see
        # extraction/pipeline.py's docstring.
        all_facts, _extraction_status = extract_all_facts(
            ingestion, log_dir=log_dir, save_path=args.facts_cache
        )

    if args.scenario:
        if args.scenario not in ingestion.scenarios:
            print(f"Unknown scenario_id {args.scenario!r}. Known: {sorted(ingestion.scenarios)}")
            return 1
        bundle = ingestion.scenarios[args.scenario]
        linked = link_scenario(
            args.scenario, bundle.account_id, ingestion.ledger, all_facts[args.scenario], log_dir=log_dir
        )
        print_linking_summary(linked)
        return 0

    linked_by_scenario, status = link_all_scenarios(ingestion, all_facts, log_dir=log_dir)
    for scenario_id in sorted(linked_by_scenario):
        print_linking_summary(linked_by_scenario[scenario_id])

    ok = [sid for sid, s in status.items() if s == "ok"]
    failed = {sid: s for sid, s in status.items() if s != "ok"}
    print(f"\nLinking status: {len(ok)}/{len(status)} scenario(s) ok.")
    if failed:
        print(f"  FAILED ({len(failed)}): {list(failed)} — re-run with --scenario <id>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
