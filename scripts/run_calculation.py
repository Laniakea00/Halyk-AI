#!/usr/bin/env python3
"""CLI entrypoint for Block 3 (Linking + Decision/Calculation).

Usage:
    python scripts/run_calculation.py --facts-cache cache/scenario_facts.json \\
        --ground-truth agentic-bank-public/ground_truth.json -v

Runs Block 1 (ingestion) -> Block 2 (extraction, or loaded from
--facts-cache) -> Block 3a (linking) -> Block 3b/3c (calculation), prints
every scenario's full chain (facts -> actual/status -> evidence), and, if
--ground-truth is given (only meaningful against the public dataset),
self-scores the result using the case's own scoring formula.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from covenant_agent.calculation.pipeline import calculate_all, calculate_scenario  # noqa: E402
from covenant_agent.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR  # noqa: E402
from covenant_agent.extraction.cache import load_scenario_facts  # noqa: E402
from covenant_agent.extraction.pipeline import extract_all_facts, extract_scenario_facts  # noqa: E402
from covenant_agent.ingestion.template import required_covenant_keys  # noqa: E402
from covenant_agent.linking.pipeline import link_all_scenarios, link_scenario  # noqa: E402
from covenant_agent.llm_client import print_usage_summary  # noqa: E402
from covenant_agent.models import CovenantResult  # noqa: E402
from covenant_agent.resolution.pipeline import run_ingestion  # noqa: E402
from covenant_agent.scoring import score_submission  # noqa: E402


def print_results(results: dict[str, dict[str, CovenantResult]]) -> None:
    for scenario_id in sorted(results):
        print(f"\n[{scenario_id}]")
        for covenant_key in sorted(results[scenario_id]):
            r = results[scenario_id][covenant_key]
            flag = "  [FALLBACK: " + r.fallback_reason + "]" if r.used_fallback else ""
            # Structural-surprises audit: metric_type="other" is the
            # generic best-effort catch-all (see calculation/pipeline.py),
            # not a truly general handler — flagged distinctly from a
            # normal fallback so it's visible even skimming the console.
            if r.metric_type == "other":
                flag += "  [OTHER-TYPE: generic catch-all, verify manually]"
            print(
                f"  {covenant_key}: status={r.status:10s} actual={r.actual:>14.2f}  "
                f"evidence={r.evidence_txn_id}{flag}"
            )
            for note in r.calculation_notes:
                print(f"        note: {note}")


def _results_to_plain(results: dict[str, dict[str, CovenantResult]]) -> dict:
    return {
        sid: {
            key: {
                "status": r.status,
                "actual": r.actual,
                "evidence_txn_id": r.evidence_txn_id,
                "metric_type": r.metric_type,
            }
            for key, r in covenants.items()
        }
        for sid, covenants in results.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--facts-cache", type=Path, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground_truth.json (public dataset only) — self-scores the run",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write plain results JSON here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ingestion = run_ingestion(args.data_dir, args.cache_dir)
    log_dir = args.cache_dir / "llm_logs"
    extraction_status: dict[str, str] | None = None
    linking_status: dict[str, str] | None = None

    if args.facts_cache and args.facts_cache.exists():
        all_facts = load_scenario_facts(args.facts_cache)
    elif args.scenario:
        # Single-scenario mode: let a failure here propagate and crash with
        # a full traceback — this is the debug/iteration path, not the
        # batch path finding #1/#2 in the code review were about.
        if args.scenario not in ingestion.scenarios:
            print(f"Unknown scenario_id {args.scenario!r}. Known: {sorted(ingestion.scenarios)}")
            return 1
        bundle = ingestion.scenarios[args.scenario]
        keys = required_covenant_keys(ingestion.template, args.scenario)
        all_facts = {args.scenario: extract_scenario_facts(bundle, keys, log_dir=log_dir)}
    else:
        # save_path -> written incrementally, once per scenario, not once
        # at the end — see extraction/pipeline.py's docstring.
        all_facts, extraction_status = extract_all_facts(
            ingestion, log_dir=log_dir, save_path=args.facts_cache
        )

    if args.scenario:
        if args.scenario not in ingestion.scenarios:
            print(f"Unknown scenario_id {args.scenario!r}. Known: {sorted(ingestion.scenarios)}")
            return 1
        bundle = ingestion.scenarios[args.scenario]
        linked = {
            args.scenario: link_scenario(
                args.scenario, bundle.account_id, ingestion.ledger, all_facts[args.scenario], log_dir=log_dir
            )
        }
        keys = required_covenant_keys(ingestion.template, args.scenario)
        results = {args.scenario: calculate_scenario(args.scenario, keys, all_facts[args.scenario], linked[args.scenario])}
    else:
        linked, linking_status = link_all_scenarios(ingestion, all_facts, log_dir=log_dir)
        results = calculate_all(ingestion, all_facts, linked)

    print_results(results)

    for label, status in (("Extraction", extraction_status), ("Linking", linking_status)):
        if not status:
            continue
        ok = [sid for sid, s in status.items() if s == "ok"]
        failed = {sid: s for sid, s in status.items() if s != "ok"}
        print(f"\n{label} status: {len(ok)}/{len(status)} scenario(s) ok.")
        if failed:
            print(f"  FAILED ({len(failed)}): {list(failed)} — re-run with --scenario <id>")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(_results_to_plain(results), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nReport written to {args.report}")

    if args.ground_truth:
        ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
        mean_score, cell_scores = score_submission(_results_to_plain(results), ground_truth)
        print(f"\n{'=' * 70}\nSelf-score (public dataset only): {mean_score:.4f} / 1.0000")
        wrong_status = [s for s in cell_scores if not s.status_correct]
        if wrong_status:
            print(f"Wrong status ({len(wrong_status)}):")
            for s in wrong_status:
                print(f"  {s.scenario_id} {s.covenant_key}")
        partial = [
            s for s in cell_scores if s.status_correct and s.total < 0.999 and s not in wrong_status
        ]
        if partial:
            print(f"Correct status but imperfect actual/evidence ({len(partial)}):")
            for s in partial:
                print(
                    f"  {s.scenario_id} {s.covenant_key}: total={s.total:.3f} "
                    f"(actual_score={s.actual_score:.3f}, evidence_score={s.evidence_score:.3f}, "
                    f"rel_err={s.relative_error})"
                )

    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
