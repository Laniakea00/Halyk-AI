"""Block 3 orchestration: guarantee exactly one CovenantResult per required
(scenario, covenant_key) cell — the hard requirement driving this module's
whole shape: an empty submission cell scores 0 exactly like a wrong one, so
every failure mode upstream (missing credit agreement, a covenant key 2a
never returned, a calculation that raises) must still produce *something*
here, loudly logged, rather than propagate into a missing cell.
"""

from __future__ import annotations

import logging

from covenant_agent.calculation.evidence import find_evidence
from covenant_agent.calculation.formulas import InsufficientDataError
from covenant_agent.ingestion.template import required_covenant_keys
from covenant_agent.models import CovenantResult, IngestionResult, LinkedScenarioData, ScenarioFacts

logger = logging.getLogger(__name__)

# There is genuinely no information to compute from when this fires (no
# covenant rule extracted at all) — any guess is uninformed. COMPLIANT is
# the fallback status simply because, absent any signal, most covenants on
# most real loans are satisfied most of the time; it is a documented
# coin-flip default, not a considered guess, and it only ever fires when
# something upstream has already failed loudly (see the log line next to
# every use).
FALLBACK_STATUS = "COMPLIANT"
FALLBACK_ACTUAL = 0.0


def _fallback(scenario_id: str, covenant_key: str, reason: str) -> CovenantResult:
    logger.error(
        "Scenario %s covenant %s: FALLBACK (status=%s, actual=%s) — %s",
        scenario_id,
        covenant_key,
        FALLBACK_STATUS,
        FALLBACK_ACTUAL,
        reason,
    )
    return CovenantResult(
        covenant_key=covenant_key,
        status=FALLBACK_STATUS,
        actual=FALLBACK_ACTUAL,
        evidence_txn_id=None,
        used_fallback=True,
        fallback_reason=reason,
    )


def calculate_covenant(
    scenario_id: str,
    covenant_key: str,
    facts: ScenarioFacts | None,
    linked: LinkedScenarioData | None,
) -> CovenantResult:
    if facts is None:
        return _fallback(scenario_id, covenant_key, "no ScenarioFacts available for this scenario")
    if facts.covenants is None:
        return _fallback(
            scenario_id,
            covenant_key,
            "no covenant clauses extracted for this scenario (missing credit agreement or "
            "extraction failure)",
        )
    clause = next((c for c in facts.covenants.covenants if c.covenant_key == covenant_key), None)
    if clause is None:
        return _fallback(
            scenario_id, covenant_key, f"covenant key {covenant_key!r} was not returned by 2a extraction"
        )
    if linked is None:
        return _fallback(scenario_id, covenant_key, "no linked ledger data available for this scenario")

    if clause.carve_outs:
        # Idea 2 (offline audit): `carve_outs` is populated by 2a extraction
        # for real covenants across the public dataset (confirmed on
        # P2/P3/P4/P7/P9/P10) but has no reader anywhere in formulas.py/
        # evidence.py — every carve-out condition (springing-test
        # thresholds, FX conversion notes, materiality floors, KYC-based
        # affiliate scoping...) is silently dropped on the floor at
        # calculation time. This does not change `actual`/`status` — it
        # only makes the drop visible, so it stops being silent. Whether to
        # act on any given carve-out is a separate, larger decision (some
        # of these are the already-deferred springing-covenant/FX scope).
        logger.warning(
            "Scenario %s covenant %s: %d carve-out condition(s) extracted but not applied in "
            "calculation — %s",
            scenario_id,
            covenant_key,
            len(clause.carve_outs),
            clause.carve_outs,
        )

    try:
        result = find_evidence(clause, linked)
    except InsufficientDataError as exc:
        return _fallback(
            scenario_id,
            covenant_key,
            f"ratio denominator matched zero transactions — insufficient data to compute a "
            f"meaningful value ({exc})",
        )
    except Exception:
        logger.exception(
            "Scenario %s covenant %s: calculation raised unexpectedly", scenario_id, covenant_key
        )
        return _fallback(scenario_id, covenant_key, "calculation raised an unexpected exception — see logs")

    return CovenantResult(
        covenant_key=covenant_key,
        status=result.status,
        actual=result.actual,
        evidence_txn_id=result.evidence_txn_id,
        used_fallback=False,
        fallback_reason=None,
        calculation_notes=result.notes,
    )


def calculate_scenario(
    scenario_id: str,
    covenant_keys: list[str],
    facts: ScenarioFacts | None,
    linked: LinkedScenarioData | None,
) -> dict[str, CovenantResult]:
    return {
        key: calculate_covenant(scenario_id, key, facts, linked)
        for key in covenant_keys
    }


def calculate_all(
    ingestion: IngestionResult,
    facts_by_scenario: dict[str, ScenarioFacts],
    linked_by_scenario: dict[str, LinkedScenarioData],
) -> dict[str, dict[str, CovenantResult]]:
    results: dict[str, dict[str, CovenantResult]] = {}
    for scenario_id in ingestion.scenarios:
        keys = required_covenant_keys(ingestion.template, scenario_id)
        results[scenario_id] = calculate_scenario(
            scenario_id,
            keys,
            facts_by_scenario.get(scenario_id),
            linked_by_scenario.get(scenario_id),
        )
    return results
