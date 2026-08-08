"""Round-trippable save/load for Block 2's output (dict[scenario_id, ScenarioFacts]).

Block 2 is the expensive, non-deterministic, real-money part of this
pipeline (dozens of LLM calls). Block 3 development needs to iterate on
linking/calculation logic against real extracted facts without re-running
all of Block 2 on every change — this is that cache. Distinct from
llm_client.py's per-call debug logs (which capture one call's raw
instructions/input/output for the Explanation layer) and from
scripts/run_extraction.py's `--report` (a flat summary dict, write-only) —
this one is specifically built to load back into real ScenarioFacts objects
via pydantic's model_validate, not just for human inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

from covenant_agent.models import ScenarioFacts
from covenant_agent.schemas import (
    AuditExtractionResult,
    CovenantExtractionResult,
    KycExtractionResult,
    OtherFactsExtractionResult,
)


def save_scenario_facts(facts_by_scenario: dict[str, ScenarioFacts], path: Path) -> None:
    payload = {}
    for scenario_id, facts in facts_by_scenario.items():
        payload[scenario_id] = {
            "covenants": facts.covenants.model_dump() if facts.covenants else None,
            "kyc": facts.kyc.model_dump() if facts.kyc else None,
            "audit_reports": [
                [doc_id, result.model_dump()] for doc_id, result in facts.audit_reports
            ],
            "other_facts": [
                [doc_id, result.model_dump()] for doc_id, result in facts.other_facts
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def status_path_for(facts_path: Path) -> Path:
    """The sidecar status file for a given facts-cache path — same name,
    `.status.json` appended, so `cache/scenario_facts.json` pairs with
    `cache/scenario_facts.json.status.json`.
    """
    return facts_path.with_name(facts_path.name + ".status.json")


def save_run_status(status_by_scenario: dict[str, str], path: Path) -> None:
    """`status_by_scenario[scenario_id]` is "ok" or an error message.

    Written alongside the facts cache after every scenario (not just at
    the end) — see extraction/pipeline.py and linking/pipeline.py — so a
    human resuming a partial run can see at a glance which scenario_ids
    still need a `--scenario` re-run, without grepping logs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status_by_scenario, indent=2, ensure_ascii=False), encoding="utf-8")


def load_scenario_facts(path: Path) -> dict[str, ScenarioFacts]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    facts_by_scenario: dict[str, ScenarioFacts] = {}
    for scenario_id, data in payload.items():
        facts_by_scenario[scenario_id] = ScenarioFacts(
            scenario_id=scenario_id,
            covenants=CovenantExtractionResult.model_validate(data["covenants"])
            if data["covenants"]
            else None,
            kyc=KycExtractionResult.model_validate(data["kyc"]) if data["kyc"] else None,
            audit_reports=tuple(
                (doc_id, AuditExtractionResult.model_validate(result))
                for doc_id, result in data["audit_reports"]
            ),
            other_facts=tuple(
                (doc_id, OtherFactsExtractionResult.model_validate(result))
                for doc_id, result in data["other_facts"]
            ),
        )
    return facts_by_scenario
