"""Assemble the final submission JSON — the exact shape submission_template.json
requires (top-level team/contact_email/model, plus answers) — from
run_calculation.py's plain results output.

Pure, offline, no LLM calls: cross-checks against the dataset's own
required scenario_ids/covenant_keys (via ingestion/template.py) so a
missing or extra cell is always visible to the caller, never silently
dropped. A missing submission cell scores exactly like a wrong one (see
template.py's own docstring), so this never pretends a partial result is
complete — it fills gaps with null and hands the caller an explicit list
of what's missing instead.
"""

from __future__ import annotations

from covenant_agent.ingestion.template import required_covenant_keys, required_scenario_ids

_REQUIRED_CELL_FIELDS = ("status", "actual", "evidence_txn_id")


def build_submission(
    template: dict,
    results: dict,
    *,
    team: str,
    contact_email: str,
    model: str,
) -> tuple[dict, list[str], list[str]]:
    """Returns (submission_dict, missing_cells, extra_cells).

    `submission_dict` always has the full shape the template requires —
    one entry per required scenario_id/covenant_key, even if `results`
    didn't have it (filled with null in that case).

    `missing_cells` (list of "scenario_id.covenant_key" strings): required
    cells `results` had no answer for. The caller MUST treat a non-empty
    list here as "not ready to submit" — these will score 0.

    `extra_cells`: cells present in `results` but not required by
    `template` (dropped from the output) — most likely a stale or
    mismatched results file, surfaced rather than silently ignored.
    """
    scenario_ids = required_scenario_ids(template)
    required_by_scenario = {sid: required_covenant_keys(template, sid) for sid in scenario_ids}

    answers: dict[str, dict[str, dict]] = {}
    missing_cells: list[str] = []
    for scenario_id, covenant_keys in required_by_scenario.items():
        answers[scenario_id] = {}
        submitted_scenario = results.get(scenario_id, {})
        for covenant_key in covenant_keys:
            cell = submitted_scenario.get(covenant_key)
            if cell is None:
                missing_cells.append(f"{scenario_id}.{covenant_key}")
                answers[scenario_id][covenant_key] = {field: None for field in _REQUIRED_CELL_FIELDS}
            else:
                answers[scenario_id][covenant_key] = {field: cell.get(field) for field in _REQUIRED_CELL_FIELDS}

    required_keys_by_scenario = {sid: set(keys) for sid, keys in required_by_scenario.items()}
    extra_cells = [
        f"{scenario_id}.{covenant_key}"
        for scenario_id, covenants in results.items()
        for covenant_key in covenants
        if covenant_key not in required_keys_by_scenario.get(scenario_id, set())
    ]

    submission = {
        "team": team,
        "contact_email": contact_email,
        "model": model,
        "answers": answers,
    }
    return submission, missing_cells, extra_cells
