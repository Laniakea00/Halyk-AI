"""Self-scorer implementing the case's own scoring formula (CASE.ru.md §4).

Only usable against the public dataset, which ships `ground_truth.json` —
the private dataset on the day will not. Its value is entirely as a
pre-submission sanity check: knowing our approximate score on data we can
verify, before ever touching data we can't.

    status   0.50 — exact match required; wrong status zeroes the whole cell
    actual   0.30 — decays linearly with relative error, 0 at >=5% off
    evidence 0.20 — exact txn_id match if the key has one; if the key's
                    evidence is null, these 0.20 decay on the *same* scale
                    as actual instead of being free — see CASE.ru.md's own
                    wording: "эти 0.20 не начисляются автоматически, а
                    зарабатываются: они убывают вместе с actual".
"""

from __future__ import annotations

from dataclasses import dataclass

RELATIVE_ERROR_ZERO_POINT = 0.05


@dataclass(frozen=True)
class CellScore:
    scenario_id: str
    covenant_key: str
    total: float  # 0..1
    status_score: float
    actual_score: float
    evidence_score: float
    status_correct: bool
    relative_error: float | None  # None if actual was missing/non-numeric


def _relative_error(submitted_actual, key_actual: float) -> float | None:
    if submitted_actual is None or isinstance(submitted_actual, bool) or not isinstance(
        submitted_actual, (int, float)
    ):
        return None
    if key_actual == 0:
        return 0.0 if submitted_actual == 0 else 1.0
    return abs(submitted_actual - key_actual) / abs(key_actual)


def score_cell(
    scenario_id: str,
    covenant_key: str,
    key: dict,
    submitted: dict,
) -> CellScore:
    status_correct = submitted.get("status") == key["status"]
    if not status_correct:
        return CellScore(
            scenario_id=scenario_id,
            covenant_key=covenant_key,
            total=0.0,
            status_score=0.0,
            actual_score=0.0,
            evidence_score=0.0,
            status_correct=False,
            relative_error=None,
        )

    status_score = 0.50
    rel_err = _relative_error(submitted.get("actual"), key["actual"])
    actual_score = (
        0.30 * max(0.0, 1 - rel_err / RELATIVE_ERROR_ZERO_POINT) if rel_err is not None else 0.0
    )

    key_evidence = key.get("evidence_txn_id")
    if key_evidence is None:
        evidence_score = (
            0.20 * max(0.0, 1 - rel_err / RELATIVE_ERROR_ZERO_POINT) if rel_err is not None else 0.0
        )
    else:
        evidence_score = 0.20 if submitted.get("evidence_txn_id") == key_evidence else 0.0

    return CellScore(
        scenario_id=scenario_id,
        covenant_key=covenant_key,
        total=status_score + actual_score + evidence_score,
        status_score=status_score,
        actual_score=actual_score,
        evidence_score=evidence_score,
        status_correct=True,
        relative_error=rel_err,
    )


def score_submission(
    results: dict[str, dict[str, dict]], ground_truth: dict
) -> tuple[float, list[CellScore]]:
    """`results[scenario_id][covenant_key] = {"status", "actual", "evidence_txn_id"}`.

    Returns (mean score across every scored cell, per-cell breakdown).
    Cells present in ground_truth but missing from `results` score 0 — the
    same as the real grader treats a missing submission cell.
    """
    scores: list[CellScore] = []
    for scenario_id, scenario_gt in ground_truth["scenarios"].items():
        submitted_scenario = results.get(scenario_id, {})
        for covenant_key, key in scenario_gt["covenants"].items():
            submitted = submitted_scenario.get(covenant_key, {})
            scores.append(score_cell(scenario_id, covenant_key, key, submitted))

    mean_score = sum(s.total for s in scores) / len(scores) if scores else 0.0
    return mean_score, scores
