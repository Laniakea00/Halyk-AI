"""Load submission_template.json — the authoritative list of what we must answer.

This is the single source of truth for *which* scenario_ids and covenant
keys exist. We never hardcode "P1..P10, B1, B4" anywhere else in the
codebase; everything downstream asks this module instead, so pointing the
pipeline at the private dataset's template on Aug 9 is enough to retarget
the whole thing.

That "everything downstream trusts this file's shape" property is exactly
why it's validated up front, loudly, before any other work happens: without
this check, a template whose top-level structure differs even slightly
(confirmed as a real risk in code review, not hypothetical) makes
`required_scenario_ids()` silently return `[]` via a bare `.get("answers",
{})`, and the entire pipeline quietly processes zero scenarios — no
exception anywhere, discovered only when looking at an empty final result,
with the least possible time left to react. Failing here, immediately,
with a message naming exactly what's missing, is strictly better than
failing (or not failing at all) three blocks later.
"""

from __future__ import annotations

import json
from pathlib import Path

_REQUIRED_CELL_FIELDS = {"status", "actual", "evidence_txn_id"}


class TemplateValidationError(RuntimeError):
    """Raised when submission_template.json doesn't have the shape every
    downstream layer assumes. Meant to fail fast, at ingestion time, not
    silently degrade into "zero scenarios processed" three blocks later.
    """


def load_template(template_path: Path) -> dict:
    with template_path.open(encoding="utf-8") as f:
        raw_text = f.read()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Deliberately NOT a fallback-to-{} here — that would reintroduce
        # exactly the failure mode this module exists to prevent (see
        # module docstring): required_scenario_ids() silently returning
        # [] and the whole pipeline quietly processing zero scenarios.
        # Malformed JSON gets the same loud, immediate, precisely-located
        # failure as a malformed *structure* does — just with a clearer
        # message than the bare traceback json.JSONDecodeError gives on
        # its own (line/column instead of a byte offset buried in repr).
        raise TemplateValidationError(
            f"{template_path}: not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    _validate_template_structure(data, template_path)
    return data


def _validate_template_structure(data: object, source: Path) -> None:
    if not isinstance(data, dict):
        raise TemplateValidationError(
            f"{source}: expected a JSON object at the top level, got {type(data).__name__}"
        )

    answers = data.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise TemplateValidationError(
            f"{source}: top-level key 'answers' must be a non-empty object mapping "
            f"scenario_id -> covenant cells (found: {type(answers).__name__ if answers is not None else 'missing'})"
        )

    for scenario_id, covenants in answers.items():
        if not isinstance(covenants, dict) or not covenants:
            raise TemplateValidationError(
                f"{source}: answers[{scenario_id!r}] must be a non-empty object mapping "
                f"covenant_key -> cell"
            )
        for covenant_key, cell in covenants.items():
            if not isinstance(cell, dict):
                raise TemplateValidationError(
                    f"{source}: answers[{scenario_id!r}][{covenant_key!r}] must be an object, "
                    f"got {type(cell).__name__}"
                )
            missing = _REQUIRED_CELL_FIELDS - cell.keys()
            if missing:
                raise TemplateValidationError(
                    f"{source}: answers[{scenario_id!r}][{covenant_key!r}] is missing "
                    f"required field(s) {sorted(missing)} — expected exactly "
                    f"{sorted(_REQUIRED_CELL_FIELDS)}"
                )


def required_scenario_ids(template: dict) -> list[str]:
    return list(template.get("answers", {}).keys())


def required_covenant_keys(template: dict, scenario_id: str) -> list[str]:
    return list(template.get("answers", {}).get(scenario_id, {}).keys())
