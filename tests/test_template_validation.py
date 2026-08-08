"""Offline tests for submission_template.json structural validation.

Covers the code-review finding: `.get("answers", {})` used to silently
return an empty scenario list for any malformed template instead of
failing loudly at load time.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from covenant_agent.ingestion.template import (
    TemplateValidationError,
    load_template,
    required_covenant_keys,
    required_scenario_ids,
)

VALID_TEMPLATE = {
    "team": "",
    "contact_email": "",
    "model": "",
    "answers": {
        "P1": {
            "6.1": {"status": None, "actual": None, "evidence_txn_id": None},
            "6.2": {"status": None, "actual": None, "evidence_txn_id": None},
        }
    },
}


def _write(tmpdir: str, data: object) -> Path:
    path = Path(tmpdir) / "submission_template.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class LoadTemplateValidationTest(unittest.TestCase):
    def test_valid_template_loads_and_derives_correctly(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _write(tmp, VALID_TEMPLATE)
            template = load_template(path)
        self.assertEqual(required_scenario_ids(template), ["P1"])
        self.assertEqual(required_covenant_keys(template, "P1"), ["6.1", "6.2"])

    def test_top_level_not_an_object_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _write(tmp, ["not", "a", "dict"])
            with self.assertRaises(TemplateValidationError):
                load_template(path)

    def test_missing_answers_key_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _write(tmp, {"team": "", "contact_email": "", "model": ""})
            with self.assertRaises(TemplateValidationError):
                load_template(path)

    def test_empty_answers_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _write(tmp, {"answers": {}})
            with self.assertRaises(TemplateValidationError):
                load_template(path)

    def test_scenario_with_empty_covenants_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _write(tmp, {"answers": {"P1": {}}})
            with self.assertRaises(TemplateValidationError):
                load_template(path)

    def test_cell_missing_required_field_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            data = {"answers": {"P1": {"6.1": {"status": None, "actual": None}}}}  # missing evidence_txn_id
            path = _write(tmp, data)
            with self.assertRaises(TemplateValidationError):
                load_template(path)

    def test_cell_not_an_object_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            data = {"answers": {"P1": {"6.1": "not a dict"}}}
            path = _write(tmp, data)
            with self.assertRaises(TemplateValidationError):
                load_template(path)

    def test_error_message_names_the_missing_piece(self) -> None:
        with TemporaryDirectory() as tmp:
            path = _write(tmp, {"team": ""})
            try:
                load_template(path)
                self.fail("expected TemplateValidationError")
            except TemplateValidationError as exc:
                self.assertIn("answers", str(exc))

    def test_malformed_json_raises_template_validation_error_not_json_decode_error(self) -> None:
        # Red-team finding M4: json.load() wasn't wrapped at all — a
        # truncated/malformed submission_template.json crashed with a bare
        # JSONDecodeError instead of this module's own, more diagnosable
        # error type. Deliberately NOT falling back to an empty dict/{} —
        # that would reintroduce the exact silent-zero-scenarios failure
        # this module exists to prevent (see module docstring).
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission_template.json"
            path.write_text('{"answers": {"P1": ', encoding="utf-8")  # truncated
            with self.assertRaises(TemplateValidationError) as cm:
                load_template(path)
            self.assertIn("not valid JSON", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
