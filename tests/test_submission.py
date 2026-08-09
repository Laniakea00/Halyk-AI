"""Offline tests for covenant_agent/submission.py's build_submission — the
pure assembly logic behind scripts/build_submission.py. No API calls.
"""

from __future__ import annotations

import unittest

from covenant_agent.submission import build_submission

_TEMPLATE = {
    "team": "",
    "contact_email": "",
    "model": "",
    "answers": {
        "P1": {
            "6.1": {"status": None, "actual": None, "evidence_txn_id": None},
            "6.2": {"status": None, "actual": None, "evidence_txn_id": None},
        },
        "P2": {"6.1": {"status": None, "actual": None, "evidence_txn_id": None}},
    },
}


class BuildSubmissionTest(unittest.TestCase):
    def test_complete_results_produce_no_missing_or_extra_cells(self) -> None:
        results = {
            "P1": {
                "6.1": {"status": "BREACH", "actual": 1.5, "evidence_txn_id": "TXN-P1-0001"},
                "6.2": {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None},
            },
            "P2": {"6.1": {"status": "COMPLIANT", "actual": 100.0, "evidence_txn_id": None}},
        }
        submission, missing, extra = build_submission(
            _TEMPLATE, results, team="Team", contact_email="a@b.com", model="gpt-5.4"
        )
        self.assertEqual(missing, [])
        self.assertEqual(extra, [])
        self.assertEqual(submission["team"], "Team")
        self.assertEqual(submission["contact_email"], "a@b.com")
        self.assertEqual(submission["model"], "gpt-5.4")
        self.assertEqual(submission["answers"]["P1"]["6.1"]["status"], "BREACH")
        self.assertEqual(submission["answers"]["P1"]["6.1"]["actual"], 1.5)
        self.assertEqual(submission["answers"]["P1"]["6.1"]["evidence_txn_id"], "TXN-P1-0001")

    def test_missing_cell_is_named_and_filled_with_null(self) -> None:
        results = {
            "P1": {"6.1": {"status": "BREACH", "actual": 1.5, "evidence_txn_id": None}},
            # P1 6.2 and all of P2 are absent from results entirely.
        }
        submission, missing, extra = build_submission(
            _TEMPLATE, results, team="Team", contact_email="a@b.com", model="gpt-5.4"
        )
        self.assertEqual(sorted(missing), ["P1.6.2", "P2.6.1"])
        self.assertEqual(submission["answers"]["P1"]["6.2"], {"status": None, "actual": None, "evidence_txn_id": None})
        self.assertEqual(submission["answers"]["P2"]["6.1"], {"status": None, "actual": None, "evidence_txn_id": None})
        # The one real cell must still come through untouched.
        self.assertEqual(submission["answers"]["P1"]["6.1"]["status"], "BREACH")

    def test_extra_cell_not_required_by_template_is_named_and_dropped(self) -> None:
        results = {
            "P1": {
                "6.1": {"status": "BREACH", "actual": 1.5, "evidence_txn_id": None},
                "6.2": {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None},
                "6.3": {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None},  # not in template
            },
            "P2": {"6.1": {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None}},
            "P99": {"6.1": {"status": "BREACH", "actual": 0.0, "evidence_txn_id": None}},  # unknown scenario
        }
        submission, missing, extra = build_submission(
            _TEMPLATE, results, team="Team", contact_email="a@b.com", model="gpt-5.4"
        )
        self.assertEqual(missing, [])
        self.assertEqual(sorted(extra), ["P1.6.3", "P99.6.1"])
        self.assertNotIn("6.3", submission["answers"]["P1"])
        self.assertNotIn("P99", submission["answers"])

    def test_extra_field_on_a_cell_is_not_carried_through(self) -> None:
        # Only the three required fields survive, even if results has more.
        results = {
            "P1": {
                "6.1": {
                    "status": "BREACH",
                    "actual": 1.5,
                    "evidence_txn_id": None,
                    "used_fallback": True,
                    "notes": "some internal debugging field",
                }
            },
            "P2": {"6.1": {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None}},
        }
        submission, _missing, _extra = build_submission(
            _TEMPLATE, results, team="Team", contact_email="a@b.com", model="gpt-5.4"
        )
        self.assertEqual(set(submission["answers"]["P1"]["6.1"].keys()), {"status", "actual", "evidence_txn_id"})

    def test_output_shape_matches_every_required_scenario_and_covenant(self) -> None:
        submission, _missing, _extra = build_submission(
            _TEMPLATE, {}, team="Team", contact_email="a@b.com", model="gpt-5.4"
        )
        self.assertEqual(set(submission["answers"].keys()), {"P1", "P2"})
        self.assertEqual(set(submission["answers"]["P1"].keys()), {"6.1", "6.2"})
        self.assertEqual(set(submission["answers"]["P2"].keys()), {"6.1"})


if __name__ == "__main__":
    unittest.main()
