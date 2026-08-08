"""Offline tests for llm_client.py's usage accounting — no real API call
(client.responses.parse is mocked).

Added 2026-08-08 alongside extending token tracking to every model, not
just the budget-constrained top tier — see llm_client.py's
_tokens_used_by_model docstring for why: a single evening's mini-tier
usage alone was comparable to a colleague's entire week of "normal" usage,
and there was no automated visibility into that total at all until it was
manually reconstructed from cache/llm_logs after the fact.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

import covenant_agent.llm_client as llm_client
from covenant_agent.llm_client import ModelConfig, extract_structured, log_usage_summary, usage_summary


class _FakeFact(BaseModel):
    value: int = 0


def _fake_response(model: str, total_tokens: int) -> MagicMock:
    response = MagicMock()
    response.id = "resp_1"
    response.model = model
    response.status = "completed"
    response.incomplete_details = None
    response.usage = MagicMock()
    response.usage.model_dump.return_value = {"total_tokens": total_tokens}
    response.usage.total_tokens = total_tokens
    response.output_text = "{}"
    parsed = _FakeFact()
    response.output_parsed = parsed
    return response


class UsageAccountingTest(unittest.TestCase):
    def setUp(self) -> None:
        # Module-level counters are process-global by design (see
        # llm_client.py) — reset between tests so they don't leak.
        llm_client._tokens_used_by_model.clear()
        llm_client._calls_by_model.clear()
        llm_client._top_tier_tokens_used = 0

    def _call(self, model: str, tokens: int) -> None:
        fake_client = MagicMock()
        fake_client.responses.parse.return_value = _fake_response(model, tokens)
        with patch("covenant_agent.llm_client.get_client", return_value=fake_client):
            extract_structured(
                instructions="test",
                input_text="test",
                response_model=_FakeFact,
                config=ModelConfig(model=model),
                log_tag="test",
            )

    def test_mini_tier_usage_is_tracked_not_just_top_tier(self) -> None:
        self._call("gpt-5.4-mini", 1000)
        summary = usage_summary()
        self.assertIn("gpt-5.4-mini", summary)
        self.assertEqual(summary["gpt-5.4-mini"], {"calls": 1, "tokens": 1000})

    def test_multiple_calls_to_the_same_model_accumulate(self) -> None:
        self._call("gpt-5.4-mini", 1000)
        self._call("gpt-5.4-mini", 500)
        summary = usage_summary()
        self.assertEqual(summary["gpt-5.4-mini"], {"calls": 2, "tokens": 1500})

    def test_different_models_are_tracked_separately(self) -> None:
        self._call("gpt-5.4-mini", 1000)
        self._call("gpt-5.4", 2000)
        summary = usage_summary()
        self.assertEqual(summary["gpt-5.4-mini"], {"calls": 1, "tokens": 1000})
        self.assertEqual(summary["gpt-5.4"], {"calls": 1, "tokens": 2000})

    def test_log_usage_summary_does_not_raise_when_nothing_called_yet(self) -> None:
        log_usage_summary()  # must not raise on an empty summary

    def test_log_usage_summary_does_not_raise_after_calls(self) -> None:
        self._call("gpt-5.4-mini", 1000)
        log_usage_summary()  # must not raise


if __name__ == "__main__":
    unittest.main()
