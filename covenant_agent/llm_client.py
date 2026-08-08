"""Single point of contact with the OpenAI API.

Every extraction module (covenant_extraction.py, fact_extraction.py) calls
through `extract_structured()` below — none of them import `openai`
directly, and none of them parse free text with regexes. Structured
Outputs (`text_format=<pydantic model>` on the Responses API) makes the SDK
itself guarantee the response matches our schema; if it can't, we get a
refusal or a validation error back, not a malformed guess to silently limp
forward with.

Model selection lives entirely here, one place, overridable by environment
variable — so swapping "the model this call type uses" never means hunting
through extraction/linking code. See "Model profiles" below for the
dev/final split and the three call-type constants
(COVENANT_EXTRACTION_MODEL / TRANSACTION_CATEGORIZATION_MODEL /
FACT_EXTRACTION_MODEL) that replace the old two-tier CRITICAL/STANDARD
naming.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Loads .env (repo root, if present) into os.environ on first import of this
# module — the one place OPENAI_API_KEY is read from, so every entry point
# (CLI scripts, tests) picks it up without each having to remember to call
# this itself. Never overrides an already-exported shell variable.
load_dotenv()

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ModelConfig:
    model: str
    temperature: float = 0.0  # extraction should be as deterministic as the API allows


# --- Model profiles --------------------------------------------------------
#
# The key active as of 2026-08-08 has two very differently-priced pools:
# mini/nano-tier models (gpt-5.4-mini/nano, gpt-4.1-mini/nano, o1-mini,
# o3-mini, o4-mini, ...) are practically unlimited; top-tier models
# (gpt-5.4/5.2/5.1, gpt-5-codex, gpt-5-chat-latest, gpt-4.1, gpt-4o, o1, o3,
# ...) share one hard ~250k-token combined budget for the whole evaluation
# window. Per direct instruction, that budget is reserved for exactly two
# things: one or two transaction-categorization stability checks near Aug 9
# (does a top-tier model categorize more consistently run-to-run than
# mini?), and the actual submission run against the private dataset on
# Aug 9. Everything else — all iteration, all of Step 2's 5x2 comparison
# runs — must stay on the mini/nano tier.
#
# COVENANT_AGENT_PROFILE picks the default tier for all three call types at
# once; each call type also has its own override, so (for example) the
# categorization stability check can promote just that one call to a
# top-tier model without moving the profile switch — which would otherwise
# also pull covenant/fact extraction onto the budgeted tier for no reason:
#
#   COVENANT_AGENT_PROFILE=dev|final                   (default: dev)
#   COVENANT_AGENT_MODEL_COVENANT_EXTRACTION
#   COVENANT_AGENT_MODEL_TRANSACTION_CATEGORIZATION
#   COVENANT_AGENT_MODEL_FACT_EXTRACTION
#
# "dev" uses the same best-available mini-tier model for all three call
# types. No reason to sub-tier within mini/nano the way "final" does below
# — cost isn't the constraint there, so there's nothing to economize by
# picking a worse model for the lower-stakes calls.
#
# "final" keeps fact extraction (task 2b: KYC/audit/other facts — several
# calls per scenario, individually the lowest-stakes of the three, see
# fact_extraction.py) on the mini tier even in the final profile, and only
# promotes covenant extraction and transaction categorization — the two
# call types where a wrong answer silently corrupts `actual`/`status` — to
# top-tier. Same CRITICAL/STANDARD rationale the pipeline already used for
# these two calls, now doubly justified: fact extraction alone would burn a
# meaningful share of a hard-capped, non-renewable budget on calls that
# don't need it.
_DEV_MODEL = "gpt-5.4-mini"
_FINAL_TOP_MODEL = "gpt-5.4"
_FINAL_STANDARD_MODEL = "gpt-5.4-mini"

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "dev": {
        "covenant_extraction": _DEV_MODEL,
        "transaction_categorization": _DEV_MODEL,
        "fact_extraction": _DEV_MODEL,
    },
    "final": {
        "covenant_extraction": _FINAL_TOP_MODEL,
        "transaction_categorization": _FINAL_TOP_MODEL,
        "fact_extraction": _FINAL_STANDARD_MODEL,
    },
}

PROFILE = os.environ.get("COVENANT_AGENT_PROFILE", "dev")
if PROFILE not in _PROFILE_DEFAULTS:
    raise RuntimeError(
        f"COVENANT_AGENT_PROFILE={PROFILE!r} is not 'dev' or 'final' "
        f"(set the COVENANT_AGENT_PROFILE env var to one of those, or unset it for 'dev')."
    )


def _resolve_model(call_type: str) -> ModelConfig:
    env_var = f"COVENANT_AGENT_MODEL_{call_type.upper()}"
    model = os.environ.get(env_var, _PROFILE_DEFAULTS[PROFILE][call_type])
    return ModelConfig(model=model)


COVENANT_EXTRACTION_MODEL = _resolve_model("covenant_extraction")
TRANSACTION_CATEGORIZATION_MODEL = _resolve_model("transaction_categorization")
FACT_EXTRACTION_MODEL = _resolve_model("fact_extraction")


# Advisory only — a per-process running total, not a hard stop and not
# persisted across separate script runs. That's a deliberate match to how
# the top-tier budget is actually spent: a handful of rare, manual,
# single-invocation events (the calibration run(s), the final run), each
# already watched directly by a human against the real OpenAI usage
# dashboard, not an unattended long-lived process that needs its own
# enforcement.
def _is_budget_constrained(model: str) -> bool:
    lowered = model.lower()
    return "mini" not in lowered and "nano" not in lowered


TOP_TIER_TOKEN_BUDGET = 250_000  # approximate, user-stated combined cap
_TOP_TIER_WARN_FRACTION = 0.8
_top_tier_tokens_used = 0

# Every call's usage, on every model — not just the budget-constrained
# top tier above. Added 2026-08-08 after the top-tier-only counter gave a
# false sense of safety: a single evening's mini-tier usage alone (~1.74M
# tokens, confirmed by manually reconstructing cache/llm_logs after the
# fact) was comparable to a colleague's entire week of "normal" usage, and
# there was zero automated visibility into that total until it was
# reconstructed by hand. "Practically unlimited" was an assumption that had
# never actually been instrumented — this closes that gap for every model,
# not just the one tier a hard dollar/token cap happened to be stated for.
_tokens_used_by_model: dict[str, int] = {}
_calls_by_model: dict[str, int] = {}


def usage_summary() -> dict[str, dict[str, int]]:
    """Cumulative {model: {"calls": N, "tokens": N}} for every model used
    by this process so far. Pure data — see log_usage_summary() to print
    it, or scripts/*.py's main() for the "print at the end of every run"
    pattern this exists for.
    """
    return {
        model: {"calls": _calls_by_model[model], "tokens": tokens}
        for model, tokens in _tokens_used_by_model.items()
    }


def log_usage_summary(level: int = logging.INFO) -> None:
    """Log a one-line-per-model breakdown plus a grand total.

    Intended to be called once, unconditionally, at the end of every CLI
    script's main() — so every run ends with an honest, automatic answer
    to "how many tokens did that just cost", never another manual
    cache/llm_logs reconstruction after the fact.
    """
    if not _tokens_used_by_model:
        logger.log(level, "No LLM calls made this run.")
        return
    total_tokens = sum(_tokens_used_by_model.values())
    total_calls = sum(_calls_by_model.values())
    logger.log(level, "LLM usage this run: %d call(s), %d token(s) total.", total_calls, total_tokens)
    for model, tokens in sorted(_tokens_used_by_model.items(), key=lambda kv: -kv[1]):
        logger.log(level, "  %s: %d call(s), %d token(s)", model, _calls_by_model[model], tokens)


def print_usage_summary() -> None:
    """Same breakdown as log_usage_summary, but via print() — always
    visible on stdout regardless of -v/logging level, since "what did this
    run cost" matters even in the default (non-verbose) invocation. Called
    unconditionally at the end of every scripts/run_*.py's main().
    """
    summary = usage_summary()
    if not summary:
        print("\nLLM usage this run: no calls made.")
        return
    total_tokens = sum(v["tokens"] for v in summary.values())
    total_calls = sum(v["calls"] for v in summary.values())
    print(f"\nLLM usage this run: {total_calls} call(s), {total_tokens} token(s) total.")
    for model, v in sorted(summary.items(), key=lambda kv: -kv[1]["tokens"]):
        print(f"  {model}: {v['calls']} call(s), {v['tokens']} token(s)")

_client: OpenAI | None = None

# The org key used to build/test this pipeline sits on a low tier (30k
# tokens/minute for gpt-4.1) that a single scenario's credit agreement plus
# its supporting documents can exhaust within a few calls — confirmed by a
# live 429 four scenarios into a 12-scenario run. The SDK's own built-in
# retry (2 attempts, short backoff) wasn't enough to ride that out, so we
# add our own longer, rate-limit-aware retry around every call. This also
# protects the run on the day if the private-dataset key ends up on a
# similarly low tier.
#
# Also confirmed live: a bare openai.APIConnectionError (an SSL read
# failure mid-stream, "SSLV3_ALERT_BAD_RECORD_MAC" — a transient TLS glitch,
# nothing to do with rate limits) killed a 12-scenario run on scenario 9/12
# because it isn't a RateLimitError or an APIStatusError with a status
# code, so the retry loop's first version let it propagate uncaught. Any
# network-level failure without an HTTP response is exactly as retriable as
# a 5xx, so it gets the same treatment below.
RATE_LIMIT_MAX_ATTEMPTS = 6
RATE_LIMIT_FALLBACK_WAIT_SECONDS = 15.0
CONNECTION_ERROR_WAIT_SECONDS = 5.0
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before running any extraction step, "
                "e.g.: export OPENAI_API_KEY=sk-..."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def _rate_limit_wait_seconds(error: RateLimitError, attempt: int) -> float:
    """Prefer the wait time OpenAI's own error message states; otherwise back off."""
    message = str(error)
    match = _RETRY_AFTER_RE.search(message)
    if match:
        return float(match.group(1)) + 1.0  # small safety margin
    return RATE_LIMIT_FALLBACK_WAIT_SECONDS * attempt


class ExtractionError(RuntimeError):
    """Raised when the model refuses, or returns no parsed output.

    Deliberately not caught anywhere in this module — a failed extraction
    must surface as a visible failure for that scenario/document, not as a
    quietly-empty or default-valued result that later looks like "we
    checked and found nothing."
    """


def extract_structured(
    *,
    instructions: str,
    input_text: str,
    response_model: type[T],
    config: ModelConfig,
    log_dir: Path | None = None,
    log_tag: str = "call",
) -> tuple[T, dict]:
    """Call the model with Structured Outputs; return (parsed_result, raw_response_dict).

    `instructions` is the system-level task description; `input_text` is the
    per-call content (the document text plus any per-call framing).
    """
    client = get_client()
    response = None
    last_error: Exception | None = None
    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            response = client.responses.parse(
                model=config.model,
                temperature=config.temperature,
                instructions=instructions,
                input=input_text,
                text_format=response_model,
            )
            break
        except RateLimitError as exc:
            last_error = exc
            if attempt == RATE_LIMIT_MAX_ATTEMPTS:
                break
            wait = _rate_limit_wait_seconds(exc, attempt)
            logger.warning(
                "[%s] rate limited (attempt %d/%d), waiting %.1fs before retrying.",
                log_tag,
                attempt,
                RATE_LIMIT_MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
        except APIStatusError as exc:
            if exc.status_code < 500 or attempt == RATE_LIMIT_MAX_ATTEMPTS:
                raise
            last_error = exc
            wait = RATE_LIMIT_FALLBACK_WAIT_SECONDS
            logger.warning(
                "[%s] server error %d (attempt %d/%d), waiting %.1fs before retrying.",
                log_tag,
                exc.status_code,
                attempt,
                RATE_LIMIT_MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
        except APIConnectionError as exc:
            last_error = exc
            if attempt == RATE_LIMIT_MAX_ATTEMPTS:
                break
            wait = CONNECTION_ERROR_WAIT_SECONDS * attempt
            logger.warning(
                "[%s] connection error (attempt %d/%d), waiting %.1fs before retrying: %s",
                log_tag,
                attempt,
                RATE_LIMIT_MAX_ATTEMPTS,
                wait,
                exc,
            )
            time.sleep(wait)

    if response is None:
        raise ExtractionError(
            f"[{log_tag}] gave up after {RATE_LIMIT_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    # Deliberately not response.model_dump(): the SDK's Response.output is a
    # union over every possible item type (function calls, MCP calls, tool
    # calls, ...), and pydantic emits one "unexpected value" warning per
    # non-matching union member for every item — pure noise for a plain
    # Structured Outputs call with no tools. Log exactly what's useful for
    # debugging/evidence instead: the parsed result, the raw text it came
    # from, and token usage for cost tracking.
    raw = {
        "id": response.id,
        "model": response.model,
        "status": response.status,
        "usage": response.usage.model_dump() if response.usage else None,
        "output_text": response.output_text,
        "output_parsed": response.output_parsed.model_dump() if response.output_parsed else None,
    }
    if log_dir is not None:
        _log_call(log_dir, log_tag, instructions, input_text, raw)

    if response.usage:
        _tokens_used_by_model[config.model] = (
            _tokens_used_by_model.get(config.model, 0) + response.usage.total_tokens
        )
        _calls_by_model[config.model] = _calls_by_model.get(config.model, 0) + 1

    if response.usage and _is_budget_constrained(config.model):
        global _top_tier_tokens_used
        _top_tier_tokens_used += response.usage.total_tokens
        logger.info(
            "[%s] top-tier call (model=%s) used %d tokens; cumulative top-tier usage "
            "this process: %d/%d.",
            log_tag,
            config.model,
            response.usage.total_tokens,
            _top_tier_tokens_used,
            TOP_TIER_TOKEN_BUDGET,
        )
        if _top_tier_tokens_used >= TOP_TIER_TOKEN_BUDGET * _TOP_TIER_WARN_FRACTION:
            logger.warning(
                "Top-tier token usage at %d/%d (%.0f%%) this process — remaining "
                "top-tier calls should be deliberate. (Advisory only — the real "
                "source of truth is the OpenAI usage dashboard.)",
                _top_tier_tokens_used,
                TOP_TIER_TOKEN_BUDGET,
                100 * _top_tier_tokens_used / TOP_TIER_TOKEN_BUDGET,
            )

    if response.status == "incomplete":
        raise ExtractionError(
            f"[{log_tag}] response incomplete: {response.incomplete_details}"
        )

    parsed = response.output_parsed
    if parsed is None:
        raise ExtractionError(
            f"[{log_tag}] model returned no parsed structured output "
            f"(status={response.status}); see logged raw response for details."
        )
    return parsed, raw


def _log_call(log_dir: Path, tag: str, instructions: str, input_text: str, raw: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = log_dir / f"{ts}_{tag}_{uuid.uuid4().hex[:8]}.json"
    path.write_text(
        json.dumps(
            {"instructions": instructions, "input": input_text, "raw_response": raw},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.debug("Logged LLM call to %s", path)
