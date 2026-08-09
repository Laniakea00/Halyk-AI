# Halyk AI Challenge — Covenant Compliance Agent

A batch pipeline (no web server, no framework) that reads a borrower's credit
agreement, KYC file, audit reports and transaction ledger, and decides
whether each financial covenant is `COMPLIANT` or `BREACH` — with the
supporting number and, where applicable, the one transaction that proves it.

The decision logic lives in code. An LLM is used only where the task is
genuinely "read this paragraph and tell me what it says" (covenant
definitions, KYC facts, auditor reclassifications) — never to decide
compliance itself. See `agentic-bank-public/CASE.ru.md` for the full rules.

Entry points so far: `run_ingestion.py` (Block 1), `run_extraction.py`
(Block 2), `run_linking.py` (Block 3a alone), `run_calculation.py` (Block
3a+3b+3c, with an optional self-score against `ground_truth.json`). Blocks
4 (Explanation) and 5 (`submission.json` assembly + validation) are not
built yet — see "Running the full pipeline end-to-end" below for exactly
what today's `HEAD` can and can't produce on its own.

## Setup

```bash
brew install poppler        # provides `pdftotext`, used for PDF extraction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in OPENAI_API_KEY
```

`pdftotext` is a hard requirement — the pipeline calls it as a subprocess
rather than depending on a Python PDF library. All 200 PDFs in the public
dataset carry a real text layer; `run_ingestion.py -v` will warn loudly
about any PDF whose extraction comes back near-empty (i.e. likely scanned),
which the private dataset could still contain.

`OPENAI_API_KEY` is required from Block 2 onward (covenant/fact
extraction). It's read from the environment or from a `.env` file in the
repo root (loaded automatically, never committed — see `.env.example`).
`COVENANT_AGENT_PROFILE=dev|final` picks the model tier for all three call
types at once (default `dev` — mini/nano, no token-budget concern);
`COVENANT_AGENT_MODEL_COVENANT_EXTRACTION` /
`COVENANT_AGENT_MODEL_TRANSACTION_CATEGORIZATION` /
`COVENANT_AGENT_MODEL_FACT_EXTRACTION` override any one call type
individually. See "Design decisions" below and `llm_client.py`'s "Model
profiles" docstring for why the split exists and what it's reserved for.

## Running the full pipeline end-to-end

This is the sequence a cold-start person should follow — e.g. picking this
repo back up on Aug 9 with the private dataset just dropped and the clock
running.

```bash
# 0. One-time setup (see above) — confirm the venv is active and the key is set:
source .venv/bin/activate
python -c "import os; assert os.environ.get('OPENAI_API_KEY'), 'no key'"

# 1. Point everything at the real dataset (defaults to agentic-bank-public/;
#    override once here rather than passing --data-dir to every command):
export DATA_DIR=/path/to/private-dataset   # only if it's not agentic-bank-public/

# 2. Sanity-check ingestion alone first — cheap, local, no API calls, and it's
#    where a private-dataset structural mismatch (see "known instability"
#    below) would show up immediately instead of 20 minutes into Block 2/3:
python scripts/run_ingestion.py -v --data-dir "$DATA_DIR" --report cache/ingestion_report.json

# 3. Run extraction (Block 2) ONCE, save it to a facts cache. This is the
#    expensive, non-deterministic, real-money step — every later command
#    should load from --facts-cache instead of re-running this:
python scripts/run_extraction.py -v --data-dir "$DATA_DIR" \
    --report cache/scenario_facts.json

# 4. Run linking + calculation (Block 3), loading the facts cache from step 3,
#    and self-score if ground_truth.json exists for this dataset (public only):
python scripts/run_calculation.py -v --data-dir "$DATA_DIR" \
    --facts-cache cache/scenario_facts.json \
    --report cache/calculation_results.json \
    --ground-truth "$DATA_DIR/ground_truth.json"   # omit for the private dataset
```

**What step 4's output actually is, honestly:** `cache/calculation_results.json`
is `{scenario_id: {covenant_key: {status, actual, evidence_txn_id}}}` for
every required cell — the exact payload `submission.json`'s `answers` field
needs. **Block 5 (assembling this into the literal `submission.json` shape
— `team`/`contact_email`/`model` header, exact key layout, JSON validity
check against `submission_template.json`) is not built yet.** Until it is,
turning step 4's report into a valid submission is a manual (or
soon-to-be-scripted) reshape, not a single command. Blocks 1–3 are the
part that's genuinely "run and forget"; treat this README as needing a
follow-up pass once Block 5 exists.

### Known sources of instability — read before assuming something's broken

- **Transaction categorization (Block 3a) is not perfectly reproducible,
  even at temperature=0.** Measured across 5 back-to-back runs on the
  dev-profile model, *zero* code changes between them: self-scores of
  0.5542, 0.5681, 0.5404, 0.6094, 0.5265 (mean 0.560, min 0.527, max
  0.609). The deterministic layers (linking arithmetic, formulas,
  evidence, fallback) are unit-tested and stable — the instability is
  specifically in which transactions the LLM puts in which category on a
  given pass. **Cross-referencing all 5 runs' wrong-status cells showed
  most of that spread isn't random** — roughly half the wrong cells were
  wrong in literally every one of the 5 runs (systematic misses, mostly
  since root-caused: dirty ledger rows, undefined "EBITDA", missing
  document links — see Findings), and only a handful of cells actually
  flip between runs. Don't chase a single-run score swing as if it were a
  regression; compare against a multi-run average, and know that a
  "systematic" miss needs a different kind of fix than a "flips between
  runs" one.
- **Full 12-scenario runs are now budget-limited, not just time-limited.**
  A single evening's worth of point-checks and full runs (this session)
  used ~3.75M tokens total, and per-model usage wasn't being tracked at
  all until it was reconstructed by hand afterward — see
  `llm_client.py`'s `print_usage_summary()` (now printed automatically at
  the end of every `scripts/run_*.py` invocation, precisely so that never
  has to happen again). Going forward: `--scenario <id>` point-checks are
  cheap and fine to run freely; full 12-scenario runs are reserved (at
  most 1–2 before the actual submission run); a stability re-check needs
  2–3 runs, not 5. Output tokens are the scarcer remaining resource —
  prompts that ask for long free-text explanations without a real need
  for them are a candidate to trim if budget gets tighter.
- **Rate limits.** `llm_client.py` retries with real backoff (up to ~6
  attempts, growing delay) — a run that logs several `rate limited`
  warnings and keeps going is working as intended, not stuck.
- **`insufficient_quota` looks identical to a rate limit at first glance**
  (both surface as HTTP 429) but isn't — it means the account has no
  credits left, and no amount of retrying fixes it. If a run keeps
  retrying for minutes with growing backoff and then finally raises
  `ExtractionError: ... insufficient_quota ... credit_balance_exhausted`,
  that's the account, not the network — go add credits, don't just rerun.

### If the API dies mid-run — can I safely restart from the middle?

**Yes, since the High-risk code-review findings were fixed (see below) —
this used to be "partially", it isn't anymore.**

- **Block 1 (ingestion)** is fully local and re-runs in seconds — never a
  concern.
- **Block 2 (extraction)** now saves after *every* scenario, not once at
  the end (`--report cache/scenario_facts.json` is written incrementally).
  A crash on scenario 9 of 12 leaves scenarios 1–8's real, paid-for
  results on disk. `extract_all_facts` also catches any exception
  per-scenario (not just the expected `ExtractionError` shape) and
  degrades that one scenario to an empty result rather than aborting the
  batch — the run finishes with a clear "N/M ok, FAILED: [...]" summary
  naming exactly which scenarios to re-run with `--scenario <id>`.
- **Block 3a (linking / transaction categorization)** has the same
  per-scenario exception isolation now — a categorization-call failure
  degrades that one scenario to "everything unclassified" (which
  `InsufficientDataError` then handles the normal way) instead of
  aborting `run_calculation.py` mid-run.
- **Practical flow today:** run Block 2 to completion (or until it
  reports failures), confirm `--facts-cache` was written, re-run just the
  named failures with `--scenario <id>`, then proceed to Block 3 — safe
  to call repeatedly either way.

## Running Block 1 (Ingestion + Entity/Version resolution)

```bash
python scripts/run_ingestion.py -v --report cache/ingestion_report.json
```

- Prints a per-scenario summary: which account it resolved to, which
  document is currently in force for each kind (credit agreement, KYC
  dossier, audit report, treasury memo), and how many older/draft versions
  were excluded.
- `--report` dumps full metadata (minus raw text) as JSON for inspection.
- `--data-dir` / `--cache-dir` override the defaults
  (`agentic-bank-public/`, `cache/`) — point these at the private dataset on
  the day without touching any code.
- Parsed PDF text is cached under `cache/pdf_text/`, keyed by a hash of the
  file's bytes, so re-running the pipeline doesn't re-shell out to
  `pdftotext` for unchanged files.

Tests — everything below runs free, fast, and deterministic, no API key
needed (see "Testing philosophy" at the end of this section for why real-API
correctness is deliberately *not* part of this suite):

```bash
python -m unittest discover -s tests -v   # all of the below in one go
```

| File | Covers |
|---|---|
| `test_resolution.py` | Block 1, against the real public dataset |
| `test_extraction_pipeline.py` | Block 2 orchestration wiring, mocked LLM calls |
| `test_linking_utils.py` | Block 3a's non-LLM pieces: compound-description splitting, fuzzy counterparty matching, date/period parsing, reclassification linking, related-party threshold resolution |
| `test_calculation.py` | Block 3b/3c: formulas by `metric_type`, period filtering, the `InsufficientDataError` fallback, and evidence's counterfactual logic — all on synthetic `LinkedScenarioData`, no real dataset needed |

**Testing philosophy:** the LLM-facing steps (2a/2b extraction, 3a
categorization) are checked for *wiring* only — which document/category
routes to which function, missing data degrades instead of raising. Whether
the model extracts *correctly* from real documents is a separate, manual
concern via `run_extraction.py` / `run_calculation.py` against the public
dataset (with `ground_truth.json` to self-score against) — that correctness
is inherently non-deterministic and costs real tokens, so it can't be a
green/red CI check the way the deterministic layers can.

## Running Block 2 (Covenant clause + Fact extraction)

```bash
python scripts/run_extraction.py -v --report cache/extraction_debug_all.json
python scripts/run_extraction.py --scenario P1 -v   # just one scenario, for fast iteration
```

Runs Block 1 first, then for every scenario: one Structured Outputs call to
extract its covenant clauses from the current credit agreement, plus one
call each for its current KYC dossier / audit report(s) / anything else.
Prints a human-readable summary per scenario and, with `--report`, writes
the full structured output (every field, not just the summary) as JSON,
in a format `--load-cache`/`--facts-cache` (see Block 3 below) can read
back in — this is the recommended way to run it, so Block 3 iteration
doesn't re-pay for extraction every time. Every individual LLM call's
instructions/input/output are also logged to `cache/llm_logs/` regardless
of `--report` — that's the trail the Explanation layer (Block 4) will read
from.

## Running Block 3 (Linking + Decision/Calculation)

```bash
# 3a alone, for inspecting the linking output directly (category counts,
# linked/unmatched reclassifications, resolved related parties):
python scripts/run_linking.py -v --facts-cache cache/scenario_facts.json

# 3a+3b+3c together, with the public dataset's self-scorer:
python scripts/run_calculation.py -v --facts-cache cache/scenario_facts.json \
    --report cache/calculation_results.json \
    --ground-truth agentic-bank-public/ground_truth.json
```

Both load `--facts-cache` if it already exists (skipping Block 2 entirely)
or run Block 2 live and populate it if not. `--scenario P1` limits either
script to one scenario — genuinely limits the work done, not just the
printed output (an earlier version of `run_calculation.py` linked all 12
scenarios even with `--scenario` set, wasting ~30x the API calls for a
"quick" single-scenario check; fixed, but a reminder to verify this kind
of thing when adding scenario-scoping to a new script).

## Architecture

```
agentic-bank-public/     input data (provided, read-only)
covenant_agent/
  config.py               paths + generic constants (nothing scenario-specific)
  models.py                the dataclass contract between pipeline stages
  schemas.py                pydantic Structured Output schemas (Block 2's LLM boundary)
  llm_client.py              OpenAI Responses API wrapper: dev/final model profiles,
                                retry, raw-call logging, per-model usage accounting
                                (print_usage_summary() — every script prints this)
  ingestion/
    pdf_text.py             pdftotext subprocess wrapper, content-hash disk cache
    documents.py             loads documents/*, dispatches by extension
    ledger.py                loads the CSV, derives scenario_id -> account_id
    template.py               loads submission_template.json (source of truth for scope),
                                 validates its structure (TemplateValidationError)
  resolution/
    accounts.py              exact-token account matching (decoy-safe), normalizes a
                                letter-spaced "A C C - 7 8 0 3" PDF-rendering artifact
                                before the same exact match runs (see Findings)
    classify.py               keyword-scored document-kind classifier — credit_agreement,
                                 kyc_dossier, audit_report, treasury_memo, financial_notes
    segment_linking.py        secondary, narrower document-to-scenario linking for
                                 documents accounts.py's ACC-token match can't place at
                                 all (a Group-parent report never mentions the
                                 subsidiary's own account) — exact match on the
                                 scenario's own already-verified borrower name next to
                                 explicit subsidiary/segment language, never a general
                                 company-name similarity guess
    versioning.py              superseded/draft detection, current-doc selection
    pipeline.py                 orchestrates the above into an IngestionResult
  extraction/
    covenant_extraction.py    Block 2a: credit agreement -> CovenantExtractionResult
    fact_extraction.py         Block 2b: KYC / audit-and-financial-notes-disclosure /
                                  other -> their result schemas (one extractor,
                                  extract_audit_facts, now covers audit_report,
                                  financial_notes, AND treasury_memo — see Findings)
    pipeline.py                  orchestrates 2a+2b per scenario into ScenarioFacts
    cache.py                      round-trippable save/load for ScenarioFacts (--facts-cache)
  linking/
    categories.py              derives a per-covenant transaction-category vocabulary;
                                  compound "X net of Y" / "X and Y" description splitting
                                  (recursive — handles N-way chains, not just pairs;
                                  borrows a trailing noun back when "и" elides it, e.g.
                                  "Арендных и Коммунальных расходов"); strips a retained
                                  "EBITDA ... как" preamble after a netting split;
                                  match_category_by_text (reclassification -> category)
    transaction_categorization.py  Block 3a: one LLM call/scenario, ledger -> category
    related_parties.py          KYC ownership % vs. threshold comparison — code, no LLM
    reclassification_linking.py  Block 3a proper: audit/financial-notes findings ->
                                    ledger txn_id, either directly (when the finding
                                    states one) or by counterparty+amount(+date) fuzzy
                                    match; three action shapes — recategorize,
                                    exclude_from_period, no_change (informational only)
    fuzzy_match.py, dates.py     shared normalization/matching utilities
    pipeline.py                   orchestrates the above into LinkedScenarioData —
                                     also patches a dirty ledger row's amount by txn_id
                                     (_apply_amount_corrections) before categorization
                                     runs, and collects other_facts for calculation
  calculation/
    formulas.py                 Block 3b: metric_type-branched arithmetic, no LLM at
                                   all; exclude_from_period drops a transaction from
                                   every sum regardless of category; other_facts add
                                   into a matching side via the same text-match
                                   machinery reclassifications use
    evidence.py                  Block 3c: the counterfactual evidence test
    pipeline.py                   guarantees one CovenantResult per required cell,
                                     with an explicit, logged fallback on any failure
  scoring.py                  self-scorer implementing the case's own scoring formula
scripts/
  run_ingestion.py          CLI for Block 1
  run_extraction.py          CLI for Block 2
  run_linking.py             CLI for Block 3a alone
  run_calculation.py          CLI for Block 3a+3b+3c (+ optional self-scoring)
  (all three print a per-model LLM usage summary unconditionally at the end)
tests/
  test_resolution.py         Block 1, against the real dataset (includes the
                                financial_notes/segment_linking regression tests)
  test_accounts.py            the letter-spaced-ACC-token normalization, synthetic
  test_segment_linking.py     the P5-shaped secondary linking mechanism, synthetic
  test_template_validation.py submission_template.json structural validation
  test_extraction_pipeline.py Block 2 orchestration, mocked LLM calls
  test_transaction_categorization.py  categorization prompt content (decoy guidance)
  test_linking_utils.py       Block 3a's non-LLM pieces, synthetic data
  test_linking_pipeline.py    Block 3a batch resilience + amount-correction patching
  test_calculation.py          Block 3b/3c, synthetic data (incl. other_facts,
                                  exclude_from_period, quarter-period inference)
  test_llm_client.py          per-model usage accounting, mocked (no real API call)
```

Planned remaining blocks (not yet built): Explanation (assembling the
evidence chain into a human-readable rationale per cell, from the
provenance already threaded through every layer) and Submission assembly
(reshaping `calculation_results.json` into the literal `submission.json`
shape + validating it against `submission_template.json`).

## Design decisions worth knowing about

- **Plain `@dataclass(frozen=True)`, not Pydantic.** This layer only reads
  local files we fully control — there's no untrusted JSON boundary yet.
  Pydantic (or a hand-written validator) belongs at the submission-assembly
  boundary instead, where the output schema is externally imposed and
  strict.
- **pandas is the only real dependency.** PDF parsing shells out to the
  system `pdftotext` binary (zero Python PDF-library fragility); the ledger
  is small enough that pandas is a convenience, not a necessity, but it will
  pay for itself in the calculation layer (grouping/summing).
- **Nothing scenario-specific is hardcoded.** The set of required
  `scenario_id`s comes from `submission_template.json`; each one's
  `account_id` is derived from the ledger's own `TXN-<scenario>-<seq>`
  encoding. Pointing `--data-dir` at the private dataset should work with
  zero code changes, *if* its structure matches — the ingestion layer is
  the thing that will surface it immediately (via exceptions/warnings) if it
  doesn't.
- **Account matching is exact-token, never substring.** `ACC-7801-08` is a
  different token from `ACC-7801` — matching by substring would silently
  merge a scenario's real account with an unrelated decoy sub-account
  belonging to a different legal entity. See "Findings" below.
- **Document-kind classification is keyword-scored, not hardcoded to the
  companies we've seen.** Markers describe a document's *function*
  ("отчёт о выполнении согласованных процедур", "знай своего клиента"),
  not any specific borrower — these should recur in the private dataset even
  though the companies won't.
- **Version resolution never forces an arbitrary pick.** Within a
  (scenario, document kind) group, ties are broken by: explicit
  supersede/draft markers → highest revision number → latest date found →
  if still tied, keep every remaining candidate and log a warning. An
  arbitrary silent pick would be worse than a visible ambiguity.
- **Dirty ledger rows are kept, not dropped.** Two rows in the public
  ledger have an empty `amount`. They're preserved with `amount=None` and a
  loud warning rather than silently vanishing (which would understate
  transaction counts) or crashing — recovering the true value from a
  supporting document is plausibly part of the challenge, not a bug to
  paper over.
- **OpenAI Responses API with Structured Outputs (`text_format=<pydantic
  model>`), never regex-parsed free text.** The SDK itself guarantees the
  response matches the schema; a malformed reply surfaces as an exception
  (`ExtractionError`), not a value we silently trust. All schemas live in
  `schemas.py`, and every field there was written to *never* ask the model
  for a judgment it isn't positioned to make (see that file's docstring) —
  the model reports the ownership percentage and the document's own
  threshold, for instance, never a computed "is this a related party"
  boolean. Comparing the two is Block 3's job, in code.
- **Three named call types, each independently model-selectable, picked
  once, in one place (`llm_client.py`).** `COVENANT_EXTRACTION_MODEL` and
  `TRANSACTION_CATEGORIZATION_MODEL` are for the two calls where a wrong
  answer silently corrupts a downstream number — covenant clause
  extraction and transaction categorization respectively.
  `FACT_EXTRACTION_MODEL` is for KYC/audit/other fact extraction —
  individually lower-stakes, and there are more of these calls per
  scenario. All three resolve from a `COVENANT_AGENT_PROFILE=dev|final`
  switch (dev = mini/nano tier for everything, final = top tier for the
  first two call types only) and are each also individually overridable
  via environment variable with no code change. Added 2026-08-08 when a
  new key arrived with a hard combined token budget on top-tier models but
  effectively unlimited mini/nano access — see `llm_client.py`'s "Model
  profiles" docstring for the full reasoning and why fact extraction
  deliberately stays off the budgeted tier even in the final profile.
- **Pydantic, finally, right at this boundary.** Block 1's README note said
  Pydantic belongs "at the boundary where we're handed a strict external
  schema to satisfy" — that's exactly what Structured Outputs is, just from
  the other direction (we impose the schema on the API, rather than the API
  imposing one on us). `schemas.py` is the only place it's used; the rest
  of the pipeline is still plain dataclasses.
- **Rate limits and transient network failures are retried with real
  backoff, not the SDK's default.** Confirmed live: a full 12-scenario run
  hit a `429` four scenarios in (the evaluation org key sits on a 30k
  tokens/minute tier for `gpt-4.1`, which a single credit agreement plus
  its supporting documents can exhaust in a few calls) and, separately, an
  `openai.APIConnectionError` (a mid-stream SSL read failure — nothing to
  do with rate limits) killed the run at scenario 9/12 because it isn't a
  `RateLimitError` or a `APIStatusError` with a status code, so the first
  version of the retry loop let it propagate uncaught. `llm_client.py` now
  retries both explicitly, respecting the server's own suggested wait time
  for 429s where it's given. This matters beyond convenience: on the day,
  a pipeline that dies on transient network noise partway through 12
  scenarios is worse than one that's merely slow.
- **The raw LLM call log is curated, not `response.model_dump()`.** The
  SDK's `Response.output` is a big union over every possible item type
  (function calls, MCP calls, tool calls, ...); dumping it wholesale
  produces a wall of pydantic "unexpected value" warnings for every
  non-matching union member on a plain no-tools call. The log instead
  records exactly what's useful for debugging and for the future
  Explanation layer: instructions, input, parsed output, token usage.
- **Transaction categorization is the one linking-layer LLM call grouped
  with covenant extraction (`TRANSACTION_CATEGORIZATION_MODEL`), not with
  the cheaper fact-extraction group, breaking Block 2's original tiering
  rule.** A miscategorized transaction directly corrupts a sum that feeds
  `actual` and potentially `status` — the exact failure mode that tier
  exists to guard against, not a "supporting fact" the way a KYC
  disclosure is.
- **Self-consistency voting (3 independent categorization calls, majority
  vote) was tried and deliberately reverted — a negative result worth
  recording so it isn't retried blind.** The theory was sound (damp
  temperature=0 non-determinism), but measured worse in practice: the
  prompt explicitly biases toward `unclassified` on any doubt (decoys
  punish over-inclusion harder than under-inclusion), so a real match that
  only survives the model's conservative bias 2-of-3 times loses to
  `unclassified`, which only needs the model to hedge twice. Voting
  amplified the bias instead of averaging out the noise. A single
  well-prompted call scored higher across repeated runs than best-of-3.
- **No artificial sign-flipping when netting a compound category (e.g.
  "revenue net of operating expenses").** The ledger already records
  outflows as negative — summing revenue (positive) with opex (already
  negative) subtracts correctly via plain addition. Confirmed the
  opposite (multiplying the "minus" side by −1) live: it doubled expenses
  onto the numerator instead of subtracting them, producing a ~30x-too-high
  coverage ratio.
- **Compound-description splitting is recursive, not one-shot, and only
  ever applied to `numerator_description`/`denominator_description`, never
  `formula_description`.** A single split turns "A net of B, net of C, net
  of D" into one clean part and one still-compound blob covering B+C+D —
  pushes the same problem one level deeper instead of solving it, so
  splitting runs to a fixed point (capped at 6 parts against pathological
  input). It's restricted to the two short, schema-guaranteed-focused
  fields because splitting `formula_description` (a full multi-sentence
  paraphrase) once caused a real regression: a connector word buried in an
  unrelated carve-out sentence split the description into one giant
  unfocused blob and one useless fragment.
- **A ratio's denominator raises `InsufficientDataError` on zero matched
  transactions instead of dividing through a small epsilon.** The
  original epsilon-fallback was itself the single most damaging bug found
  in this block — see "Findings" below — and the fix is structural, not a
  bigger epsilon: zero classified transactions for a concept like
  "operating expenses" is never a real business fact (no operating company
  has zero), so it's now surfaced as "insufficient data" and routed through
  the same explicit, logged fallback as every other missing-information
  case, not divided by. Deliberately *not* applied to `aggregate_amount`/
  `max_single_component`/related-party-sourced sides — see the code-review
  findings below for why that's a known, not-yet-closed gap.

## Findings baked into the code (validated against the real public dataset)

- **Decoy sub-accounts.** Aktau Port Services JSC (scenario P1, `ACC-7801`)
  has internal ops reports referencing `ACC-7801-08` / `ACC-7801-02` /
  `ACC-7801-05` — the same numeric prefix, but tied to differently-named
  legal entities (`... AG`, `... LLP`) and never to the real account.
  Exact-token matching (not substring) keeps these out.
- **Explicitly superseded agreements.** Every scenario has a 2024 credit
  agreement stamped `НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ ... НЕ ПРИМЕНЯЕТСЯ`, superseded
  by a 2025 agreement. Confirmed this phrase appears in exactly the 12
  outdated agreements and nowhere else in the 200-document corpus.
- **Draft audit workpapers with no final counterpart.** 4 of 5
  audit-reclassification-shaped documents (P3, P6, P8, P9) are interim
  workpapers explicitly marked "не является окончательной позицией
  аудитора" — and, unlike the credit agreements, most have *no* competing
  final version in the dataset at all. A naive "only demote on a tie"
  approach would have silently accepted these four as authoritative;
  they're now correctly excluded (those four scenarios end up with **no**
  current audit report, which is the honest answer, not a bug).
- **PDF headers are letter-spaced.** Stylized headers extract from
  `pdftotext` with a space between every letter/syllable (e.g. "С л уже б н
  а я з ап и с ка"), so naive substring matches against the "clean" phrase
  fail on headers but succeed on normal body prose. The classifier's marker
  phrases were chosen/verified against actual extracted text, not the
  visual PDF layout.
- **Not every scenario has every document kind.** P6 has no KYC dossier
  anywhere in the public corpus; only one treasury memo exists in total
  (P7's). Both are gaps in the source data, not resolution bugs — Block 2
  needs to degrade gracefully (e.g. no related-party disclosures found ⇒
  can't positively identify any related party, not ⇒ crash).
- **Multi-currency ledger — a rate exists for exactly one scenario, found
  late.** Some scenario accounts have EUR-denominated rows alongside USD
  ones. Initially no FX rate document had turned up; later confirmed (after
  the organizers stated in the hackathon chat that a rate does exist
  somewhere in the files) that P3's own `financial_notes` document states
  one, implied by a real settlement (72,146.75 EUR paid as $83,690.23).
  Checked all 12 scenarios' own documents specifically for this — only P3
  has one. Currency is preserved verbatim per transaction; conversion is
  deliberately not implemented (P3's rate hasn't been applied yet either)
  rather than guessed at for the other 11 scenarios with no rate source.

### Block 2 findings

- **Clause 6.1/6.2/6.3 really is a different formula per borrower —
  confirmed across all 12, not just the two spot-checked in Block 1.**
  `metric_type` came back mixed at every position across scenarios: e.g.
  6.1 is `ratio` for most borrowers but a flat `aggregate_amount` for P8;
  6.2 is `max_single_component` ("Individual Overhead Line Ceiling" — judged
  by whichever named line item is larger, explicitly *not* their sum) for
  B1 and P10 but `aggregate_amount` for most others. A calculation layer
  keyed by clause number instead of by extracted `metric_type` would be
  wrong on roughly half these scenarios.
- **The same related-party test can be phrased as a flat cap or as a ratio
  of revenue.** P1/B1's clause 6.3 is "related-party payments must not
  exceed $X". P2's clause 6.3 is "related-party payments must not exceed
  3% of revenue" (`metric_type: ratio`, threshold `0.03`) — same underlying
  concept (related-party payments), structurally different test. Both
  still need the same underlying fact (which counterparties are related
  parties), just combined differently.
- **Not every KYC dossier in the corpus has an ownership-percentage table.**
  P1/B1/etc.'s KYC documents have an explicit "Бенефициарное владение и
  контроль" table with named counterparties and voting percentages. P2's
  KYC document is a same-template-looking but purely narrative/procedural
  dossier — identification, sanctions screening, risk rating — with no
  ownership table and no numeric related-party threshold anywhere in the
  text. The extractor correctly returned `disclosures: []` and
  `related_party_threshold_pct: null` rather than inventing structure that
  isn't there; Block 3 needs to treat that as "no disclosures found in this
  document", a real gap to reason about, not "zero related parties
  confirmed".
- **The Ekibastuz Energy (B1) reclassification hits two different covenants
  in two different ways, exactly as suspected in Block 1.** The auditor
  report reclassifies a $592,296.10 payment to *Irtysh Advisory Bureau*
  from "Консультационные услуги" to "Процентные расходы" — which affects
  6.1 (interest coverage ratio) by moving that amount into the interest
  expense line. Separately, the KYC dossier discloses Irtysh Advisory
  Bureau at 18.6% voting rights against a stated 20% related-party
  threshold — *below* the threshold, so it does **not** count toward 6.3
  (related-party payments) even though it's the same counterparty being
  reclassified for 6.1. Block 2 extracted both facts (the percentage and
  the document's own threshold) without ever computing the boolean itself
  — Block 3 does that comparison and will need to get this one right in
  both directions.

### Block 3 findings

- **The ledger has no category column — by design, per the case brief —
  which turned out to require an entire extra classification pass that
  the original Block 1–2 architecture hadn't accounted for.** Computing
  "capital expenditure", "operating expenses", "revenue", etc. means
  reading every transaction's free-text `description` and deciding what
  it is, for every covenant that sums/ratios by concept. This is
  `linking/transaction_categorization.py` — the one LLM call in the whole
  of Block 3, everything downstream of it is pure arithmetic.
- **The epsilon-fallback for a near-zero ratio denominator was the single
  most damaging bug found in this block, and it was silent by
  construction.** `numerator / max(denominator, 0.01)` turns "the
  classifier found zero matching transactions" into a fabricated
  actual value in the hundreds of millions — not an obviously-wrong
  number by inspection alone once you're staring at 12 scenarios' worth of
  output, and *worse*, `status` often came out looking plausible anyway
  (an absurdly huge ratio trivially breaches almost any max-direction cap,
  so the coin-flip nature of the bug was masked). Confirmed on P3/P4/P6/P9
  simultaneously in one run. Replaced with `InsufficientDataError` — see
  "Design decisions" above. This class of bug (a quiet default standing in
  for "we don't actually know") is exactly what the code-review pass
  below went looking for elsewhere in the codebase.
- **Ledger sign convention is genuinely load-bearing, not incidental.**
  Getting "revenue net of operating expenses" right required realizing the
  ledger's own negative-for-outflows convention already performs the
  subtraction — an explicit +1/−1 sign flip on top of it silently
  *doubles* the expense instead of subtracting it. Caught by manually
  checking B1's real numbers against `ground_truth.json` (a 30x-too-high
  ratio is easy to spot), not by code review — worth remembering that this
  class of arithmetic-direction bug doesn't announce itself in the types.
- **The dataset embeds decoy transactions on both sides of the ledger's
  sign, not just the expense side.** Block 1 already found expense-flavored
  decoys (marketing/insurance/tax that sound like but aren't "operating
  expenses"). Block 3 found the mirror trap on inflows: refunds, rebates,
  tax credits/overpayments returned, deposits returned, lease incentives,
  and "marketing co-op funding received" are all positive-amount
  transactions that are *not* revenue, and an unguided classifier
  readily miscounts them as such (confirmed live: it inflated B1's revenue
  numerator with a marketing-funding receipt, flipping a correct BREACH to
  an incorrect COMPLIANT). The categorization prompt now names this trap
  explicitly on both sides.
- **The case's evidence-disqualification rule ("a transaction that merely
  tips an accumulated sum over the threshold is not evidence") had to be
  taken completely literally, not just as a tie-breaking preference.** An
  early version of `evidence.py` tested *every* category-contributing
  transaction for exclusion and found a real swing candidate for P1's 6.1
  — an ordinary, never-reclassified opex line that happens to be large
  enough that removing it flips BREACH to COMPLIANT. The ground truth key
  for that cell is `null`. The fix wasn't a smarter tie-break; it was
  narrowing which transactions are ever tested for exclusion at all, to
  only the two cases the case's own wording allows: a reclassification
  reversal, or a related-party inclusion/exclusion call. Both are
  *treatment* determinations; plain sum membership isn't.
- **"EBITDA" used without a local definition splits into two genuinely
  different situations, confirmed by reading the source credit agreements
  directly — only one of them is safe to fix in code.** Some borrowers'
  agreements spell it out inline ("EBITDA (Выручка за вычетом
  Операционных расходов)" — P5); the *shape* of resolving that mattered as
  much as resolving it at all — the transaction classifier was finding
  zero matches until `_split_compound`'s output stopped retaining the
  "EBITDA ... как" preamble as part of the category label (fixed:
  `_strip_defined_as_prefix`, confirmed via `match_category_by_text` no
  longer needing to match against a polluted label). Other borrowers'
  agreements (P3, P7) just say "EBITDA" with **no definition anywhere in
  the document at all** — confirmed by grepping the full agreement text,
  not assumed. Ground truth confirms a real number was expected there
  (P3's actual=1.71), meaning some standard-definition fallback is
  probably the intended answer — but deliberately **not implemented**:
  auto-substituting a textbook EBITDA formula the source document never
  states is arguably the model contributing outside knowledge rather than
  extracting a fact, which cuts close to the case's core "LLM extracts,
  code decides" rule. Flagged for a separate, careful discussion before
  ever touching it, not silently fixed.
- **LLM categorization non-determinism is large enough to dominate the
  self-score, not just add noise around the edges.** Two runs against
  identical code and prompts scored 0.61 and 0.50 on the public dataset.
  Self-consistency voting (3-vote majority) was the obvious mitigation and
  measurably made it *worse* — see "Design decisions" above for why. This
  remains open; see the code-review findings' note on splitting the single
  large categorization call into a cheap deterministic pre-filter plus a
  narrower LLM pass for the genuinely ambiguous remainder, which hasn't
  been attempted yet.
- **Even a well-scoped OpenAI account can run out of credits mid-session,
  and the error is easy to misread as a transient rate limit.** Both
  surface as HTTP 429; `insufficient_quota`/`credit_balance_exhausted` in
  the error body is the tell that no amount of retrying will fix it. Cost
  real development time before being identified — see "Known sources of
  instability" above.

### Block 3 findings — the `financial_notes` discovery (2026-08-08, later session)

The single largest finding of the project, found by chasing an
organizer-confirmed hint ("the FX rate is in the files") with the same
method as everything else here — read the actual documents, don't reason
abstractly — rather than assuming it was already ruled out.

- **A PDF letter-spacing rendering artifact hid one entire document kind
  for all 12 scenarios simultaneously.** Each scenario has its own
  "Примечания к финансовой отчётности" (Notes to Financial Statements) —
  an audit-firm-issued document with the *same* "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ
  КОВЕНАНТОВ" section shape as a standalone audit report. Its account
  number renders as `A C C - 7 8 0 3` (a space between every character —
  the same class of artifact already documented for stylized headers like
  "Д О ГО В О Р", but never applied to `accounts.py`'s regex match).
  `ACCOUNT_TOKEN_RE`'s exact `\bACC-\d+\b` match can't see through that,
  so all 12 of these documents sat in `unmatched_documents`, completely
  invisible, for the entire project until this was found. Fixed by
  normalizing whitespace out of an `"A C C - ..."`-shaped span *before*
  the same exact-token regex runs — not a new or looser matching
  mechanism, a repaired input to the existing one (`accounts.py`'s
  `_normalize_spaced_account_tokens`).
- **What was hiding inside, once found:** P8's true amount for its
  dirty ledger row (`TXN-P8-0031` → $884,204.16) *and* an off-ledger
  severance obligation ($918,447.52) — both needed for 6.1, both in one
  document; P3's FX settlement rate; P7's own treasury memo similarly held
  `TXN-P7-0033`'s true amount; B4/P1's findings that a specific transaction
  should be excluded from the covenant period entirely (not
  recategorized — a genuinely new finding *shape*, "исключена из
  ковенантного периода"); P10/P2's ordinary reclassifications. Confirmed
  with the strongest evidence available short of the actual answer key:
  P8's 6.1, computed with all of the above wired in, came out
  **actual=4,221,314.95 / BREACH — bit-exact against `ground_truth.json`**
  (`2,418,663.27` real payroll txn + `884,204.16` patched txn +
  `918,447.52` severance fact, exactly).
- **One related but separate document was found the same evening by a
  different route: P5's Group-parent consolidated report** (Sarybel
  Energy Holding JSC — English-language, a different legal entity from
  the borrower). Not an ACC-token case at all — a Group-parent report
  will never mention the subsidiary's own account number. Deliberately
  **not** generalized into a company-name similarity matcher (this
  dataset has a confirmed, deliberate near-duplicate-entity-name trap —
  see `accounts.py`'s decoy-subaccount docstring). Fixed narrowly instead:
  an exact match of the scenario's own *already-verified* borrower name
  (from its own ACC-matched credit agreement, never guessed from the
  candidate document's own name) next to explicit "this document names
  the borrower as its own subsidiary/segment" language
  (`segment_linking.py`). A real, confirmed false-positive source was
  caught and excluded before this shipped: "Kazakhstan JSC", a fragment
  of every document's "Halyk Bank of Kazakhstan JSC" letterhead, extracted
  by the same best-effort company-name regex as every real borrower name —
  trusting it would have made the matcher fire near-universally.
- **Extracting the facts wasn't enough on its own — a whole class of
  "extracted but never consumed" fact existed and had to be wired into
  `calculation/` by shape, not by scenario.** Three genuinely different
  shapes, one underlying mechanism each: a plain off-ledger figure adds
  into a matching covenant side's sum via the same `match_category_by_text`
  stem-overlap machinery reclassifications already used
  (`AuditExtractionResult.other_facts`); a dirty-row correction patches
  `Transaction.amount` by `txn_id` *before* categorization ever runs, not
  after (`_apply_amount_corrections`, logged explicitly per patch); a
  period-exclusion finding is a new `AuditReclassification.action` value
  (`"exclude_from_period"`, alongside the original `"recategorize"` and a
  new `"no_change"` for a reclassification explicitly considered and
  rejected — must never be treated as a real one). All three routed
  through `extract_audit_facts` — the *same* extractor and schema
  `audit_report` already used, extended with two new list fields, not a
  parallel pipeline — since `financial_notes` and `treasury_memo` turned
  out to mix all three finding shapes in one document exactly like a
  standalone audit report does.

## Code review: places that could fail the same way the epsilon bug did

Requested explicitly after the epsilon bug was found and fixed: a pass over
`covenant_agent/` looking for *other* silent fallbacks, defaults, or
divisions that could produce a confidently-wrong answer instead of a loud
failure — specifically ones that haven't shown up yet because the public
dataset hasn't happened to trigger them, not because they're provably safe.
Risk-rated for triage. **Findings 1–4 (High) are now fixed**, each with its
own offline test — see the fix descriptions inline below; 5–12
(Medium/Low) are still open findings, not fixes, by explicit instruction
not to keep tuning defaults on a small sample. A second wave of findings
from a later session (the `financial_notes` discovery) is appended after
this original list.

**High — fixed:**

1. **No per-scenario exception isolation in Block 2 (`extraction/pipeline.py`)
   or Block 3a (`linking/pipeline.py`) — fixed.** `extraction/pipeline.py`
   used to catch only `ExtractionError` around each of its four extraction
   calls; any other exception (a pydantic quirk, an unanticipated SDK
   exception type, a bug in our own matching code) propagated uncaught
   through the entire `extract_all_facts` loop, aborting all 12 scenarios
   at once — `linking/pipeline.py`'s call to `categorize_transactions`
   wasn't wrapped in *any* try/except, so it was even more exposed. Not
   hypothetical: an `openai.APIConnectionError` did exactly this before it
   was added to the retry list (see "Design decisions" above). Both loops
   now catch `Exception` per-scenario, log clearly, degrade to an
   empty/safe result for that scenario, and continue — see
   `test_extraction_pipeline.py`'s `ExtractAllFactsBatchResilienceTest` and
   `test_linking_pipeline.py`.
2. **No incremental checkpointing — fixed.** `save_scenario_facts` (Block 2)
   used to be called once, after the full 12-scenario loop completed;
   Block 3a had no save/cache mechanism at all. Combined with finding #1: a
   crash on scenario 9 of 12 lost scenarios 1–8's real, paid-for API calls,
   not just scenario 9's. Now saves after *every* scenario, not once at
   the end — `test_progress_is_saved_after_every_scenario_not_once_at_the_end`.
3. **Silent zero for `aggregate_amount`/`max_single_component`/`other`
   when categorization finds nothing — fixed.** The `InsufficientDataError`
   fix originally only covered `ratio` denominators. The other three
   metric types used to get a plain `actual=0.0` on zero matched
   transactions with no signal at all — and unlike the ratio case, `0.0`
   didn't look obviously wrong: it read as a perfectly plausible "no capex
   this year" / "no overhead this quarter" answer, for a max-direction
   covenant that's `COMPLIANT`. Same underlying bug class as the epsilon
   fallback, just harder to catch by inspection because it didn't produce
   an absurd number. Now raises the same `InsufficientDataError` for all
   four metric types — `test_aggregate_amount_zero_matches_raises_insufficient_data`,
   `test_max_single_component_all_zero_raises_insufficient_data`.
4. **`ingestion/template.py`'s `template.get("answers", {})` had no
   structural validation — fixed.** If the private dataset's
   `submission_template.json` had even a slightly different top-level
   shape, `required_scenario_ids` would silently return `[]`, and the
   entire pipeline would process zero scenarios with no exception
   anywhere — the failure would only become visible at the very end,
   looking at an empty result, with the least time available to react.
   `load_template` now validates structure explicitly and raises
   `TemplateValidationError` with a message naming the missing piece — see
   `test_template_validation.py`.

**Medium — narrower or lower-probability, worth knowing about:**

5. **A residual, narrower version of the epsilon bug still exists in
   `_safe_ratio`.** It only runs once `InsufficientDataError` has already
   ruled out zero matched transactions, but if a denominator's *real*,
   counted transactions happen to net to ~$0 (e.g. a charge and its exact
   reversal landing in the same category), the epsilon path still
   triggers. Narrow — needs near-perfect cancellation — but the same
   failure mode.
6. **`match_category_by_text`'s 0.35 threshold doesn't distinguish "barely
   cleared" from "exact match."** A reclassification mapped at 0.36
   confidence is trusted identically to one at 1.0. The stopword list
   (`_STOPWORD_STEMS`) that makes this threshold usable at all was hand-built
   from one confirmed false-positive (`"расходы"` in `"Процентные
   расходы"`) — it may not cover whatever generic financial vocabulary the
   private dataset's reclassification reports happen to use.
7. **`in_period` silently assumes no lower bound when 2a fails to extract
   an explicit `period_start` and nothing can be inferred** (see
   `_effective_period_start` for the one case it *can* now infer — "N-й
   квартал, оканчивающийся ДАТА" — added after B4's Q4-only revenue test
   was found silently summing the whole year; confirmed necessary, see
   Block 3 findings below). Outside that one recognized shape, the
   assumption is still silent and unlogged — if a private-dataset covenant
   is genuinely period-scoped in some other phrasing extraction missed,
   there's no signal that the "no lower bound" default fired.
8. **`FALLBACK_STATUS = "COMPLIANT"` is a documented coin-flip, not a
   considered guess — worth remembering it's a strategic choice, not a
   neutral one.** If the private dataset's true status distribution skews
   toward `BREACH`, this default loses points systematically every time it
   fires, rather than being a wash. **Revisited with data**, not just
   reasoned about: the public `ground_truth.json` splits 19/17
   COMPLIANT/BREACH (52.8%/47.2%) across all 36 cells — close to even, so
   `COMPLIANT` isn't free points but isn't a clear loser either. Among the
   ~6 cells where the pipeline actually fell back on a real run, the split
   was an exact 3/3 (a sample far too small to justify anything smarter
   than the population-level default). Decision: keep `COMPLIANT` — no
   evidence supports deviating from it.
9. **A scenario with no KYC document at all produces a "legitimate-looking"
   zero for related-party-based covenants, indistinguishable from a
   confirmed zero related parties.** Found while investigating finding #8:
   P2's covenant 6.3 and P6's covenant 6.1 both have no KYC dossier
   anywhere in the corpus for that scenario, so `resolve_related_parties`
   correctly returns an empty map, the related-party-exempt branch of
   `InsufficientDataError` (see `calculation/formulas.py`) treats that as a
   genuine, honest zero, and `actual=0.0` comes out looking like a normal,
   confident answer rather than a fallback. Both cells are `BREACH` in
   ground truth — the honest zero is wrong for both. The deeper issue:
   полное отсутствие KYC-документа у сценария — не обязательно значит
   "связанных сторон нет", может значить "документ не нашёлся/не
   распарсился", и это неотличимо друг от друга в текущей логике. Not
   fixed — documented here so it isn't forgotten, per explicit instruction
   to prioritize categorization-stability work (below) over further
   default-tuning on a 6-cell sample.

**Low — real, but small blast radius:**

10. **`llm_client._log_call` isn't wrapped in try/except.** A disk-write
    failure (permissions, full disk) there would discard an already-successful,
    already-parsed API response purely because the *logging* side-effect
    failed.
11. **`extraction/cache.py` load/save has no format validation** — a
    stale or hand-edited `scenario_facts.json` fails with a raw
    `KeyError`/pydantic `ValidationError`, not a clear message pointing at
    the cache file.
12. **The related-party sum silently drops any transaction recorded with
    a non-negative amount** (`if txn.amount >= 0: continue`), even to a
    known related party — correct under the ledger's stated sign
    convention, but undefended against a data-entry sign anomaly.

### Second wave — the `financial_notes` discovery (2026-08-08, later session)

Found while chasing a genuinely different question — the organizers
confirmed in the hackathon chat that a FX conversion rate *does* exist
somewhere in the dataset — using the same method as everything else in
this section: read the actual files, don't reason abstractly. See
"Findings" below for the full story (a PDF letter-spacing artifact hid a
whole undiscovered document kind for all 12 scenarios at once). Findings
from wiring the newly-discovered documents' facts into calculation:

13. **`other_facts` matching is scenario-wide, not exclusivity-checked
    across covenants.** `_resolve_side_sum` matches every one of a
    scenario's `other_facts` against every covenant/role it's asked to
    resolve, independently each time — there's no check that a given fact
    is claimed by at most one covenant side scenario-wide. Confirmed safe
    on the one real case exercised so far (P8's severance fact only
    clears `match_category_by_text`'s threshold against 6.1's own
    category, verified by hand) but not defended against a private-dataset
    fact whose vocabulary happens to overlap two different covenants'
    descriptions — it would silently double-count into both.
14. **`other_facts` are only wired into `_resolve_side_sum`, not
    `max_single_component`'s per-component sums.** `compute_metric`'s
    `max_single_component` branch calls `_category_signed_sum` directly,
    bypassing `_resolve_side_sum` entirely — an off-ledger fact that
    should count toward one named component of a max-single-component
    test (B1/P10's "Individual Overhead Line Ceiling" shape) currently
    can't reach it. Not confirmed as a live gap on the public dataset (no
    max_single_component covenant has needed an other_fact so far), but a
    known, undefended gap in coverage.
15. **FX conversion is scoped to P3 only, deliberately, not implemented
    generally.** Re-checked all 12 scenarios' own `financial_notes`
    documents for an FX rate disclosure (not just P3's) — only P3's has
    one (`Примечание 9`, the Rheinland Katalyse Service GmbH settlement:
    72,146.75 EUR → $83,690.23). The other 11 scenarios' EUR-denominated
    ledger rows (1–3 each, mostly decoy-flavored by description) have no
    disclosed rate anywhere in their own documents — non-USD transactions
    are still filtered out entirely (`test_non_usd_transactions_excluded`),
    not converted, and that remains the right call for every scenario
    except possibly P3, where a rate now exists but hasn't been applied.
16. **P8's covenant 6.3 still falls back** (`ratio denominator matched 0
    transactions` — the revenue category, not anything this session
    touched) even after the `financial_notes` fix resolved 6.1 to a
    bit-exact ground-truth match. Confirms the `financial_notes` fix and
    the categorization-noise problem are separate issues, not the same
    one — fixing document linking didn't and wasn't expected to fix
    transaction-categorization misses.
17. **Two designed-but-not-yet-built extensions, flagged so they aren't
    silently assumed done:** "Unrestricted Subsidiary" style covenants
    (P9's 6.1 — a KYC/ownership-style question phrased without any of
    `is_related_party_text`'s trigger words, so it's currently routed to
    the ordinary transaction classifier instead of a related-party-style
    resolution) and springing/conditional covenants (P3's 6.1 — "applies
    only if financing proceeds exceed $X", which `CovenantClause` has no
    field to represent, so it's always evaluated unconditionally). Design
    docs for both, not implementations, are below.

### Third wave — the 9-cell forensic review and fix batch (2026-08-09)

After two full 12-scenario dev-profile runs (self-score 0.6557 and 0.6283,
mean 0.6420 vs. the 0.6237 post-`financial_notes`-fix baseline — not
directly comparable to earlier 0.560/0.6149 figures, which predate that
fix), 9 cells were wrong identically in both runs — not noise, the same
class of "systematic, not random" signal documented earlier in this file.
Each was traced offline, log by log (no new API calls), to a specific root
cause before any fix was written. Full per-cell evidence (2a input,
categorization output, ledger rows, exact numbers) lives in the session
transcript; summarized findings and what got fixed:

- **B1 6.1, P2 6.2** — plain transaction-categorization misses on
  textbook-clean descriptions ("Plant operating and maintenance expenses",
  "Purchase of blast freezer equipment") that the prompt's own existing
  guidance already covers. Addressed by adding concrete worked few-shot
  examples to `transaction_categorization.py`'s `SYSTEM_PROMPT` (a
  different lever than the abstract rule that was already present and
  still got missed) plus an explicit "re-check before unclassified"
  instruction. **Checked whether the already-designed rule-based
  pre-filter (below) could close this more cheaply — it cannot**: both
  confirmed misses are cross-language (English transaction description,
  Russian category description), and the prefilter's stem-overlap
  machinery is language-blind by construction (`"опера"` and `"opera"`
  share zero characters as Unicode strings, regardless of what they mean).
  Not fixed by, or evidence against, the prefilter design itself — it
  remains valid for same-language matches, just doesn't reach this case.
- **P10 6.3** — a real, code-level bug, not a data limitation: a
  `max_single_component` sub-category's description was built as
  `f"{clause.metric_name}: {label}"`, so a *payroll* component inherited
  the word "выручка" from its own covenant's unrelated metric name,
  scoring a false-positive 1.0 sibling-borrow match against a different
  covenant's real revenue category. Fixed in `categories.py`
  (`_specs_for_clause`) — component descriptions are now just the clean
  label.
- **P4 6.1** — a ratio numerator matched zero transactions on both of its
  sub-specs; `compute_metric` only ever checked the *denominator* for
  `InsufficientDataError`, so the miss silently became a plausible-looking
  `actual=0.0` (BREACH against a min threshold) instead of a loud
  fallback. Fixed in `formulas.py` — the same zero-count check now applies
  to the numerator too.
- **P3 6.1, P7 6.1** — both covenants' EBITDA denominator is the bare,
  undefined acronym "EBITDA", confirmed via full-text search to appear
  nowhere else in either source document (2a's own prompt already
  instructs decomposing an undefined acronym into "revenue minus operating
  expenses", and still didn't in these two cases). Added a code-level
  backstop in `categories.py`: a denominator/numerator that reduces to
  bare "EBITDA" after splitting is now deterministically expanded into the
  same "Выручка"/"Операционных расходов" convention every other
  EBITDA-based covenant in this dataset already uses. P3 6.1 separately
  involves an unapplied springing-covenant condition (Task C, still not
  implemented — see below) and P3's own unapplied FX rate (see finding 15
  above); the bare-EBITDA fix addresses only the denominator piece.
- **P5 6.1** — two independent `categories.py`/`formulas.py` bugs, not
  one: (1) `_split_compound`'s additive-connector regex mistook a
  descriptive continuation clause ("...определяемые по консолидированной
  отчётности... **и включающие** затраты всех участников Группы") for a
  genuine two-concept list, spuriously splitting the capex numerator and
  letting an unrelated operating-cost transaction into it — narrowed to
  exclude continuations starting with `включа-`/`определя-`/etc.; (2) when
  the whole "revenue net of opex" denominator missed, sibling-borrow only
  ever found ONE sibling (revenue) using the combined description text,
  silently dropping the opex deduction entirely — `_borrow_sibling_category_sum`
  now borrows each part of a multi-spec role separately, so revenue and
  opex can each find their own donor.
- **P10 6.2** — a schema/extraction gap, not a categorization bug: the
  clause measures "Выручка за вычетом наибольшей из величин..." (revenue
  *minus* the larger of two components), but `CovenantClause` had no field
  to represent the "other amount," so `max_single_component` could only
  ever return the bare `max()` of the named components. Added
  `net_against_description` to the schema, updated 2a's prompt to
  populate it, and wired `formulas.py` to subtract the largest component
  from that side's resolved sum (reusing the same sibling-borrow/rescue/
  other_facts/addback machinery every ratio side already gets).
- **P2 6.3, P6 6.1 — left as open questions, not fixed.** Both traced to a
  likely structural data limitation rather than a code or prompt bug:
  - *P2 6.3*: ground truth cites `TXN-P2-0012` (Zhetysu Capital Partners
    LLP) as a related-party payment. P2's own KYC dossier is pure
    boilerplate (identification/sanctions-screening process language, zero
    named counterparties or ownership percentages) — checked every other
    document in P2's corpus for a mention of that entity; the only hit is
    an unrelated coincidence (the audit firm is also named "Zhetysu Audit &
    Advisory LLP"). No document our pipeline reads discloses this
    relationship.
  - *P6 6.1*: ground truth cites `TXN-P6-0040` as a related-party payment,
    but P6 has **no `kyc_dossier` document at all** in the 200-document
    corpus — confirmed by grepping every cached PDF mentioning `ACC-7806`;
    the only other two are a correctly-excluded draft audit workpaper and
    a correctly-excluded superseded 2024 credit agreement, neither a KYC
    document.

  Since related-party determination is deliberately KYC-dossier-only by
  design (matching each covenant's own stated rule — "определяется в
  соответствии с... досье Заёмщика по идентификации клиента"), there is no
  code or prompt fix available for either cell without inventing a
  relationship the given documents don't disclose. **Worth raising with
  the organizers directly** (there's apparently a channel for exactly this)
  rather than guessing — it's possible the private dataset's KYC coverage
  is more complete, or that the public dataset's ground truth relies on
  information genuinely outside the provided corpus for these two cells
  specifically.

## Draft design: rule-based pre-filter for transaction categorization (not implemented)

Design only — no code changed for this section. Written now, while OpenAI
credits are unavailable, so implementation can start immediately once they're
back, instead of starting with design. Priority context: per-run self-score
variance (0.50–0.61 across identical-code runs) is a bigger source of error
than any remaining fallback-default tuning, and its root cause is
transaction-categorization non-determinism (see `transaction_categorization.py`'s
docstring on the reverted self-consistency-voting experiment). The plan
agreed with the user: once credits return, run the Step 2 methodology (5×2
self-score comparison, current vs. the 0.6149-scoring baseline) before
touching categorization further, then use *this* design if that comparison
says the categorization approach itself needs to change, not just the prompt.

**What this is actually trying to reduce.** Not API cost — fewer, harder
decisions for the LLM to make. Right now every transaction in a scenario,
easy or hard, goes through the same single LLM call and inherits whatever
noise that call has on that particular run. If the transactions with an
unambiguous, rule-decidable answer are pulled out and decided in code
instead, two things happen: the LLM only ever sees the genuinely ambiguous
remainder (its majority failure mode — flip-flopping on a call that was
never confident either way — has less surface area to occur on), and the
rule-based portion of the classification becomes perfectly reproducible
run-to-run, which is itself a real reduction in the pipeline's total
variance even before any prompt tuning.

**Which of the three candidate signal types are actually in scope here —
this matters more than it looks.** The original framing (exact keyword
matches, amount patterns, known KYC counterparties) suggests three
independent rule families. In practice, checking `derive_category_specs`
(categories.py) and how it's used shows only one of the three is actually
this call's job:

- **Known KYC counterparties are already fully rule-based and already
  excluded from this call.** `is_related_party_text()` routes any
  related-party-flavored covenant side to `related_parties.py` instead of
  generating a `CategorySpec` for it at all (see categories.py's module
  docstring) — `transaction_categorization.py` never sees related-party
  membership as a question it's supposed to answer. There is nothing to
  prefilter here; the design decision was already made correctly one layer
  up. Same story for reclassifications — `reclassification_linking.py`
  already resolves counterparty+amount(+date) matches in code, independent
  of this call. **So a prefilter that reuses the KYC counterparty list
  would be solving a problem that doesn't exist at this call site** — worth
  stating explicitly so it isn't "rediscovered" and rebuilt as
  dead-weight duplication later.
- **Amount patterns are a real but secondary signal, not a standalone
  classifier.** The existing prompt already says as much ("amount sign is
  a supporting clue... but the description is the primary signal") for a
  reason confirmed on this dataset: the ledger deliberately contains
  positive-amount non-revenue rows (refunds, rebates, lease incentives)
  specifically to punish sign-based shortcuts. Amount belongs in the
  design only as a *downgrade* trigger (see below), never as grounds to
  auto-accept a match on its own.
- **Keyword/description matching against each category's own text is the
  one signal that's both genuinely available and actually does this call's
  job.** This is where the design effort goes.

**Two-tier split, not a single yes/no rule.**

1. *Auto-classify without an LLM call* — a transaction's description
   clears a **high, conservative** stem-overlap threshold against exactly
   one category's own description, using the same stemming machinery
   already built and validated for `match_category_by_text`
   (`_stems()`/`_STOPWORD_STEMS`/`_STEM_CHARS` in categories.py — reused,
   not reinvented), and does **not** also hit a decoy-keyword stem. The
   decoy list isn't new — it's the existing prompt's own enumerated decoys
   (marketing, insurance, tax, refund, rebate, credit, return, incentive,
   reimbursement, marketing co-op funding, overbilling correction) lifted
   into an actual code constant so the prefilter and the prompt stay in
   sync instead of drifting apart. The threshold here must be well above
   `CATEGORY_MATCH_THRESHOLD` (0.35) — that number was tuned for "worth
   showing as a candidate," a much weaker bar than "trust it with zero
   downstream check," which is what auto-classification actually needs.
   A plausible starting point is "every one of the category's own
   non-stopword content stems is present in the transaction description"
   (i.e. overlap = 1.0 relative to the category side, not the transaction
   side) — genuinely conservative, tuned down from there only against
   measured evidence, the same posture as the epsilon-bug fix.
2. *Auto-classify as `unclassified` without an LLM call* — a transaction's
   description hits a decoy stem strongly and has **no** stem overlap with
   any category at all. This is the "most transactions in this ledger
   match none of the categories" case the prompt already describes as the
   expected default outcome — safe to shortcut in code because the answer
   ("none of these") is symmetric with what the LLM is already told to
   default to when unsure.
3. *Everything else goes to the LLM unchanged* — no signal, conflicting
   signal (decoy stem AND category stem both present), or a tie between
   two categories' overlap scores. Ambiguity is not resolved by picking
   the higher score in code; that's exactly the judgment call this
   pre-filter is designed to *not* make outside the LLM.

**Amount sign's actual role: a downgrade trigger, not a promoter.** If a
transaction would otherwise clear the auto-classify threshold for a
category whose role/description implies a sign expectation (e.g. a
revenue-flavored category expecting inflows) but the transaction's amount
has the opposite sign, drop it back into tier 3 (send to the LLM) instead
of auto-accepting. Never the reverse — matching sign never promotes a
weak text match into an auto-accept on its own.

**Implementation sketch (for when credits return):**
- New module `covenant_agent/linking/category_prefilter.py`, exporting
  `prefilter_transactions(transactions, category_specs) -> tuple[dict[str, str], list[Transaction]]`
  — confident auto-classifications (including confident `unclassified`
  calls) plus the residual list still needing the LLM.
- `categorize_transactions` (`transaction_categorization.py`) calls it
  first, sends only the residual list to `extract_structured(...)`, then
  merges the two dicts. Logged per scenario (e.g. "42/60 auto-classified
  by rule, 18 sent to LLM") so the auto-accept rate is visible per run.
- New constants: an `AUTO_ACCEPT_THRESHOLD` well above
  `CATEGORY_MATCH_THRESHOLD`, and a `_DECOY_STEMS` set transcribed from the
  existing prompt text (single source of truth going forward — the prompt
  should probably generate its decoy-list prose from the same constant,
  not duplicate it by hand).
- Offline tests, same style as the rest of this repo (no API key needed):
  synthetic transactions with an unambiguous exact-concept description
  (must auto-classify), synthetic decoy-only descriptions (must
  auto-classify to `unclassified`), and synthetic genuinely-ambiguous ones
  (must land in the residual list, not be auto-decided either way).

**How this gets validated, not just shipped.** Same Step 2 machinery
already agreed on: 5 self-scorer runs with the prefilter enabled vs. 5
without, compare means (not single runs — this dataset's per-run variance
is exactly why that discipline exists). Two independent things to check,
not one: whether the mean score improves, and whether the *variance*
across the 5 runs shrinks (the actual goal — a prefilter that keeps the
same mean but with less run-to-run spread is still a real win, since it
means the LLM is being asked fewer genuinely-uncertain questions). The
auto-accept rate itself should be exactly reproducible run-to-run (it's
pure code, no LLM call) — that's a cheap free sanity check that the
prefilter is wired correctly, checkable before even looking at scores.
Also worth re-testing self-consistency voting *on the post-prefilter
residual only* — the earlier revert was measured on the full,
mostly-easy transaction mix, where voting amplified conservative bias on
easy cases; voting on a residual that's ambiguous *by construction* might
behave differently, but that's a hypothesis for after this ships, not
part of this design.

## Draft design: "Unrestricted Subsidiary" routing (Task B — not implemented)

Design only, per explicit instruction to get this approved before writing
any code. P9's covenant 6.1 ("Максимальная доля активов, переданных
неограниченным дочерним организациям") caps the value of capital assets
transferred to "Unrestricted Subsidiaries" at 0.15x of total capex. The
numerator names a specific counterparty-identity question — *which*
counterparties are designated Unrestricted Subsidiaries — but
`is_related_party_text()`'s trigger regex (`связанн|аффилиров|related.?
part|affiliat`) doesn't match "дочерним" at all, confirmed by direct test
(`is_related_party_text(...)` on the real numerator text returns `False`).
So this currently routes to the ordinary transaction-description
classifier instead of any identity-based resolution — structurally the
wrong tool, since no ledger row's free-text description would ever say
"transferred to an Unrestricted Subsidiary".

**A checked assumption changes the recommendation here.** The obvious fix
looked like "add a `дочерн`-style trigger and a new
`resolve_unrestricted_subsidiaries()` mirroring `related_parties.py`'s
shape." Before designing that, checked whether *any* document in P9's
corpus (KYC dossier, `financial_notes`, credit agreement — all three, full
text) actually discloses which counterparty, if any, carries the
"Unrestricted Subsidiary" designation. **None does.** The term appears
exactly three times in the whole corpus, all three inside the credit
agreement's own covenant-clause *definition* of the concept — never as an
actual disclosure naming a real entity. This is a materially different
situation from related-party resolution, where every scenario's KYC
dossier (when one exists) *does* carry a real ownership table to compare
against a threshold.

**Two options, ranked by how well-supported they are by actual data:**

1. **Recommended for now — extend the existing zero-exemption, don't build
   a parallel resolution mechanism.** `InsufficientDataError`'s
   related-party exemption already encodes the principle "a genuine zero
   for an identity-based side is a normal business fact, not a
   categorization miss, when we have no positive evidence otherwise" (see
   `_needs_data_check`). Recognizing "Unrestricted Subsidiary"-flavored
   text (a narrow, specific phrase match — not broadening
   `is_related_party_text` itself, which would risk misrouting unrelated
   "дочерн" mentions elsewhere in the private dataset) as exempt the same
   way gets the *currently correct* answer (a numerator of 0.0, since
   nothing discloses a transfer to any such entity, is `COMPLIANT` against
   a 0.15x cap) without inventing a resolution path that has zero real
   examples to validate against. Small, safe, matches the same logging
   discipline as the existing exemption (`calculation_notes` states
   plainly that this is an absence-of-disclosure zero, not a confirmed
   one — the same honesty gap already flagged for the KYC case, finding
   #9 above, applies here too and should reuse the same framing).
2. **If the private dataset actually discloses Unrestricted Subsidiary
   designations somewhere** (a KYC dossier field, a credit-agreement
   schedule, anything) — build the fuller mechanism: extend
   `RelatedPartyDisclosure` (or a sibling schema) with an
   `explicitly_labeled_unrestricted_subsidiary: bool` field (mirroring
   `explicitly_labeled_related_party`'s "literal reading of the source
   text, never a computed judgment" pattern), add a narrow
   `is_unrestricted_subsidiary_text()` matcher (specific to "неограниченн
   ... дочерн" / "unrestricted subsidiar", not a bare "дочерн" check), and
   a `resolve_unrestricted_subsidiaries()` resolution function alongside
   `related_parties.py`. Only worth building once there's a real
   disclosure to extract and test against — building it against zero
   examples risks the same kind of untested-assumption bug this project
   has twice found the hard way this session (epsilon bug, ACC-token
   letter-spacing).

**Recommendation: do option 1 now (cheap, safe, matches current data);
revisit option 2 only if a private-dataset document actually discloses
this, using the same "read the files first" discipline that found the
`financial_notes` documents.**

## Draft design: springing/conditional covenants (Task C — not implemented)

Design only, same reasoning: get this approved before writing code. P3's
covenant 6.1 ("Springing Drawdown Leverage Test") states the ratio test
"применяется... только при условии, что совокупные поступления по
финансированию превышают $4,000,000.00" — the covenant only binds if a
condition (itself a computable aggregate-amount test) is met.
`CovenantClause` has no field to represent this at all, so the pipeline
always evaluates the ratio unconditionally. Not currently a confirmed
self-score error (P3's own condition happens to be met — its financing
proceeds do exceed $4M, so the covenant applies and the unconditional
evaluation gives the right answer by coincidence of this particular
scenario), but a real schema-completeness gap: a private-dataset scenario
whose condition genuinely *isn't* met would still get evaluated as if it
always applies.

**Proposed schema extension** — the condition itself has the same shape as
a covenant's own test (a description, a threshold, a direction), so
reuse that shape rather than inventing a new one:

```python
class CovenantClause(BaseModel):
    ...
    applicability_condition_description: Optional[str] = Field(
        description="If this covenant only applies when some other "
        "condition is met (a 'springing' covenant), describe that "
        "condition's own measurable test here, in the source language "
        "(e.g. 'совокупные поступления по финансированию'). Null if the "
        "covenant applies unconditionally — the normal case."
    )
    applicability_condition_threshold: Optional[float] = ...
    applicability_condition_direction: Optional[Literal["max", "min"]] = ...
```

Extraction (2a) fills these from the source text exactly like the main
threshold/direction fields — pure extraction, no judgment about whether
the condition is *currently* met (that's computed from the ledger, in
code, same as everything else).

**Proposed calculation flow** — in `compute_metric` or a thin wrapper
around it: if `applicability_condition_description` is set, run the
*same* `aggregate_amount`-style resolution (`_resolve_side_sum`) against
it first, compare to `applicability_condition_threshold` via the same
`compare_to_threshold` logic already used for the main test, and only
evaluate/report the main ratio if the condition says the covenant
applies.

**The genuinely open design question: what does a non-applicable covenant
report as `status`?** `submission_template.json`/`ground_truth.json` only
ever show `COMPLIANT`/`BREACH`, no "not applicable" option, and there's no
confirmed real example (in the public dataset) of a springing covenant
whose condition isn't met to check against. Leaning `COMPLIANT` (a
covenant that doesn't bind can't be breached) with an explicit
`calculation_notes` entry stating the condition wasn't met and why — the
same transparent-fallback discipline as `FALLBACK_STATUS`, not a silent
guess — but this is exactly the kind of assumption worth confirming
before implementing, not after.

**Cost/benefit honestly:** moderate implementation cost (a second
`compute_metric`-shaped evaluation per springing covenant), for a shape
confirmed exactly once in 12 public scenarios (P3 only), with the
riskiest part (the non-applicable-status question) unvalidatable against
any real example either way. Worth doing if time allows post-submission-
critical-path work; not worth displacing higher-confidence fixes for.

## LLM provider portability — what it'd take to switch off OpenAI

Assessed, not implemented, in case OpenAI quota becomes a blocker again on
the day (as it did during development — see "Known sources of instability").

**The abstraction holds up well.** Every call into an LLM anywhere in this
codebase goes through exactly one function, `llm_client.extract_structured()`
— `covenant_extraction.py`, `fact_extraction.py`, and
`transaction_categorization.py` (the only three callers) pass it
`instructions` (a plain string), `input_text` (a plain string), a
`response_model` (a plain pydantic class), and a `ModelConfig`, and get
back `(parsed_instance, raw_dict)`. None of them import `openai`, catch an
OpenAI-specific exception, or know the API shape underneath. Swapping
providers means rewriting the *inside* of `llm_client.py` — roughly the
`get_client()` function and the body of `extract_structured()` — and
nothing else in the repo.

**What would actually need to change**, if switching to the Anthropic API:

- `from openai import OpenAI, ...` → `from anthropic import Anthropic, ...`;
  `get_client()` constructs an `Anthropic(api_key=...)` instead, reading
  `ANTHROPIC_API_KEY`.
- The request/response mechanics differ and would need rewriting, not just
  renaming: OpenAI's Responses API takes `text_format=<pydantic model>`
  directly; Anthropic's structured-output path goes through forced
  tool-use (define a tool whose input schema is the pydantic model's JSON
  schema, force `tool_choice`, and parse the model's `tool_use` block back
  into the pydantic type — the `anthropic` SDK has helpers for this, but
  it's a different call shape, not a drop-in argument swap).
- Exception types are named similarly but come from a different module
  (`anthropic.RateLimitError`, `anthropic.APIConnectionError`,
  `anthropic.APIStatusError` mirror the OpenAI names closely enough that
  the *retry structure* in `extract_structured` — attempt loop, backoff,
  which exceptions are retried vs. re-raised — carries over essentially
  unchanged; only the `except` clause's imports need updating).
- The `_PROFILE_DEFAULTS` model names change (e.g. `claude-sonnet-5` in
  place of `gpt-5.4-mini`) — already isolated behind
  environment-variable-overridable constants (`COVENANT_EXTRACTION_MODEL` /
  `TRANSACTION_CATEGORIZATION_MODEL` / `FACT_EXTRACTION_MODEL`), a
  same-shape change either way.
- Prompts themselves need no changes — they're plain English task
  descriptions with no OpenAI-specific formatting conventions.

**Net assessment: genuinely easy, on the order of an hour of focused work**
to reimplement `extract_structured`/`get_client` against `anthropic` and
re-verify against a couple of scenarios — not a pipeline rewrite. If OpenAI
becomes unusable on the day: install `anthropic`, rewrite those two
functions in `llm_client.py` against it, set `ANTHROPIC_API_KEY`, and
re-point the `_PROFILE_DEFAULTS` model names at Claude model names. Every
other file in the repo is unaffected by construction.
