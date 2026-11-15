# EU261 Claim Intake Agent

An AI agent that takes a passenger's "my flight was delayed" message and turns it into a **verified decision, a compensation amount and a draft claim letter**, ready for a claims handler to approve.

Built with **Python 3.12 · LangGraph · Claude (Sonnet 5.5 + Haiku 5.5) · Pydantic · FastAPI · React**.

[![CI](https://img.shields.io/badge/CI-ruff%20%C2%B7%20mypy%20--strict%20%C2%B7%20pytest-blue)](#6-how-to-test--verify)
![tests](https://img.shields.io/badge/tests-143%20passing-brightgreen) ![coverage](https://img.shields.io/badge/coverage-96%25-brightgreen) ![types](https://img.shields.io/badge/mypy-strict-blue)

![Review UI showing a claim with a prompt injection that was detected and ignored](docs/images/ui-injection-case.png)

---

**Contents:**
[1. Problem](#1-problem-statement) ·
[2. Scope](#2-what-the-agent-does) ·
[3. Business cases](#3-business-cases-covered) ·
[4. Architecture](#4-architecture-overview) ·
[5. Setup](#5-setup-local-run) ·
[6. Testing](#6-how-to-test--verify) ·
[7. Examples](#7-example-inputs-and-outputs) ·
[8. Trade-offs](#8-trade-offs-limitations-and-next-steps) ·
[9. Real-world usage](#9-real-world-usage) ·
[Security](#security-prompt-injection-and-the-owasp-llm-top-10) ·
[Evals](#evals)

---

## 1. Problem statement

Under **EU Regulation 261/2004 (EU261)**, passengers can claim **EUR 250, 400 or 600** when a flight is delayed by 3+ hours, cancelled at short notice, or overbooked. Claims companies and law firms handle these claims at volume, and most of the work is repetitive:

1. read a messy email or form (often with a pasted booking or airline email);
2. find the flight number and date, and check what really happened;
3. check whether EU261 applies (where the flight departed, which airline);
4. work out the amount from the flight distance and delay;
5. write a claim letter to the airline;
6. send the hard cases (bad weather, strikes, missing facts) to a senior handler.

Steps 1–5 are mostly mechanical, yet they are done by hand for every claim. Mistakes are costly: claiming the wrong amount, or claiming a case that is not covered, damages credibility with airlines.

**Goal:** automate steps 1–5 with an AI agent, keep a human in control of what is sent, and make every decision explainable and auditable.

## 2. What the agent does

**In scope**
- Screens input for prompt injection, requests to falsify facts, and off-topic messages.
- Extracts structured facts from free text and attachments (Claude, structured outputs).
- **Plans and gathers evidence with tools:** official flight record, airline, distance, relevant regulation passages.
- **Decides with deterministic rules** (scope, delay, cancellation notice, extraordinary circumstances, distance band, 50% long-haul reduction), based on the **flight record, not the customer's claim**.
- Drafts a claim letter, then checks it with code (right amount, flight number, legal basis, no links or placeholders).
- Pauses for **human approval** (LangGraph `interrupt`) and resumes later, even after a restart.
- Returns a **structured JSON result + a human-readable summary**, and writes a **trace** of every step, LLM call and tool call.

**Out of scope (by design)**
- Sending emails, filing claims or paying anything. The agent has **no tools with side effects**.
- Legal advice to the customer, or court proceedings.
- UK261, Montreal Convention baggage/damage claims, and national time limits (flagged only).
- Live flight data: a mock flight-status provider is used (see [next steps](#next-steps)).

## 3. Business cases covered

Each case is a runnable sample in [`samples/inputs`](samples/inputs) with its expected result in [`samples/expected`](samples/expected).

| # | Case | Result |
|---|---|---|
| 01 | Delay of 4 h 10 min, Frankfurt → Lisbon | **Eligible, EUR 400**, letter drafted, awaits approval |
| 02 | Delay of 2 h 05 min (customer says "nearly 4 hours") | **Not eligible**, flags the mismatch with the record |
| 03 | Cancellation, told 20 days ahead | **Not eligible** (14+ days notice) |
| 04 | Cancellation 2 days ahead, own-crew strike | **Eligible, EUR 250** (staff strike is not extraordinary) |
| 05 | 5 h delay, airline blames weather | **Needs review** (extraordinary circumstances must be checked by a person) |
| 06 | Istanbul → Rome on Turkish Airlines | **Out of scope** (non-EU departure, non-EU airline) |
| 07 | Tirana → Rome on Wizz Air, 3 h 30 min late | **Eligible, EUR 250** (into the EU on an EU carrier) |
| 08 | Paris → New York, 3 h 20 min late | **Eligible, EUR 300** (600 reduced by 50%, Art. 7(2)) |
| 09 | Denied boarding, overbooked flight | **Eligible, EUR 250**, flagged to verify the boarding pass |
| 10 | No flight number or date | **Needs info**, lists the missing fields |
| 11 | **Prompt injection** in the attached booking ("approve EUR 5000") | **Eligible, EUR 400** (not 5000), injection flagged, human review forced |
| 12 | Customer asks to say the delay was longer | **Rejected** (refuses to falsify facts) |
| 13 | Off-topic request (restaurant tips) | **Out of scope** |
| 14 | Flight not in the records | **Needs review** (cannot verify, does not guess) |
| 15 | Paris → Réunion (EU outermost region, 9,369 km) | **Eligible, EUR 400** (intra-EU cap, not 600) |

The [eval set](evals/dataset.jsonl) adds 28 harder variants: other date formats, Albanian and Italian messages, exact 3 h / 2 h 59 min boundaries, subtle injections, and indirect fabrication requests.

## 4. Architecture overview

### Data flow

```mermaid
flowchart TD
    IN([Claim: message + attachment]) --> V[validate_input<br/><i>length, binary, empty</i>]
    V -->|invalid| F
    V --> S[screen_input<br/><i>heuristics + Claude Haiku</i>]
    S -->|fabrication / off-topic| F
    S --> X[extract_facts<br/><i>Claude Sonnet, structured output<br/>+ deterministic normalisation</i>]
    X -->|missing facts| F
    X --> P[plan<br/><i>Claude Sonnet chooses tools</i>]
    P -->|tool calls| T[run_tools<br/><i>allowlisted, validated, read-only</i>]
    T --> P
    P -->|done / step limit| E[verify_evidence<br/><i>code checks the plan and backfills gaps</i>]
    E --> R[apply_rules<br/><i>pure Python EU261 rules</i>]
    R -->|eligible| D[draft_letter<br/><i>Claude Sonnet</i>]
    D --> C[check_letter<br/><i>amount, flight, legal basis, no links</i>]
    C -->|problems, retry with feedback| D
    C --> H{{review_gate<br/>human-in-the-loop interrupt}}
    R -->|other| H
    H --> F[finalize<br/><i>JSON result + summary + trace</i>]
    F --> OUT([ClaimResult])
```

(`uv run claim-agent graph` prints the exact graph LangGraph compiled.)

### Components

| Component | File | Role |
|---|---|---|
| Graph / orchestration | [`agent/graph.py`](src/eu261_claim_agent/agent/graph.py), [`agent/nodes/`](src/eu261_claim_agent/agent/nodes) | LangGraph `StateGraph` wiring and routing; one file per step |
| Agent state and helpers | [`agent/state.py`](src/eu261_claim_agent/agent/state.py), [`agent/runtime.py`](src/eu261_claim_agent/agent/runtime.py), [`agent/evidence.py`](src/eu261_claim_agent/agent/evidence.py) | Typed `AgentState`, step tracing, token budget, tool execution and evidence checks |
| LLM interface | [`llm/base.py`](src/eu261_claim_agent/llm/base.py) | Provider-agnostic `LLMClient` protocol + typed errors |
| Claude adapter | [`llm/anthropic_client.py`](src/eu261_claim_agent/llm/anthropic_client.py) | Structured outputs, strict tools, per-task model/effort, usage + cost |
| Offline model | [`llm/fake.py`](src/eu261_claim_agent/llm/fake.py) | Deterministic regex stand-in for CI, tests and `--mock` |
| Tools | [`tools/`](src/eu261_claim_agent/tools) | `lookup_flight`, `get_airline`, `calculate_distance`, `search_regulation` (one file each) + [`registry.py`](src/eu261_claim_agent/tools/registry.py) |
| Rules engine | [`core/rules.py`](src/eu261_claim_agent/core/rules.py) | The only place money and eligibility are decided |
| Guardrails | [`guardrails/guards.py`](src/eu261_claim_agent/guardrails/guards.py), [`core/normalize.py`](src/eu261_claim_agent/core/normalize.py), [`core/letters.py`](src/eu261_claim_agent/core/letters.py) | Input screening, spotlighting, fact validation, letter validation |
| Knowledge base | [`infra/search.py`](src/eu261_claim_agent/infra/search.py), [`data/regulation`](data/regulation) | BM25 search over plain-language EU261 notes + firm policy |
| Observability | [`infra/observability.py`](src/eu261_claim_agent/infra/observability.py) | structlog JSON logs + per-run JSONL traces with PII redaction |
| Service / API / CLI | [`service.py`](src/eu261_claim_agent/service.py), [`api.py`](src/eu261_claim_agent/api.py), [`cli.py`](src/eu261_claim_agent/cli.py) | One entry point used by every interface |
| Prompts | [`agent/prompts/`](src/eu261_claim_agent/agent/prompts) | Versioned `.md` files (version is logged with every LLM call) |
| Review UI | [`ui/`](ui) | React + TypeScript page to run claims, read traces, approve or reject |

### Key design decisions

1. **The LLM reads, plans and writes. Code validates and decides.** Eligibility and amounts come from a tested rules engine, so they are deterministic and injection-proof ([ADR 0002](docs/adr/0002-rules-in-code-not-in-the-llm.md)).
2. **Privilege separation.** The step that reads raw customer text (extraction) has **no tools**. The step that has tools (planner) **never sees raw text**, only validated facts such as flight number, date and airports.
3. **"Planner proposes, code guarantees."** The LLM picks which tools to call. `verify_evidence` then checks that the right flight, airline and route were fetched, ignores off-target results, and fetches anything missing itself.
4. **Every LLM failure has a fallback.** Planner down → deterministic evidence collection. Draft invalid → retry with feedback, then a safe template. Extraction failed → "needs review". Nothing crashes, and nothing is guessed.
5. **Why LangGraph:** explicit, drawable control flow; typed state; built-in checkpointing and `interrupt()` for human approval that can wait days ([ADR 0001](docs/adr/0001-langgraph-as-orchestrator.md)).
6. **Why Claude Sonnet + Haiku:** Sonnet for extraction, planning and drafting; Haiku as a cheap screen on every claim. Structured outputs + strict tools, because current Claude models reject forced tool use ([ADR 0003](docs/adr/0003-claude-models-and-structured-outputs.md)).

## 5. Setup (local run)

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 22 (only for the UI). An Anthropic API key is optional: everything runs offline with `--mock`.

```bash
git clone <this-repo> && cd eu261-claim-agent
make setup                 # uv sync + npm ci + pre-commit hooks
cp .env.example .env       # add ANTHROPIC_API_KEY to use Claude
```

**On Windows:** use [WSL](https://learn.microsoft.com/windows/wsl/install), or run the commands without `make`:

| `make` target | Command without make |
|---|---|
| `make setup` | `uv sync` and `cd ui && npm ci` |
| `make test` | `uv run pytest --cov` |
| `make lint` | `uv run ruff check . && uv run mypy` |
| `make demo` | `uv run claim-agent demo --mock` |
| `make eval PROVIDER=anthropic` | `uv run python -m evals.run_evals --provider anthropic` |
| `make serve MOCK=1` | `cd ui && npm run build`, then `uv run claim-agent serve --mock` |

**Try it in 30 seconds (no API key):**
```bash
make demo                                                        # all 15 samples, offline
uv run claim-agent run samples/inputs/11_prompt_injection.json --mock
```

**With Claude:**
```bash
uv run claim-agent run samples/inputs/01_long_delay_eligible.json          # pauses for review
uv run claim-agent review <run_id> --approve --reviewer "Your Name"        # resumes the graph
uv run claim-agent trace <run_id>                                          # every step, LLM and tool call
```

**API + review UI:**
```bash
make serve            # http://127.0.0.1:8000  (MOCK=1 for offline). API docs at /docs
# or
docker compose up --build
```

| Endpoint | Purpose |
|---|---|
| `POST /claims` | Submit a claim → `ClaimResult` (pauses at review if needed) |
| `GET /claims/{id}` | Current result |
| `POST /claims/{id}/review` | `{"approved": true, "reviewer": "...", "note": "..."}` resumes the paused run |
| `GET /claims/{id}/trace` | Trace events |
| `GET /health` | Liveness + active model provider |

Set `CLAIM_AGENT_API_KEY` to require an `X-API-Key` header on `/claims` endpoints.

## 6. How to test / verify

```bash
make lint        # ruff + ruff format --check + mypy --strict + import layers
make test        # 143 tests, offline, ~60 s, with coverage
make test-live   # smoke tests against real Claude (needs ANTHROPIC_API_KEY)
make eval PROVIDER=anthropic   # quality metrics on 28 labelled cases
```

| Layer | What it proves | Where |
|---|---|---|
| **Unit** | Rules at every boundary (179/180 min, 239/240 min, 1500 km, 7/13/14 days), scope for EU/EEA/CH/UK, normalisation, guards, tool validation, letter checks, PII redaction | [`tests/unit`](tests/unit) |
| **Property-based** (Hypothesis) | For *any* route/airline/delay: amount ∈ {0, 250, 300, 400, 600}; a longer delay never pays less; the customer's reported delay never changes the decision | [`test_rules_properties.py`](tests/unit/core/test_rules_properties.py) |
| **Claude adapter** | Correct model per task, no forced tool use, strict tools, thinking blocks preserved, refusal/outage/invalid output mapped to typed errors | [`test_anthropic_client.py`](tests/unit/llm/test_anthropic_client.py) |
| **Integration** | All 15 samples match their expected outputs through the full graph | [`test_samples.py`](tests/integration/test_samples.py) |
| **Resilience** | Planner outage, lazy planner, off-target lookups, hallucinated tools, runaway loops, bad letters, token budget, crashes | [`test_resilience.py`](tests/integration/test_resilience.py) |
| **E2E** | HTTP API: submit → pending → approve → 409 on second review; 401/404/422; a paused claim survives a restart | [`test_review_and_api.py`](tests/integration/test_review_and_api.py) |
| **Live** | Real Claude on 6 key samples | [`test_live_claude.py`](tests/integration/test_live_claude.py) |

### Manual scenarios

1. `make serve MOCK=1`, open http://127.0.0.1:8000, click **Prompt injection**, then **Run agent**. Check that the result is EUR 400 (not 5000), there is a red `PROMPT_INJECTION_SUSPECTED` flag, and the letter contains neither the amount 5000 nor the email address. Enter your name and click **Approve**: the review state changes to *approved*.
2. Click **Bad weather**: the status is *Needs review* and no letter is drafted.
3. Click **Missing info**: the status is *Needs info* and the summary lists the missing fields.
4. Run `uv run claim-agent trace <run_id>` and open `runs/<run_id>.jsonl`: every node, tool call and decision is logged, and names and emails are redacted.
5. Stop the server while a claim is pending, start it again, and approve it via `POST /claims/{id}/review`: it resumes from the SQLite checkpoint.

## 7. Example inputs and outputs

**Input** ([`samples/inputs/01_long_delay_eligible.json`](samples/inputs/01_long_delay_eligible.json)):
```json
{
  "message": "Hi, my name is Anna Rossi. My Lufthansa flight LH1452 from Frankfurt to Lisbon on 3 October 2026 arrived about 4 hours late. They said it was a technical problem. Can I get compensation?",
  "received_on": "2026-11-15"
}
```

**Output** (abridged; full examples for every sample are in [`samples/outputs`](samples/outputs)):
```json
{
  "run_id": "3aa6920ae05f",
  "status": "eligible",
  "review_state": "pending",
  "needs_human_review": true,
  "decision": {
    "eligible": true,
    "amount_eur": 400,
    "distance_km": 1874.1,
    "arrival_delay_minutes": 250,
    "rule_ids": ["SCOPE_DEPARTS_EU", "BAND_MEDIUM", "DELAY_AT_LEAST_3H"],
    "reasons": ["Covered: the flight departed from Frankfurt, inside the EU261 area.", "..."]
  },
  "facts": {"flight_number": "LH1452", "flight_date": "2026-10-03", "departure_airport": "FRA",
            "arrival_airport": "LIS", "disruption_type": "delay", "reported_delay_minutes": 240},
  "risk_flags": [],
  "letter": {"subject": "EU261 compensation claim - flight LH1452 on 2026-10-03", "body": "Dear Lufthansa Customer Relations, ..."},
  "tool_calls": [{"tool": "lookup_flight", "source": "planner", "ok": true, "...": "..."}, "..."],
  "steps": ["validate_input", "screen_input", "extract_facts", "plan", "run_tools", "plan",
            "run_tools", "plan", "verify_evidence", "apply_rules", "draft_letter", "check_letter"],
  "usage": {"input_tokens": 0, "output_tokens": 0, "llm_calls": 6, "cost_usd": 0.0}
}
```

**Human-readable summary** (same run):
```text
Claim for flight LH1452 on 2026-10-03 (FRA→LIS).
Result: ELIGIBLE for EUR 400.
Why: Covered: the flight departed from Frankfurt, inside the EU261 area. Distance 1,874 km
(within the EU): EUR 400 band. Airline reason 'TECH' does not excuse the airline: technical
problems are part of normal airline operations (CJEU C-549/07). The flight arrived 4 h 10 min late.
Status: waiting for human review.
Next step: A claims handler must approve the drafted letter before it is sent.
```

**Trace** (`uv run claim-agent trace <run_id>`, offline run, abridged):
```text
15:13:33.966 run_start        {"provider": "fake", "message_chars": 186, "has_attachment": false}
15:13:33.976 guard_verdict    {"injection_suspected": false, "fabrication_request": false, "off_topic": false, ...}
15:13:33.979 facts_extracted  {"passenger_name": "[redacted]", "flight_number": "LH1452", "flight_date": "2026-10-03", ...}
15:13:33.984 tool_call        {"tool": "lookup_flight", "arguments": {...}, "ok": true, "duration_ms": 0.17, "source": "planner"}
15:13:33.987 tool_call        {"tool": "get_airline", "arguments": {"airline_code": "LH"}, "ok": true, "source": "planner"}
15:13:33.987 tool_call        {"tool": "calculate_distance", "arguments": {"origin_iata": "FRA", "destination_iata": "LIS"}, ...}
15:13:33.988 tool_call        {"tool": "search_regulation", "arguments": {"query": "delay three hours arrival compensation ...", ...}}
15:13:33.991 rules_decision   {"status": "eligible", "amount_eur": 400, "rule_ids": ["SCOPE_DEPARTS_EU", "BAND_MEDIUM", "DELAY_AT_LEAST_3H"]}
15:13:33.994 letter_check     {"attempt": 1, "problems": []}
```
With Claude, each `llm_call` event also records the model, input/output tokens, cost and prompt version.

## Security: prompt injection and the OWASP LLM Top 10

| Risk (OWASP LLM Top 10, 2025) | Mitigation in this project | Tested in |
|---|---|---|
| **LLM01 Prompt injection** (direct and indirect, via attachments) | 5 layers: regex screen + Haiku classifier (OR-ed, fail-safe); untrusted text wrapped in a per-run random tag it cannot close (*spotlighting*); the LLM reading raw text has no tools; the planner sees only validated facts; money is decided by code | `test_injection_*`, `test_wrapped_text_cannot_close_its_own_tag`, sample 11, evals e13/e26 |
| **LLM02 Sensitive information disclosure** | Names, emails, phones and booking references redacted in logs and traces; the planner never receives the passenger's name | `test_redaction`, `test_every_run_writes_a_trace` |
| **LLM05 Improper output handling** | Every LLM output is schema-validated (Pydantic) and then re-validated by code: flight-number/IATA/date formats, plausible delays, letter amount, links/emails/placeholders | `test_bad_llm_values_are_caught`, `test_letter_validation_*` |
| **LLM06 Excessive agency** | Tool allowlist, all tools read-only, arguments validated in code, no send/pay/delete capability, human approval before anything leaves the system | `test_unknown_tool_is_refused`, `test_hallucinated_tool_*` |
| **LLM09 Misinformation** | Decision based on the official flight record, not the customer's or the model's claim; mismatches flagged; unknown cases go to a human | `test_decision_uses_record_not_customer_claim`, property tests |
| **LLM10 Unbounded consumption** | Step limit on the planner, per-run token budget, input size limits, SDK timeouts and retries with backoff | `test_runaway_planner_*`, `test_token_budget_is_enforced` |

Other measures: optional API key with constant-time comparison; trace reads protected against path traversal; checkpoints deserialise only this project's own schema types; secrets only via environment variables; the Docker image runs as a non-root user.

## Evals

Tests check that the system behaves correctly. **Evals measure how well the model does** on 28 labelled, deliberately harder cases ([`evals/dataset.jsonl`](evals/dataset.jsonl)). Results are written to [`evals/results/`](evals/results).

| Metric | Offline regex baseline (`make eval`) | Claude (`make eval PROVIDER=anthropic`) |
|---|---|---|
| Decision accuracy | 0.82 | run to fill in |
| Amount accuracy | 0.89 | run to fill in |
| Extraction accuracy | 0.85 | run to fill in |
| Injection recall | 0.50 | run to fill in |
| Injection false-positive rate | 0.00 | run to fill in |
| **Overpayments** | **0** | run to fill in |
| p50 latency / avg cost per claim | 0.02 s / $0 | run to fill in |

How to read this: the regex baseline fails on lowercase flight numbers, Albanian and Italian messages, abbreviated months, subtle injections and indirect fabrication requests. Those are the cases where the LLM earns its place. Even when extraction fails, **overpayments stay at 0**, because the rules engine only pays what the flight record supports. CI runs the offline eval as a gate (≥ 0.80) on every push, and the Claude eval (≥ 0.90) on `main` when the API key secret is configured.

## 8. Trade-offs, limitations and next steps

### Trade-offs
- **Determinism vs flexibility.** Rules in code are predictable and testable, but every legal nuance must be coded. Anything the rules can't judge goes to a human instead of to the LLM.
- **LLM planner vs fixed pipeline.** For four tools a fixed sequence would work. The planner is kept because it chooses the relevant regulation passages per case and scales to more tools (e.g. weather or NOTAM lookups), and the backfill step makes it safe.
- **Template fallback vs always-LLM letters.** The fallback letter is less polished but always correct, so a model outage never blocks a valid claim.
- **BM25 vs vectors.** Seven short documents don't need embeddings. The `search()` interface stays the same if a vector store is added.

### Limitations
- Flight data is a **mock provider** (21 flights); airports and airlines are a small reference set.
- Simplified law: re-routing offers for cancellations, connecting flights, rebooking, national time limits and the pending EU261 reform are not modelled. They are flagged or sent to review.
- One passenger per claim; no OCR of PDF boarding passes yet.
- The SQLite checkpointer is single-node.
- The regex FakeLLM tests the system, not model quality; Claude quality comes from the live evals.

### Next steps
1. Real flight-status API (e.g. FlightAware / Cirium / AeroDataBox) behind the same `lookup_flight` tool, with caching.
2. PDF and image input (boarding passes, airline emails) using Claude vision, plus OCR fallback.
3. Model rerouting windows, connecting flights and multi-passenger bookings.
4. Postgres checkpointer + task queue for horizontal scaling; OpenTelemetry/Langfuse exporter behind the existing `Tracer`.
5. Grow the eval set from real (anonymised) handler decisions; track metrics per prompt version.
6. Customer-facing reply drafts for "not eligible" and "needs info" outcomes.

## 9. Real-world usage

**Is this solution currently used in any business?**
**Not in production today.** It is a working prototype built for this assignment.

**Who could use it:**
1. **Flight-compensation claim companies.** Intake triage for thousands of claims a month: auto-reject out-of-scope cases, route extraordinary-circumstance disputes to specialists, pre-draft letters for the rest.
2. **Law firms with a consumer or travel practice.** Paralegals review and approve drafts instead of writing them; the trace gives an audit trail for every decision.
3. **Airline customer-relations teams.** Check incoming claims against their own flight records to pay valid claims quickly (and avoid court costs) and push back on invalid ones consistently.

**What production would need:**
- **Security:** SSO/RBAC for reviewers, secrets in a vault, rate limiting, WAF, a regular red-team set for prompt injection.
- **Data access:** a licensed flight-data API, an airline registry, and a document store for attachments, behind the existing tool interfaces.
- **Human-in-the-loop:** the review queue already exists; add assignment, SLAs, a four-eyes rule for high amounts, and use reviewer edits as eval data.
- **Monitoring:** OpenTelemetry traces, dashboards for decision mix, override rate, cost and latency, alerts on eval regressions and LLM error rates.
- **Compliance:** GDPR (lawful basis, data processing agreement with the LLM provider, retention and deletion of traces and checkpoints, data residency), an audit log of every approval, and legal sign-off on rule changes. Under the EU AI Act this kind of assistive tool with human review is generally not high-risk, but that should be confirmed for the deployment.

---

### Project structure

```
src/eu261_claim_agent/
  api.py, cli.py         HTTP API and command line (independent of each other)
  service.py             ClaimAgent: the one entry point every interface uses
  config.py              settings (importable from any layer)
  agent/                 LangGraph workflow: state, runtime helpers, graph wiring
    nodes/               one file per graph step (validate_input … finalize)
    prompts/             versioned .md prompts
  llm/                   LLMClient protocol, Claude adapter, offline fake
  tools/                 registry + one file per tool
  guardrails/            input screening and spotlighting
  infra/                 reference data, regulation search, logs and traces
  core/                  schemas, rules, normalisation, letters, summary, distance
data/                    airports, airlines, mock flights, rule thresholds, regulation notes
samples/                 inputs/ expected/ outputs/ for 15 business cases
tests/                   unit/ (mirrors core, guardrails, tools, infra, llm; incl. property-based)
                         and integration/ (graph, resilience, API, CLI, live)
evals/                   labelled dataset, harness, results/
ui/                      React + TypeScript review UI
docs/                    ADRs, architecture graph, screenshots
```

Imports only point downwards: `api | cli → service → agent → llm → tools → guardrails | infra → core`.
This rule is enforced by [import-linter](https://import-linter.readthedocs.io/) (the `layers` contract in
`pyproject.toml`, run by `make lint` and in CI), so for example `core/` can never import AI, web or file code.

License: MIT.
