# ADR 0003: Claude models, structured outputs and an offline stand-in

**Status:** accepted · **Date:** 2026-11-15

## Decision
- **Claude Sonnet 5.5** for extraction, planning and drafting: strong instruction following at moderate cost.
- **Claude Haiku 5.5** for input screening: a cheap, fast classifier call on every claim.
- Structured answers use **structured outputs** (`messages.parse` with a Pydantic model), and the planner uses **strict tools** with `tool_choice: auto`. The current Claude models reject forced tool use, so the older "force a tool to get JSON" trick is not used.
- Thinking blocks are passed back unchanged inside the tool loop, and content blocks are read by `type`, never by position.
- Transport retries (429/5xx/timeouts) are done by the SDK with exponential backoff; validation retries are ours.
- A deterministic **FakeLLM** implements the same `LLMClient` protocol. It powers CI, the `--mock` demo and reproducible tests without a key. Quality is measured separately with `make eval PROVIDER=anthropic`.

## Consequences
- Swapping provider means writing one class that implements `LLMClient`.
- Tests are fast and free, but they test the *system*, not the model; model quality is the job of the eval set.
