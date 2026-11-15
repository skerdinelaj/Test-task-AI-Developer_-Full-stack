# ADR 0001: LangGraph as the agent orchestrator

**Status:** accepted · **Date:** 2026-11-15

## Context
The workflow has fixed stages (screen → extract → gather evidence → decide → draft → human review), one open-ended loop (the planner choosing tools), retry loops (letter validation), and a pause that can last days (human approval). We need typed state, explicit control flow that reviewers can read, and durable pause/resume.

## Options
| Option | For | Against |
|---|---|---|
| Plain prompt + function calling loop | Simple, no dependency | Control flow hides in prompts; no checkpointing; human-in-the-loop must be built by hand |
| Claude Agent SDK / OpenAI Agents SDK | Fast to start, good tool loop | Optimised for autonomous agents; less explicit about fixed business stages and deterministic gates |
| **LangGraph** | Explicit state machine, conditional edges, built-in checkpointers and `interrupt()` for human-in-the-loop, graph can be drawn | Extra dependency and concepts |
| Custom state machine | Full control | Rebuilding checkpointing and resume is not where the value is |

## Decision
Use LangGraph `StateGraph` with a typed `AgentState`, a SQLite checkpointer, and `interrupt()` for review. The LLM is used *inside* nodes; the graph, not the prompt, decides what happens next.

## Consequences
- The flow is visible (`claim-agent graph` prints it) and testable node by node.
- Paused claims survive restarts (tested in `test_paused_claim_survives_a_restart`).
- The LLM provider is behind our own `LLMClient` protocol, so LangGraph is not coupled to a vendor SDK.
