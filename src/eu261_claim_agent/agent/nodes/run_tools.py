"""run_tools: execute the planner's tool calls and send the results back to it."""

from __future__ import annotations

from typing import Any

from eu261_claim_agent.agent.evidence import execute
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update
from eu261_claim_agent.core.schemas import Evidence, RiskFlag, ToolCallRecord


def run(state: AgentState, deps: AgentDeps) -> Update:
    evidence = state.get("evidence", Evidence()).model_copy(deep=True)
    results: list[dict[str, Any]] = []
    records: list[ToolCallRecord] = []
    flags: list[RiskFlag] = []
    for call in state.get("pending_calls", []):
        content, ok, record, new_flags = execute(
            deps, state, evidence, call["name"], call["arguments"], "planner"
        )
        records.append(record)
        flags += new_flags
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": call["id"],
            "content": content,
        }
        if not ok:
            block["is_error"] = True
        results.append(block)
    messages = [*state["planner_messages"], {"role": "user", "content": results}]
    return {
        "evidence": evidence,
        "planner_messages": messages,
        "pending_calls": [],
        "tool_calls": records,
        "risk_flags": flags,
    }
