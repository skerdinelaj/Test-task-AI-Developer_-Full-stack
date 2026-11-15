"""finalize: build the plain-language summary and log the final decision."""

from __future__ import annotations

from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update
from eu261_claim_agent.core.schemas import ReviewState, Usage
from eu261_claim_agent.core.summary import build_summary


def run(state: AgentState, deps: AgentDeps) -> Update:
    review_state = state.get("review_state", ReviewState.NOT_REQUIRED)
    summary = build_summary(
        state["status"],
        state.get("facts"),
        state.get("decision"),
        state.get("terminal_reasons", []),
        state.get("missing_fields", []),
        state.get("risk_flags", []),
        review_state,
        state.get("review"),
    )
    deps.tracer.event(
        state["run_id"],
        "final_decision",
        status=state["status"],
        review_state=review_state,
        usage=state.get("usage", Usage()).model_dump(),
    )
    return {"summary": summary, "review_state": review_state}
