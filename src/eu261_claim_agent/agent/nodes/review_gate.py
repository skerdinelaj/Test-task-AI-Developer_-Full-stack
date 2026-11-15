"""review_gate: pause for a human decision when the outcome needs one (human-in-the-loop)."""

from __future__ import annotations

from langgraph.types import interrupt

from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update
from eu261_claim_agent.core.schemas import ClaimStatus, ReviewOutcome, ReviewState, Severity


def run(state: AgentState, deps: AgentDeps) -> Update:
    status = state["status"]
    high = any(f.severity == Severity.HIGH for f in state.get("risk_flags", []))
    needed = status in (ClaimStatus.ELIGIBLE, ClaimStatus.NEEDS_REVIEW) or (
        status == ClaimStatus.NOT_ELIGIBLE and high
    )
    if not needed:
        return {"review_state": ReviewState.NOT_REQUIRED}
    # Pauses the graph; state is checkpointed. Resumed by ClaimAgent.review().
    answer = interrupt(
        {
            "run_id": state["run_id"],
            "status": status.value,
            "amount_eur": state["decision"].amount_eur if state.get("decision") else 0,
        }
    )
    review = ReviewOutcome.model_validate(answer)
    deps.tracer.event(
        state["run_id"], "human_review", approved=review.approved, reviewer=review.reviewer
    )
    return {
        "review": review,
        "review_state": ReviewState.APPROVED if review.approved else ReviewState.REJECTED,
    }
