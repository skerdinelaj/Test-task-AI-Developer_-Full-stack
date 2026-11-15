"""validate_input: reject requests that break the size and shape limits."""

from __future__ import annotations

from eu261_claim_agent.agent.runtime import halt
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update, flag
from eu261_claim_agent.core.schemas import ClaimStatus, Severity
from eu261_claim_agent.guardrails.guards import check_request


def run(state: AgentState, deps: AgentDeps) -> Update:
    check = check_request(
        state["request"], deps.settings.max_message_chars, deps.settings.max_attachment_chars
    )
    if not check.ok:
        return halt(
            ClaimStatus.REJECTED,
            check.errors,
            risk_flags=[flag("INVALID_INPUT", Severity.WARNING, " ".join(check.errors))],
        )
    return {"halt": False}
