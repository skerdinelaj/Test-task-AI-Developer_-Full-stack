"""apply_rules: the deterministic rules engine decides eligibility and the amount."""

from __future__ import annotations

from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update
from eu261_claim_agent.core.rules import evaluate_claim


def run(state: AgentState, deps: AgentDeps) -> Update:
    outcome = evaluate_claim(state["facts"], state["evidence"], deps.ref.rules)
    deps.tracer.event(
        state["run_id"],
        "rules_decision",
        status=outcome.decision.status,
        amount_eur=outcome.decision.amount_eur,
        rule_ids=outcome.decision.rule_ids,
    )
    return {
        "decision": outcome.decision,
        "status": outcome.decision.status,
        "risk_flags": outcome.flags,
    }
