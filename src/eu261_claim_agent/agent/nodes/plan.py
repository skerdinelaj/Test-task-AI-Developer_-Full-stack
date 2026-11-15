"""plan: the LLM chooses which tools to call, seeing only validated, structured facts."""

from __future__ import annotations

from eu261_claim_agent.agent.prompts import load_prompt
from eu261_claim_agent.agent.runtime import log_llm, over_budget
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update, flag
from eu261_claim_agent.core.schemas import LLMTask, Severity
from eu261_claim_agent.llm.base import LLMError


def run(state: AgentState, deps: AgentDeps) -> Update:
    steps = state.get("plan_steps", 0)
    if steps >= deps.settings.max_plan_steps or over_budget(state, deps):
        return {
            "pending_calls": [],
            "risk_flags": [
                flag(
                    "PLAN_LIMIT_REACHED",
                    Severity.INFO,
                    "Planner stopped at the step/token limit.",
                )
            ],
        }
    messages = state.get("planner_messages") or [
        {
            "role": "user",
            "content": "<facts>\n"
            + state["facts"].model_dump_json(
                include={
                    "flight_number",
                    "flight_date",
                    "departure_airport",
                    "arrival_airport",
                    "disruption_type",
                    "reported_delay_minutes",
                    "cancellation_notice_days",
                },
                indent=2,
            )
            + "\n</facts>\nCollect the evidence for this claim.",
        }
    ]
    try:
        turn = deps.llm.tool_turn(load_prompt("plan"), messages, deps.tools.specs())
    except LLMError as exc:
        # Degrade gracefully: verify_evidence will fetch what is needed deterministically.
        return {
            "pending_calls": [],
            "risk_flags": [
                flag(
                    "PLANNER_DEGRADED",
                    Severity.WARNING,
                    f"Planner unavailable ({exc}); evidence "
                    "was collected by the deterministic fallback.",
                )
            ],
        }
    log_llm(
        deps,
        state,
        LLMTask.PLAN,
        turn.usage,
        turn.model,
        step=steps + 1,
        tool_calls=[c.name for c in turn.tool_calls],
        stop_reason=turn.stop_reason,
    )
    return {
        "planner_messages": [
            *messages,
            {"role": "assistant", "content": turn.assistant_content},
        ],
        "pending_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls
        ],
        "plan_steps": steps + 1,
        "usage": turn.usage,
    }
