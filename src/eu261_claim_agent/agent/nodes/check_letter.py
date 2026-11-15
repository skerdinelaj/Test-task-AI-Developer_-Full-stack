"""check_letter: code validates the draft; retry with feedback, then fall back to the template."""

from __future__ import annotations

from eu261_claim_agent.agent.runtime import case_file
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update, flag
from eu261_claim_agent.core.letters import render_template_letter, validate_letter
from eu261_claim_agent.core.schemas import Severity


def run(state: AgentState, deps: AgentDeps) -> Update:
    case = case_file(state)
    problems = validate_letter(state["letter"], case)
    deps.tracer.event(
        state["run_id"], "letter_check", attempt=state["letter_attempts"], problems=problems
    )
    if not problems:
        return {"letter_problems": []}
    if state["letter_attempts"] <= deps.settings.max_validation_retries:
        return {"letter_problems": problems}  # loop back to draft_letter with feedback
    return {
        "letter": render_template_letter(case),
        "letter_problems": [],
        "risk_flags": [
            flag(
                "LETTER_TEMPLATE_FALLBACK",
                Severity.WARNING,
                "The AI draft failed validation after retries; the safe template was used.",
            )
        ],
    }
