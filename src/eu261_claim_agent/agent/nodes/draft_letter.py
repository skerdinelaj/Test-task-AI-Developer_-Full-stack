"""draft_letter: the LLM writes the claim letter from the verified case file."""

from __future__ import annotations

from eu261_claim_agent.agent.prompts import load_prompt
from eu261_claim_agent.agent.runtime import case_file, log_llm, over_budget
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update, flag
from eu261_claim_agent.core.letters import render_template_letter
from eu261_claim_agent.core.schemas import LetterDraft, LLMTask, Severity
from eu261_claim_agent.llm.base import LLMError


def run(state: AgentState, deps: AgentDeps) -> Update:
    attempts = state.get("letter_attempts", 0) + 1
    case = case_file(state)
    if over_budget(state, deps):
        return {
            "letter": render_template_letter(case),
            "letter_attempts": attempts,
            "letter_problems": [],
            "risk_flags": [
                flag(
                    "LETTER_TEMPLATE_FALLBACK",
                    Severity.INFO,
                    "Token budget reached; template used.",
                )
            ],
        }
    prompt = f"<case_file>\n{case.model_dump_json(indent=2)}\n</case_file>"
    if problems := state.get("letter_problems"):
        prompt += "\n\nYour previous draft was rejected for these reasons:\n- " + "\n- ".join(
            problems
        )
    try:
        result = deps.llm.structured(LLMTask.DRAFT, load_prompt("draft"), prompt, LetterDraft)
    except LLMError as exc:
        return {
            "letter": render_template_letter(case),
            "letter_attempts": attempts,
            "letter_problems": [],
            "risk_flags": [
                flag(
                    "LETTER_TEMPLATE_FALLBACK",
                    Severity.INFO,
                    f"Drafting failed ({exc}); template used.",
                )
            ],
        }
    log_llm(deps, state, LLMTask.DRAFT, result.usage, result.model, attempt=attempts)
    return {"letter": result.value, "letter_attempts": attempts, "usage": result.usage}
