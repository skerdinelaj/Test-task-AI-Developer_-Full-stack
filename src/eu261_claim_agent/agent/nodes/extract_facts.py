"""extract_facts: the LLM reads the customer's text (no tools); code normalises the result."""

from __future__ import annotations

from eu261_claim_agent.agent.prompts import load_prompt
from eu261_claim_agent.agent.runtime import halt, log_llm, over_budget
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update, flag
from eu261_claim_agent.core.normalize import normalize_facts
from eu261_claim_agent.core.schemas import ClaimStatus, ExtractedFacts, LLMTask, Severity
from eu261_claim_agent.guardrails.guards import wrap_untrusted
from eu261_claim_agent.llm.base import LLMError


def run(state: AgentState, deps: AgentDeps) -> Update:
    if over_budget(state, deps):
        return halt(
            ClaimStatus.NEEDS_REVIEW,
            ["Token budget exceeded."],
            risk_flags=[flag("BUDGET_EXCEEDED", Severity.HIGH, "Token budget exceeded.")],
        )
    req = state["request"]
    parts = [
        f"Reference date (today): {state['today'].isoformat()}",
        "Customer message:",
        wrap_untrusted("message", req.message, state["boundary"]),
    ]
    if req.attachment_text:
        parts += [
            "Attachment:",
            wrap_untrusted("attachment", req.attachment_text, state["boundary"]),
        ]
    try:
        result = deps.llm.structured(
            LLMTask.EXTRACT, load_prompt("extract"), "\n".join(parts), ExtractedFacts
        )
    except LLMError as exc:
        return halt(
            ClaimStatus.NEEDS_REVIEW,
            ["The AI service could not read the claim."],
            risk_flags=[flag("LLM_FAILURE", Severity.HIGH, f"Extraction failed: {exc}")],
        )
    log_llm(deps, state, LLMTask.EXTRACT, result.usage, result.model, attempts=result.attempts)
    normalized = normalize_facts(result.value, state["today"])
    deps.tracer.event(
        state["run_id"],
        "facts_extracted",
        **normalized.facts.model_dump(mode="json"),
        missing=normalized.missing_fields,
    )
    update: Update = {
        "facts": normalized.facts,
        "missing_fields": normalized.missing_fields,
        "usage": result.usage,
        "risk_flags": normalized.flags,
    }
    if normalized.missing_fields:
        update |= halt(ClaimStatus.NEEDS_INFO, ["Some required details are missing."])
    return update
