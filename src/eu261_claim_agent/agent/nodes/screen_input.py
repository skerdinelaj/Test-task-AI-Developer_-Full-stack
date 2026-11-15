"""screen_input: heuristic + LLM screen for prompt injection, fabrication and off-topic input."""

from __future__ import annotations

from eu261_claim_agent.agent.prompts import load_prompt
from eu261_claim_agent.agent.runtime import halt, log_llm
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update, flag
from eu261_claim_agent.core.schemas import (
    ClaimStatus,
    GuardVerdict,
    LLMTask,
    RiskFlag,
    Severity,
    Usage,
)
from eu261_claim_agent.guardrails.guards import (
    combine_verdicts,
    heuristic_screen,
    request_text,
    wrap_untrusted,
)
from eu261_claim_agent.llm.base import LLMError, LLMRefusal


def run(state: AgentState, deps: AgentDeps) -> Update:
    text = request_text(state["request"])
    heuristic = heuristic_screen(text)
    flags: list[RiskFlag] = []
    llm_verdict: GuardVerdict | None = None
    usage = Usage()
    try:
        result = deps.llm.structured(
            LLMTask.GUARD,
            load_prompt("guard"),
            wrap_untrusted("text", text, state["boundary"]),
            GuardVerdict,
        )
        llm_verdict, usage = result.value, result.usage
        log_llm(deps, state, LLMTask.GUARD, usage, result.model)
    except LLMRefusal:
        # The screening model refused to classify: treat as suspicious (fail safe).
        llm_verdict = GuardVerdict(
            injection_suspected=True,
            fabrication_request=False,
            off_topic=False,
            rationale="screening model refused",
        )
    except LLMError as exc:
        flags.append(
            flag(
                "GUARD_DEGRADED",
                Severity.WARNING,
                f"LLM screen unavailable, heuristic screen only ({exc}).",
            )
        )
    verdict = combine_verdicts(heuristic, llm_verdict)
    deps.tracer.event(state["run_id"], "guard_verdict", **verdict.model_dump())
    if verdict.fabrication_request:
        return halt(
            ClaimStatus.REJECTED,
            [
                "The request asks to present facts differently from what happened. We only "
                "submit claims based on the real facts; the customer can resubmit them "
                "as they are."
            ],
            guard=verdict,
            usage=usage,
            risk_flags=[*flags, flag("FABRICATION_REQUEST", Severity.HIGH, verdict.rationale)],
        )
    if verdict.off_topic:
        return halt(
            ClaimStatus.OUT_OF_SCOPE,
            ["This message is not about a flight delay, cancellation or denied boarding."],
            guard=verdict,
            usage=usage,
            risk_flags=flags,
        )
    if verdict.injection_suspected:
        flags.append(
            flag(
                "PROMPT_INJECTION_SUSPECTED",
                Severity.HIGH,
                "The input contains instructions aimed at the AI. They were "
                "ignored and the case is routed to a human.",
            )
        )
    return {"guard": verdict, "usage": usage, "risk_flags": flags}
