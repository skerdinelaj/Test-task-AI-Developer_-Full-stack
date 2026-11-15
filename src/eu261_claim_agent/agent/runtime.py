"""Helpers every node can use: tracing, the token budget, LLM logging and halting."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from eu261_claim_agent.agent.prompts import PROMPT_VERSION
from eu261_claim_agent.agent.state import AgentDeps, AgentState, NodeFn, Update
from eu261_claim_agent.core.letters import CaseFile
from eu261_claim_agent.core.schemas import ClaimStatus, LLMTask, StepRecord, Usage


def traced(name: str, fn: Callable[[AgentState, AgentDeps], Update], deps: AgentDeps) -> NodeFn:
    """Bind deps to a node, time it, log start/end, and record the step in state."""

    def wrapper(state: AgentState) -> Update:
        start = time.perf_counter()
        deps.tracer.event(state["run_id"], "node_start", node=name)
        update = fn(state, deps)
        ms = round((time.perf_counter() - start) * 1000, 2)
        deps.tracer.event(
            state["run_id"],
            "node_end",
            node=name,
            duration_ms=ms,
            status=update.get("status"),
            halt=update.get("halt", False),
        )
        update["steps"] = [StepRecord(node=name, duration_ms=ms)]
        return update

    wrapper.__name__ = name
    return wrapper


def over_budget(state: AgentState, deps: AgentDeps) -> bool:
    return state.get("usage", Usage()).total_tokens >= deps.settings.max_run_tokens


def log_llm(
    deps: AgentDeps, state: AgentState, task: LLMTask, usage: Usage, model: str, **extra: Any
) -> None:
    deps.tracer.event(
        state["run_id"],
        "llm_call",
        task=task.value,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        prompt_version=PROMPT_VERSION,
        **extra,
    )


def halt(status: ClaimStatus, reasons: list[str], **extra: Any) -> Update:
    return {"halt": True, "status": status, "terminal_reasons": reasons, **extra}


def case_file(state: AgentState) -> CaseFile:
    return CaseFile.build(state["facts"], state["decision"], state["evidence"])
