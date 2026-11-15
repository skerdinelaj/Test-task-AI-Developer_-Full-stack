"""The agent as an explicit LangGraph state machine. This file only wires the graph.

    validate_input → screen_input → extract_facts → plan ⇄ run_tools → verify_evidence
      → apply_rules → [draft_letter ⇄ check_letter] → review_gate (human-in-the-loop) → finalize

Design principles
* The LLM *reads* (extraction), *plans* (which evidence to fetch) and *writes* (letter).
  Code *validates* every LLM output and *decides* eligibility and money.
* Privilege separation: the extraction step reads raw customer text but has no tools; the planner
  has tools but only sees validated, structured facts.
* Every LLM failure has a deterministic fallback, so the agent degrades to "needs review"
  instead of crashing or guessing.

Each step lives in ``agent/nodes/<node>.py``; shared helpers are in ``runtime.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from eu261_claim_agent.agent.nodes import (
    apply_rules,
    check_letter,
    draft_letter,
    extract_facts,
    finalize,
    plan,
    review_gate,
    run_tools,
    screen_input,
    validate_input,
    verify_evidence,
)
from eu261_claim_agent.agent.runtime import traced
from eu261_claim_agent.agent.state import AgentDeps, AgentState
from eu261_claim_agent.core.schemas import ClaimStatus

NODES = {
    "validate_input": validate_input.run,
    "screen_input": screen_input.run,
    "extract_facts": extract_facts.run,
    "plan": plan.run,
    "run_tools": run_tools.run,
    "verify_evidence": verify_evidence.run,
    "apply_rules": apply_rules.run,
    "draft_letter": draft_letter.run,
    "check_letter": check_letter.run,
    "review_gate": review_gate.run,
    "finalize": finalize.run,
}


# ---------------------------------------------------------------------- routing
def after(next_node: str) -> Callable[[AgentState], str]:
    def route(state: AgentState) -> str:
        return "finalize" if state.get("halt") else next_node

    route.__name__ = f"route_to_{next_node}"
    return route


def after_plan(state: AgentState) -> str:
    return "run_tools" if state.get("pending_calls") else "verify_evidence"


def after_rules(state: AgentState) -> str:
    return "draft_letter" if state["status"] == ClaimStatus.ELIGIBLE else "review_gate"


def after_check(state: AgentState) -> str:
    return "draft_letter" if state.get("letter_problems") else "review_gate"


# ---------------------------------------------------------------------- wiring
def build_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    g = StateGraph(AgentState)
    for name, run in NODES.items():
        # LangGraph's overloads can't infer a TypedDict with reducer annotations.
        g.add_node(name, traced(name, run, deps))  # type: ignore[call-overload]
    g.add_edge(START, "validate_input")
    g.add_conditional_edges("validate_input", after("screen_input"), ["screen_input", "finalize"])
    g.add_conditional_edges("screen_input", after("extract_facts"), ["extract_facts", "finalize"])
    g.add_conditional_edges("extract_facts", after("plan"), ["plan", "finalize"])
    g.add_conditional_edges("plan", after_plan, ["run_tools", "verify_evidence"])
    g.add_edge("run_tools", "plan")
    g.add_edge("verify_evidence", "apply_rules")
    g.add_conditional_edges("apply_rules", after_rules, ["draft_letter", "review_gate"])
    g.add_edge("draft_letter", "check_letter")
    g.add_conditional_edges("check_letter", after_check, ["draft_letter", "review_gate"])
    g.add_edge("review_gate", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
