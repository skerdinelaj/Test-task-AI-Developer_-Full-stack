"""The graph's typed state, its dependencies and small shared types."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypedDict

from eu261_claim_agent.config import Settings
from eu261_claim_agent.core.schemas import (
    ClaimRequest,
    ClaimStatus,
    EligibilityDecision,
    Evidence,
    GuardVerdict,
    LetterDraft,
    NormalizedFacts,
    ReviewOutcome,
    ReviewState,
    RiskFlag,
    Severity,
    StepRecord,
    ToolCallRecord,
    Usage,
)
from eu261_claim_agent.infra.observability import Tracer
from eu261_claim_agent.infra.reference_data import ReferenceData
from eu261_claim_agent.llm.base import LLMClient
from eu261_claim_agent.tools import ToolRegistry


def add_usage(left: Usage, right: Usage) -> Usage:
    """Reducer: token usage from every step is summed."""
    return left + right


class AgentState(TypedDict, total=False):
    # inputs
    run_id: str
    request: ClaimRequest
    today: date
    boundary: str
    # control
    halt: bool
    status: ClaimStatus
    terminal_reasons: list[str]
    # working data
    guard: GuardVerdict
    facts: NormalizedFacts
    missing_fields: list[str]
    planner_messages: list[dict[str, Any]]
    pending_calls: list[dict[str, Any]]
    plan_steps: int
    evidence: Evidence
    decision: EligibilityDecision
    letter: LetterDraft
    letter_attempts: int
    letter_problems: list[str]
    review: ReviewOutcome
    review_state: ReviewState
    summary: str
    # append-only (reducers)
    risk_flags: Annotated[list[RiskFlag], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    steps: Annotated[list[StepRecord], operator.add]
    usage: Annotated[Usage, add_usage]


Update = dict[str, Any]
NodeFn = Callable[[AgentState], Update]


@dataclass
class AgentDeps:
    settings: Settings
    llm: LLMClient
    tools: ToolRegistry
    ref: ReferenceData
    tracer: Tracer


def flag(code: str, severity: Severity, message: str) -> RiskFlag:
    return RiskFlag(code=code, severity=severity, message=message)
