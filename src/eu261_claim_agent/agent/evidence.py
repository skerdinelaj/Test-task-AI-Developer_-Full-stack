"""Running a tool and folding its result into the evidence (used by run_tools and
verify_evidence)."""

from __future__ import annotations

from typing import Any

from eu261_claim_agent.agent.state import AgentDeps, AgentState, flag
from eu261_claim_agent.core.schemas import (
    Airline,
    Evidence,
    FlightRecord,
    NormalizedFacts,
    RiskFlag,
    Severity,
    ToolCallRecord,
)
from eu261_claim_agent.tools import DistanceResult, SearchResult


def absorb(evidence: Evidence, facts: NormalizedFacts, result: Any) -> list[RiskFlag]:
    """Fold a tool result into the evidence, accepting only what matches the claim."""
    flags: list[RiskFlag] = []
    if isinstance(result, FlightRecord):
        if (result.flight_number, result.date) == (facts.flight_number, facts.flight_date):
            evidence.flight = result
        else:
            flags.append(
                flag(
                    "PLANNER_OFF_TARGET",
                    Severity.WARNING,
                    f"Planner looked up {result.flight_number}; result ignored.",
                )
            )
    elif isinstance(result, Airline):
        if evidence.flight is None or evidence.flight.airline == result.iata:
            evidence.airline = result
    elif isinstance(result, DistanceResult):
        route = (result.origin.iata, result.destination.iata)
        if evidence.flight is None or route == (
            evidence.flight.origin,
            evidence.flight.destination,
        ):
            evidence.origin, evidence.destination = result.origin, result.destination
            evidence.distance_km = result.distance_km
    elif isinstance(result, SearchResult):
        seen = {(s.doc_id, s.text) for s in evidence.regulation}
        evidence.regulation += [p for p in result.passages if (p.doc_id, p.text) not in seen]
    return flags


def execute(
    deps: AgentDeps,
    state: AgentState,
    evidence: Evidence,
    name: str,
    args: dict[str, Any],
    source: str,
) -> tuple[str, bool, ToolCallRecord, list[RiskFlag]]:
    execution = deps.tools.execute(name, args)
    record = ToolCallRecord(
        tool=name,
        arguments=execution.arguments,
        ok=execution.ok,
        duration_ms=execution.duration_ms,
        error=execution.error,
        source=source,
    )
    deps.tracer.event(state["run_id"], "tool_call", **record.model_dump())
    flags = absorb(evidence, state["facts"], execution.result) if execution.ok else []
    return execution.content, execution.ok, record, flags
