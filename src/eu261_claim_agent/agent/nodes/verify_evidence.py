"""verify_evidence: code checks the planner's work and fetches anything missing (planner
proposes, code guarantees)."""

from __future__ import annotations

from typing import Any

from eu261_claim_agent.agent.evidence import execute
from eu261_claim_agent.agent.state import AgentDeps, AgentState, Update, flag
from eu261_claim_agent.core.schemas import (
    DisruptionType,
    Evidence,
    RiskFlag,
    Severity,
    ToolCallRecord,
)


def run(state: AgentState, deps: AgentDeps) -> Update:
    facts = state["facts"]
    evidence = state.get("evidence", Evidence()).model_copy(deep=True)
    records: list[ToolCallRecord] = []
    flags: list[RiskFlag] = []

    def backfill(name: str, args: dict[str, Any]) -> None:
        _, _, record, new_flags = execute(deps, state, evidence, name, args, "backfill")
        records.append(record)
        flags.extend(new_flags)
        evidence.backfilled.append(name)

    attempted = any(
        c.tool == "lookup_flight"
        and c.ok is False
        and c.arguments.get("flight_number") == facts.flight_number
        for c in state.get("tool_calls", [])
    )
    if evidence.flight is None and not attempted and facts.flight_number and facts.flight_date:
        backfill(
            "lookup_flight",
            {
                "flight_number": facts.flight_number,
                "flight_date": facts.flight_date.isoformat(),
            },
        )
    flight = evidence.flight
    if flight is not None:
        if evidence.airline is None or evidence.airline.iata != flight.airline:
            evidence.airline = None
            backfill("get_airline", {"airline_code": flight.airline})
        if (
            evidence.distance_km is None
            or evidence.origin is None
            or evidence.destination is None
            or (evidence.origin.iata, evidence.destination.iata)
            != (flight.origin, flight.destination)
        ):
            evidence.distance_km = None
            backfill(
                "calculate_distance",
                {"origin_iata": flight.origin, "destination_iata": flight.destination},
            )
    if not evidence.regulation:
        query = {
            DisruptionType.CANCELLATION: "cancellation notice compensation",
            DisruptionType.DENIED_BOARDING: "denied boarding compensation",
        }.get(facts.disruption_type, "delay three hours compensation")
        backfill("search_regulation", {"query": query, "top_k": 3})
    if evidence.backfilled:
        flags.append(
            flag(
                "EVIDENCE_BACKFILLED",
                Severity.INFO,
                "Code fetched evidence the planner did not: "
                + ", ".join(evidence.backfilled)
                + ".",
            )
        )
    return {"evidence": evidence, "tool_calls": records, "risk_flags": flags}
