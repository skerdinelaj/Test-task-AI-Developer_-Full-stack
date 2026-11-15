"""Tools the planner LLM may call, one module per tool. Read-only by design."""

from __future__ import annotations

from eu261_claim_agent.infra.reference_data import ReferenceData
from eu261_claim_agent.infra.search import RegulationIndex
from eu261_claim_agent.tools import (
    calculate_distance,
    get_airline,
    lookup_flight,
    search_regulation,
)
from eu261_claim_agent.tools.calculate_distance import DistanceArgs, DistanceResult
from eu261_claim_agent.tools.get_airline import GetAirlineArgs
from eu261_claim_agent.tools.lookup_flight import LookupFlightArgs
from eu261_claim_agent.tools.registry import (
    Args,
    Tool,
    ToolError,
    ToolExecution,
    ToolRegistry,
    ToolSpec,
)
from eu261_claim_agent.tools.search_regulation import SearchRegulationArgs, SearchResult

__all__ = [
    "Args",
    "DistanceArgs",
    "DistanceResult",
    "GetAirlineArgs",
    "LookupFlightArgs",
    "SearchRegulationArgs",
    "SearchResult",
    "Tool",
    "ToolError",
    "ToolExecution",
    "ToolRegistry",
    "ToolSpec",
    "build_tools",
]


def build_tools(ref: ReferenceData, index: RegulationIndex) -> ToolRegistry:
    """The allowlist the planner works with, in a fixed order."""
    return ToolRegistry(
        [
            lookup_flight.make(ref),
            get_airline.make(ref),
            calculate_distance.make(ref),
            search_regulation.make(index),
        ]
    )
