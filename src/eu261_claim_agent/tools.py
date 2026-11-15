"""Tools the planner LLM may call. Read-only by design: no tool can send, write or pay.

Each tool has a Pydantic argument model. The JSON schema sent to Claude is generated from it and
marked ``strict`` (schema-valid calls), and arguments are validated *again* in code before the
handler runs, because the model's output is never trusted on its own.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from eu261_claim_agent.geo import great_circle_km
from eu261_claim_agent.normalize import normalize_flight_number, normalize_iata
from eu261_claim_agent.reference_data import ReferenceData
from eu261_claim_agent.schemas import Airline, Airport, FlightRecord, RegulationSnippet
from eu261_claim_agent.search import RegulationIndex


class ToolError(Exception):
    """A tool failed in a way the planner should be told about (not a crash)."""


def _flight_number(value: str) -> str:
    normalized = normalize_flight_number(value)
    if normalized is None:
        raise ValueError("not a valid flight number")
    return normalized


def _iata(value: str) -> str:
    normalized = normalize_iata(value)
    if normalized is None:
        raise ValueError("must be a 3-letter IATA airport code")
    return normalized


def _iso_date(value: str) -> str:
    date.fromisoformat(value)
    return value


def _airline_code(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 2 or not code.isalnum():
        raise ValueError("must be a 2-character IATA airline code")
    return code


def _top_k(value: int) -> int:
    if not 1 <= value <= 5:
        raise ValueError("top_k must be between 1 and 5")
    return value


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LookupFlightArgs(_Args):
    flight_number: Annotated[str, AfterValidator(_flight_number)] = Field(
        description="Flight number, e.g. 'LH1452'."
    )
    flight_date: Annotated[str, AfterValidator(_iso_date)] = Field(
        description="Scheduled date, YYYY-MM-DD."
    )


class GetAirlineArgs(_Args):
    airline_code: Annotated[str, AfterValidator(_airline_code)] = Field(
        description="2-character IATA airline code, e.g. 'LH'."
    )


class DistanceArgs(_Args):
    origin_iata: Annotated[str, AfterValidator(_iata)] = Field(
        description="Origin airport IATA code."
    )
    destination_iata: Annotated[str, AfterValidator(_iata)] = Field(
        description="Destination airport IATA code."
    )


class SearchRegulationArgs(_Args):
    query: str = Field(description="Short search query, e.g. 'cancellation notice 14 days'.")
    top_k: Annotated[int, AfterValidator(_top_k)] = Field(description="Number of passages, 1-5.")


class DistanceResult(BaseModel):
    origin: Airport
    destination: Airport
    distance_km: float


class SearchResult(BaseModel):
    passages: list[RegulationSnippet]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


@dataclass(frozen=True)
class ToolExecution:
    ok: bool
    content: str  # JSON string sent back to the LLM
    result: BaseModel | None
    error: str | None
    duration_ms: float
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[_Args]
    handler: Callable[[Any], BaseModel]

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        props = {
            k: {kk: vv for kk, vv in v.items() if kk in {"type", "description"}}
            for k, v in schema["properties"].items()
        }
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema={
                "type": "object",
                "properties": props,
                "required": list(props),
                "additionalProperties": False,
            },
        )


class ToolRegistry:
    """Allowlist of tools. Anything not registered here cannot be executed."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        start = time.perf_counter()

        def done(
            ok: bool,
            content: str,
            result: BaseModel | None,
            error: str | None,
            args: dict[str, Any],
        ) -> ToolExecution:
            return ToolExecution(
                ok, content, result, error, round((time.perf_counter() - start) * 1000, 2), args
            )

        tool = self._tools.get(name)
        if tool is None:
            msg = f"Unknown tool '{name}'. Allowed tools: {', '.join(self.names)}."
            return done(False, json.dumps({"error": msg}), None, msg, arguments)
        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            msg = "Invalid arguments: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            return done(False, json.dumps({"error": msg}), None, msg, arguments)
        try:
            result = tool.handler(args)
        except ToolError as exc:
            return done(False, json.dumps({"error": str(exc)}), None, str(exc), args.model_dump())
        return done(True, result.model_dump_json(), result, None, args.model_dump())


def build_tools(ref: ReferenceData, index: RegulationIndex) -> ToolRegistry:
    def lookup_flight(args: LookupFlightArgs) -> FlightRecord:
        record = ref.find_flight(args.flight_number, date.fromisoformat(args.flight_date))
        if record is None:
            raise ToolError(f"No flight {args.flight_number} on {args.flight_date} in the records.")
        return record

    def get_airline(args: GetAirlineArgs) -> Airline:
        airline = ref.airlines.get(args.airline_code)
        if airline is None:
            raise ToolError(f"Unknown airline code {args.airline_code}.")
        return airline

    def calculate_distance(args: DistanceArgs) -> DistanceResult:
        origin = ref.airports.get(args.origin_iata)
        dest = ref.airports.get(args.destination_iata)
        if origin is None or dest is None:
            unknown = args.origin_iata if origin is None else args.destination_iata
            raise ToolError(f"Unknown airport {unknown}.")
        km = great_circle_km(origin.lat, origin.lon, dest.lat, dest.lon)
        return DistanceResult(origin=origin, destination=dest, distance_km=round(km, 1))

    def search_regulation(args: SearchRegulationArgs) -> SearchResult:
        return SearchResult(passages=index.search(args.query, args.top_k))

    return ToolRegistry(
        [
            Tool(
                "lookup_flight",
                "Look up the official status of a flight on a date: route, operating airline, "
                "scheduled and actual arrival, cancellation and the reason code the airline "
                "reported.",
                LookupFlightArgs,
                lookup_flight,
            ),
            Tool(
                "get_airline",
                "Get an airline's name, home country and whether it is an EU (Community) carrier.",
                GetAirlineArgs,
                get_airline,
            ),
            Tool(
                "calculate_distance",
                "Get both airports (city, country) and the great-circle distance in km "
                "between them.",
                DistanceArgs,
                calculate_distance,
            ),
            Tool(
                "search_regulation",
                "Search the EU261 knowledge base and the firm's claim-handling policy. Returns the "
                "most relevant passages.",
                SearchRegulationArgs,
                search_regulation,
            ),
        ]
    )
