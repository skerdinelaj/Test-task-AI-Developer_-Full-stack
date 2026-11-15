"""calculate_distance: both airports and the great-circle distance between them."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from eu261_claim_agent.core.distance import great_circle_km
from eu261_claim_agent.core.schemas import Airport
from eu261_claim_agent.infra.reference_data import ReferenceData
from eu261_claim_agent.tools.registry import Args, Tool, ToolError, check_iata

NAME = "calculate_distance"
DESCRIPTION = "Get both airports (city, country) and the great-circle distance in km between them."


class DistanceArgs(Args):
    origin_iata: Annotated[str, AfterValidator(check_iata)] = Field(
        description="Origin airport IATA code."
    )
    destination_iata: Annotated[str, AfterValidator(check_iata)] = Field(
        description="Destination airport IATA code."
    )


class DistanceResult(BaseModel):
    origin: Airport
    destination: Airport
    distance_km: float


def make(ref: ReferenceData) -> Tool:
    def calculate_distance(args: DistanceArgs) -> DistanceResult:
        origin = ref.airports.get(args.origin_iata)
        dest = ref.airports.get(args.destination_iata)
        if origin is None or dest is None:
            unknown = args.origin_iata if origin is None else args.destination_iata
            raise ToolError(f"Unknown airport {unknown}.")
        km = great_circle_km(origin.lat, origin.lon, dest.lat, dest.lon)
        return DistanceResult(origin=origin, destination=dest, distance_km=round(km, 1))

    return Tool(NAME, DESCRIPTION, DistanceArgs, calculate_distance)
