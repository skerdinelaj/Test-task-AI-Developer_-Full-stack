"""get_airline: an airline's name, home country and EU-carrier status."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field

from eu261_claim_agent.core.schemas import Airline
from eu261_claim_agent.infra.reference_data import ReferenceData
from eu261_claim_agent.tools.registry import Args, Tool, ToolError, check_airline_code

NAME = "get_airline"
DESCRIPTION = "Get an airline's name, home country and whether it is an EU (Community) carrier."


class GetAirlineArgs(Args):
    airline_code: Annotated[str, AfterValidator(check_airline_code)] = Field(
        description="2-character IATA airline code, e.g. 'LH'."
    )


def make(ref: ReferenceData) -> Tool:
    def get_airline(args: GetAirlineArgs) -> Airline:
        airline = ref.airlines.get(args.airline_code)
        if airline is None:
            raise ToolError(f"Unknown airline code {args.airline_code}.")
        return airline

    return Tool(NAME, DESCRIPTION, GetAirlineArgs, get_airline)
