"""lookup_flight: the official record of a flight on a date."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import AfterValidator, Field

from eu261_claim_agent.core.schemas import FlightRecord
from eu261_claim_agent.infra.reference_data import ReferenceData
from eu261_claim_agent.tools.registry import (
    Args,
    Tool,
    ToolError,
    check_flight_number,
    check_iso_date,
)

NAME = "lookup_flight"
DESCRIPTION = (
    "Look up the official status of a flight on a date: route, operating airline, "
    "scheduled and actual arrival, cancellation and the reason code the airline "
    "reported."
)


class LookupFlightArgs(Args):
    flight_number: Annotated[str, AfterValidator(check_flight_number)] = Field(
        description="Flight number, e.g. 'LH1452'."
    )
    flight_date: Annotated[str, AfterValidator(check_iso_date)] = Field(
        description="Scheduled date, YYYY-MM-DD."
    )


def make(ref: ReferenceData) -> Tool:
    def lookup_flight(args: LookupFlightArgs) -> FlightRecord:
        record = ref.find_flight(args.flight_number, date.fromisoformat(args.flight_date))
        if record is None:
            raise ToolError(f"No flight {args.flight_number} on {args.flight_date} in the records.")
        return record

    return Tool(NAME, DESCRIPTION, LookupFlightArgs, lookup_flight)
