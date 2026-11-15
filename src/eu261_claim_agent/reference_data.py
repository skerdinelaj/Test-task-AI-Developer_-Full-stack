"""Read-only access to reference data (airports, airlines, mock flight provider, rule thresholds).

In production these would be a flight-data API, an airline registry and a config service.
Keeping them behind this small repository class means swapping the source is a one-file change.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from eu261_claim_agent.normalize import normalize_flight_number
from eu261_claim_agent.schemas import Airline, Airport, FlightRecord


class DistanceBands(BaseModel):
    short_max: int
    medium_max: int


class Amounts(BaseModel):
    short: int
    medium: int
    long: int


class RulesConfig(BaseModel):
    eu261_countries: frozenset[str]
    delay_threshold_minutes: int
    long_haul_reduction_max_delay_minutes: int
    long_haul_reduction_factor: float
    cancellation_full_notice_days: int
    cancellation_partial_notice_days: int
    distance_bands_km: DistanceBands
    amounts_eur: Amounts
    extraordinary_reason_codes: frozenset[str]
    not_extraordinary_reason_codes: frozenset[str]
    reported_vs_record_tolerance_minutes: int


class ReferenceData:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.airports: dict[str, Airport] = {
            a.iata: a
            for a in TypeAdapter(list[Airport]).validate_json(
                (data_dir / "airports.json").read_text(encoding="utf-8")
            )
        }
        self.airlines: dict[str, Airline] = {
            a.iata: a
            for a in TypeAdapter(list[Airline]).validate_json(
                (data_dir / "airlines.json").read_text(encoding="utf-8")
            )
        }
        raw_flights = json.loads((data_dir / "flights.json").read_text(encoding="utf-8"))["flights"]
        flights = TypeAdapter(list[FlightRecord]).validate_python(raw_flights)
        # Canonical form ("AF006" -> "AF6") so lookups match however the number was written.
        self._flights: dict[tuple[str, date], FlightRecord] = {}
        for f in flights:
            canonical = normalize_flight_number(f.flight_number) or f.flight_number
            self._flights[(canonical, f.date)] = f.model_copy(update={"flight_number": canonical})
        self.rules = RulesConfig.model_validate_json(
            (data_dir / "rules_config.json").read_text(encoding="utf-8")
        )

    def find_flight(self, flight_number: str, flight_date: date) -> FlightRecord | None:
        return self._flights.get((flight_number, flight_date))

    def is_eu261_country(self, country: str) -> bool:
        return country in self.rules.eu261_countries
