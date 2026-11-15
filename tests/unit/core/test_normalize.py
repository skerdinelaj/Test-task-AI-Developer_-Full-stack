"""Normalisation and validation of LLM-extracted facts."""

from __future__ import annotations

from datetime import date

import pytest

from eu261_claim_agent.core.normalize import normalize_facts, normalize_flight_number
from eu261_claim_agent.core.schemas import (
    DisruptionType,
    ExtractedFacts,
)

TODAY = date(2026, 11, 15)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LH1452", "LH1452"),
        ("lh 1452", "LH1452"),
        ("AF006", "AF6"),
        ("W6-2301", "W62301"),
        ("A3 600", "A3600"),
        ("1452", None),
        ("HELLO", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_flight_number(raw: str | None, expected: str | None) -> None:
    assert normalize_flight_number(raw) == expected


def extracted(**kw: object) -> ExtractedFacts:
    base: dict[str, object] = dict(
        passenger_name=None,
        flight_number="LH1452",
        flight_date="2026-10-03",
        departure_airport="fra",
        arrival_airport="LIS",
        disruption_type=DisruptionType.DELAY,
        reported_delay_hours=4.0,
        cancellation_notice_days=None,
        reason_given_by_airline=None,
    )
    return ExtractedFacts.model_validate(base | kw)


def test_normalize_happy_path() -> None:
    out = normalize_facts(extracted(), TODAY)
    assert out.facts.departure_airport == "FRA"
    assert out.facts.reported_delay_minutes == 240
    assert out.missing_fields == [] and out.flags == []


def test_future_date_is_rejected_and_reported_missing() -> None:
    out = normalize_facts(extracted(flight_date="2027-01-01"), TODAY)
    assert out.facts.flight_date is None
    assert "flight_date" in out.missing_fields
    assert out.flags[0].code == "FLIGHT_DATE_IN_FUTURE"


@pytest.mark.parametrize(
    ("field", "value", "flag"),
    [
        ("flight_date", "03/10/2026", "INVALID_FLIGHT_DATE"),
        ("flight_number", "not a flight", "INVALID_FLIGHT_NUMBER"),
        ("reported_delay_hours", -3.0, "IMPLAUSIBLE_DELAY"),
        ("reported_delay_hours", 500.0, "IMPLAUSIBLE_DELAY"),
        ("flight_date", "2020-01-01", "OLD_CLAIM"),
    ],
)
def test_bad_llm_values_are_caught(field: str, value: object, flag: str) -> None:
    out = normalize_facts(extracted(**{field: value}), TODAY)
    assert flag in [f.code for f in out.flags]


def test_long_free_text_is_clipped() -> None:
    out = normalize_facts(extracted(reason_given_by_airline="x" * 1000), TODAY)
    assert out.facts.reason_given_by_airline is not None
    assert len(out.facts.reason_given_by_airline) == 200
