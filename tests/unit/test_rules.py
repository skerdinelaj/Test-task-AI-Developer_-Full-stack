"""The rules engine decides money, so it gets the most thorough tests, including boundaries."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from eu261_claim_agent.geo import great_circle_km
from eu261_claim_agent.reference_data import ReferenceData
from eu261_claim_agent.rules import compensation_band, evaluate_claim
from eu261_claim_agent.schemas import (
    ClaimStatus,
    DisruptionType,
    Evidence,
    FlightRecord,
    NormalizedFacts,
)

D = date(2026, 10, 1)


def evidence_for(
    ref: ReferenceData,
    origin: str,
    dest: str,
    airline: str,
    *,
    delay: int | None = 0,
    cancelled: bool = False,
    notice: int | None = None,
    reason: str | None = None,
    distance: float | None = None,
) -> Evidence:
    sched = datetime(2026, 10, 1, 12, 0)
    flight = FlightRecord(
        flight_number=f"{airline}100",
        date=D,
        airline=airline,
        origin=origin,
        destination=dest,
        scheduled_arrival=sched,
        actual_arrival=None if cancelled or delay is None else sched + timedelta(minutes=delay),
        status="cancelled" if cancelled else "landed",
        cancellation_notice_days=notice,
        disruption_reason_code=reason,
    )
    o, d = ref.airports[origin], ref.airports[dest]
    km = distance if distance is not None else great_circle_km(o.lat, o.lon, d.lat, d.lon)
    return Evidence(
        flight=flight, airline=ref.airlines[airline], origin=o, destination=d, distance_km=km
    )


def facts(kind: DisruptionType = DisruptionType.DELAY, **kw: object) -> NormalizedFacts:
    return NormalizedFacts(flight_number="XX100", flight_date=D, disruption_type=kind, **kw)


# --------------------------------------------------------------------------- bands


@pytest.mark.parametrize(
    ("km", "intra", "expected"),
    [
        (100, False, 250),
        (1500, False, 250),
        (1500.1, False, 400),
        (3500, False, 400),
        (3500.1, False, 600),
        (3500.1, True, 400),
        (9000, True, 400),
    ],
)
def test_compensation_band_boundaries(
    ref: ReferenceData, km: float, intra: bool, expected: int
) -> None:
    assert compensation_band(km, intra, ref.rules)[1] == expected


# --------------------------------------------------------------------------- delays


@pytest.mark.parametrize(
    ("delay", "status", "amount"),
    [
        (179, ClaimStatus.NOT_ELIGIBLE, 0),
        (180, ClaimStatus.ELIGIBLE, 400),
        (600, ClaimStatus.ELIGIBLE, 400),
        (-10, ClaimStatus.NOT_ELIGIBLE, 0),  # early arrival
    ],
)
def test_delay_threshold_is_exactly_three_hours(
    ref: ReferenceData, delay: int, status: ClaimStatus, amount: int
) -> None:
    out = evaluate_claim(facts(), evidence_for(ref, "FRA", "LIS", "LH", delay=delay), ref.rules)
    assert (out.decision.status, out.decision.amount_eur) == (status, amount)


@pytest.mark.parametrize(("delay", "amount"), [(180, 300), (239, 300), (240, 600)])
def test_long_haul_reduction_between_three_and_four_hours(
    ref: ReferenceData, delay: int, amount: int
) -> None:
    out = evaluate_claim(facts(), evidence_for(ref, "CDG", "JFK", "AF", delay=delay), ref.rules)
    assert out.decision.amount_eur == amount
    assert ("LONG_HAUL_REDUCTION_50" in out.decision.rule_ids) == (amount == 300)


def test_intra_eu_over_3500_km_is_400_without_reduction(ref: ReferenceData) -> None:
    out = evaluate_claim(facts(), evidence_for(ref, "CDG", "RUN", "AF", delay=200), ref.rules)
    assert out.decision.amount_eur == 400
    assert "LONG_HAUL_REDUCTION_50" not in out.decision.rule_ids


def test_decision_uses_record_not_customer_claim(ref: ReferenceData) -> None:
    out = evaluate_claim(
        facts(reported_delay_minutes=300),
        evidence_for(ref, "FCO", "CDG", "AZ", delay=120),
        ref.rules,
    )
    assert out.decision.status == ClaimStatus.NOT_ELIGIBLE
    assert [f.code for f in out.flags] == ["REPORTED_DELAY_MISMATCH"]


def test_reported_delay_within_tolerance_is_not_flagged(ref: ReferenceData) -> None:
    out = evaluate_claim(
        facts(reported_delay_minutes=240),
        evidence_for(ref, "FRA", "LIS", "LH", delay=250),
        ref.rules,
    )
    assert out.flags == []


# --------------------------------------------------------------------------- scope


@pytest.mark.parametrize(
    ("origin", "dest", "airline", "status"),
    [
        ("FRA", "JFK", "DL", ClaimStatus.ELIGIBLE),  # departs EU, non-EU airline: covered
        ("TIA", "FCO", "W6", ClaimStatus.ELIGIBLE),  # into EU on EU carrier: covered
        ("IST", "FCO", "TK", ClaimStatus.OUT_OF_SCOPE),  # into EU on non-EU carrier
        ("TIA", "IST", "TK", ClaimStatus.OUT_OF_SCOPE),  # never touches the EU
        ("LHR", "JFK", "BA", ClaimStatus.OUT_OF_SCOPE),  # UK261, not EU261
        ("OSL", "CPH", "SK", ClaimStatus.ELIGIBLE),  # Norway is in scope (EEA)
        ("ZRH", "ATH", "LX", ClaimStatus.ELIGIBLE),  # Switzerland is in scope
    ],
)
def test_scope(
    ref: ReferenceData, origin: str, dest: str, airline: str, status: ClaimStatus
) -> None:
    out = evaluate_claim(facts(), evidence_for(ref, origin, dest, airline, delay=300), ref.rules)
    assert out.decision.status == status


def test_uk_out_of_scope_reason_mentions_uk261(ref: ReferenceData) -> None:
    out = evaluate_claim(facts(), evidence_for(ref, "LHR", "JFK", "BA", delay=300), ref.rules)
    assert any("UK261" in r for r in out.decision.reasons)


# --------------------------------------------------------------------------- cancellations


@pytest.mark.parametrize(
    ("notice", "status"),
    [
        (0, ClaimStatus.ELIGIBLE),
        (6, ClaimStatus.ELIGIBLE),
        (7, ClaimStatus.NEEDS_REVIEW),
        (13, ClaimStatus.NEEDS_REVIEW),
        (14, ClaimStatus.NOT_ELIGIBLE),
        (60, ClaimStatus.NOT_ELIGIBLE),
    ],
)
def test_cancellation_notice_windows(ref: ReferenceData, notice: int, status: ClaimStatus) -> None:
    ev = evidence_for(ref, "BUD", "TIA", "W6", cancelled=True, notice=notice)
    assert (
        evaluate_claim(facts(DisruptionType.CANCELLATION), ev, ref.rules).decision.status == status
    )


def test_cancellation_falls_back_to_customer_notice(ref: ReferenceData) -> None:
    ev = evidence_for(ref, "BUD", "TIA", "W6", cancelled=True, notice=None)
    out = evaluate_claim(
        facts(DisruptionType.CANCELLATION, cancellation_notice_days=3), ev, ref.rules
    )
    assert out.decision.status == ClaimStatus.ELIGIBLE


def test_cancellation_unknown_notice_needs_review(ref: ReferenceData) -> None:
    ev = evidence_for(ref, "BUD", "TIA", "W6", cancelled=True, notice=None)
    assert (
        evaluate_claim(facts(DisruptionType.CANCELLATION), ev, ref.rules).decision.status
        == ClaimStatus.NEEDS_REVIEW
    )


# ----------------------------------------------------------------- extraordinary circumstances


@pytest.mark.parametrize("reason", ["WX", "ATC", "SEC"])
def test_extraordinary_circumstances_go_to_human(ref: ReferenceData, reason: str) -> None:
    out = evaluate_claim(
        facts(), evidence_for(ref, "CPH", "OSL", "SK", delay=300, reason=reason), ref.rules
    )
    assert out.decision.status == ClaimStatus.NEEDS_REVIEW
    assert out.decision.amount_eur == 0


@pytest.mark.parametrize("reason", ["TECH", "CREW_STRIKE", "LATE_INBOUND"])
def test_ordinary_reasons_do_not_excuse_airline(ref: ReferenceData, reason: str) -> None:
    out = evaluate_claim(
        facts(), evidence_for(ref, "FRA", "LIS", "LH", delay=300, reason=reason), ref.rules
    )
    assert out.decision.status == ClaimStatus.ELIGIBLE


def test_unknown_reason_code_is_flagged_not_fatal(ref: ReferenceData) -> None:
    out = evaluate_claim(
        facts(), evidence_for(ref, "FRA", "LIS", "LH", delay=300, reason="ZZZ"), ref.rules
    )
    assert out.decision.status == ClaimStatus.ELIGIBLE
    assert "UNKNOWN_REASON_CODE" in [f.code for f in out.flags]


# --------------------------------------------------------------------------- other paths


def test_denied_boarding_is_eligible_but_flagged(ref: ReferenceData) -> None:
    out = evaluate_claim(
        facts(DisruptionType.DENIED_BOARDING),
        evidence_for(ref, "BCN", "DUB", "FR", delay=0),
        ref.rules,
    )
    assert (out.decision.status, out.decision.amount_eur) == (ClaimStatus.ELIGIBLE, 250)
    assert "DENIED_BOARDING_UNVERIFIED" in [f.code for f in out.flags]


def test_missing_flight_record_needs_review(ref: ReferenceData) -> None:
    out = evaluate_claim(facts(), Evidence(), ref.rules)
    assert out.decision.status == ClaimStatus.NEEDS_REVIEW
    assert out.flags[0].code == "FLIGHT_NOT_VERIFIED"


def test_route_mismatch_is_flagged(ref: ReferenceData) -> None:
    out = evaluate_claim(
        facts(departure_airport="MUC", arrival_airport="LIS"),
        evidence_for(ref, "FRA", "LIS", "LH", delay=300),
        ref.rules,
    )
    assert "ROUTE_MISMATCH" in [f.code for f in out.flags]


def test_customer_says_delay_record_says_cancelled(ref: ReferenceData) -> None:
    ev = evidence_for(ref, "BUD", "TIA", "W6", cancelled=True, notice=1)
    out = evaluate_claim(facts(DisruptionType.DELAY), ev, ref.rules)
    assert out.decision.status == ClaimStatus.ELIGIBLE
    assert "DISRUPTION_TYPE_MISMATCH" in [f.code for f in out.flags]
