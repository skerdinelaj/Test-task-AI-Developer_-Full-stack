"""Deterministic validation of what the LLM extracted. Nothing reaches the tools unvalidated."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from eu261_claim_agent.schemas import ExtractedFacts, NormalizedFacts, RiskFlag, Severity

FLIGHT_NUMBER_RE = re.compile(r"^([A-Z][A-Z0-9]|[0-9][A-Z])(\d{1,4})$")
IATA_RE = re.compile(r"^[A-Z]{3}$")
MAX_REASONABLE_DELAY_MIN = 72 * 60
OLD_CLAIM_YEARS = 3
REQUIRED_FIELDS = ("flight_number", "flight_date")


@dataclass
class NormalizationResult:
    facts: NormalizedFacts
    missing_fields: list[str] = field(default_factory=list)
    flags: list[RiskFlag] = field(default_factory=list)


def normalize_flight_number(raw: str | None) -> str | None:
    """'lh 01452' -> 'LH1452'. Returns None if it does not look like a flight number."""
    if not raw:
        return None
    cleaned = re.sub(r"[\s\-]", "", raw).upper()
    match = FLIGHT_NUMBER_RE.match(cleaned)
    if not match:
        return None
    return f"{match.group(1)}{int(match.group(2))}"


def normalize_iata(raw: str | None) -> str | None:
    if not raw:
        return None
    code = raw.strip().upper()
    return code if IATA_RE.match(code) else None


def _clip(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text[:limit] or None


def normalize_facts(extracted: ExtractedFacts, today: date) -> NormalizationResult:
    flags: list[RiskFlag] = []
    missing: list[str] = []

    flight_number = normalize_flight_number(extracted.flight_number)
    if extracted.flight_number and flight_number is None:
        flags.append(
            RiskFlag(
                code="INVALID_FLIGHT_NUMBER",
                severity=Severity.WARNING,
                message="The flight number given is not in a valid format.",
            )
        )

    flight_date: date | None = None
    if extracted.flight_date:
        try:
            flight_date = date.fromisoformat(extracted.flight_date)
        except ValueError:
            flags.append(
                RiskFlag(
                    code="INVALID_FLIGHT_DATE",
                    severity=Severity.WARNING,
                    message="The flight date could not be read.",
                )
            )
    if flight_date and flight_date > today:
        flags.append(
            RiskFlag(
                code="FLIGHT_DATE_IN_FUTURE",
                severity=Severity.WARNING,
                message="The flight date is in the future; a claim is not possible yet.",
            )
        )
        flight_date = None
    if flight_date and flight_date < today - timedelta(days=365 * OLD_CLAIM_YEARS):
        flags.append(
            RiskFlag(
                code="OLD_CLAIM",
                severity=Severity.INFO,
                message="Flight is over 3 years old; the national time limit may "
                "have passed (it varies by country).",
            )
        )

    delay_minutes: int | None = None
    if extracted.reported_delay_hours is not None:
        minutes = round(extracted.reported_delay_hours * 60)
        if 0 <= minutes <= MAX_REASONABLE_DELAY_MIN:
            delay_minutes = minutes
        else:
            flags.append(
                RiskFlag(
                    code="IMPLAUSIBLE_DELAY",
                    severity=Severity.WARNING,
                    message="The reported delay is not plausible and was ignored.",
                )
            )

    notice = extracted.cancellation_notice_days
    if notice is not None and not 0 <= notice <= 365:
        notice = None

    facts = NormalizedFacts(
        passenger_name=_clip(extracted.passenger_name, 100),
        flight_number=flight_number,
        flight_date=flight_date,
        departure_airport=normalize_iata(extracted.departure_airport),
        arrival_airport=normalize_iata(extracted.arrival_airport),
        disruption_type=extracted.disruption_type,
        reported_delay_minutes=delay_minutes,
        cancellation_notice_days=notice,
        reason_given_by_airline=_clip(extracted.reason_given_by_airline, 200),
    )
    for name in REQUIRED_FIELDS:
        if getattr(facts, name) is None:
            missing.append(name)
    return NormalizationResult(facts=facts, missing_fields=missing, flags=flags)
