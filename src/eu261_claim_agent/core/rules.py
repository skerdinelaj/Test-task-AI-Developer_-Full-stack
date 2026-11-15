"""Deterministic EU261 rules engine.

This is the only place where eligibility and money are decided. It is pure Python with no LLM
involvement, so it is fully unit-testable and a prompt injection cannot change the outcome.
Decisions are always based on the *flight record*, never on what the customer claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from eu261_claim_agent.core.schemas import (
    ClaimStatus,
    DisruptionType,
    EligibilityDecision,
    Evidence,
    NormalizedFacts,
    RiskFlag,
    RulesConfig,
    Severity,
)

Band = Literal["short", "medium", "long"]

REASON_EXPLANATIONS: dict[str, str] = {
    "TECH": "technical problems are part of normal airline operations (CJEU C-549/07)",
    "CREW_STRIKE": "a strike by the airline's own staff is not extraordinary (CJEU C-28/20)",
    "LATE_INBOUND": "a late incoming aircraft is an ordinary operational issue",
    "SCHEDULE_CHANGE": "a commercial schedule change is within the airline's control",
    "OVERBOOKING": "overbooking is within the airline's control",
}


@dataclass
class RulesOutcome:
    decision: EligibilityDecision
    flags: list[RiskFlag] = field(default_factory=list)


def compensation_band(distance_km: float, intra_eu: bool, rules: RulesConfig) -> tuple[Band, int]:
    """Article 7(1): pick the distance band and its fixed amount."""
    bands, amounts = rules.distance_bands_km, rules.amounts_eur
    if distance_km <= bands.short_max:
        return "short", amounts.short
    if intra_eu or distance_km <= bands.medium_max:
        return "medium", amounts.medium
    return "long", amounts.long


def _fmt_minutes(minutes: int) -> str:
    return f"{minutes // 60} h {minutes % 60:02d} min"


def evaluate_claim(facts: NormalizedFacts, evidence: Evidence, rules: RulesConfig) -> RulesOutcome:
    flags: list[RiskFlag] = []
    reasons: list[str] = []
    rule_ids: list[str] = []

    def decide(
        status: ClaimStatus, eligible: bool | None, amount: int = 0, delay: int | None = None
    ) -> RulesOutcome:
        return RulesOutcome(
            decision=EligibilityDecision(
                status=status,
                eligible=eligible,
                amount_eur=amount,
                distance_km=round(evidence.distance_km, 1) if evidence.distance_km else None,
                arrival_delay_minutes=delay,
                reasons=reasons,
                rule_ids=rule_ids,
            ),
            flags=flags,
        )

    flight = evidence.flight
    if flight is None:
        rule_ids.append("FLIGHT_NOT_FOUND")
        reasons.append(
            "The flight could not be found in the flight records, so the facts cannot be "
            "verified. A handler must check it manually."
        )
        flags.append(
            RiskFlag(
                code="FLIGHT_NOT_VERIFIED",
                severity=Severity.HIGH,
                message="No flight record matches the flight number and date.",
            )
        )
        return decide(ClaimStatus.NEEDS_REVIEW, None)

    origin, destination, airline = evidence.origin, evidence.destination, evidence.airline
    if origin is None or destination is None or airline is None or evidence.distance_km is None:
        rule_ids.append("REFERENCE_DATA_MISSING")
        reasons.append("Airport or airline reference data is missing for this route.")
        return decide(ClaimStatus.NEEDS_REVIEW, None)

    # --- consistency between the customer's story and the record -----------------
    claimed_route = (facts.departure_airport, facts.arrival_airport)
    if all(claimed_route) and claimed_route != (flight.origin, flight.destination):
        flags.append(
            RiskFlag(
                code="ROUTE_MISMATCH",
                severity=Severity.WARNING,
                message=f"Customer said {claimed_route[0]}→{claimed_route[1]}, record shows "
                f"{flight.origin}→{flight.destination}. Decision uses the record.",
            )
        )

    # --- scope (Article 3) ----------------------------------------------------------
    origin_eu = origin.country in rules.eu261_countries
    dest_eu = destination.country in rules.eu261_countries
    if origin_eu:
        rule_ids.append("SCOPE_DEPARTS_EU")
        reasons.append(f"Covered: the flight departed from {origin.city}, inside the EU261 area.")
    elif dest_eu and airline.eu_carrier:
        rule_ids.append("SCOPE_EU_CARRIER_INBOUND")
        reasons.append(
            f"Covered: the flight arrived in the EU and {airline.name} is an EU carrier."
        )
    else:
        rule_ids.append("SCOPE_NOT_COVERED")
        why = (
            f"the flight departed from {origin.city} ({origin.country}), outside the EU261 area, "
            f"and {airline.name} is not an EU carrier"
        )
        if origin.country == "GB":
            why += "; flights from the UK fall under the UK's own rules (UK261), not EU261"
        reasons.append(f"Not covered by EU261: {why}.")
        return decide(ClaimStatus.OUT_OF_SCOPE, False)

    intra_eu = origin_eu and dest_eu
    band, base_amount = compensation_band(evidence.distance_km, intra_eu, rules)
    rule_ids.append(f"BAND_{band.upper()}")
    reasons.append(
        f"Distance {evidence.distance_km:,.0f} km"
        f"{' (within the EU)' if intra_eu else ''}: EUR {base_amount} band."
    )

    # --- what actually happened, according to the record ---------------------------
    delay = flight.arrival_delay_minutes
    if flight.status == "cancelled":
        disruption = DisruptionType.CANCELLATION
    elif facts.disruption_type == DisruptionType.DENIED_BOARDING:
        disruption = DisruptionType.DENIED_BOARDING
    else:
        disruption = DisruptionType.DELAY

    if facts.disruption_type not in (DisruptionType.UNKNOWN, disruption):
        flags.append(
            RiskFlag(
                code="DISRUPTION_TYPE_MISMATCH",
                severity=Severity.WARNING,
                message=f"Customer described a {facts.disruption_type.value}, the record shows a "
                f"{disruption.value}. Decision uses the record.",
            )
        )

    # --- extraordinary circumstances (Article 5(3)) ---------------------------------
    reason_code = flight.disruption_reason_code
    if reason_code and reason_code in rules.extraordinary_reason_codes:
        rule_ids.append("EXTRAORDINARY_CIRCUMSTANCES_CLAIMED")
        reasons.append(
            f"The airline reports reason '{reason_code}', which is usually accepted as an "
            f"extraordinary circumstance. A handler should check whether it really applies; if "
            f"not, the claim would be worth EUR {base_amount}."
        )
        flags.append(
            RiskFlag(
                code="EXTRAORDINARY_CIRCUMSTANCES",
                severity=Severity.WARNING,
                message=f"Airline cites '{reason_code}'.",
            )
        )
        return decide(ClaimStatus.NEEDS_REVIEW, None, delay=delay)
    if reason_code and reason_code in REASON_EXPLANATIONS:
        reasons.append(
            f"Airline reason '{reason_code}' does not excuse the airline: "
            f"{REASON_EXPLANATIONS[reason_code]}."
        )
    elif reason_code and reason_code not in rules.not_extraordinary_reason_codes:
        flags.append(
            RiskFlag(
                code="UNKNOWN_REASON_CODE",
                severity=Severity.INFO,
                message=f"Unrecognised airline reason code '{reason_code}'.",
            )
        )

    # --- delay -----------------------------------------------------------------------
    if disruption == DisruptionType.DELAY:
        if delay is None:
            rule_ids.append("DELAY_UNKNOWN")
            reasons.append("The record has no actual arrival time.")
            return decide(ClaimStatus.NEEDS_REVIEW, None)
        reported = facts.reported_delay_minutes
        tolerance = rules.reported_vs_record_tolerance_minutes
        if reported is not None and abs(reported - delay) > tolerance:
            flags.append(
                RiskFlag(
                    code="REPORTED_DELAY_MISMATCH",
                    severity=Severity.WARNING,
                    message=f"Customer reported {_fmt_minutes(reported)}, record shows "
                    f"{_fmt_minutes(max(delay, 0))}. Decision uses the record.",
                )
            )
        if delay < rules.delay_threshold_minutes:
            rule_ids.append("DELAY_BELOW_3H")
            reasons.append(
                f"The flight arrived {_fmt_minutes(max(delay, 0))} late; compensation "
                f"starts at 3 h 00 min."
            )
            return decide(ClaimStatus.NOT_ELIGIBLE, False, delay=delay)
        rule_ids.append("DELAY_AT_LEAST_3H")
        reasons.append(f"The flight arrived {_fmt_minutes(delay)} late (3 h or more).")
        amount = base_amount
        if band == "long" and not intra_eu and delay <= rules.long_haul_reduction_max_delay_minutes:
            amount = int(base_amount * rules.long_haul_reduction_factor)
            rule_ids.append("LONG_HAUL_REDUCTION_50")
            reasons.append(
                "Long flight delayed by less than 4 h: amount reduced by 50% (Art. 7(2))."
            )
        return decide(ClaimStatus.ELIGIBLE, True, amount, delay=delay)

    # --- cancellation ------------------------------------------------------------------
    if disruption == DisruptionType.CANCELLATION:
        notice = flight.cancellation_notice_days
        if notice is None:
            notice = facts.cancellation_notice_days
        if notice is None:
            rule_ids.append("CANCELLATION_NOTICE_UNKNOWN")
            reasons.append(
                "It is not known how early the passenger was told about the cancellation."
            )
            return decide(ClaimStatus.NEEDS_REVIEW, None)
        if notice >= rules.cancellation_full_notice_days:
            rule_ids.append("CANCELLATION_NOTICE_14D_PLUS")
            reasons.append(
                f"The passenger was told {notice} days ahead (14 or more): no compensation."
            )
            return decide(ClaimStatus.NOT_ELIGIBLE, False)
        if notice >= rules.cancellation_partial_notice_days:
            rule_ids.append("CANCELLATION_NOTICE_7_13D")
            reasons.append(
                f"The passenger was told {notice} days ahead. Between 7 and 13 days, compensation "
                f"depends on the times of the re-routing offer, which a handler must check."
            )
            return decide(ClaimStatus.NEEDS_REVIEW, None)
        rule_ids.append("CANCELLATION_NOTICE_UNDER_7D")
        reasons.append(f"The passenger was told only {notice} day(s) ahead (less than 7).")
        flags.append(
            RiskFlag(
                code="REROUTING_NOT_VERIFIED",
                severity=Severity.INFO,
                message="Check that no re-routing within the Art. 5(1)(c)(iii) "
                "time window was offered.",
            )
        )
        return decide(ClaimStatus.ELIGIBLE, True, base_amount)

    # --- denied boarding -------------------------------------------------------------------
    rule_ids.append("DENIED_BOARDING_INVOLUNTARY")
    reasons.append("The passenger reports being denied boarding against their will.")
    flags.append(
        RiskFlag(
            code="DENIED_BOARDING_UNVERIFIED",
            severity=Severity.WARNING,
            message="Flight records cannot confirm denied boarding; check the "
            "boarding pass and airline correspondence.",
        )
    )
    return decide(ClaimStatus.ELIGIBLE, True, base_amount)
