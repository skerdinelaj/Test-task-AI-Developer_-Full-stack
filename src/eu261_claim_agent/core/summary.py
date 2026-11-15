"""Human-readable summary, built by code from the structured result (so it can't contradict it)."""

from __future__ import annotations

from eu261_claim_agent.core.schemas import (
    ClaimStatus,
    EligibilityDecision,
    NormalizedFacts,
    ReviewOutcome,
    ReviewState,
    RiskFlag,
    Severity,
)

NEXT_STEPS: dict[ClaimStatus, str] = {
    ClaimStatus.ELIGIBLE: "A claims handler must approve the drafted letter before it is sent.",
    ClaimStatus.NOT_ELIGIBLE: "Tell the customer why no compensation is due.",
    ClaimStatus.NEEDS_INFO: "Ask the customer for the missing details, then run the claim again.",
    ClaimStatus.NEEDS_REVIEW: "A claims handler must decide this case manually.",
    ClaimStatus.OUT_OF_SCOPE: "Tell the customer this case is not covered; no letter is drafted.",
    ClaimStatus.REJECTED: "No action; the request was not processed.",
}

FIELD_LABELS = {"flight_number": "flight number", "flight_date": "flight date"}


def build_summary(
    status: ClaimStatus,
    facts: NormalizedFacts | None,
    decision: EligibilityDecision | None,
    reasons: list[str],
    missing_fields: list[str],
    flags: list[RiskFlag],
    review_state: ReviewState,
    review: ReviewOutcome | None,
) -> str:
    lines: list[str] = []
    if facts and facts.flight_number:
        route = ""
        if facts.departure_airport and facts.arrival_airport:
            route = f" ({facts.departure_airport}→{facts.arrival_airport})"
        when = f" on {facts.flight_date.isoformat()}" if facts.flight_date else ""
        lines.append(f"Claim for flight {facts.flight_number}{when}{route}.")
    headline = status.value.replace("_", " ").upper()
    if decision and status == ClaimStatus.ELIGIBLE:
        headline += f" for EUR {decision.amount_eur}"
    lines.append(f"Result: {headline}.")
    why = decision.reasons if decision else reasons
    if why:
        lines.append("Why: " + " ".join(why))
    if missing_fields:
        lines.append("Missing: " + ", ".join(FIELD_LABELS.get(f, f) for f in missing_fields) + ".")
    serious = [f for f in flags if f.severity in (Severity.HIGH, Severity.WARNING)]
    if serious:
        lines.append("Watch out: " + " ".join(f"[{f.code}] {f.message}" for f in serious))
    if review_state == ReviewState.PENDING:
        lines.append("Status: waiting for human review.")
    elif review and review_state in (ReviewState.APPROVED, ReviewState.REJECTED):
        lines.append(
            f"Review: {review_state.value} by {review.reviewer}"
            + (f" ({review.note})" if review.note else "")
            + "."
        )
    if review_state == ReviewState.APPROVED:
        next_step = (
            "Send the approved letter to the airline."
            if status == ClaimStatus.ELIGIBLE
            else "Act on the handler's decision."
        )
    elif review_state == ReviewState.REJECTED:
        next_step = "The handler rejected the agent's output; revise the case manually."
    else:
        next_step = NEXT_STEPS[status]
    lines.append("Next step: " + next_step)
    return "\n".join(lines)
