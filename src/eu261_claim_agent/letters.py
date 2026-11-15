"""Claim-letter case file, deterministic output checks, and the safe template fallback."""

from __future__ import annotations

import re

from pydantic import BaseModel

from eu261_claim_agent.schemas import EligibilityDecision, Evidence, LetterDraft, NormalizedFacts

AMOUNT_RE = re.compile(
    r"(?:€|\bEUR\b)\s?(\d{1,3}(?:[.,\s]\d{3})*|\d+)(?:[.,]\d{2})?"
    r"|(\d{1,3}(?:[.,\s]\d{3})*|\d+)(?:[.,]\d{2})?\s?(?:€|\bEUR\b|\beuros?\b)",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(
    r"\[[^\]]{1,40}\]|\{\{|\}\}|<[A-Za-z_ ]{2,30}>|\bTBD\b|\bXXX\b|\blorem\b", re.IGNORECASE
)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class CaseFile(BaseModel):
    """Everything the drafting step is allowed to know. Built by code from verified data only."""

    flight_number: str
    flight_date: str
    airline_name: str
    origin: str
    destination: str
    disruption: str
    arrival_delay_minutes: int | None
    distance_km: float | None
    amount_eur: int
    reasons: list[str]
    passenger_name: str | None
    legal_basis_passages: list[str]

    @classmethod
    def build(
        cls, facts: NormalizedFacts, decision: EligibilityDecision, evidence: Evidence
    ) -> CaseFile:
        flight, airline = evidence.flight, evidence.airline
        # guaranteed by the rules engine: a letter is only drafted for a verified flight
        assert flight is not None
        assert airline is not None
        if flight.status == "cancelled":
            disruption = "cancellation"
        elif facts.disruption_type.value == "denied_boarding":
            disruption = "denied boarding"
        else:
            disruption = "long delay"
        origin = evidence.origin.city if evidence.origin else flight.origin
        destination = evidence.destination.city if evidence.destination else flight.destination
        return cls(
            flight_number=flight.flight_number,
            flight_date=flight.date.isoformat(),
            airline_name=airline.name,
            origin=f"{origin} ({flight.origin})",
            destination=f"{destination} ({flight.destination})",
            disruption=disruption,
            arrival_delay_minutes=decision.arrival_delay_minutes,
            distance_km=decision.distance_km,
            amount_eur=decision.amount_eur,
            reasons=decision.reasons,
            passenger_name=facts.passenger_name,
            legal_basis_passages=[s.text for s in evidence.regulation[:4]],
        )


def _amounts_in(text: str) -> set[int]:
    found: set[int] = set()
    for match in AMOUNT_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        digits = re.sub(r"[.,\s]", "", raw)
        if digits.isdigit():
            found.add(int(digits))
    return found


def validate_letter(letter: LetterDraft, case: CaseFile) -> list[str]:
    """Return a list of problems; empty means the letter is safe to show a handler."""
    problems: list[str] = []
    text = f"{letter.subject}\n{letter.body}"
    if case.flight_number not in text.replace(" ", ""):
        problems.append(f"The letter must mention flight {case.flight_number}.")
    amounts = _amounts_in(text)
    if case.amount_eur not in amounts:
        problems.append(f"The letter must claim exactly EUR {case.amount_eur}.")
    wrong = sorted(a for a in amounts if a != case.amount_eur)
    if wrong:
        problems.append(
            f"The letter mentions other amounts {wrong}; only EUR {case.amount_eur} is allowed."
        )
    if "261/2004" not in text:
        problems.append("The letter must cite Regulation (EC) No 261/2004.")
    if PLACEHOLDER_RE.search(text):
        problems.append("The letter contains a placeholder; write the final text.")
    if URL_RE.search(text) or EMAIL_RE.search(text):
        problems.append("The letter must not contain links or email addresses.")
    if not 300 <= len(letter.body) <= 6_000:
        problems.append("The letter body must be between 300 and 6000 characters.")
    return problems


def _delay_text(minutes: int | None) -> str:
    if minutes is None:
        return ""
    return f", arriving {minutes // 60} hours and {minutes % 60} minutes late"


def render_template_letter(case: CaseFile) -> LetterDraft:
    """Deterministic fallback used when the LLM draft fails validation (and by the offline mode)."""
    passenger = case.passenger_name or "our client"
    what = {
        "long delay": f"was delayed{_delay_text(case.arrival_delay_minutes)}",
        "cancellation": "was cancelled at short notice",
        "denied boarding": "was operated, but the passenger was denied boarding against their will",
    }.get(case.disruption, "was disrupted")
    body = (
        f"Dear {case.airline_name} Customer Relations,\n\n"
        f"We act on behalf of {passenger} regarding flight {case.flight_number} on "
        f"{case.flight_date} from {case.origin} to {case.destination}, which {what}.\n\n"
        f"Under Regulation (EC) No 261/2004, Articles 5, 7 and the case law of the Court of "
        f"Justice of the EU, the passenger is entitled to compensation of EUR {case.amount_eur} "
        f"for this {case.disruption}. The flight distance is "
        f"{case.distance_km:,.0f} km.\n\n"
        "Based on the information available to us, no extraordinary circumstances apply.\n\n"
        f"Please pay EUR {case.amount_eur} within 14 days of this letter and send your reply "
        "to our claims team.\n\n"
        "Yours faithfully,\nClaims Team"
    )
    return LetterDraft(
        subject=f"EU261 compensation claim - flight {case.flight_number} on {case.flight_date}",
        body=body,
    )
