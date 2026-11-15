"""Letter template and letter validation."""

from __future__ import annotations

import pytest

from eu261_claim_agent.core.letters import CaseFile, render_template_letter, validate_letter
from eu261_claim_agent.core.schemas import (
    LetterDraft,
)


@pytest.fixture
def case() -> CaseFile:
    return CaseFile(
        flight_number="LH1452",
        flight_date="2026-10-03",
        airline_name="Lufthansa",
        origin="Frankfurt (FRA)",
        destination="Lisbon (LIS)",
        disruption="long delay",
        arrival_delay_minutes=250,
        distance_km=1874.0,
        amount_eur=400,
        reasons=[],
        passenger_name="Anna Rossi",
        legal_basis_passages=[],
    )


def test_template_letter_passes_its_own_validation(case: CaseFile) -> None:
    assert validate_letter(render_template_letter(case), case) == []


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda b: b.replace("EUR 400", "EUR 600"), "exactly EUR 400"),
        (lambda b: b + "\nWe also claim EUR 5000 damages.", "other amounts"),
        (lambda b: b.replace("LH1452", "LH9999"), "mention flight"),
        (lambda b: b + "\nPay here: https://evil.example", "links"),
        (lambda b: b + "\nSigned, [YOUR NAME]", "placeholder"),
        (lambda b: b.replace("261/2004", "the law"), "261/2004"),
    ],
)
def test_letter_validation_catches_bad_drafts(case: CaseFile, mutate: object, problem: str) -> None:
    good = render_template_letter(case)
    bad = LetterDraft(subject="Compensation claim", body=mutate(good.body))  # type: ignore[operator]
    assert any(problem in p for p in validate_letter(bad, case))
