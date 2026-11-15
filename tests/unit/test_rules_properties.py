"""Property-based tests: invariants that must hold for *any* flight, not just hand-picked ones."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from eu261_claim_agent.config import PROJECT_ROOT
from eu261_claim_agent.reference_data import ReferenceData
from eu261_claim_agent.rules import evaluate_claim
from eu261_claim_agent.schemas import ClaimStatus, DisruptionType
from tests.unit.test_rules import evidence_for, facts

REF = ReferenceData(PROJECT_ROOT / "data")
AIRPORTS = sorted(REF.airports)
AIRLINES = sorted(REF.airlines)
VALID_AMOUNTS = {0, 250, 300, 400, 600}

routes = st.tuples(st.sampled_from(AIRPORTS), st.sampled_from(AIRPORTS)).filter(
    lambda r: r[0] != r[1]
)
delays = st.integers(min_value=-60, max_value=24 * 60)
reasons = st.sampled_from([None, "TECH", "WX", "ATC", "CREW_STRIKE", "LATE_INBOUND"])


@settings(max_examples=300, deadline=None)
@given(
    route=routes,
    airline=st.sampled_from(AIRLINES),
    delay=delays,
    reason=reasons,
    kind=st.sampled_from(list(DisruptionType)),
)
def test_amount_is_always_a_legal_value(
    route: tuple[str, str], airline: str, delay: int, reason: str | None, kind: DisruptionType
) -> None:
    out = evaluate_claim(
        facts(kind), evidence_for(REF, *route, airline, delay=delay, reason=reason), REF.rules
    )
    d = out.decision
    assert d.amount_eur in VALID_AMOUNTS
    assert (d.status == ClaimStatus.ELIGIBLE) == (d.amount_eur > 0)
    assert (d.eligible is True) == (d.status == ClaimStatus.ELIGIBLE)
    assert d.reasons, "every decision must explain itself"


@settings(max_examples=200, deadline=None)
@given(route=routes, airline=st.sampled_from(AIRLINES), d1=delays, d2=delays)
def test_longer_delay_never_pays_less(
    route: tuple[str, str], airline: str, d1: int, d2: int
) -> None:
    short, long_ = sorted((d1, d2))
    a = evaluate_claim(facts(), evidence_for(REF, *route, airline, delay=short), REF.rules).decision
    b = evaluate_claim(facts(), evidence_for(REF, *route, airline, delay=long_), REF.rules).decision
    assert b.amount_eur >= a.amount_eur


@settings(max_examples=200, deadline=None)
@given(
    route=routes,
    airline=st.sampled_from(AIRLINES),
    delay=delays,
    reported=st.integers(min_value=0, max_value=72 * 60),
)
def test_customer_reported_delay_never_changes_the_decision(
    route: tuple[str, str], airline: str, delay: int, reported: int
) -> None:
    ev = evidence_for(REF, *route, airline, delay=delay)
    base = evaluate_claim(facts(), ev, REF.rules).decision
    claimed = evaluate_claim(facts(reported_delay_minutes=reported), ev, REF.rules).decision
    assert (claimed.status, claimed.amount_eur) == (base.status, base.amount_eur)
