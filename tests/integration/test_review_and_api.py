"""Human-in-the-loop flow, durability across restarts, and the HTTP API (end to end)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from eu261_claim_agent.api import create_app
from eu261_claim_agent.config import Settings
from eu261_claim_agent.core.schemas import ReviewOutcome, ReviewState
from eu261_claim_agent.service import ClaimAgent, ReviewNotPendingError
from tests.conftest import AgentFactory, load_sample


def test_review_approve_flow(make_agent: AgentFactory) -> None:
    agent = make_agent()
    pending = agent.submit(load_sample("01_long_delay_eligible"))
    assert pending.review_state == ReviewState.PENDING
    assert [s.node for s in pending.steps][
        -1
    ] == "check_letter"  # paused before review_gate finished

    done = agent.review(pending.run_id, ReviewOutcome(approved=True, reviewer="maria", note="ok"))
    assert done.review_state == ReviewState.APPROVED
    assert done.review is not None and done.review.reviewer == "maria"
    assert [s.node for s in done.steps][-2:] == ["review_gate", "finalize"]
    assert "Send the approved letter" in done.summary

    with pytest.raises(ReviewNotPendingError):
        agent.review(pending.run_id, ReviewOutcome(approved=False, reviewer="maria"))


def test_review_reject_flow(make_agent: AgentFactory) -> None:
    agent = make_agent()
    pending = agent.submit(load_sample("05_weather_needs_review"))
    done = agent.review(
        pending.run_id, ReviewOutcome(approved=False, reviewer="li", note="weather was real")
    )
    assert done.review_state == ReviewState.REJECTED


def test_paused_claim_survives_a_restart(settings: Settings) -> None:
    """The SQLite checkpointer lets a claim wait days for a reviewer, across deployments."""
    with ClaimAgent(settings, persist=True) as first:
        run_id = first.submit(load_sample("04_cancellation_crew_strike")).run_id
    with ClaimAgent(settings, persist=True) as second:  # brand-new process, same database
        assert second.get(run_id).review_state == ReviewState.PENDING
        done = second.review(run_id, ReviewOutcome(approved=True, reviewer="ops"))
        assert done.review_state == ReviewState.APPROVED


# --------------------------------------------------------------------------- API


@pytest.fixture
def client(make_agent: AgentFactory) -> Iterator[TestClient]:
    with TestClient(create_app(agent=make_agent())) as c:
        yield c


def test_api_end_to_end(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    body = load_sample("07_tirana_rome_eu_carrier").model_dump(mode="json")
    created = client.post("/claims", json=body)
    assert created.status_code == 201
    claim = created.json()
    assert (claim["status"], claim["decision"]["amount_eur"], claim["review_state"]) == (
        "eligible",
        250,
        "pending",
    )

    run_id = claim["run_id"]
    assert client.get(f"/claims/{run_id}").json()["run_id"] == run_id
    trace = client.get(f"/claims/{run_id}/trace").json()
    assert any(e["event"] == "tool_call" for e in trace)

    reviewed = client.post(f"/claims/{run_id}/review", json={"approved": True, "reviewer": "ana"})
    assert reviewed.json()["review_state"] == "approved"
    again = client.post(f"/claims/{run_id}/review", json={"approved": True, "reviewer": "ana"})
    assert again.status_code == 409


def test_api_validation_and_not_found(client: TestClient) -> None:
    assert client.post("/claims", json={"message": ""}).status_code == 422
    assert client.post("/claims", json={"message": "x" * 8_001}).status_code == 422
    assert client.get("/claims/does-not-exist").status_code == 404
    assert client.get("/claims/does-not-exist/trace").status_code == 404
    bad_review = client.post("/claims/x/review", json={"approved": True, "reviewer": ""})
    assert bad_review.status_code == 422


def test_api_key_is_enforced_when_configured(make_agent: AgentFactory) -> None:
    agent = make_agent(api_key=SecretStr("s3cret"))
    with TestClient(create_app(agent=agent)) as c:
        body = {"message": "LH1452 on 2026-10-03 was 4 hours late", "received_on": "2026-11-15"}
        assert c.post("/claims", json=body).status_code == 401
        assert c.post("/claims", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/claims", json=body, headers={"X-API-Key": "s3cret"}).status_code == 201
