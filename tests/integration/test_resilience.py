"""What happens when the LLM misbehaves or fails: the agent must degrade safely, never guess."""

from __future__ import annotations

import pytest

from eu261_claim_agent.llm.base import LLMOutputInvalid, LLMRefusal, LLMUnavailable
from eu261_claim_agent.llm.fake import FakeLLM
from eu261_claim_agent.schemas import ClaimStatus, LetterDraft, LLMTask, ReviewState
from tests.conftest import AgentFactory, load_sample
from tests.helpers import ScriptedLLM, call_tools, finish

LONG_DELAY = "01_long_delay_eligible"


def codes(result: object) -> set[str]:
    return {f.code for f in result.risk_flags}  # type: ignore[attr-defined]


def test_planner_outage_falls_back_to_deterministic_evidence(
    make_agent: AgentFactory, fake_llm: FakeLLM
) -> None:
    llm = ScriptedLLM(fake_llm, plan=[LLMUnavailable("503")])
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert (result.status, result.decision and result.decision.amount_eur) == (
        ClaimStatus.ELIGIBLE,
        400,
    )
    assert {"PLANNER_DEGRADED", "EVIDENCE_BACKFILLED"} <= codes(result)
    assert {c.source for c in result.tool_calls} == {"backfill"}


def test_lazy_planner_is_backfilled(make_agent: AgentFactory, fake_llm: FakeLLM) -> None:
    llm = ScriptedLLM(
        fake_llm,
        plan=[
            call_tools(("lookup_flight", {"flight_number": "LH1452", "flight_date": "2026-10-03"})),
            finish("I think that is enough."),
        ],
    )
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert result.status == ClaimStatus.ELIGIBLE
    backfilled = {c.tool for c in result.tool_calls if c.source == "backfill"}
    assert backfilled == {"get_airline", "calculate_distance", "search_regulation"}


def test_off_target_lookup_is_ignored(make_agent: AgentFactory, fake_llm: FakeLLM) -> None:
    llm = ScriptedLLM(
        fake_llm,
        plan=[
            call_tools(("lookup_flight", {"flight_number": "DL45", "flight_date": "2026-10-08"})),
            finish(),
        ],
    )
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert "PLANNER_OFF_TARGET" in codes(result)
    assert (
        result.decision is not None and result.decision.amount_eur == 400
    )  # from the right flight


def test_hallucinated_tool_and_bad_arguments_are_contained(
    make_agent: AgentFactory, fake_llm: FakeLLM
) -> None:
    llm = ScriptedLLM(
        fake_llm,
        plan=[
            call_tools(
                ("send_email", {"to": "x@example.com"}),
                ("lookup_flight", {"flight_number": "???", "flight_date": "yesterday"}),
            ),
            finish(),
        ],
    )
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    failed = [c for c in result.tool_calls if not c.ok]
    assert {c.tool for c in failed} == {"send_email", "lookup_flight"}
    assert result.status == ClaimStatus.ELIGIBLE


def test_runaway_planner_stops_at_step_limit(make_agent: AgentFactory, fake_llm: FakeLLM) -> None:
    loop = call_tools(("search_regulation", {"query": "delay", "top_k": 1}))
    llm = ScriptedLLM(fake_llm, plan=[loop], repeat_last_plan=True)
    result = make_agent(llm=llm, max_plan_steps=3).submit(load_sample(LONG_DELAY))
    assert llm.calls.count("plan") == 3
    assert "PLAN_LIMIT_REACHED" in codes(result)
    assert result.status == ClaimStatus.ELIGIBLE


def test_bad_letter_is_retried_with_feedback(make_agent: AgentFactory, fake_llm: FakeLLM) -> None:
    bad = LetterDraft(subject="Claim", body="Please pay EUR 600 for flight LH1452. " * 10)
    llm = ScriptedLLM(fake_llm, structured={LLMTask.DRAFT: [bad]})  # 2nd attempt uses the good fake
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert llm.calls.count("draft") == 2
    assert result.letter is not None and "EUR 400" in result.letter.body
    assert "LETTER_TEMPLATE_FALLBACK" not in codes(result)


def test_persistently_bad_letter_falls_back_to_template(
    make_agent: AgentFactory, fake_llm: FakeLLM
) -> None:
    bad = LetterDraft(subject="Claim", body="Pay EUR 5000 now. See https://x.example " * 10)
    llm = ScriptedLLM(fake_llm, structured={LLMTask.DRAFT: [bad, bad, bad]})
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert "LETTER_TEMPLATE_FALLBACK" in codes(result)
    assert (
        result.letter is not None
        and "EUR 400" in result.letter.body
        and "5000" not in result.letter.body
    )


def test_extraction_failure_routes_to_human(make_agent: AgentFactory, fake_llm: FakeLLM) -> None:
    llm = ScriptedLLM(fake_llm, structured={LLMTask.EXTRACT: [LLMOutputInvalid("garbage")]})
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert result.status == ClaimStatus.NEEDS_REVIEW and result.needs_human_review
    assert "LLM_FAILURE" in codes(result)


@pytest.mark.parametrize(
    ("error", "flag"),
    [
        (LLMRefusal("no"), "PROMPT_INJECTION_SUSPECTED"),  # refusing to screen = suspicious
        (LLMUnavailable("down"), "GUARD_DEGRADED"),  # heuristic screen still runs
    ],
)
def test_guard_failures_are_fail_safe(
    make_agent: AgentFactory, fake_llm: FakeLLM, error: Exception, flag: str
) -> None:
    llm = ScriptedLLM(fake_llm, structured={LLMTask.GUARD: [error]})
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert flag in codes(result)


def test_token_budget_is_enforced(make_agent: AgentFactory, fake_llm: FakeLLM) -> None:
    llm = ScriptedLLM(fake_llm)
    result = make_agent(llm=llm, max_run_tokens=0).submit(load_sample(LONG_DELAY))
    assert result.status == ClaimStatus.NEEDS_REVIEW
    assert "BUDGET_EXCEEDED" in codes(result)


def test_unexpected_crash_never_loses_the_claim(
    make_agent: AgentFactory, fake_llm: FakeLLM
) -> None:
    llm = ScriptedLLM(fake_llm, structured={LLMTask.EXTRACT: [ZeroDivisionError("bug")]})
    result = make_agent(llm=llm).submit(load_sample(LONG_DELAY))
    assert result.status == ClaimStatus.NEEDS_REVIEW
    assert result.review_state == ReviewState.NOT_REQUIRED
    assert "INTERNAL_ERROR" in codes(result)


def test_oversized_input_is_rejected_before_any_llm_call(
    make_agent: AgentFactory, fake_llm: FakeLLM
) -> None:
    llm = ScriptedLLM(fake_llm)
    request = load_sample(LONG_DELAY).model_copy(update={"message": "x" * 9_000})
    result = make_agent(llm=llm).submit(request)
    assert result.status == ClaimStatus.REJECTED
    assert llm.calls == []
