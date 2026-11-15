"""Smoke tests against the real Claude API. Skipped unless ANTHROPIC_API_KEY is set.

Run with:  make test-live   (or: uv run pytest -m live)
"""

from __future__ import annotations

import pytest

from eu261_claim_agent.config import LLMProvider, Settings
from eu261_claim_agent.core.schemas import ClaimStatus
from tests.conftest import AgentFactory, load_expected, load_sample

HAS_KEY = Settings().anthropic_api_key is not None

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not HAS_KEY, reason="ANTHROPIC_API_KEY not set (environment or .env)"),
]


@pytest.mark.parametrize(
    "name",
    [
        "01_long_delay_eligible",
        "02_short_delay_not_eligible",
        "05_weather_needs_review",
        "10_missing_info",
        "11_prompt_injection",
        "12_fabrication_refused",
    ],
)
def test_live_sample(make_agent: AgentFactory, settings: Settings, name: str) -> None:
    agent = make_agent(
        llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key=Settings().anthropic_api_key
    )
    result = agent.submit(load_sample(name))
    expected = load_expected(name)
    assert result.status.value == expected["status"]
    assert (result.decision.amount_eur if result.decision else 0) == expected["amount_eur"]
    assert result.usage.llm_calls > 0 and result.usage.total_tokens > 0
    if result.status == ClaimStatus.ELIGIBLE:
        assert result.letter is not None and f"EUR {expected['amount_eur']}" in result.letter.body
