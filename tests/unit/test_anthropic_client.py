"""Tests for the Claude adapter using a stub SDK client (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from eu261_claim_agent.config import Settings
from eu261_claim_agent.llm.anthropic_client import AnthropicLLM
from eu261_claim_agent.llm.base import LLMOutputInvalid, LLMRefusal, LLMUnavailable
from eu261_claim_agent.schemas import LLMTask
from eu261_claim_agent.tools import ToolRegistry


class Answer(BaseModel):
    value: int


def usage(i: int = 10, o: int = 5) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=i, output_tokens=o)


class StubMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def _next(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    parse = _next
    create = _next


def make(responses: list[Any], settings: Settings) -> tuple[AnthropicLLM, StubMessages]:
    messages = StubMessages(responses)
    client = SimpleNamespace(messages=messages)
    return AnthropicLLM(settings, client=client), messages  # type: ignore[arg-type]


def parsed(value: Answer | None, stop: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(parsed_output=value, stop_reason=stop, usage=usage())


def test_structured_success_tracks_usage_and_cost(settings: Settings) -> None:
    llm, stub = make([parsed(Answer(value=1))], settings)
    result = llm.structured(LLMTask.EXTRACT, "sys", "user", Answer)
    assert result.value.value == 1
    assert result.usage.input_tokens == 10 and result.usage.cost_usd > 0
    call = stub.calls[0]
    assert call["model"] == settings.main_model
    assert call["output_format"] is Answer
    assert "tool_choice" not in call  # forced tool use is not supported on current models


def test_guard_uses_cheap_model_without_thinking(settings: Settings) -> None:
    llm, stub = make([parsed(Answer(value=1))], settings)
    llm.structured(LLMTask.GUARD, "sys", "user", Answer)
    assert stub.calls[0]["model"] == settings.guard_model
    assert stub.calls[0]["thinking"] == {"type": "disabled"}


def test_structured_retries_then_gives_up(settings: Settings) -> None:
    llm, stub = make([parsed(None, "max_tokens")] * 3, settings)
    with pytest.raises(LLMOutputInvalid):
        llm.structured(LLMTask.EXTRACT, "sys", "user", Answer)
    assert len(stub.calls) == settings.max_validation_retries + 1


def test_structured_recovers_on_second_attempt(settings: Settings) -> None:
    llm, _ = make([parsed(None, "max_tokens"), parsed(Answer(value=2))], settings)
    result = llm.structured(LLMTask.EXTRACT, "sys", "user", Answer)
    assert result.attempts == 2 and result.usage.llm_calls == 2


def test_refusal_is_typed(settings: Settings) -> None:
    llm, _ = make([parsed(None, "refusal")], settings)
    with pytest.raises(LLMRefusal):
        llm.structured(LLMTask.GUARD, "sys", "user", Answer)


def test_api_errors_become_llm_unavailable(settings: Settings) -> None:
    request: Any = httpx.Request("POST", "https://api.anthropic.com")
    error = anthropic.APIConnectionError(request=request)
    llm, _ = make([error], settings)
    with pytest.raises(LLMUnavailable):
        llm.structured(LLMTask.EXTRACT, "sys", "user", Answer)


def test_tool_turn_sends_strict_tools_with_auto_choice(
    settings: Settings, registry: ToolRegistry
) -> None:
    thinking = SimpleNamespace(
        type="thinking",
        model_dump=lambda **_: {"type": "thinking", "thinking": "", "signature": "sig"},
    )
    tool_use = SimpleNamespace(
        type="tool_use",
        id="t1",
        name="lookup_flight",
        input={"flight_number": "LH1452", "flight_date": "2026-10-03"},
        model_dump=lambda **_: {
            "type": "tool_use",
            "id": "t1",
            "name": "lookup_flight",
            "input": {"flight_number": "LH1452", "flight_date": "2026-10-03"},
        },
    )
    response = SimpleNamespace(content=[thinking, tool_use], stop_reason="tool_use", usage=usage())
    llm, stub = make([response], settings)
    turn = llm.tool_turn("sys", [{"role": "user", "content": "go"}], registry.specs())
    sent = stub.calls[0]
    assert sent["tool_choice"]["type"] == "auto"
    assert all(t["strict"] for t in sent["tools"])
    assert turn.tool_calls[0].name == "lookup_flight"
    assert turn.assistant_content[0]["type"] == "thinking"  # thinking blocks are kept for the loop


def test_missing_api_key_gives_clear_error(settings: Settings) -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM(settings.model_copy(update={"anthropic_api_key": None}))
