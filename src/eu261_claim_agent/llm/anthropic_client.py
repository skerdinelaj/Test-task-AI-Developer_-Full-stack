"""Claude implementation of ``LLMClient``.

Notes on the current Claude models (Sonnet/Haiku 5.5):
* forced tool use (``tool_choice`` any/tool) is rejected, so structured answers use
  **structured outputs** (``messages.parse`` with a Pydantic model) and the planner uses
  ``tool_choice=auto`` with ``strict`` tools;
* thinking blocks returned during a tool loop must be passed back unchanged, so we keep the
  assistant content exactly as returned;
* content blocks are selected by ``type``, never by position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from eu261_claim_agent.config import Settings
from eu261_claim_agent.llm.base import (
    LLMOutputInvalid,
    LLMRefusal,
    LLMUnavailable,
    StructuredResult,
    ToolCall,
    ToolTurn,
)
from eu261_claim_agent.schemas import LLMTask, Usage
from eu261_claim_agent.tools import ToolSpec


@dataclass(frozen=True)
class TaskProfile:
    model_attr: str  # which Settings attribute holds the model id
    max_tokens: int
    effort: str
    thinking: dict[str, str] | None  # None = model default (adaptive)


# Cheap model + no thinking for screening; main model with little up-front thinking for
# extraction/drafting (the task is well specified); default adaptive thinking for planning.
PROFILES: dict[LLMTask, TaskProfile] = {
    LLMTask.GUARD: TaskProfile("guard_model", 512, "low", {"type": "disabled"}),
    LLMTask.EXTRACT: TaskProfile("main_model", 2_048, "medium", {"type": "between_tools"}),
    LLMTask.PLAN: TaskProfile("main_model", 4_096, "low", None),
    LLMTask.DRAFT: TaskProfile("main_model", 4_096, "medium", {"type": "between_tools"}),
}


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        if client is None:
            if settings.anthropic_api_key is None:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Add it to .env, or run with --mock / "
                    "CLAIM_AGENT_LLM_PROVIDER=fake to use the offline stand-in."
                )
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value(),
                max_retries=settings.sdk_max_retries,  # exponential backoff on 429/5xx/timeouts
                timeout=settings.request_timeout_s,
            )
        self._client = client
        # The SDK's TypedDict overloads are stricter than our plain-dict message history; the
        # history is built by us from SDK output, so we type the call boundary loosely here.
        self._messages: Any = client.messages
        self._settings = settings

    # ------------------------------------------------------------------ helpers
    def _model(self, task: LLMTask) -> str:
        return str(getattr(self._settings, PROFILES[task].model_attr))

    def _kwargs(self, task: LLMTask) -> dict[str, Any]:
        profile = PROFILES[task]
        kwargs: dict[str, Any] = {
            "model": self._model(task),
            "max_tokens": profile.max_tokens,
            "output_config": {"effort": profile.effort},
        }
        if profile.thinking is not None:
            kwargs["thinking"] = profile.thinking
        return kwargs

    def _usage(self, model: str, usage: Any) -> Usage:
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        price_in = self._settings.price_in_per_mtok.get(model, 0.0)
        price_out = self._settings.price_out_per_mtok.get(model, 0.0)
        return Usage(
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            llm_calls=1,
            cost_usd=round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6),
        )

    # ------------------------------------------------------------------ API
    def structured[T: BaseModel](
        self, task: LLMTask, system: str, user: str, schema: type[T]
    ) -> StructuredResult[T]:
        kwargs = self._kwargs(task)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        total = Usage()
        attempts = self._settings.max_validation_retries + 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                response = self._messages.parse(
                    system=system, messages=messages, output_format=schema, **kwargs
                )
            except ValidationError as exc:  # SDK parsed the JSON but Pydantic rejected it
                last_error = str(exc)
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{user}\n\nYour previous answer failed validation:\n{last_error}\n"
                            "Answer again, following the schema exactly."
                        ),
                    }
                ]
                continue
            except anthropic.APIError as exc:
                raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
            total = total + self._usage(kwargs["model"], response.usage)
            if response.stop_reason == "refusal":
                raise LLMRefusal(f"{task.value}: model declined to answer")
            parsed = response.parsed_output
            if parsed is not None:
                return StructuredResult(parsed, total, kwargs["model"], attempt)
            last_error = f"no parsed output (stop_reason={response.stop_reason})"
        raise LLMOutputInvalid(f"{task.value}: {last_error}")

    def tool_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> ToolTurn:
        kwargs = self._kwargs(LLMTask.PLAN)
        try:
            response = self._messages.create(
                system=system,
                messages=messages,
                tools=[t.to_anthropic() for t in tools],
                tool_choice={"type": "auto", "disable_parallel_tool_use": False},
                **kwargs,
            )
        except anthropic.APIError as exc:
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMRefusal("plan: model declined to answer")
        content = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
        calls = [
            ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"
        ]
        text = "".join(b.text for b in response.content if b.type == "text")
        return ToolTurn(
            content,
            calls,
            text,
            str(response.stop_reason),
            self._usage(kwargs["model"], response.usage),
            kwargs["model"],
        )
