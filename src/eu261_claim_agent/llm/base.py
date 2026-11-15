"""Provider-agnostic LLM interface. The agent only depends on this Protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from eu261_claim_agent.schemas import LLMTask, Usage
from eu261_claim_agent.tools import ToolSpec


class LLMError(Exception):
    """Base class for LLM failures the agent knows how to degrade from."""


class LLMUnavailable(LLMError):
    """Network / rate limit / server error after the SDK's own retries."""


class LLMRefusal(LLMError):
    """The model declined to answer (stop_reason == 'refusal')."""


class LLMOutputInvalid(LLMError):
    """The output still failed validation after all retries."""


@dataclass
class StructuredResult[T: BaseModel]:
    value: T
    usage: Usage
    model: str
    attempts: int = 1


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolTurn:
    assistant_content: list[
        dict[str, Any]
    ]  # append to the conversation as-is (keeps thinking blocks)
    tool_calls: list[ToolCall]
    text: str
    stop_reason: str
    usage: Usage
    model: str
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    provider: str

    def structured[T: BaseModel](
        self, task: LLMTask, system: str, user: str, schema: type[T]
    ) -> StructuredResult[T]:
        """One call whose answer must match ``schema``; retries on validation failure."""
        ...

    def tool_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> ToolTurn:
        """One planning turn: the model either requests tool calls or finishes with text."""
        ...
