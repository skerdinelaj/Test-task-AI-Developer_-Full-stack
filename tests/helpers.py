"""ScriptedLLM: wraps the FakeLLM but lets a test replace any answer or inject failures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from eu261_claim_agent.llm.base import StructuredResult, ToolCall, ToolTurn
from eu261_claim_agent.llm.fake import FakeLLM
from eu261_claim_agent.schemas import LLMTask, Usage
from eu261_claim_agent.tools import ToolSpec

Scripted = BaseModel | Exception


class ScriptedLLM:
    provider = "scripted"

    def __init__(
        self,
        base: FakeLLM,
        structured: dict[LLMTask, list[Scripted]] | None = None,
        plan: list[ToolTurn | Exception] | None = None,
        repeat_last_plan: bool = False,
    ) -> None:
        self.base = base
        self.structured_script = structured or {}
        self.plan_script = plan
        self.repeat_last_plan = repeat_last_plan
        self.calls: list[str] = []

    def structured[T: BaseModel](
        self, task: LLMTask, system: str, user: str, schema: type[T]
    ) -> StructuredResult[T]:
        self.calls.append(task.value)
        queue = self.structured_script.get(task)
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, schema)
            return StructuredResult(
                item, Usage(input_tokens=100, output_tokens=50, llm_calls=1), "scripted"
            )
        return self.base.structured(task, system, user, schema)

    def tool_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> ToolTurn:
        self.calls.append("plan")
        if self.plan_script is None:
            return self.base.tool_turn(system, messages, tools)
        if not self.plan_script:
            return finish()
        item = (
            self.plan_script[0]
            if (self.repeat_last_plan and len(self.plan_script) == 1)
            else self.plan_script.pop(0)
        )
        if isinstance(item, Exception):
            raise item
        return item


def call_tools(*calls: tuple[str, dict[str, Any]], step: int = 0) -> ToolTurn:
    tool_calls = [ToolCall(f"call_{step}_{i}", name, args) for i, (name, args) in enumerate(calls)]
    content = [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in tool_calls
    ]
    return ToolTurn(content, tool_calls, "", "tool_use", Usage(llm_calls=1), "scripted")


def finish(text: str = "Done.") -> ToolTurn:
    return ToolTurn(
        [{"type": "text", "text": text}], [], text, "end_turn", Usage(llm_calls=1), "scripted"
    )
