"""Tool plumbing: the allowlist registry, specs, execution and shared argument validators.

Read-only by design: no tool can send, write or pay.

Each tool has a Pydantic argument model. The JSON schema sent to Claude is generated from it and
marked ``strict`` (schema-valid calls), and arguments are validated *again* in code before the
handler runs, because the model's output is never trusted on its own.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from eu261_claim_agent.core.normalize import normalize_flight_number, normalize_iata


class ToolError(Exception):
    """A tool failed in a way the planner should be told about (not a crash)."""


def check_flight_number(value: str) -> str:
    normalized = normalize_flight_number(value)
    if normalized is None:
        raise ValueError("not a valid flight number")
    return normalized


def check_iata(value: str) -> str:
    normalized = normalize_iata(value)
    if normalized is None:
        raise ValueError("must be a 3-letter IATA airport code")
    return normalized


def check_iso_date(value: str) -> str:
    date.fromisoformat(value)
    return value


def check_airline_code(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 2 or not code.isalnum():
        raise ValueError("must be a 2-character IATA airline code")
    return code


def check_top_k(value: int) -> int:
    if not 1 <= value <= 5:
        raise ValueError("top_k must be between 1 and 5")
    return value


class Args(BaseModel):
    """Base for tool arguments: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


@dataclass(frozen=True)
class ToolExecution:
    ok: bool
    content: str  # JSON string sent back to the LLM
    result: BaseModel | None
    error: str | None
    duration_ms: float
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[Args]
    handler: Callable[[Any], BaseModel]

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        props = {
            k: {kk: vv for kk, vv in v.items() if kk in {"type", "description"}}
            for k, v in schema["properties"].items()
        }
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema={
                "type": "object",
                "properties": props,
                "required": list(props),
                "additionalProperties": False,
            },
        )


class ToolRegistry:
    """Allowlist of tools. Anything not registered here cannot be executed."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        start = time.perf_counter()

        def done(
            ok: bool,
            content: str,
            result: BaseModel | None,
            error: str | None,
            args: dict[str, Any],
        ) -> ToolExecution:
            return ToolExecution(
                ok, content, result, error, round((time.perf_counter() - start) * 1000, 2), args
            )

        tool = self._tools.get(name)
        if tool is None:
            msg = f"Unknown tool '{name}'. Allowed tools: {', '.join(self.names)}."
            return done(False, json.dumps({"error": msg}), None, msg, arguments)
        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            msg = "Invalid arguments: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            return done(False, json.dumps({"error": msg}), None, msg, arguments)
        try:
            result = tool.handler(args)
        except ToolError as exc:
            return done(False, json.dumps({"error": str(exc)}), None, str(exc), args.model_dump())
        return done(True, result.model_dump_json(), result, None, args.model_dump())
