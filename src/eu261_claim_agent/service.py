"""Application service: the one entry point used by the CLI, the API, tests and evals."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import AbstractContextManager
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from eu261_claim_agent.agent.graph import build_graph
from eu261_claim_agent.agent.state import AgentDeps, AgentState
from eu261_claim_agent.config import Settings, get_settings
from eu261_claim_agent.core import schemas
from eu261_claim_agent.core.schemas import (
    ClaimRequest,
    ClaimResult,
    ClaimStatus,
    ReviewOutcome,
    ReviewState,
    RiskFlag,
    Severity,
    Usage,
)
from eu261_claim_agent.guardrails.guards import new_boundary
from eu261_claim_agent.infra.observability import Tracer
from eu261_claim_agent.infra.reference_data import ReferenceData
from eu261_claim_agent.infra.search import RegulationIndex
from eu261_claim_agent.llm import LLMClient, build_llm
from eu261_claim_agent.tools import build_tools


class ClaimNotFoundError(KeyError):
    pass


class ReviewNotPendingError(RuntimeError):
    pass


def _serializer() -> JsonPlusSerializer:
    # Only our own schema types may be revived from checkpoints (safe deserialisation).
    allowed = [
        ("eu261_claim_agent.core.schemas", name)
        for name in dir(schemas)
        if isinstance(getattr(schemas, name), type)
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


class ClaimAgent(AbstractContextManager["ClaimAgent"]):
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        persist: bool = True,
        trace_to_disk: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.ref = ReferenceData(self.settings.data_dir)
        self.llm = llm or build_llm(self.settings, self.ref)
        self.tracer = Tracer(self.settings.runs_dir, enabled=trace_to_disk)
        tools = build_tools(self.ref, RegulationIndex(self.settings.data_dir / "regulation"))
        self._conn: sqlite3.Connection | None = None
        checkpointer: BaseCheckpointSaver[Any]
        if persist:
            Path(self.settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.settings.checkpoint_db, check_same_thread=False)
            checkpointer = SqliteSaver(self._conn, serde=_serializer())
        else:
            checkpointer = InMemorySaver(serde=_serializer())
        self.graph = build_graph(
            AgentDeps(self.settings, self.llm, tools, self.ref, self.tracer), checkpointer
        )

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ use cases
    def submit(self, request: ClaimRequest, run_id: str | None = None) -> ClaimResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        initial: AgentState = {
            "run_id": run_id,
            "request": request,
            "today": request.received_on or date.today(),
            "boundary": new_boundary(),
            "usage": Usage(),
            "risk_flags": [],
            "tool_calls": [],
            "steps": [],
        }
        self.tracer.event(
            run_id,
            "run_start",
            provider=self.llm.provider,
            message_chars=len(request.message),
            has_attachment=bool(request.attachment_text),
        )
        try:
            self.graph.invoke(initial, self._config(run_id))
        except Exception as exc:  # last-resort safety net: never lose a claim silently
            self.tracer.event(run_id, "run_error", error=f"{type(exc).__name__}: {exc}")
            return self._error_result(run_id, exc)
        return self.get(run_id)

    def review(self, run_id: str, outcome: ReviewOutcome) -> ClaimResult:
        result = self.get(run_id)
        if result.review_state != ReviewState.PENDING:
            raise ReviewNotPendingError(f"Claim {run_id} is not waiting for review.")
        self.graph.invoke(Command(resume=outcome.model_dump()), self._config(run_id))
        return self.get(run_id)

    def get(self, run_id: str) -> ClaimResult:
        snapshot = self.graph.get_state(self._config(run_id))
        state: AgentState = snapshot.values  # type: ignore[assignment]
        if not state:
            raise ClaimNotFoundError(run_id)
        pending = any(task.interrupts for task in snapshot.tasks)
        review_state = (
            ReviewState.PENDING if pending else state.get("review_state", ReviewState.NOT_REQUIRED)
        )
        status = state.get("status", ClaimStatus.NEEDS_REVIEW)
        summary = state.get("summary")
        if summary is None:
            from eu261_claim_agent.core.summary import build_summary

            summary = build_summary(
                status,
                state.get("facts"),
                state.get("decision"),
                state.get("terminal_reasons", []),
                state.get("missing_fields", []),
                state.get("risk_flags", []),
                review_state,
                state.get("review"),
            )
        return ClaimResult(
            run_id=run_id,
            status=status,
            review_state=review_state,
            needs_human_review=status == ClaimStatus.NEEDS_REVIEW
            or review_state != ReviewState.NOT_REQUIRED,
            summary=summary,
            decision=state.get("decision"),
            facts=state.get("facts"),
            missing_fields=state.get("missing_fields", []),
            risk_flags=state.get("risk_flags", []),
            letter=state.get("letter"),
            review=state.get("review"),
            tool_calls=state.get("tool_calls", []),
            steps=state.get("steps", []),
            usage=state.get("usage", Usage()),
            provider=self.llm.provider,
        )

    def trace(self, run_id: str) -> list[dict[str, Any]]:
        return self.tracer.read(run_id)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _config(run_id: str) -> Any:
        return {"configurable": {"thread_id": run_id}}

    def _error_result(self, run_id: str, exc: Exception) -> ClaimResult:
        return ClaimResult(
            run_id=run_id,
            status=ClaimStatus.NEEDS_REVIEW,
            review_state=ReviewState.NOT_REQUIRED,
            needs_human_review=True,
            summary="Result: NEEDS REVIEW.\nThe agent hit an internal error; a handler must "
            "process this claim manually.",
            decision=None,
            facts=None,
            missing_fields=[],
            letter=None,
            review=None,
            risk_flags=[
                RiskFlag(code="INTERNAL_ERROR", severity=Severity.HIGH, message=type(exc).__name__)
            ],
            tool_calls=[],
            steps=[],
            usage=Usage(),
            provider=self.llm.provider,
        )
