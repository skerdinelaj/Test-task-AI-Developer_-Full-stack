"""HTTP API. The review endpoint resumes the paused LangGraph run (human-in-the-loop)."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eu261_claim_agent.config import PROJECT_ROOT, Settings, get_settings
from eu261_claim_agent.core.schemas import ClaimRequest, ClaimResult, ReviewOutcome
from eu261_claim_agent.infra.observability import configure_logging
from eu261_claim_agent.service import ClaimAgent, ClaimNotFoundError, ReviewNotPendingError


class ClaimIn(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)
    attachment_text: str | None = Field(default=None, max_length=20_000)
    received_on: date | None = None


def get_agent(request: Request) -> ClaimAgent:
    return request.app.state.agent  # type: ignore[no-any-return]


AgentDep = Annotated[ClaimAgent, Depends(get_agent)]


def check_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.agent.settings.api_key
    if expected is not None and not (
        x_api_key and secrets.compare_digest(x_api_key, expected.get_secret_value())
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key.")


def create_app(agent: ClaimAgent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (agent.settings if agent else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.agent = agent or ClaimAgent(settings)
        yield
        app.state.agent.close()

    app = FastAPI(title="EU261 Claim Agent", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    auth = [Depends(check_key)]

    @app.get("/health")
    def health(agent: AgentDep) -> dict[str, str]:
        return {"status": "ok", "provider": agent.llm.provider}

    @app.post(
        "/claims",
        response_model=ClaimResult,
        dependencies=auth,
        status_code=status.HTTP_201_CREATED,
    )
    def submit_claim(body: ClaimIn, agent: AgentDep) -> ClaimResult:
        return agent.submit(ClaimRequest(**body.model_dump()))

    @app.get("/claims/{run_id}", response_model=ClaimResult, dependencies=auth)
    def get_claim(run_id: str, agent: AgentDep) -> ClaimResult:
        try:
            return agent.get(run_id)
        except ClaimNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found.") from exc

    @app.post("/claims/{run_id}/review", response_model=ClaimResult, dependencies=auth)
    def review_claim(run_id: str, body: ReviewOutcome, agent: AgentDep) -> ClaimResult:
        try:
            return agent.review(run_id, body)
        except ClaimNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Claim not found.") from exc
        except ReviewNotPendingError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    @app.get("/claims/{run_id}/trace", dependencies=auth)
    def get_trace(run_id: str, agent: AgentDep) -> list[dict[str, Any]]:
        events = agent.trace(run_id)
        if not events:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Trace not found.")
        return events

    ui_dist = PROJECT_ROOT / "ui" / "dist"
    if ui_dist.exists():
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
    return app
