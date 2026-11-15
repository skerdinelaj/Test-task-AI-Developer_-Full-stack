from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from eu261_claim_agent.config import PROJECT_ROOT, LLMProvider, Settings
from eu261_claim_agent.llm.base import LLMClient
from eu261_claim_agent.llm.fake import FakeLLM
from eu261_claim_agent.reference_data import ReferenceData
from eu261_claim_agent.schemas import ClaimRequest
from eu261_claim_agent.search import RegulationIndex
from eu261_claim_agent.service import ClaimAgent
from eu261_claim_agent.tools import ToolRegistry, build_tools

SAMPLES = PROJECT_ROOT / "samples"
TODAY = date(2026, 11, 15)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # tests never read the developer's .env
        llm_provider=LLMProvider.FAKE,
        runs_dir=tmp_path / "runs",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
    )


@pytest.fixture(scope="session")
def ref() -> ReferenceData:
    return ReferenceData(PROJECT_ROOT / "data")


@pytest.fixture(scope="session")
def index() -> RegulationIndex:
    return RegulationIndex(PROJECT_ROOT / "data" / "regulation")


@pytest.fixture(scope="session")
def registry(ref: ReferenceData, index: RegulationIndex) -> ToolRegistry:
    return build_tools(ref, index)


@pytest.fixture
def fake_llm(ref: ReferenceData) -> FakeLLM:
    return FakeLLM(ref)


AgentFactory = Callable[..., ClaimAgent]


@pytest.fixture
def make_agent(settings: Settings) -> Iterator[AgentFactory]:
    agents: list[ClaimAgent] = []

    def factory(
        llm: LLMClient | None = None, persist: bool = False, **overrides: Any
    ) -> ClaimAgent:
        s = settings.model_copy(update=overrides) if overrides else settings
        agent = ClaimAgent(s, llm=llm, persist=persist)
        agents.append(agent)
        return agent

    yield factory
    for agent in agents:
        agent.close()


def load_sample(name: str) -> ClaimRequest:
    return ClaimRequest.model_validate_json((SAMPLES / "inputs" / f"{name}.json").read_text())


def load_expected(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((SAMPLES / "expected" / f"{name}.json").read_text())
    return data


def sample_names() -> list[str]:
    return sorted(p.stem for p in (SAMPLES / "inputs").glob("*.json"))


def claim(message: str, attachment: str | None = None) -> ClaimRequest:
    return ClaimRequest(message=message, attachment_text=attachment, received_on=TODAY)
