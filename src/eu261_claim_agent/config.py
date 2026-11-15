"""Application settings, loaded from environment variables / .env (12-factor style)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    FAKE = "fake"  # deterministic offline stand-in, used by tests, CI and `--mock`


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAIM_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    main_model: str = "claude-sonnet-5-5"  # extraction, planning, drafting
    guard_model: str = "claude-haiku-5-5"  # cheap, fast input screening
    request_timeout_s: float = 60.0
    sdk_max_retries: int = 3  # transport-level retries (429/5xx) with backoff, done by the SDK

    # --- Agent limits (guardrails against runaway loops / cost) -------------
    max_plan_steps: int = 6  # LLM planning turns in the tool loop
    max_validation_retries: int = 2  # re-ask the LLM when its output fails validation
    max_run_tokens: int = 60_000  # hard token budget per claim
    max_message_chars: int = 8_000
    max_attachment_chars: int = 20_000

    # --- Pricing used for cost estimates in traces (USD per million tokens) --
    price_in_per_mtok: dict[str, float] = {"claude-sonnet-5-5": 2.0, "claude-haiku-5-5": 0.10}
    price_out_per_mtok: dict[str, float] = {"claude-sonnet-5-5": 10.0, "claude-haiku-5-5": 0.50}

    # --- Paths ---------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    runs_dir: Path = PROJECT_ROOT / "runs"
    checkpoint_db: Path = PROJECT_ROOT / "runs" / "checkpoints.sqlite"

    # --- API -----------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:5173"]
    api_key: SecretStr | None = None  # if set, every /claims call needs header X-API-Key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
