from eu261_claim_agent.config import LLMProvider, Settings
from eu261_claim_agent.infra.reference_data import ReferenceData
from eu261_claim_agent.llm.base import (
    LLMClient,
    LLMError,
    LLMOutputInvalid,
    LLMRefusal,
    LLMUnavailable,
    StructuredResult,
    ToolCall,
    ToolTurn,
)

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMOutputInvalid",
    "LLMRefusal",
    "LLMUnavailable",
    "StructuredResult",
    "ToolCall",
    "ToolTurn",
    "build_llm",
]


def build_llm(settings: Settings, ref: ReferenceData) -> LLMClient:
    if settings.llm_provider == LLMProvider.FAKE:
        from eu261_claim_agent.llm.fake import FakeLLM

        return FakeLLM(ref)
    from eu261_claim_agent.llm.anthropic_client import AnthropicLLM

    return AnthropicLLM(settings)
