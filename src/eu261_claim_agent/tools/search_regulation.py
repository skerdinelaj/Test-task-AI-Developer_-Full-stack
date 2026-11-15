"""search_regulation: BM25 search over the EU261 notes and the firm's policy."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from eu261_claim_agent.core.schemas import RegulationSnippet
from eu261_claim_agent.infra.search import RegulationIndex
from eu261_claim_agent.tools.registry import Args, Tool, check_top_k

NAME = "search_regulation"
DESCRIPTION = (
    "Search the EU261 knowledge base and the firm's claim-handling policy. Returns the "
    "most relevant passages."
)


class SearchRegulationArgs(Args):
    query: str = Field(description="Short search query, e.g. 'cancellation notice 14 days'.")
    top_k: Annotated[int, AfterValidator(check_top_k)] = Field(
        description="Number of passages, 1-5."
    )


class SearchResult(BaseModel):
    passages: list[RegulationSnippet]


def make(index: RegulationIndex) -> Tool:
    def search_regulation(args: SearchRegulationArgs) -> SearchResult:
        return SearchResult(passages=index.search(args.query, args.top_k))

    return Tool(NAME, DESCRIPTION, SearchRegulationArgs, search_regulation)
