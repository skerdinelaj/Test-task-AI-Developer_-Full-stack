"""Regulation search (BM25)."""

from __future__ import annotations

from eu261_claim_agent.infra.search import RegulationIndex


def test_regulation_search_ranks_relevant_doc_first(index: RegulationIndex) -> None:
    assert index.search("cancellation notice 14 days", 1)[0].doc_id == "cancellation"
    assert index.search("pilot strike extraordinary", 1)[0].doc_id == "extraordinary"
    assert index.search("zzzz qqqq", 3) == []
