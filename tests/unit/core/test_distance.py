"""Great-circle distance."""

from __future__ import annotations

import pytest

from eu261_claim_agent.core.distance import great_circle_km


def test_great_circle_known_distance_and_symmetry() -> None:
    fra, lis = (50.0379, 8.5622), (38.7742, -9.1342)
    assert great_circle_km(*fra, *lis) == pytest.approx(1874, abs=5)
    assert great_circle_km(*fra, *lis) == pytest.approx(great_circle_km(*lis, *fra))
    assert great_circle_km(*fra, *fra) == 0
