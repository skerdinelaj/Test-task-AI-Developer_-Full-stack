"""Tool registry: strict schemas, allowlist, argument validation, handlers."""

from __future__ import annotations

from eu261_claim_agent.tools import ToolRegistry


def test_tool_schemas_are_strict(registry: ToolRegistry) -> None:
    for spec in registry.specs():
        schema = spec.input_schema
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert spec.to_anthropic()["strict"] is True


def test_unknown_tool_is_refused(registry: ToolRegistry) -> None:
    result = registry.execute("send_email", {"to": "x@y.z"})
    assert not result.ok and "Unknown tool" in (result.error or "")


def test_tool_arguments_are_validated_in_code(registry: ToolRegistry) -> None:
    result = registry.execute("lookup_flight", {"flight_number": "DROP TABLE", "flight_date": "x"})
    assert not result.ok and "Invalid arguments" in (result.error or "")
    extra = registry.execute("get_airline", {"airline_code": "LH", "debug": True})
    assert not extra.ok


def test_tools_happy_path_and_not_found(registry: ToolRegistry) -> None:
    found = registry.execute(
        "lookup_flight", {"flight_number": "lh 1452", "flight_date": "2026-10-03"}
    )
    assert found.ok and '"origin":"FRA"' in found.content
    missing = registry.execute(
        "lookup_flight", {"flight_number": "FR9999", "flight_date": "2026-10-01"}
    )
    assert not missing.ok and "No flight" in missing.content
    dist = registry.execute("calculate_distance", {"origin_iata": "fra", "destination_iata": "LIS"})
    assert dist.ok and '"distance_km":1874' in dist.content
