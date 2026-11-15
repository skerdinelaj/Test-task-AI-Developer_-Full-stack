"""PII redaction and trace storage."""

from __future__ import annotations

from pathlib import Path

from eu261_claim_agent.infra.observability import Tracer, redact, redact_text


def test_redaction() -> None:
    text = (
        "Mail anna.rossi@example.com, call +355 69 123 4567, "
        "booking reference: X7K2PQ on 2026-10-03"
    )
    out = redact_text(text)
    assert "anna.rossi" not in out and "123 4567" not in out and "X7K2PQ" not in out
    assert "2026-10-03" in out  # dates are not mistaken for phone numbers
    assert redact({"passenger_name": "Anna", "nested": [{"email": "a@b.c"}]}) == {
        "passenger_name": "[redacted]",
        "nested": [{"email": "[redacted]"}],
    }


def test_tracer_rejects_path_traversal(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path)
    tracer.event("run1", "hello", passenger_name="Anna")
    assert tracer.read("run1")[0]["passenger_name"] == "[redacted]"
    assert tracer.read("../../etc/passwd") == []
