from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eu261_claim_agent.cli import app
from tests.conftest import SAMPLES

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAIM_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CLAIM_AGENT_CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))


def test_cli_run_json_and_review() -> None:
    sample = SAMPLES / "inputs" / "07_tirana_rome_eu_carrier.json"
    out = runner.invoke(app, ["run", str(sample), "--mock", "--json"])
    assert out.exit_code == 0, out.output
    result = json.loads(out.output)
    assert (result["status"], result["review_state"]) == ("eligible", "pending")

    reviewed = runner.invoke(app, ["review", result["run_id"], "--approve", "--mock"])
    assert reviewed.exit_code == 0 and "approved" in reviewed.output
    shown = runner.invoke(app, ["show", result["run_id"], "--mock"])
    assert json.loads(shown.output)["review_state"] == "approved"
    traced = runner.invoke(app, ["trace", result["run_id"]])
    assert "tool_call" in traced.output


def test_cli_run_with_inline_approval_and_demo() -> None:
    sample = SAMPLES / "inputs" / "01_long_delay_eligible.json"
    out = runner.invoke(app, ["run", str(sample), "--mock", "--approve"])
    assert out.exit_code == 0 and "Draft letter" in out.output
    demo = runner.invoke(app, ["demo", "--mock"])
    assert demo.exit_code == 0 and "11_prompt_injection" in demo.output
    graph = runner.invoke(app, ["graph"])
    assert "review_gate" in graph.output
