"""Command-line interface: run claims, review them, inspect traces, start the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from eu261_claim_agent.config import PROJECT_ROOT, LLMProvider, Settings
from eu261_claim_agent.core.schemas import ClaimRequest, ClaimResult, ReviewOutcome
from eu261_claim_agent.infra.observability import configure_logging
from eu261_claim_agent.service import ClaimAgent

app = typer.Typer(add_completion=False, help="EU261 claim intake agent.")
console = Console()

MockOpt = Annotated[
    bool, typer.Option("--mock", help="Use the offline FakeLLM (no API key needed).")
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show JSON logs on stderr.")]


def _agent(mock: bool, verbose: bool) -> ClaimAgent:
    configure_logging("INFO" if verbose else "WARNING")
    settings = Settings(llm_provider=LLMProvider.FAKE) if mock else Settings()
    try:
        return ClaimAgent(settings)
    except ValueError as exc:  # e.g. missing API key
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _print_result(result: ClaimResult, show_letter: bool = True) -> None:
    colour = {
        "eligible": "green",
        "not_eligible": "yellow",
        "needs_info": "cyan",
        "needs_review": "magenta",
        "out_of_scope": "yellow",
        "rejected": "red",
    }
    console.print(
        Panel(
            result.summary,
            title=f"run {result.run_id} · {result.status.value}",
            border_style=colour.get(result.status.value, "white"),
        )
    )
    if result.tool_calls:
        table = Table("tool", "source", "ok", "ms", "arguments", title="Tool calls")
        for call in result.tool_calls:
            table.add_row(
                call.tool,
                call.source,
                "✓" if call.ok else "✗",
                f"{call.duration_ms:.1f}",
                json.dumps(call.arguments)[:70],
            )
        console.print(table)
    if show_letter and result.letter:
        console.print(
            Panel(
                f"[bold]{result.letter.subject}[/bold]\n\n{result.letter.body}",
                title="Draft letter (not sent)",
            )
        )
    u = result.usage
    console.print(
        f"[dim]provider={result.provider} llm_calls={u.llm_calls} "
        f"tokens={u.total_tokens} cost≈${u.cost_usd:.4f} "
        f"steps={' → '.join(s.node for s in result.steps)}[/dim]"
    )


@app.command()
def run(
    file: Annotated[Path, typer.Argument(help="JSON file with message / attachment_text.")],
    mock: MockOpt = False,
    verbose: VerboseOpt = False,
    approve: Annotated[
        bool | None,
        typer.Option("--approve/--reject", help="Resolve the human review immediately."),
    ] = None,
    reviewer: str = "cli-reviewer",
    as_json: Annotated[bool, typer.Option("--json", help="Print the JSON result only.")] = False,
) -> None:
    """Process one claim."""
    request = ClaimRequest.model_validate_json(file.read_text(encoding="utf-8"))
    with _agent(mock, verbose) as agent:
        result = agent.submit(request)
        if approve is not None and result.review_state.value == "pending":
            result = agent.review(result.run_id, ReviewOutcome(approved=approve, reviewer=reviewer))
    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        _print_result(result)


@app.command()
def review(
    run_id: str,
    approve: Annotated[bool, typer.Option("--approve/--reject")],
    reviewer: str = "cli-reviewer",
    note: str | None = None,
    mock: MockOpt = False,
) -> None:
    """Approve or reject a claim that is waiting for human review."""
    with _agent(mock, False) as agent:
        _print_result(
            agent.review(run_id, ReviewOutcome(approved=approve, reviewer=reviewer, note=note))
        )


@app.command()
def show(run_id: str, mock: MockOpt = False) -> None:
    """Show the current result of a claim."""
    with _agent(mock, False) as agent:
        print(agent.get(run_id).model_dump_json(indent=2))


@app.command()
def trace(run_id: str) -> None:
    """Print the trace (every node, LLM call and tool call) of a run."""
    with _agent(True, False) as agent:
        for event in agent.trace(run_id):
            extra = {k: v for k, v in event.items() if k not in {"ts", "run_id", "event"}}
            console.print(
                f"[dim]{event['ts'][11:23]}[/dim] [bold]{event['event']:<16}[/bold] "
                f"{escape(json.dumps(extra, default=str)[:140])}"
            )


@app.command()
def demo(mock: MockOpt = False, samples: Path = PROJECT_ROOT / "samples" / "inputs") -> None:
    """Run every sample and print a one-line result per case."""
    table = Table("sample", "status", "EUR", "review", "flags", title="Sample claims")
    with _agent(mock, False) as agent:
        for path in sorted(samples.glob("*.json")):
            result = agent.submit(
                ClaimRequest.model_validate_json(path.read_text(encoding="utf-8"))
            )
            table.add_row(
                path.stem,
                result.status.value,
                str(result.decision.amount_eur) if result.decision else "-",
                result.review_state.value,
                ", ".join(f.code for f in result.risk_flags if f.severity != "info"),
            )
    console.print(table)


@app.command()
def graph() -> None:
    """Print the agent graph as Mermaid."""
    with _agent(True, False) as agent:
        print(agent.graph.get_graph().draw_mermaid())


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, mock: MockOpt = False) -> None:
    """Start the HTTP API (and the review UI if it has been built)."""
    import os

    import uvicorn

    if mock:
        os.environ["CLAIM_AGENT_LLM_PROVIDER"] = "fake"
    uvicorn.run("eu261_claim_agent.api:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
