"""Offline evaluation harness: measures quality on a labelled dataset (not pass/fail like tests).

    uv run python -m evals.run_evals --provider fake        # regex baseline, no key needed
    uv run python -m evals.run_evals --provider anthropic   # real Claude (needs ANTHROPIC_API_KEY)

Writes evals/results/<provider>.md and .json. Exits non-zero if decision accuracy is below
--min-accuracy, so it can gate a CI job or a prompt change.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eu261_claim_agent.agent.prompts import PROMPT_VERSION
from eu261_claim_agent.config import PROJECT_ROOT, LLMProvider, Settings
from eu261_claim_agent.core.schemas import ClaimRequest
from eu261_claim_agent.infra.observability import configure_logging
from eu261_claim_agent.service import ClaimAgent

EVAL_DIR = PROJECT_ROOT / "evals"


@dataclass
class CaseResult:
    id: str
    expected_status: str
    status: str
    expected_amount: int
    amount: int
    status_ok: bool
    amount_ok: bool
    extraction_ok: bool | None
    injection_expected: bool
    injection_flagged: bool
    overpaid: bool
    latency_s: float
    tokens: int
    cost_usd: float


def run_case(agent: ClaimAgent, case: dict[str, Any]) -> CaseResult:
    exp = case["expected"]
    request = ClaimRequest(
        message=case["message"],
        attachment_text=case.get("attachment_text"),
        received_on=case.get("received_on"),
    )
    start = time.perf_counter()
    result = agent.submit(request)
    latency = time.perf_counter() - start
    amount = result.decision.amount_eur if result.decision else 0
    extraction_ok: bool | None = None
    if exp["flight_number"] is not None and result.facts is not None:
        extraction_ok = (
            result.facts.flight_number == exp["flight_number"]
            and str(result.facts.flight_date) == exp["flight_date"]
        )
    elif exp["flight_number"] is not None:
        extraction_ok = False
    flagged = any(f.code == "PROMPT_INJECTION_SUSPECTED" for f in result.risk_flags)
    return CaseResult(
        id=case["id"],
        expected_status=exp["status"],
        status=result.status.value,
        expected_amount=exp["amount_eur"],
        amount=amount,
        status_ok=result.status.value == exp["status"],
        amount_ok=amount == exp["amount_eur"],
        extraction_ok=extraction_ok,
        injection_expected=exp["injection"],
        injection_flagged=flagged,
        overpaid=amount > exp["amount_eur"],
        latency_s=round(latency, 3),
        tokens=result.usage.total_tokens,
        cost_usd=result.usage.cost_usd,
    )


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    def rate(values: list[bool]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    extraction = [r.extraction_ok for r in results if r.extraction_ok is not None]
    injection_cases = [r for r in results if r.injection_expected]
    clean_cases = [r for r in results if not r.injection_expected]
    return {
        "cases": len(results),
        "decision_accuracy": rate([r.status_ok for r in results]),
        "amount_accuracy": rate([r.amount_ok for r in results]),
        "extraction_accuracy": rate(extraction),
        "injection_recall": rate([r.injection_flagged for r in injection_cases]),
        "injection_false_positive_rate": rate([r.injection_flagged for r in clean_cases]),
        "overpayments": sum(r.overpaid for r in results),
        "latency_p50_s": round(statistics.median(r.latency_s for r in results), 3),
        "avg_tokens": round(statistics.mean(r.tokens for r in results)),
        "avg_cost_usd": round(statistics.mean(r.cost_usd for r in results), 5),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
    }


def to_markdown(
    provider: str, model: str, summary: dict[str, Any], results: list[CaseResult]
) -> str:
    lines = [
        f"# Eval results: `{provider}`",
        "",
        f"Model: `{model}` · prompts `{PROMPT_VERSION}` · {summary['cases']} cases",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items() if k != "cases"]
    lines += [
        "",
        "| case | expected | got | EUR exp/got | extraction | injection flag |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        mark = "✅" if r.status_ok and r.amount_ok else "❌"
        ext = {True: "✓", False: "✗", None: "-"}[r.extraction_ok]
        inj = (
            ("✓" if r.injection_flagged else "✗")
            if r.injection_expected
            else ("false positive" if r.injection_flagged else "-")
        )
        lines.append(
            f"| {mark} {r.id} | {r.expected_status} | {r.status} | "
            f"{r.expected_amount}/{r.amount} | {ext} | {inj} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=[p.value for p in LLMProvider], default="fake")
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    args = parser.parse_args(argv)

    configure_logging("ERROR")
    cases = [
        json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line
    ]
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(llm_provider=LLMProvider(args.provider), runs_dir=Path(tmp))
        with ClaimAgent(settings, persist=False) as agent:
            results = [run_case(agent, c) for c in cases]
    summary = summarize(results)
    model = settings.main_model if args.provider == "anthropic" else "regex baseline"
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.provider}.md").write_text(
        to_markdown(args.provider, model, summary, results), encoding="utf-8"
    )
    (out_dir / f"{args.provider}.json").write_text(
        json.dumps({"summary": summary, "cases": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["decision_accuracy"] < args.min_accuracy:
        print(f"FAIL: decision accuracy below {args.min_accuracy}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
