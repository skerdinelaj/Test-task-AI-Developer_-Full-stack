"""The agent as an explicit LangGraph state machine.

    validate_input → screen_input → extract_facts → plan ⇄ run_tools → verify_evidence
      → apply_rules → [draft_letter ⇄ check_letter] → review_gate (human-in-the-loop) → finalize

Design principles
* The LLM *reads* (extraction), *plans* (which evidence to fetch) and *writes* (letter).
  Code *validates* every LLM output and *decides* eligibility and money.
* Privilege separation: the extraction step reads raw customer text but has no tools; the planner
  has tools but only sees validated, structured facts.
* Every LLM failure has a deterministic fallback, so the agent degrades to "needs review"
  instead of crashing or guessing.
"""

from __future__ import annotations

import json
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from eu261_claim_agent.config import Settings
from eu261_claim_agent.guards import (
    check_request,
    combine_verdicts,
    heuristic_screen,
    request_text,
    wrap_untrusted,
)
from eu261_claim_agent.letters import CaseFile, render_template_letter, validate_letter
from eu261_claim_agent.llm.base import LLMClient, LLMError, LLMRefusal
from eu261_claim_agent.normalize import normalize_facts
from eu261_claim_agent.observability import Tracer
from eu261_claim_agent.prompts import PROMPT_VERSION, load_prompt
from eu261_claim_agent.reference_data import ReferenceData
from eu261_claim_agent.rules import evaluate_claim
from eu261_claim_agent.schemas import (
    Airline,
    ClaimRequest,
    ClaimStatus,
    DisruptionType,
    EligibilityDecision,
    Evidence,
    ExtractedFacts,
    FlightRecord,
    GuardVerdict,
    LetterDraft,
    LLMTask,
    NormalizedFacts,
    ReviewOutcome,
    ReviewState,
    RiskFlag,
    Severity,
    StepRecord,
    ToolCallRecord,
    Usage,
)
from eu261_claim_agent.summary import build_summary
from eu261_claim_agent.tools import DistanceResult, SearchResult, ToolRegistry


def _add_usage(left: Usage, right: Usage) -> Usage:
    return left + right


class AgentState(TypedDict, total=False):
    # inputs
    run_id: str
    request: ClaimRequest
    today: date
    boundary: str
    # control
    halt: bool
    status: ClaimStatus
    terminal_reasons: list[str]
    # working data
    guard: GuardVerdict
    facts: NormalizedFacts
    missing_fields: list[str]
    planner_messages: list[dict[str, Any]]
    pending_calls: list[dict[str, Any]]
    plan_steps: int
    evidence: Evidence
    decision: EligibilityDecision
    letter: LetterDraft
    letter_attempts: int
    letter_problems: list[str]
    review: ReviewOutcome
    review_state: ReviewState
    summary: str
    # append-only (reducers)
    risk_flags: Annotated[list[RiskFlag], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    steps: Annotated[list[StepRecord], operator.add]
    usage: Annotated[Usage, _add_usage]


Update = dict[str, Any]
NodeFn = Callable[[AgentState], Update]


@dataclass
class AgentDeps:
    settings: Settings
    llm: LLMClient
    tools: ToolRegistry
    ref: ReferenceData
    tracer: Tracer


def _flag(code: str, severity: Severity, message: str) -> RiskFlag:
    return RiskFlag(code=code, severity=severity, message=message)


def build_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    settings, llm, tools, tracer = deps.settings, deps.llm, deps.tools, deps.tracer

    # ------------------------------------------------------------------ helpers
    def traced(name: str) -> Callable[[NodeFn], NodeFn]:
        """Time every node, log start/end, and record the step in state."""

        def decorator(fn: NodeFn) -> NodeFn:
            def wrapper(state: AgentState) -> Update:
                start = time.perf_counter()
                tracer.event(state["run_id"], "node_start", node=name)
                update = fn(state)
                ms = round((time.perf_counter() - start) * 1000, 2)
                tracer.event(
                    state["run_id"],
                    "node_end",
                    node=name,
                    duration_ms=ms,
                    status=update.get("status"),
                    halt=update.get("halt", False),
                )
                update["steps"] = [StepRecord(node=name, duration_ms=ms)]
                return update

            wrapper.__name__ = name
            return wrapper

        return decorator

    def over_budget(state: AgentState) -> bool:
        return state.get("usage", Usage()).total_tokens >= settings.max_run_tokens

    def log_llm(state: AgentState, task: LLMTask, usage: Usage, model: str, **extra: Any) -> None:
        tracer.event(
            state["run_id"],
            "llm_call",
            task=task.value,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            prompt_version=PROMPT_VERSION,
            **extra,
        )

    def halt(status: ClaimStatus, reasons: list[str], **extra: Any) -> Update:
        return {"halt": True, "status": status, "terminal_reasons": reasons, **extra}

    # ------------------------------------------------------------------ nodes
    @traced("validate_input")
    def validate_input(state: AgentState) -> Update:
        check = check_request(
            state["request"], settings.max_message_chars, settings.max_attachment_chars
        )
        if not check.ok:
            return halt(
                ClaimStatus.REJECTED,
                check.errors,
                risk_flags=[_flag("INVALID_INPUT", Severity.WARNING, " ".join(check.errors))],
            )
        return {"halt": False}

    @traced("screen_input")
    def screen_input(state: AgentState) -> Update:
        text = request_text(state["request"])
        heuristic = heuristic_screen(text)
        flags: list[RiskFlag] = []
        llm_verdict: GuardVerdict | None = None
        usage = Usage()
        try:
            result = llm.structured(
                LLMTask.GUARD,
                load_prompt("guard"),
                wrap_untrusted("text", text, state["boundary"]),
                GuardVerdict,
            )
            llm_verdict, usage = result.value, result.usage
            log_llm(state, LLMTask.GUARD, usage, result.model)
        except LLMRefusal:
            # The screening model refused to classify: treat as suspicious (fail safe).
            llm_verdict = GuardVerdict(
                injection_suspected=True,
                fabrication_request=False,
                off_topic=False,
                rationale="screening model refused",
            )
        except LLMError as exc:
            flags.append(
                _flag(
                    "GUARD_DEGRADED",
                    Severity.WARNING,
                    f"LLM screen unavailable, heuristic screen only ({exc}).",
                )
            )
        verdict = combine_verdicts(heuristic, llm_verdict)
        tracer.event(state["run_id"], "guard_verdict", **verdict.model_dump())
        if verdict.fabrication_request:
            return halt(
                ClaimStatus.REJECTED,
                [
                    "The request asks to present facts differently from what happened. We only "
                    "submit claims based on the real facts; the customer can resubmit them "
                    "as they are."
                ],
                guard=verdict,
                usage=usage,
                risk_flags=[*flags, _flag("FABRICATION_REQUEST", Severity.HIGH, verdict.rationale)],
            )
        if verdict.off_topic:
            return halt(
                ClaimStatus.OUT_OF_SCOPE,
                ["This message is not about a flight delay, cancellation or denied boarding."],
                guard=verdict,
                usage=usage,
                risk_flags=flags,
            )
        if verdict.injection_suspected:
            flags.append(
                _flag(
                    "PROMPT_INJECTION_SUSPECTED",
                    Severity.HIGH,
                    "The input contains instructions aimed at the AI. They were "
                    "ignored and the case is routed to a human.",
                )
            )
        return {"guard": verdict, "usage": usage, "risk_flags": flags}

    @traced("extract_facts")
    def extract_facts(state: AgentState) -> Update:
        if over_budget(state):
            return halt(
                ClaimStatus.NEEDS_REVIEW,
                ["Token budget exceeded."],
                risk_flags=[_flag("BUDGET_EXCEEDED", Severity.HIGH, "Token budget exceeded.")],
            )
        req = state["request"]
        parts = [
            f"Reference date (today): {state['today'].isoformat()}",
            "Customer message:",
            wrap_untrusted("message", req.message, state["boundary"]),
        ]
        if req.attachment_text:
            parts += [
                "Attachment:",
                wrap_untrusted("attachment", req.attachment_text, state["boundary"]),
            ]
        try:
            result = llm.structured(
                LLMTask.EXTRACT, load_prompt("extract"), "\n".join(parts), ExtractedFacts
            )
        except LLMError as exc:
            return halt(
                ClaimStatus.NEEDS_REVIEW,
                ["The AI service could not read the claim."],
                risk_flags=[_flag("LLM_FAILURE", Severity.HIGH, f"Extraction failed: {exc}")],
            )
        log_llm(state, LLMTask.EXTRACT, result.usage, result.model, attempts=result.attempts)
        normalized = normalize_facts(result.value, state["today"])
        tracer.event(
            state["run_id"],
            "facts_extracted",
            **normalized.facts.model_dump(mode="json"),
            missing=normalized.missing_fields,
        )
        update: Update = {
            "facts": normalized.facts,
            "missing_fields": normalized.missing_fields,
            "usage": result.usage,
            "risk_flags": normalized.flags,
        }
        if normalized.missing_fields:
            update |= halt(ClaimStatus.NEEDS_INFO, ["Some required details are missing."])
        return update

    @traced("plan")
    def plan(state: AgentState) -> Update:
        steps = state.get("plan_steps", 0)
        if steps >= settings.max_plan_steps or over_budget(state):
            return {
                "pending_calls": [],
                "risk_flags": [
                    _flag(
                        "PLAN_LIMIT_REACHED",
                        Severity.INFO,
                        "Planner stopped at the step/token limit.",
                    )
                ],
            }
        messages = state.get("planner_messages") or [
            {
                "role": "user",
                "content": "<facts>\n"
                + state["facts"].model_dump_json(
                    include={
                        "flight_number",
                        "flight_date",
                        "departure_airport",
                        "arrival_airport",
                        "disruption_type",
                        "reported_delay_minutes",
                        "cancellation_notice_days",
                    },
                    indent=2,
                )
                + "\n</facts>\nCollect the evidence for this claim.",
            }
        ]
        try:
            turn = llm.tool_turn(load_prompt("plan"), messages, tools.specs())
        except LLMError as exc:
            # Degrade gracefully: verify_evidence will fetch what is needed deterministically.
            return {
                "pending_calls": [],
                "risk_flags": [
                    _flag(
                        "PLANNER_DEGRADED",
                        Severity.WARNING,
                        f"Planner unavailable ({exc}); evidence "
                        "was collected by the deterministic fallback.",
                    )
                ],
            }
        log_llm(
            state,
            LLMTask.PLAN,
            turn.usage,
            turn.model,
            step=steps + 1,
            tool_calls=[c.name for c in turn.tool_calls],
            stop_reason=turn.stop_reason,
        )
        return {
            "planner_messages": [
                *messages,
                {"role": "assistant", "content": turn.assistant_content},
            ],
            "pending_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls
            ],
            "plan_steps": steps + 1,
            "usage": turn.usage,
        }

    def absorb(evidence: Evidence, facts: NormalizedFacts, result: Any) -> list[RiskFlag]:
        """Fold a tool result into the evidence, accepting only what matches the claim."""
        flags: list[RiskFlag] = []
        if isinstance(result, FlightRecord):
            if (result.flight_number, result.date) == (facts.flight_number, facts.flight_date):
                evidence.flight = result
            else:
                flags.append(
                    _flag(
                        "PLANNER_OFF_TARGET",
                        Severity.WARNING,
                        f"Planner looked up {result.flight_number}; result ignored.",
                    )
                )
        elif isinstance(result, Airline):
            if evidence.flight is None or evidence.flight.airline == result.iata:
                evidence.airline = result
        elif isinstance(result, DistanceResult):
            route = (result.origin.iata, result.destination.iata)
            if evidence.flight is None or route == (
                evidence.flight.origin,
                evidence.flight.destination,
            ):
                evidence.origin, evidence.destination = result.origin, result.destination
                evidence.distance_km = result.distance_km
        elif isinstance(result, SearchResult):
            seen = {(s.doc_id, s.text) for s in evidence.regulation}
            evidence.regulation += [p for p in result.passages if (p.doc_id, p.text) not in seen]
        return flags

    def execute(
        state: AgentState, evidence: Evidence, name: str, args: dict[str, Any], source: str
    ) -> tuple[str, bool, ToolCallRecord, list[RiskFlag]]:
        execution = tools.execute(name, args)
        record = ToolCallRecord(
            tool=name,
            arguments=execution.arguments,
            ok=execution.ok,
            duration_ms=execution.duration_ms,
            error=execution.error,
            source=source,
        )
        tracer.event(state["run_id"], "tool_call", **record.model_dump())
        flags = absorb(evidence, state["facts"], execution.result) if execution.ok else []
        return execution.content, execution.ok, record, flags

    @traced("run_tools")
    def run_tools(state: AgentState) -> Update:
        evidence = state.get("evidence", Evidence()).model_copy(deep=True)
        results: list[dict[str, Any]] = []
        records: list[ToolCallRecord] = []
        flags: list[RiskFlag] = []
        for call in state.get("pending_calls", []):
            content, ok, record, new_flags = execute(
                state, evidence, call["name"], call["arguments"], "planner"
            )
            records.append(record)
            flags += new_flags
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": content,
            }
            if not ok:
                block["is_error"] = True
            results.append(block)
        messages = [*state["planner_messages"], {"role": "user", "content": results}]
        return {
            "evidence": evidence,
            "planner_messages": messages,
            "pending_calls": [],
            "tool_calls": records,
            "risk_flags": flags,
        }

    @traced("verify_evidence")
    def verify_evidence(state: AgentState) -> Update:
        """Code checks the planner's work and fetches anything missing (planner proposes,
        code guarantees)."""
        facts = state["facts"]
        evidence = state.get("evidence", Evidence()).model_copy(deep=True)
        records: list[ToolCallRecord] = []
        flags: list[RiskFlag] = []

        def backfill(name: str, args: dict[str, Any]) -> None:
            _, _, record, new_flags = execute(state, evidence, name, args, "backfill")
            records.append(record)
            flags.extend(new_flags)
            evidence.backfilled.append(name)

        attempted = any(
            c.tool == "lookup_flight"
            and c.ok is False
            and c.arguments.get("flight_number") == facts.flight_number
            for c in state.get("tool_calls", [])
        )
        if evidence.flight is None and not attempted and facts.flight_number and facts.flight_date:
            backfill(
                "lookup_flight",
                {
                    "flight_number": facts.flight_number,
                    "flight_date": facts.flight_date.isoformat(),
                },
            )
        flight = evidence.flight
        if flight is not None:
            if evidence.airline is None or evidence.airline.iata != flight.airline:
                evidence.airline = None
                backfill("get_airline", {"airline_code": flight.airline})
            if (
                evidence.distance_km is None
                or evidence.origin is None
                or evidence.destination is None
                or (evidence.origin.iata, evidence.destination.iata)
                != (flight.origin, flight.destination)
            ):
                evidence.distance_km = None
                backfill(
                    "calculate_distance",
                    {"origin_iata": flight.origin, "destination_iata": flight.destination},
                )
        if not evidence.regulation:
            query = {
                DisruptionType.CANCELLATION: "cancellation notice compensation",
                DisruptionType.DENIED_BOARDING: "denied boarding compensation",
            }.get(facts.disruption_type, "delay three hours compensation")
            backfill("search_regulation", {"query": query, "top_k": 3})
        if evidence.backfilled:
            flags.append(
                _flag(
                    "EVIDENCE_BACKFILLED",
                    Severity.INFO,
                    "Code fetched evidence the planner did not: "
                    + ", ".join(evidence.backfilled)
                    + ".",
                )
            )
        return {"evidence": evidence, "tool_calls": records, "risk_flags": flags}

    @traced("apply_rules")
    def apply_rules(state: AgentState) -> Update:
        outcome = evaluate_claim(state["facts"], state["evidence"], deps.ref.rules)
        tracer.event(
            state["run_id"],
            "rules_decision",
            status=outcome.decision.status,
            amount_eur=outcome.decision.amount_eur,
            rule_ids=outcome.decision.rule_ids,
        )
        return {
            "decision": outcome.decision,
            "status": outcome.decision.status,
            "risk_flags": outcome.flags,
        }

    def case_file(state: AgentState) -> CaseFile:
        return CaseFile.build(state["facts"], state["decision"], state["evidence"])

    @traced("draft_letter")
    def draft_letter(state: AgentState) -> Update:
        attempts = state.get("letter_attempts", 0) + 1
        case = case_file(state)
        if over_budget(state):
            return {
                "letter": render_template_letter(case),
                "letter_attempts": attempts,
                "letter_problems": [],
                "risk_flags": [
                    _flag(
                        "LETTER_TEMPLATE_FALLBACK",
                        Severity.INFO,
                        "Token budget reached; template used.",
                    )
                ],
            }
        prompt = f"<case_file>\n{case.model_dump_json(indent=2)}\n</case_file>"
        if problems := state.get("letter_problems"):
            prompt += "\n\nYour previous draft was rejected for these reasons:\n- " + "\n- ".join(
                problems
            )
        try:
            result = llm.structured(LLMTask.DRAFT, load_prompt("draft"), prompt, LetterDraft)
        except LLMError as exc:
            return {
                "letter": render_template_letter(case),
                "letter_attempts": attempts,
                "letter_problems": [],
                "risk_flags": [
                    _flag(
                        "LETTER_TEMPLATE_FALLBACK",
                        Severity.INFO,
                        f"Drafting failed ({exc}); template used.",
                    )
                ],
            }
        log_llm(state, LLMTask.DRAFT, result.usage, result.model, attempt=attempts)
        return {"letter": result.value, "letter_attempts": attempts, "usage": result.usage}

    @traced("check_letter")
    def check_letter(state: AgentState) -> Update:
        case = case_file(state)
        problems = validate_letter(state["letter"], case)
        tracer.event(
            state["run_id"], "letter_check", attempt=state["letter_attempts"], problems=problems
        )
        if not problems:
            return {"letter_problems": []}
        if state["letter_attempts"] <= settings.max_validation_retries:
            return {"letter_problems": problems}  # loop back to draft_letter with feedback
        return {
            "letter": render_template_letter(case),
            "letter_problems": [],
            "risk_flags": [
                _flag(
                    "LETTER_TEMPLATE_FALLBACK",
                    Severity.WARNING,
                    "The AI draft failed validation after retries; the safe template was used.",
                )
            ],
        }

    @traced("review_gate")
    def review_gate(state: AgentState) -> Update:
        status = state["status"]
        high = any(f.severity == Severity.HIGH for f in state.get("risk_flags", []))
        needed = status in (ClaimStatus.ELIGIBLE, ClaimStatus.NEEDS_REVIEW) or (
            status == ClaimStatus.NOT_ELIGIBLE and high
        )
        if not needed:
            return {"review_state": ReviewState.NOT_REQUIRED}
        # Pauses the graph; state is checkpointed. Resumed by ClaimAgent.review().
        answer = interrupt(
            {
                "run_id": state["run_id"],
                "status": status.value,
                "amount_eur": state["decision"].amount_eur if state.get("decision") else 0,
            }
        )
        review = ReviewOutcome.model_validate(answer)
        tracer.event(
            state["run_id"], "human_review", approved=review.approved, reviewer=review.reviewer
        )
        return {
            "review": review,
            "review_state": ReviewState.APPROVED if review.approved else ReviewState.REJECTED,
        }

    @traced("finalize")
    def finalize(state: AgentState) -> Update:
        review_state = state.get("review_state", ReviewState.NOT_REQUIRED)
        summary = build_summary(
            state["status"],
            state.get("facts"),
            state.get("decision"),
            state.get("terminal_reasons", []),
            state.get("missing_fields", []),
            state.get("risk_flags", []),
            review_state,
            state.get("review"),
        )
        tracer.event(
            state["run_id"],
            "final_decision",
            status=state["status"],
            review_state=review_state,
            usage=state.get("usage", Usage()).model_dump(),
        )
        return {"summary": summary, "review_state": review_state}

    # ------------------------------------------------------------------ routing
    def after(next_node: str) -> Callable[[AgentState], str]:
        def route(state: AgentState) -> str:
            return "finalize" if state.get("halt") else next_node

        route.__name__ = f"route_to_{next_node}"
        return route

    def after_plan(state: AgentState) -> str:
        return "run_tools" if state.get("pending_calls") else "verify_evidence"

    def after_rules(state: AgentState) -> str:
        return "draft_letter" if state["status"] == ClaimStatus.ELIGIBLE else "review_gate"

    def after_check(state: AgentState) -> str:
        return "draft_letter" if state.get("letter_problems") else "review_gate"

    # ------------------------------------------------------------------ wiring
    g = StateGraph(AgentState)
    for fn in (
        validate_input,
        screen_input,
        extract_facts,
        plan,
        run_tools,
        verify_evidence,
        apply_rules,
        draft_letter,
        check_letter,
        review_gate,
        finalize,
    ):
        # LangGraph's overloads can't infer a TypedDict with reducer annotations.
        g.add_node(fn.__name__, fn)  # type: ignore[call-overload]
    g.add_edge(START, "validate_input")
    g.add_conditional_edges("validate_input", after("screen_input"), ["screen_input", "finalize"])
    g.add_conditional_edges("screen_input", after("extract_facts"), ["extract_facts", "finalize"])
    g.add_conditional_edges("extract_facts", after("plan"), ["plan", "finalize"])
    g.add_conditional_edges("plan", after_plan, ["run_tools", "verify_evidence"])
    g.add_edge("run_tools", "plan")
    g.add_edge("verify_evidence", "apply_rules")
    g.add_conditional_edges("apply_rules", after_rules, ["draft_letter", "review_gate"])
    g.add_edge("draft_letter", "check_letter")
    g.add_conditional_edges("check_letter", after_check, ["draft_letter", "review_gate"])
    g.add_edge("review_gate", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def planner_transcript(state: AgentState) -> str:
    """Debug helper: the planner conversation as pretty JSON."""
    return json.dumps(state.get("planner_messages", []), indent=2, default=str)
