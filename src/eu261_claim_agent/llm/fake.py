"""Deterministic offline stand-in for Claude.

Not intelligent: it uses regular expressions and a fixed tool plan. It exists so that the whole
pipeline (graph, tools, guards, rules, review, API) can be run, demoed and tested in CI without
an API key or network, and so that tests are reproducible. Real quality is measured with
``make eval PROVIDER=anthropic``.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from eu261_claim_agent.guards import heuristic_screen
from eu261_claim_agent.letters import CaseFile, render_template_letter
from eu261_claim_agent.llm.base import StructuredResult, ToolCall, ToolTurn
from eu261_claim_agent.reference_data import ReferenceData
from eu261_claim_agent.schemas import DisruptionType, ExtractedFacts, LLMTask, Usage
from eu261_claim_agent.tools import ToolSpec

UNTRUSTED_RE = re.compile(r"<(untrusted_[a-z]+_[0-9a-f]+)>\n(.*?)\n</\1>", re.DOTALL)
FLIGHT_RE = re.compile(r"\b([A-Z][A-Z0-9]|[0-9][A-Z])\s?-?(\d{1,4})\b")
MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}
MONTH_ALT = "|".join(MONTHS)
DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "ymd"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_ALT})\s+(\d{{4}})\b", re.I), "dmy_name"),
    (
        re.compile(rf"\b({MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I),
        "mdy_name",
    ),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "dmy"),
)
HOURS_MIN_RE = re.compile(
    r"(\d+)\s*(?:h|hours?|hrs?)\s*(?:and\s*)?(\d+)\s*(?:m|min|mins|minutes)\b", re.I
)
HOURS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:h|hours?|hrs?)\b", re.I)
MINUTES_RE = re.compile(r"(\d+)\s*(?:min|mins|minutes)\b", re.I)
NOTICE_RE = re.compile(r"(\d+)\s*days?\s*(?:before|prior|in advance|ahead|notice|earlier)", re.I)
NAME_RE = re.compile(
    r"(?i:my name is|name:|passenger:)\s*([A-Z][a-zA-Z'-]+(?:[ \t]+[A-Z][a-zA-Z'-]+)+)"
)
REASON_RE = re.compile(
    r"(?:because of|due to|reason:|citing|blamed it on)\s+(?:an?\s+|the\s+)?([a-z][a-z \-]{3,40})",
    re.I,
)
EXCLUDED_PREFIXES = frozenset({"EU", "EC", "NO"})


def _untrusted_text(prompt: str) -> str:
    parts = [m.group(2) for m in UNTRUSTED_RE.finditer(prompt)]
    return "\n".join(parts) if parts else prompt


class FakeLLM:
    provider = "fake"

    def __init__(self, ref: ReferenceData) -> None:
        self._ref = ref
        self._city_to_iata = {a.city.lower(): a.iata for a in ref.airports.values()}

    # ------------------------------------------------------------------ structured
    def structured[T: BaseModel](
        self, task: LLMTask, system: str, user: str, schema: type[T]
    ) -> StructuredResult[T]:
        if task == LLMTask.GUARD:
            value: BaseModel = heuristic_screen(_untrusted_text(user))
        elif task == LLMTask.EXTRACT:
            value = self.extract(_untrusted_text(user))
        elif task == LLMTask.DRAFT:
            match = re.search(r"<case_file>\s*(\{.*\})\s*</case_file>", user, re.DOTALL)
            if match is None:
                raise ValueError("draft prompt has no <case_file>")
            value = render_template_letter(CaseFile.model_validate_json(match.group(1)))
        else:
            raise NotImplementedError(task)
        assert isinstance(value, schema)
        return StructuredResult(value, Usage(llm_calls=1), "fake")

    def extract(self, text: str) -> ExtractedFacts:
        lowered = text.lower()
        if re.search(
            r"denied boarding|overbook|not allowed to board|bumped|refused (us )?boarding", lowered
        ):
            disruption = DisruptionType.DENIED_BOARDING
        elif "cancel" in lowered:
            disruption = DisruptionType.CANCELLATION
        elif re.search(r"delay|late\b|hours late", lowered):
            disruption = DisruptionType.DELAY
        else:
            disruption = DisruptionType.UNKNOWN
        departure, arrival = self._airports(text)
        notice = NOTICE_RE.search(text)
        name = NAME_RE.search(text)
        reason = REASON_RE.search(text)
        return ExtractedFacts(
            passenger_name=name.group(1) if name else None,
            flight_number=self._flight_number(text),
            flight_date=self._date(text),
            departure_airport=departure,
            arrival_airport=arrival,
            disruption_type=disruption,
            reported_delay_hours=self._delay_hours(text)
            if disruption == DisruptionType.DELAY
            else None,
            cancellation_notice_days=int(notice.group(1)) if notice else None,
            reason_given_by_airline=reason.group(1).strip() if reason else None,
        )

    def _flight_number(self, text: str) -> str | None:
        candidates = [
            (m.group(1), m.group(2))
            for m in FLIGHT_RE.finditer(text)
            if m.group(1) not in EXCLUDED_PREFIXES
        ]
        known = [c for c in candidates if c[0] in self._ref.airlines]
        pool = known or candidates
        return f"{pool[0][0]}{pool[0][1]}" if pool else None

    def _date(self, text: str) -> str | None:
        for pattern, kind in DATE_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            try:
                if kind == "ymd":
                    d = date(int(m[1]), int(m[2]), int(m[3]))
                elif kind == "dmy_name":
                    d = date(int(m[3]), MONTHS[m[2].lower()], int(m[1]))
                elif kind == "mdy_name":
                    d = date(int(m[3]), MONTHS[m[1].lower()], int(m[2]))
                else:
                    d = date(int(m[3]), int(m[2]), int(m[1]))
            except ValueError:
                continue
            return d.isoformat()
        return None

    def _airports(self, text: str) -> tuple[str | None, str | None]:
        hits: list[tuple[int, str]] = [
            (m.start(), m.group(0))
            for m in re.finditer(r"\b[A-Z]{3}\b", text)
            if m.group(0) in self._ref.airports
        ]
        lowered = text.lower()
        for city, iata in self._city_to_iata.items():
            for m in re.finditer(rf"\b{re.escape(city)}\b", lowered):
                hits.append((m.start(), iata))
        ordered: list[str] = []
        for _, code in sorted(hits):
            if code not in ordered:
                ordered.append(code)
        return (ordered[0] if ordered else None, ordered[1] if len(ordered) > 1 else None)

    @staticmethod
    def _delay_hours(text: str) -> float | None:
        if m := HOURS_MIN_RE.search(text):
            return int(m[1]) + int(m[2]) / 60
        if m := HOURS_RE.search(text):
            return float(m[1].replace(",", "."))
        if m := MINUTES_RE.search(text):
            return int(m[1]) / 60
        return None

    # ------------------------------------------------------------------ planning
    def tool_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]
    ) -> ToolTurn:
        facts_match = re.search(
            r"<facts>\s*(\{.*?\})\s*</facts>", str(messages[0]["content"]), re.DOTALL
        )
        facts: dict[str, Any] = json.loads(facts_match.group(1)) if facts_match else {}
        names_by_id: dict[str, str] = {}
        results: dict[str, dict[str, Any]] = {}
        for msg in messages[1:]:
            for block in msg["content"] if isinstance(msg["content"], list) else []:
                if block.get("type") == "tool_use":
                    names_by_id[block["id"]] = block["name"]
                elif block.get("type") == "tool_result":
                    results[names_by_id[block["tool_use_id"]]] = json.loads(block["content"])
        step = len([m for m in messages if m["role"] == "assistant"])
        calls: list[ToolCall] = []
        if (
            "lookup_flight" not in results
            and facts.get("flight_number")
            and facts.get("flight_date")
        ):
            calls.append(
                ToolCall(
                    f"fake_{step}_0",
                    "lookup_flight",
                    {"flight_number": facts["flight_number"], "flight_date": facts["flight_date"]},
                )
            )
        elif "lookup_flight" in results and "search_regulation" not in results:
            record = results["lookup_flight"]
            if "error" in record:
                calls.append(
                    ToolCall(
                        f"fake_{step}_0",
                        "search_regulation",
                        {"query": "flight verification scope", "top_k": 2},
                    )
                )
            else:
                calls.extend(
                    [
                        ToolCall(
                            f"fake_{step}_0", "get_airline", {"airline_code": record["airline"]}
                        ),
                        ToolCall(
                            f"fake_{step}_1",
                            "calculate_distance",
                            {
                                "origin_iata": record["origin"],
                                "destination_iata": record["destination"],
                            },
                        ),
                        ToolCall(
                            f"fake_{step}_2",
                            "search_regulation",
                            {"query": self._query(facts, record), "top_k": 3},
                        ),
                    ]
                )
        content: list[dict[str, Any]] = [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in calls
        ]
        text = "" if calls else "Evidence collected."
        if text:
            content.append({"type": "text", "text": text})
        return ToolTurn(
            content, calls, text, "tool_use" if calls else "end_turn", Usage(llm_calls=1), "fake"
        )

    @staticmethod
    def _query(facts: dict[str, Any], record: dict[str, Any]) -> str:
        if record.get("status") == "cancelled":
            query = "cancellation notice days compensation"
        elif facts.get("disruption_type") == "denied_boarding":
            query = "denied boarding overbooked compensation"
        else:
            sched = datetime.fromisoformat(record["scheduled_arrival"])
            late = (
                record.get("actual_arrival")
                and datetime.fromisoformat(record["actual_arrival"]) > sched
            )
            query = "delay three hours arrival compensation" if late else "compensation amounts"
        if record.get("disruption_reason_code"):
            query += f" extraordinary circumstances {record['disruption_reason_code'].lower()}"
        return query
