"""Input guardrails: deterministic checks, heuristic screening and safe wrapping of untrusted text.

Layered defence against prompt injection (see README "Security"):
1. deterministic input limits (this module);
2. heuristic + LLM screening (this module + the guard prompt);
3. *spotlighting*: untrusted text is wrapped in a per-run random tag and described as data;
4. privilege separation: the LLM that reads raw customer text has no tools, and the LLM that has
   tools only sees validated, structured facts;
5. money and eligibility are decided by deterministic code, never by the LLM.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

from eu261_claim_agent.schemas import ClaimRequest, GuardVerdict

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
        r"disregard (all |any )?(the )?(previous|prior|above|system)",
        r"\b(system|developer) (prompt|message|instruction)s?\b",
        r"you are now\b",
        r"new instructions?:",
        r"^\s*(system|assistant)\s*:",
        r"\bact as (an?|the)\b",
        r"(approve|grant|pay) (this|the) claim (automatically|immediately|regardless)",
        r"(reveal|print|show|leak) (your|the) (prompt|instructions|system)",
        r"(send|forward|email) (all|every|the) (claims?|data|records?)",
        r"</?(untrusted|customer|instructions?|system)[^>]*>",
    )
)

FABRICATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(say|write|state|claim|put|pretend) (that )?(it|the (flight|delay))? ?was\b"
        r".*\b(hours?|cancel)",
        r"\b(make up|invent|fake|exaggerate|inflate)\b",
        r"\beven though (it|the (flight|delay)) (was|wasn'?t)\b",
        r"\b(lie|lying) (to|about)\b",
        r"\bchange the (date|delay|times?|facts)\b",
    )
)

FLIGHT_KEYWORDS = re.compile(
    r"\b(flight|flew|fly|delay(ed)?|late|cancel(l?ed|lation)?|boarding|overbook(ed)?|airline|"
    r"airport|plane|aircraft|pilot|crew|gate|layover|land(ed|ing)|took off|departure|"
    r"arriv(al|ed)|compensation|eu ?261)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InputCheck:
    ok: bool
    errors: list[str]


def check_request(
    request: ClaimRequest, max_message_chars: int, max_attachment_chars: int
) -> InputCheck:
    errors: list[str] = []
    if not request.message.strip():
        errors.append("The message is empty.")
    if len(request.message) > max_message_chars:
        errors.append(f"The message is longer than {max_message_chars} characters.")
    if request.attachment_text and len(request.attachment_text) > max_attachment_chars:
        errors.append(f"The attachment text is longer than {max_attachment_chars} characters.")
    if "\x00" in request.message or (request.attachment_text and "\x00" in request.attachment_text):
        errors.append("The input contains binary data.")
    return InputCheck(ok=not errors, errors=errors)


def heuristic_screen(text: str) -> GuardVerdict:
    """Cheap, deterministic screen. Used on its own offline and OR-ed with the LLM screen online."""
    injection_hits = [p.pattern for p in INJECTION_PATTERNS if p.search(text)]
    fabrication_hits = [p.pattern for p in FABRICATION_PATTERNS if p.search(text)]
    off_topic = FLIGHT_KEYWORDS.search(text) is None
    parts = []
    if injection_hits:
        parts.append(f"{len(injection_hits)} injection pattern(s) matched")
    if fabrication_hits:
        parts.append("asks to alter or invent facts")
    if off_topic:
        parts.append("no flight-related content found")
    return GuardVerdict(
        injection_suspected=bool(injection_hits),
        fabrication_request=bool(fabrication_hits),
        off_topic=off_topic,
        rationale="; ".join(parts) or "no issues found by heuristic screen",
    )


def combine_verdicts(heuristic: GuardVerdict, llm: GuardVerdict | None) -> GuardVerdict:
    """Injection and fabrication are OR-ed (fail safe). Topicality trusts the LLM when available,
    because keyword matching is too blunt for that."""
    if llm is None:
        return heuristic
    return GuardVerdict(
        injection_suspected=heuristic.injection_suspected or llm.injection_suspected,
        fabrication_request=heuristic.fabrication_request or llm.fabrication_request,
        off_topic=llm.off_topic,
        rationale=f"LLM: {llm.rationale} | heuristic: {heuristic.rationale}",
    )


def new_boundary() -> str:
    return secrets.token_hex(6)


def wrap_untrusted(label: str, text: str, boundary: str) -> str:
    """Spotlighting: wrap untrusted text in a tag the attacker cannot predict or close early."""
    tag = f"untrusted_{label}_{boundary}"
    sanitized = text.replace(tag, "[removed]")
    return f"<{tag}>\n{sanitized}\n</{tag}>"


def request_text(request: ClaimRequest) -> str:
    return request.message + ("\n\n" + request.attachment_text if request.attachment_text else "")
