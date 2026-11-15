"""Structured logging and per-run traces (JSONL), with PII redaction.

Every node, LLM call and tool call is recorded as an event in ``runs/<run_id>.jsonl`` and logged
as JSON via structlog. The ``Tracer`` interface is the seam for OpenTelemetry / Langfuse later.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# International format only (+country code), so dates and times are never mistaken for phones.
PHONE_RE = re.compile(r"\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}")
BOOKING_REF_RE = re.compile(
    r"(booking (reference|ref|code)|pnr|reservation (code|number))\s*[:#]?\s*[A-Z0-9]{5,8}",
    re.IGNORECASE,
)
PII_KEYS = frozenset({"passenger_name", "customer_email", "email", "phone"})


def redact_text(text: str) -> str:
    text = EMAIL_RE.sub("[email]", text)
    text = BOOKING_REF_RE.sub(lambda m: m.group(0).split()[0] + " reference [redacted]", text)
    return PHONE_RE.sub("[phone]", text)


def redact(value: Any) -> Any:
    """Recursively redact PII from anything we are about to log."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: ("[redacted]" if k in PII_KEYS and v else redact(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


class Tracer:
    """Append-only trace store, one JSONL file per run."""

    def __init__(self, runs_dir: Path, enabled: bool = True) -> None:
        self.runs_dir = runs_dir
        self.enabled = enabled
        self._lock = threading.Lock()
        self._log = structlog.get_logger("claim_agent")
        if enabled:
            runs_dir.mkdir(parents=True, exist_ok=True)

    def event(self, run_id: str, event: str, **data: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "event": event,
            **redact(data),
        }
        self._log.info(
            event,
            run_id=run_id,
            **{k: v for k, v in record.items() if k not in {"ts", "run_id", "event"}},
        )
        if not self.enabled:
            return
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, (self.runs_dir / f"{run_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self, run_id: str) -> list[dict[str, Any]]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):  # no path traversal
            return []
        path = self.runs_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
