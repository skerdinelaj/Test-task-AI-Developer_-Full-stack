"""Prompts live in versioned .md files so they can be reviewed and diffed like code."""

from __future__ import annotations

from functools import cache
from importlib import resources

PROMPT_VERSION = "2026-11-15.2"


@cache
def load_prompt(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8").strip()
