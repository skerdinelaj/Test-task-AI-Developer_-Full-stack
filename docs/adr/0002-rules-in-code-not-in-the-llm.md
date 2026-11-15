# ADR 0002: Eligibility and money are decided by code, not by the LLM

**Status:** accepted · **Date:** 2026-11-15

## Context
EU261 eligibility is mostly mechanical (scope, delay ≥ 3 h, notice windows, distance bands), and a wrong amount has a direct financial and legal cost. LLMs are good at reading messy text and writing letters, but they are non-deterministic and can be manipulated by prompt injection.

## Decision
- The LLM extracts facts, plans which evidence to fetch, and drafts the letter.
- A pure-Python rules engine (`rules.py`) makes the decision from the **flight record**, never from the customer's claim.
- Thresholds live in `data/rules_config.json`, because an EU261 reform has been under negotiation.
- The letter is checked by code (`letters.validate_letter`): the amount must equal the rules engine's amount, the flight number must appear, no links or placeholders. Failed drafts are retried with feedback, then replaced by a safe template.

## Consequences
- A prompt injection cannot change the amount (tested: `test_injection_cannot_change_amount_or_reach_the_letter`).
- The rules are unit-tested at every boundary and property-tested with Hypothesis.
- Edge cases the code cannot judge (extraordinary circumstances, 7–13 day cancellation notice, unverified denied boarding) go to a human instead of being guessed.
