"""Independent single-candidate hypothesis-first generation for AlphaLDM."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

from alpha_research.canonical import canonical_form
from alpha_research.formula import FormulaSyntaxError
from alpha_research.methods.ldm.generator import CandidateGenerator


SCHEMA_VERSION = "alphaldm.minimal_prompt_harness.v1"
_FENCE = re.compile(r"^```[a-z]*\s*\n|\n```\s*$", re.IGNORECASE)

SYSTEM_PROMPT = """You are the candidate-generation stage of an alpha-mining search agent.

Propose ONE new cross-sectional stock-selection factor as a directly executable
Qlib expression. Use only the fields, operators, history, and train-only evidence
in the user message. Never use labels, future observations, validation results,
or test results.

Construct the candidate in this order:
1. State one concise, testable financial hypothesis.
2. Explain how the expression's fields, operators, windows, and sign implement it.
3. Write the expression. The final value must already be oriented so that a
   larger value implies a larger expected next-day return.

Output one JSON object with exactly name, hypothesis, implementation_rationale,
and expression. Do not add markdown, metric estimates, or additional keys.
"""


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "hypothesis", "implementation_rationale", "expression"],
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 48,
                "pattern": r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
            },
            "hypothesis": {"type": "string", "minLength": 1},
            "implementation_rationale": {"type": "string", "minLength": 1},
            "expression": {"type": "string", "minLength": 1},
        },
    }


class MinimalHarnessCandidateGenerator(CandidateGenerator):
    """Generate independent candidates using pure LDM's parallel sampler."""

    def _one(self, context: str) -> dict[str, str] | None:
        chat_failures = 0
        last_chat_error: Exception | None = None
        for _ in range(self.max_attempts):
            self.calls += 1
            try:
                raw = self.client.chat(SYSTEM_PROMPT, context)
            except Exception as exc:
                chat_failures += 1
                last_chat_error = exc
                continue
            record = self._parse_minimal(raw)
            if record is None:
                continue
            try:
                canonical = canonical_form(record["expression"])
            except Exception:
                canonical = record["expression"]
            try:
                self.validator.validate(canonical)
            except (FormulaSyntaxError, ValueError, TypeError):
                continue
            except Exception:
                continue
            return record
        if chat_failures >= self.max_attempts and last_chat_error is not None:
            raise RuntimeError(
                f"candidate LLM failed on all {self.max_attempts} attempt(s): "
                f"{last_chat_error}"
            ) from last_chat_error
        return None

    @staticmethod
    def _parse_minimal(payload: Any) -> dict[str, str] | None:
        if isinstance(payload, dict):
            data: Any = payload
        else:
            text = _FENCE.sub("", str(payload).strip())
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        if list(Draft202012Validator(_schema()).iter_errors(data)):
            return None
        hypothesis = str(data["hypothesis"]).strip()
        rationale = str(data["implementation_rationale"]).strip()
        return {
            "name": str(data["name"]).strip(),
            "hypothesis": hypothesis,
            "implementation_rationale": rationale,
            "expression": str(data["expression"]).strip(),
            "reason": f"Hypothesis: {hypothesis} Implementation: {rationale}",
        }
