"""Candidate generation: the ``{x_1..x_N} ~ P(x | C_0)`` step of the LDM loop.

Context ``C_0`` carries what the flowchart specifies -- the formulas evaluated so
far with their train scores, the incumbent, and the operator/field grammar the
proposal must satisfy. The grammar is not prose: it is read straight off
``alpha_research.formula``, so the prompt cannot drift from what the validator
will actually accept.

Parallelism comes from issuing N independent single-candidate calls, following
the MLS-Bench generator. Asking one context for N candidates makes them
correlated, which defeats the point of sampling a pool the surrogate can then
discriminate between.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from alpha_research.canonical import canonical_form
from alpha_research.formula import (
    ALLOWED_FUNCTIONS,
    BINARY,
    ROLLING_ONE,
    ROLLING_TWO,
    SPECIAL,
    TERNARY,
    UNARY,
    FormulaSyntaxError,
    FormulaValidator,
)


_FENCE = re.compile(r"^```[a-z]*\s*\n|\n```\s*$")

_SYSTEM_PROMPT = (
    "You are the candidate-generation stage of an alpha-mining search agent. "
    "Propose ONE new cross-sectional stock-selection factor as a Qlib "
    "expression. Output valid JSON only, with this exact shape:\n"
    '{"name": "<SHORT_UPPER_SNAKE_NAME>", '
    '"expression": "<qlib expression>", '
    '"reason": "<one sentence on the economic mechanism and why the score should improve>"}\n'
    "Do not rank, test, or score candidates. Do not emit anything outside the "
    "JSON object. Do not wrap the JSON in markdown fences."
)


def _grammar_block(validator: FormulaValidator) -> str:
    """Render the accepted grammar from the validator that will judge the output."""
    return "\n".join(
        [
            "GRAMMAR (a proposal outside this is rejected without evaluation)",
            f"  Fields: {', '.join(sorted(validator.allowed_fields))}",
            f"  Unary: {', '.join(sorted(UNARY))}",
            f"  Binary: {', '.join(sorted(BINARY))}",
            f"  Ternary: {', '.join(sorted(TERNARY))}",
            f"  Rolling(expr, window): {', '.join(sorted(ROLLING_ONE))}",
            f"  Rolling(expr, expr, window): {', '.join(sorted(ROLLING_TWO))}",
            f"  Special: {', '.join(sorted(SPECIAL))} -- Quantile(expr, window, q) with 0 <= q <= 1",
            f"  Windows must be positive integers, at most {validator.max_window}.",
            "  A negative window would read future data and is rejected.",
            f"  Maximum nesting depth {validator.max_depth}, maximum length {validator.max_length} characters.",
            "  Infix + - * / and parentheses are accepted and treated as Add/Sub/Mul/Div.",
            "  No other functions and no constants other than numeric literals.",
        ]
    )


def _format_scores(record: dict[str, Any]) -> str:
    """Format either the standard scalar score or named multi-objective scores."""
    scores = record.get("scores")
    if isinstance(scores, dict):
        parts = []
        for name, value in scores.items():
            text = f"{value:.5f}" if isinstance(value, (int, float)) else "n/a"
            parts.append(f"{name}={text}")
        if parts:
            return ", ".join(parts)
    score = record.get("score")
    return f"{score:.5f}" if isinstance(score, (int, float)) else "n/a"


class CandidateGenerator:
    """Generate N candidate formulas per round via concurrent JSON-mode calls."""

    def __init__(
        self,
        *,
        client: Any,
        validator: FormulaValidator,
        candidates_per_round: int = 8,
        max_parallel: int = 8,
        max_attempts: int = 2,
        history_shown: int = 24,
    ) -> None:
        self.client = client
        self.validator = validator
        self.candidates_per_round = max(1, int(candidates_per_round))
        self.max_parallel = max(1, int(max_parallel))
        self.max_attempts = max(1, int(max_attempts))
        self.history_shown = max(0, int(history_shown))
        self.calls = 0

    # ------------------------------------------------------------- context

    def build_context(
        self,
        *,
        evaluated: Sequence[dict[str, Any]],
        best: dict[str, Any] | None,
        objective: str,
        round_id: int,
    ) -> str:
        """Assemble C_0: grammar, incumbent, and the scored history so far.

        Only train-window scores appear here. Validation is selection-only and
        test is report-only under the static protocol's leakage policy, so
        neither may reach the generator.
        """
        lines = [
            "TASK",
            "  Propose a cross-sectional factor for Chinese A-share daily data.",
            f"  It is scored by {objective} against the next-day return on the training window.",
            "  Higher is better. A factor that merely restates an existing one adds nothing:",
            "  the pool is later combined as an equal-weight rank, so redundancy costs a slot.",
            "",
            _grammar_block(self.validator),
            "",
            f"ROUND {round_id}",
        ]

        if best:
            best_score_text = (
                _format_scores(best)
                if isinstance(best.get("scores"), dict)
                else f"{objective} = {best.get('score')}"
            )
            lines += [
                "",
                "BEST SO FAR",
                f"  {best.get('expression', '')}",
                f"  {best_score_text}",
            ]

        shown = list(evaluated)[-self.history_shown :] if self.history_shown else []
        if shown:
            lines += ["", f"EVALUATED SO FAR (most recent {len(shown)} of {len(evaluated)})"]
            for record in shown:
                lines.append(f"  {_format_scores(record)}  {record.get('expression', '')}")

        lines += [
            "",
            "Propose one new factor that is not a restatement of the above.",
        ]
        return "\n".join(lines)

    # ---------------------------------------------------------- generation

    def generate(self, context: str) -> list[dict[str, str]]:
        """Return the validated, in-round-deduplicated candidates for one round."""
        def guarded_call(_index: int):
            try:
                return self._one(context), None
            except Exception as exc:
                return None, exc

        with ThreadPoolExecutor(max_workers=min(self.max_parallel, self.candidates_per_round)) as pool:
            outcomes = list(pool.map(guarded_call, range(self.candidates_per_round)))
        records = [record for record, _error in outcomes if record is not None]
        errors = [error for _record, error in outcomes if error is not None]
        if not records and errors:
            raise RuntimeError(
                f"all {self.candidates_per_round} candidate-generation calls failed; "
                f"first error: {errors[0]}"
            ) from errors[0]
        return records

    def _one(self, context: str) -> dict[str, str] | None:
        chat_failures = 0
        last_chat_error: Exception | None = None
        for _ in range(self.max_attempts):
            self.calls += 1
            try:
                raw = self.client.chat(_SYSTEM_PROMPT, context)
            except Exception as exc:
                chat_failures += 1
                last_chat_error = exc
                continue
            record = self._parse(raw)
            if record is None:
                continue
            # Judge the canonical call form: the grammar below is written in
            # call syntax, but Qlib and the rest of the pipeline accept infix
            # arithmetic, and canonical_form rewrites one into the other.
            try:
                canonical = canonical_form(record["expression"])
            except Exception:
                canonical = record["expression"]
            try:
                self.validator.validate(canonical)
            except FormulaSyntaxError:
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
    def _parse(payload: Any) -> dict[str, str] | None:
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
        if not isinstance(data, dict):
            return None
        expression = str(data.get("expression", "")).strip()
        if not expression:
            return None
        name = str(data.get("name", "")).strip() or "LDM_CANDIDATE"
        return {
            "name": re.sub(r"[^A-Za-z0-9_]", "_", name)[:48],
            "expression": expression,
            "reason": str(data.get("reason", "")).strip(),
        }


class MockCandidateGenerator:
    """Deterministic no-network generator for the smoke config."""

    def __init__(self, *, candidates_per_round: int = 4, **_kwargs: Any) -> None:
        self.candidates_per_round = max(1, int(candidates_per_round))
        self.calls = 0
        self._round = 0

    def build_context(self, **_kwargs: Any) -> str:
        return "mock-context"

    def restore_state(self, *, completed_round: int, llm_calls_total: int) -> None:
        """Restore deterministic mock sequencing without double-counting calls."""
        self._round = max(0, int(completed_round))
        # ``calls`` counts only this process. ContinuousDiscoveryLDM adds the
        # committed total separately when it writes the next checkpoint.
        self.calls = 0

    def generate(self, _context: str) -> list[dict[str, str]]:
        self._round += 1
        records = []
        for index in range(self.candidates_per_round):
            window = 5 + 5 * ((self._round + index) % 6)
            lag = self._round + index
            records.append({
                "name": f"MOCK_LDM_R{self._round}_C{index}",
                "expression": (
                    f"Mean($close, {window}) / "
                    f"Std(Ref($volume, {lag}), {window + 5})"
                ),
                "reason": "Mock candidate for the offline smoke test.",
            })
            self.calls += 1
        return records
