"""Strict parser for the point-in-time Qlib subset used by AlphaBench prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_FIELDS = ("$open", "$high", "$low", "$close", "$volume")

UNARY = {"Abs", "Sign", "Log", "Not"}
BINARY = {
    "Add", "Sub", "Mul", "Div", "Power", "Greater", "Less",
    "And", "Or", "Gt", "Ge", "Lt", "Le", "Eq", "Ne",
}
TERNARY = {"If"}
ROLLING_ONE = {
    "Ref", "Max", "Min", "Sum", "Mean", "Std", "Var", "Skew",
    "Kurt", "Med", "Mad", "Slope", "Rsquare", "Resi", "Rank",
    "Count", "EMA", "WMA", "Delta", "IdxMax", "IdxMin",
}
ROLLING_TWO = {"Corr", "Cov"}
SPECIAL = {"Quantile"}
ALLOWED_FUNCTIONS = UNARY | BINARY | TERNARY | ROLLING_ONE | ROLLING_TWO | SPECIAL

TOKEN_RE = re.compile(
    r"\s*(?:(?P<field>\$[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|(?P<punct>[(),]))"
)


class FormulaSyntaxError(ValueError):
    """Raised when a proposal falls outside the leakage-safe formula DSL."""


@dataclass(frozen=True)
class Node:
    kind: str
    value: str
    args: tuple["Node", ...] = ()

    @property
    def numeric_value(self) -> float | None:
        if self.kind != "number":
            return None
        return float(self.value)


class FormulaValidator:
    def __init__(
        self,
        allowed_fields: tuple[str, ...] = DEFAULT_FIELDS,
        *,
        max_length: int = 800,
        max_depth: int = 12,
        max_window: int = 504,
    ) -> None:
        self.allowed_fields = frozenset(allowed_fields)
        self.max_length = int(max_length)
        self.max_depth = int(max_depth)
        self.max_window = int(max_window)

    def validate(self, expression: str) -> Node:
        raw = str(expression).strip()
        if not raw:
            raise FormulaSyntaxError("expression is empty")
        if len(raw) > self.max_length:
            raise FormulaSyntaxError(f"expression exceeds max length {self.max_length}")
        tokens = self._tokenize(raw)
        node, position = self._parse(tokens, 0, depth=0)
        if position != len(tokens):
            raise FormulaSyntaxError(f"unexpected token {tokens[position][1]!r}")
        self._validate_node(node)
        return node

    def _tokenize(self, raw: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(raw):
            match = TOKEN_RE.match(raw, position)
            if match is None:
                snippet = raw[position : position + 20]
                raise FormulaSyntaxError(f"unsupported syntax near {snippet!r}")
            kind = match.lastgroup
            if kind is None:
                raise FormulaSyntaxError("unable to tokenize expression")
            tokens.append((kind, match.group(kind)))
            position = match.end()
        return tokens

    def _parse(
        self,
        tokens: list[tuple[str, str]],
        position: int,
        *,
        depth: int,
    ) -> tuple[Node, int]:
        if depth > self.max_depth:
            raise FormulaSyntaxError(f"expression exceeds max depth {self.max_depth}")
        if position >= len(tokens):
            raise FormulaSyntaxError("unexpected end of expression")
        kind, value = tokens[position]
        if kind in {"field", "number"}:
            return Node(kind, value), position + 1
        if kind != "ident":
            raise FormulaSyntaxError(f"expected field, number, or function; got {value!r}")
        if value not in ALLOWED_FUNCTIONS:
            raise FormulaSyntaxError(f"unsupported function {value!r}")
        if position + 1 >= len(tokens) or tokens[position + 1] != ("punct", "("):
            raise FormulaSyntaxError(f"function {value!r} must be called")
        args: list[Node] = []
        cursor = position + 2
        if cursor < len(tokens) and tokens[cursor] == ("punct", ")"):
            return Node("call", value, ()), cursor + 1
        while True:
            child, cursor = self._parse(tokens, cursor, depth=depth + 1)
            args.append(child)
            if cursor >= len(tokens):
                raise FormulaSyntaxError(f"unclosed call to {value!r}")
            token = tokens[cursor]
            if token == ("punct", ")"):
                return Node("call", value, tuple(args)), cursor + 1
            if token != ("punct", ","):
                raise FormulaSyntaxError(f"expected ',' or ')'; got {token[1]!r}")
            cursor += 1

    def _validate_node(self, node: Node) -> None:
        if node.kind == "field":
            if node.value not in self.allowed_fields:
                raise FormulaSyntaxError(
                    f"field {node.value!r} is not allowed; choose from {sorted(self.allowed_fields)}"
                )
            return
        if node.kind == "number":
            return
        expected = 1 if node.value in UNARY else 2 if node.value in (BINARY | ROLLING_ONE) else 3
        if len(node.args) != expected:
            raise FormulaSyntaxError(
                f"{node.value} expects {expected} arguments, received {len(node.args)}"
            )
        if node.value in ROLLING_ONE:
            self._validate_window(node.value, node.args[1])
        elif node.value in ROLLING_TWO:
            self._validate_window(node.value, node.args[2])
        elif node.value == "Quantile":
            self._validate_window(node.value, node.args[1])
            quantile = node.args[2].numeric_value
            if quantile is None or not 0.0 <= quantile <= 1.0:
                raise FormulaSyntaxError("Quantile qscore must be a number between 0 and 1")
        for child in node.args:
            self._validate_node(child)

    def _validate_window(self, function: str, node: Node) -> None:
        value = node.numeric_value
        if value is None or not value.is_integer():
            raise FormulaSyntaxError(f"{function} window must be a positive integer")
        if value <= 0:
            raise FormulaSyntaxError(
                f"{function} window must be positive; negative Ref/Delta would leak future data"
            )
        if value > self.max_window:
            raise FormulaSyntaxError(f"{function} window exceeds maximum {self.max_window}")


def normalize_fields(raw: str | None) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_FIELDS
    fields = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not fields:
        raise ValueError("allowed fields cannot be empty")
    return tuple(field if field.startswith("$") else f"${field}" for field in fields)
