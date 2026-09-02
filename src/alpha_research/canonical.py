"""Canonical form for Qlib factor expressions.

The LLM emits the same factor in several surface syntaxes: prefix calls
(``Div(Sub($high, $low), $close)``) and infix operators
(``($high - $low) / $close``). String-level de-duplication therefore keeps
many copies of one factor, which collapses pool diversity. This module
parses both syntaxes into one tree and renders a single canonical string.

Canonicalisation rules:
  * infix operators are rewritten to their Qlib function names
  * numbers are normalised (``5`` and ``5.0`` agree, ``1e-12`` is stable)
  * commutative operators (Add, Mul and the symmetric comparisons) sort
    their arguments, so ``a + b`` and ``b + a`` agree
  * whitespace is dropped

``canonical_form`` never raises: an expression it cannot parse is returned
stripped of whitespace, so an unparsable candidate is still de-duplicated
against itself and never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Qlib binary operators, in ascending precedence order.
_INFIX_TO_FUNC = {
    "|": "Or",
    "&": "And",
    "<": "Lt",
    "<=": "Le",
    ">": "Gt",
    ">=": "Ge",
    "==": "Eq",
    "!=": "Ne",
    "+": "Add",
    "-": "Sub",
    "*": "Mul",
    "/": "Div",
    "^": "Power",
}

_PRECEDENCE = {
    "|": 1, "&": 2,
    "<": 3, "<=": 3, ">": 3, ">=": 3, "==": 3, "!=": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5,
    "^": 6,
}

# Operators whose operands may be swapped without changing the value.
_COMMUTATIVE = {"Add", "Mul", "And", "Or", "Eq", "Ne", "Greater", "Less"}

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<field>\$[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|==|!=|[-+*/^<>|&])"
    r"|(?P<punct>[(),])"
    r")"
)


class CanonicalError(ValueError):
    """Raised internally when an expression cannot be parsed."""


@dataclass(frozen=True)
class Node:
    kind: str  # "field" | "number" | "call"
    value: str
    args: tuple["Node", ...] = ()


def _tokenize(raw: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    length = len(raw)
    while position < length:
        if raw[position].isspace():
            position += 1
            continue
        match = _TOKEN_RE.match(raw, position)
        if match is None:
            raise CanonicalError(f"unsupported syntax near {raw[position:position + 20]!r}")
        kind = match.lastgroup
        tokens.append((kind, match.group(kind)))
        position = match.end()
    if not tokens:
        raise CanonicalError("expression is empty")
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def parse(self) -> Node:
        node = self.parse_expr(0)
        if self.pos != len(self.tokens):
            raise CanonicalError(f"trailing tokens at {self.tokens[self.pos]!r}")
        return node

    def parse_expr(self, min_prec: int) -> Node:
        left = self.parse_unary()
        while True:
            token = self.peek()
            if token is None or token[0] != "op":
                break
            op = token[1]
            prec = _PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break
            self.pos += 1
            # every Qlib infix operator is left-associative
            right = self.parse_expr(prec + 1)
            left = Node("call", _INFIX_TO_FUNC[op], (left, right))
        return left

    def parse_unary(self) -> Node:
        token = self.peek()
        if token is not None and token[0] == "op" and token[1] in {"-", "+"}:
            self.pos += 1
            operand = self.parse_unary()
            if token[1] == "-":
                # -x is rendered as Sub(0, x) so it has one canonical shape
                return Node("call", "Sub", (Node("number", "0"), operand))
            return operand
        return self.parse_atom()

    def parse_atom(self) -> Node:
        token = self.peek()
        if token is None:
            raise CanonicalError("unexpected end of expression")
        kind, value = token
        if kind == "field":
            self.pos += 1
            return Node("field", value)
        if kind == "number":
            self.pos += 1
            return Node("number", value)
        if kind == "punct" and value == "(":
            self.pos += 1
            node = self.parse_expr(0)
            self._expect(")")
            return node
        if kind == "ident":
            self.pos += 1
            nxt = self.peek()
            if nxt is None or nxt != ("punct", "("):
                # bare identifier: treat as an opaque leaf rather than failing
                return Node("field", value)
            self.pos += 1
            args: list[Node] = []
            if self.peek() == ("punct", ")"):
                self.pos += 1
                return Node("call", value, ())
            while True:
                args.append(self.parse_expr(0))
                token = self.peek()
                if token == ("punct", ")"):
                    self.pos += 1
                    return Node("call", value, tuple(args))
                if token != ("punct", ","):
                    raise CanonicalError(f"expected ',' or ')' in call to {value}")
                self.pos += 1
        raise CanonicalError(f"unexpected token {value!r}")

    def _expect(self, punct: str) -> None:
        if self.peek() != ("punct", punct):
            raise CanonicalError(f"expected {punct!r}")
        self.pos += 1


def _normalize_number(raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    # repr round-trips exactly and keeps 1e-12 stable across surface forms
    return repr(value)


def _render(node: Node) -> str:
    if node.kind == "field":
        return node.value
    if node.kind == "number":
        return _normalize_number(node.value)
    rendered = [_render(child) for child in node.args]
    if node.value in _COMMUTATIVE and len(rendered) == 2:
        rendered = sorted(rendered)
    return f"{node.value}({','.join(rendered)})"


def canonical_form(expression: str) -> str:
    """Return a canonical string for ``expression``.

    Two expressions that denote the same factor map to the same string.
    Unparsable input is returned with whitespace stripped rather than
    raising, so callers can use this for de-duplication unconditionally.
    """

    raw = str(expression or "").strip()
    if not raw:
        return ""
    try:
        return _render(_Parser(_tokenize(raw)).parse())
    except (CanonicalError, RecursionError):
        return re.sub(r"\s+", "", raw)
