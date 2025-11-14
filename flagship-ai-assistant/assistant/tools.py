"""Utility tools that the assistant can call.

Tools are simple Python functions that implement non-LLM capabilities such
as calculations. In a production system, this file could grow to include
HTTP calls, database queries, and more.
"""

from __future__ import annotations

import ast
import math
from typing import Any


class SafeEvalVisitor(ast.NodeVisitor):
    """Restricted AST visitor that only allows basic math expressions."""

    ALLOWED_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
    }

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self.ALLOWED_NODES:
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        super().generic_visit(node)


def safe_calculate(expression: str) -> float:
    """Safely evaluate a basic math expression.

    Args:
        expression: String containing a simple math expression.

    Returns:
        The evaluated float result.

    Raises:
        ValueError: If the expression contains unsafe operations.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    SafeEvalVisitor().visit(parsed)

    return float(eval(compile(parsed, filename="<expr>", mode="eval")))  # noqa: S307
