"""
Safe, AST-sandboxed financial mathematical calculation engine for Program-Aided Language (PAL).

Eliminates LLM arithmetic hallucination (margins, YoY growth, CAGR, basis points)
by deterministically evaluating mathematical expressions in a secure AST environment.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class CalculationResult:
    expression: str
    result: float
    formatted: str
    success: bool
    error: str = ""


class SafeFinancialCalculator:
    """
    AST-restricted math evaluator for financial formulas.
    Prevents code injection by strictly whitelisting arithmetic AST nodes
    and dedicated financial helper functions.
    """

    ALLOWED_OPERATORS: dict[type[ast.AST], Callable[..., Any]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b if abs(b) <= 20 else ValueError("Exponent too large"),
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        # Financial helpers:
        "growth": lambda old, new: ((new - old) / abs(old) * 100) if old != 0 else 0.0,
        "pct_change": lambda old, new: ((new - old) / abs(old) * 100) if old != 0 else 0.0,
        "margin": lambda num, denom: (num / denom * 100) if denom != 0 else 0.0,
        "bps": lambda old_pct, new_pct: (new_pct - old_pct) * 100,  # 1% change = 100 bps
        "cagr": lambda start, end, periods: (
            ((end / start) ** (1 / periods) - 1) * 100 if start > 0 and periods > 0 else 0.0
        ),
    }

    _CALC_BLOCK_RE = re.compile(r"```calculation\s*([\s\S]*?)\s*```", re.IGNORECASE)
    _INLINE_CALC_RE = re.compile(r"\[CALC:\s*([^\]]+)\]", re.IGNORECASE)

    def evaluate(self, expr: str) -> CalculationResult:
        """
        Safely parse and evaluate a single mathematical expression.
        """
        expr_clean = expr.strip()
        if not expr_clean:
            return CalculationResult(
                expression=expr, result=0.0, formatted="0", success=False, error="Empty expression"
            )

        try:
            tree = ast.parse(expr_clean, mode="eval")
            val = self._eval_node(tree.body)
            if isinstance(val, int | float):
                if math.isnan(val) or math.isinf(val):
                    raise ValueError("Calculation resulted in NaN or Infinity")
                # Format smartly
                if isinstance(val, float) and val.is_integer():
                    formatted = str(int(val))
                elif isinstance(val, float):
                    formatted = f"{val:.2f}".rstrip("0").rstrip(".")
                else:
                    formatted = str(val)
                return CalculationResult(
                    expression=expr_clean,
                    result=float(val),
                    formatted=formatted,
                    success=True,
                )
            raise ValueError(f"Result type {type(val)} is not numeric")
        except ZeroDivisionError:
            return CalculationResult(
                expression=expr_clean,
                result=0.0,
                formatted="ERR",
                success=False,
                error="Division by zero",
            )
        except Exception as exc:
            return CalculationResult(
                expression=expr_clean, result=0.0, formatted="ERR", success=False, error=str(exc)
            )

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return node.value
            raise ValueError(f"Literal constant {node.value!r} not permitted")

        if isinstance(node, ast.BinOp):
            bin_op_type = type(node.op)
            if bin_op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator {bin_op_type.__name__} is not permitted")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.ALLOWED_OPERATORS[bin_op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Unary operator {unary_op_type.__name__} is not permitted")
            operand = self._eval_node(node.operand)
            return self.ALLOWED_OPERATORS[unary_op_type](operand)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only standard financial helper function calls are permitted")
            func_name = node.func.id.lower()
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Function {func_name}() is not permitted")
            args = [self._eval_node(arg) for arg in node.args]
            return self.ALLOWED_FUNCTIONS[func_name](*args)

        raise ValueError(f"Syntax node {type(node).__name__} is not permitted in math engine")

    def process_text_calculations(self, text: str) -> tuple[str, list[CalculationResult]]:
        """
        Find and evaluate all calculation tags in the text, replacing them with
        their verified numeric results, and returning the audit list.
        Supports both ```calculation ... ``` and [CALC: ...].
        """
        audits: list[CalculationResult] = []

        def _replace_block(match: re.Match) -> str:
            expr = match.group(1).strip()
            # If multiple lines, evaluate each line or the last line
            lines = [ln.strip() for ln in expr.splitlines() if ln.strip()]
            if not lines:
                return ""
            res = self.evaluate(lines[-1])
            audits.append(res)
            return res.formatted if res.success else match.group(0)

        def _replace_inline(match: re.Match) -> str:
            expr = match.group(1).strip()
            res = self.evaluate(expr)
            audits.append(res)
            return res.formatted if res.success else match.group(0)

        processed = self._CALC_BLOCK_RE.sub(_replace_block, text)
        processed = self._INLINE_CALC_RE.sub(_replace_inline, processed)
        return processed, audits
