"""Safe evaluator for numeric relationship rules."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Mapping


class FormulaValidationError(ValueError):
    """Raised when a rule formula contains unsupported syntax."""


_FUNCTIONS = {
    "abs": abs,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log,
    "pow": pow,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
_CONSTANTS = {"e": math.e, "pi": math.pi}
_BIN_OPS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
    ast.Mod: lambda left, right: left % right,
}
_UNARY_OPS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


@dataclass(frozen=True)
class CompiledComparison:
    left: ast.AST
    operator: ast.cmpop
    right: ast.AST
    variable_names: frozenset[str]

    def evaluate(self, variables: Mapping[str, float], tolerance: float = 0.0) -> bool:
        left_value = float(_evaluate_node(self.left, variables))
        right_value = float(_evaluate_node(self.right, variables))

        if isinstance(self.operator, ast.Eq):
            return abs(left_value - right_value) <= tolerance
        if isinstance(self.operator, ast.NotEq):
            return abs(left_value - right_value) > tolerance
        if isinstance(self.operator, ast.GtE):
            return left_value >= right_value - tolerance
        if isinstance(self.operator, ast.LtE):
            return left_value <= right_value + tolerance
        if isinstance(self.operator, ast.Gt):
            return left_value > right_value - tolerance
        if isinstance(self.operator, ast.Lt):
            return left_value < right_value + tolerance
        raise FormulaValidationError("Unsupported comparison operator")


def compile_comparison(formula: str) -> CompiledComparison:
    """Parse and validate a single numeric comparison."""
    try:
        expression = ast.parse(formula, mode="eval").body
    except SyntaxError as exc:
        raise FormulaValidationError(f"Invalid formula syntax: {formula}") from exc

    if not isinstance(expression, ast.Compare) or len(expression.ops) != 1:
        raise FormulaValidationError("Formula must contain exactly one comparison")

    names = {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name)
        and node.id not in _FUNCTIONS
        and node.id not in _CONSTANTS
    }
    compiled = CompiledComparison(
        left=expression.left,
        operator=expression.ops[0],
        right=expression.comparators[0],
        variable_names=frozenset(names),
    )

    # Validate every node before processing any dataset row.
    _validate_node(compiled.left)
    _validate_node(compiled.right)
    return compiled


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise FormulaValidationError("Only numeric constants are allowed")
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        _validate_node(node.left)
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTIONS
            or node.keywords
        ):
            raise FormulaValidationError("Unsupported function call")
        for argument in node.args:
            _validate_node(argument)
        return
    raise FormulaValidationError(f"Unsupported formula syntax: {type(node).__name__}")


def _evaluate_node(node: ast.AST, variables: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if node.id not in variables:
            raise FormulaValidationError(f"Missing formula variable: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](
            _evaluate_node(node.left, variables),
            _evaluate_node(node.right, variables),
        )
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_evaluate_node(node.operand, variables))
    if isinstance(node, ast.Call):
        function = _FUNCTIONS[node.func.id]
        return function(*(_evaluate_node(arg, variables) for arg in node.args))
    raise FormulaValidationError(f"Unsupported formula node: {type(node).__name__}")
