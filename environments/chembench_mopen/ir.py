"""Safe executable intermediate representation for proposed ChemBench laws."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


INPUT_NAMES = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
FUNCTIONS: dict[str, Callable[[float], float]] = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
}
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY = (ast.UAdd, ast.USub)


class RateLawError(ValueError):
    """Raised when a proposed rate law is unsafe or malformed."""


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    transform: str = "log"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ParameterSpec":
        required = {"name", "low", "high"}
        if not required.issubset(payload):
            raise RateLawError(f"parameter is missing keys: {sorted(required - set(payload))}")
        extra = set(payload) - required - {"transform"}
        if extra:
            raise RateLawError(f"parameter has unexpected keys: {sorted(extra)}")
        name = payload["name"]
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise RateLawError("parameter name is invalid")
        try:
            lower = float(payload["low"])
            upper = float(payload["high"])
        except (TypeError, ValueError) as exc:
            raise RateLawError("parameter bounds must be numeric") from exc
        transform = payload.get("transform", "log")
        if transform not in {"identity", "log"}:
            raise RateLawError("parameter transform must be identity or log")
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise RateLawError("parameter bounds must be finite and increasing")
        if transform == "log" and lower <= 0:
            raise RateLawError("log-transformed parameters require a positive lower bound")
        if name in INPUT_NAMES or name in FUNCTIONS:
            raise RateLawError(f"parameter name is reserved: {name}")
        return cls(name=name, lower=lower, upper=upper, transform=transform)


def _validate_expression(node: ast.AST, allowed_names: set[str]) -> None:
    if isinstance(node, ast.Expression):
        _validate_expression(node.body, allowed_names)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINARY):
        _validate_expression(node.left, allowed_names)
        _validate_expression(node.right, allowed_names)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY):
        _validate_expression(node.operand, allowed_names)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise RateLawError("only exp, log, and sqrt calls are allowed")
        if len(node.args) != 1 or node.keywords:
            raise RateLawError("allowlisted functions require exactly one positional argument")
        _validate_expression(node.args[0], allowed_names)
        return
    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise RateLawError(f"undeclared name in expression: {node.id}")
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise RateLawError("only numeric constants are allowed")
        if not math.isfinite(float(node.value)):
            raise RateLawError("numeric constants must be finite")
        return
    raise RateLawError(f"unsupported expression node: {type(node).__name__}")


def _canonical(node: ast.AST, parameter_aliases: Mapping[str, str]) -> tuple[Any, ...]:
    if isinstance(node, ast.Expression):
        return _canonical(node.body, parameter_aliases)
    if isinstance(node, ast.Name):
        return ("name", parameter_aliases.get(node.id, node.id))
    if isinstance(node, ast.Constant):
        return ("constant", float(node.value))
    if isinstance(node, ast.UnaryOp):
        return (type(node.op).__name__, _canonical(node.operand, parameter_aliases))
    if isinstance(node, ast.Call):
        return ("call", node.func.id, _canonical(node.args[0], parameter_aliases))
    if isinstance(node, ast.BinOp):
        left = _canonical(node.left, parameter_aliases)
        right = _canonical(node.right, parameter_aliases)
        operation = type(node.op).__name__
        if isinstance(node.op, (ast.Add, ast.Mult)) and right < left:
            left, right = right, left
        return (operation, left, right)
    raise AssertionError(f"validated node was not canonicalized: {type(node).__name__}")


@dataclass(frozen=True)
class RateLaw:
    name: str
    expression: str
    parameters: tuple[ParameterSpec, ...]
    rationale: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RateLaw":
        required = {"name", "expr", "params"}
        if not required.issubset(payload):
            raise RateLawError(f"rate law is missing keys: {sorted(required - set(payload))}")
        extra = set(payload) - required - {"rationale"}
        if extra:
            raise RateLawError(f"rate law has unexpected keys: {sorted(extra)}")
        name = payload["name"]
        expression = payload["expr"]
        parameters_payload = payload["params"]
        rationale = payload.get("rationale", "")
        if not isinstance(name, str) or not name.strip():
            raise RateLawError("rate-law name must be a non-empty string")
        if not isinstance(expression, str) or not expression.strip():
            raise RateLawError("rate-law expression must be a non-empty string")
        if not isinstance(rationale, str):
            raise RateLawError("rate-law rationale must be a string")
        if not isinstance(parameters_payload, list) or not parameters_payload:
            raise RateLawError("rate-law params must be a non-empty list")
        parameters = tuple(ParameterSpec.from_payload(item) for item in parameters_payload)
        parameter_names = [item.name for item in parameters]
        if len(set(parameter_names)) != len(parameter_names):
            raise RateLawError("parameter names must be unique")
        law = cls(name=name.strip(), expression=expression.strip(), parameters=parameters, rationale=rationale)
        law.parsed_expression()
        return law

    def parsed_expression(self) -> ast.Expression:
        try:
            parsed = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise RateLawError("rate-law expression is not valid Python syntax") from exc
        allowed_names = set(INPUT_NAMES) | {item.name for item in self.parameters}
        _validate_expression(parsed, allowed_names)
        return parsed

    @property
    def canonical_key(self) -> tuple[Any, ...]:
        aliases = {item.name: f"p{index}" for index, item in enumerate(self.parameters)}
        bounds = tuple(
            (aliases[item.name], item.lower, item.upper, item.transform)
            for item in self.parameters
        )
        return (_canonical(self.parsed_expression(), aliases), bounds)

    def compile(self) -> Callable[[Mapping[str, float], Mapping[str, float]], float]:
        code = compile(self.parsed_expression(), "<chembench-rate-law>", "eval")
        parameter_names = {item.name for item in self.parameters}

        def evaluate(inputs: Mapping[str, float], parameters: Mapping[str, float]) -> float:
            if set(inputs) != set(INPUT_NAMES):
                raise RateLawError("inputs must contain exactly the seven ChemBench variables")
            if set(parameters) != parameter_names:
                raise RateLawError("parameter values do not match the declared parameters")
            namespace: dict[str, Any] = dict(FUNCTIONS)
            for key, value in inputs.items():
                namespace[key] = float(value)
            for spec in self.parameters:
                value = float(parameters[spec.name])
                if not math.isfinite(value) or value < spec.lower or value > spec.upper:
                    raise RateLawError(f"parameter is outside declared bounds: {spec.name}")
                namespace[spec.name] = value
            try:
                result = float(eval(code, {"__builtins__": {}}, namespace))
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
                raise RateLawError("rate-law evaluation failed") from exc
            if not math.isfinite(result):
                raise RateLawError("rate-law evaluation returned a non-finite value")
            return result

        return evaluate

    def stress_test(self, points: Sequence[Mapping[str, float]]) -> None:
        midpoint = {
            item.name: (math.sqrt(item.lower * item.upper) if item.transform == "log" else (item.lower + item.upper) / 2.0)
            for item in self.parameters
        }
        evaluate = self.compile()
        for point in points:
            value = evaluate(point, midpoint)
            if value < 0:
                raise RateLawError("rate law returned a negative rate on the stress grid")
