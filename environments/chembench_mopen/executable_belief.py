"""Bounded executable-law snapshots for real-history model proposals.

This is finite-particle inference conditional on a supplied model pool, not an
exact Bayesian correction for selecting structures using the same observations.
No source registry, hidden outcomes, network, or proposer is accessed here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import operator

import numpy as np

from .ir import INPUT_NAMES, RateLaw, RateLawError
from .raw_belief import GaussianParticleModel


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: np.power,
}
_FUNCTIONS = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt}


def _points(values, name, *, allow_empty=False):
    result = np.array(values, dtype=float, copy=True)
    if allow_empty and result.shape == (0,):
        result = np.empty((0, len(INPUT_NAMES)))
    if (
        result.ndim != 2
        or result.shape[1] != len(INPUT_NAMES)
        or (not allow_empty and len(result) == 0)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be finite rows of seven input variables")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _evaluate(node, namespace):
    # Floating NumPy arithmetic prevents arbitrary-precision integer powers.
    if isinstance(node, ast.Expression):
        value = _evaluate(node.body, namespace)
    elif isinstance(node, ast.Name):
        value = namespace[node.id]
    elif isinstance(node, ast.Constant):
        value = np.float64(node.value)
    elif isinstance(node, ast.BinOp):
        value = _BINARY[type(node.op)](
            _evaluate(node.left, namespace), _evaluate(node.right, namespace)
        )
    elif isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, namespace)
        value = -value if isinstance(node.op, ast.USub) else value
    elif isinstance(node, ast.Call):
        value = _FUNCTIONS[node.func.id](_evaluate(node.args[0], namespace))
    else:
        raise RateLawError("unsupported validated node")
    if not np.isfinite(value).all():
        raise RateLawError("nonfinite intermediate rate-law value")
    return value


@dataclass(frozen=True)
class ExecutableSnapshot:
    model: GaussianParticleModel
    state: tuple[float, ...]
    law_keys: tuple[str, ...]
    particle_law: tuple[str, ...]
    parameter_values: tuple[tuple[float, ...], ...]
    history_sha256: str
    conditional_log_evidence: float
    evaluated_scalar_nodes: int
    interpretation: str = "finite_pool_conditional_fit_not_selection_corrected"


class ExecutableBeliefPool:
    """Accumulate valid proposals; rebuild every snapshot from the full history.

    Each canonical law has equal prior mass and deterministic prior parameter
    draws, regardless of name, rationale, submission order or submission count.
    Newly proposed laws receive their prior mass before replay, never zero mass.
    """

    def __init__(
        self,
        *,
        particles_per_law=32,
        seed=0,
        max_laws=16,
        max_scalar_nodes=2_000_000,
        max_workspace_bytes=64 * 1024 * 1024,
    ):
        self.particles_per_law = _positive_integer(
            particles_per_law, "particles_per_law"
        )
        self.max_laws = _positive_integer(max_laws, "max_laws")
        self.max_scalar_nodes = _positive_integer(max_scalar_nodes, "max_scalar_nodes")
        self.max_workspace_bytes = _positive_integer(
            max_workspace_bytes, "max_workspace_bytes"
        )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = seed
        self._laws = {}

    def add(self, payload):
        """Validate one proposal atomically; invalid proposals never enter support."""
        if not isinstance(payload, dict):
            raise RateLawError("proposal must be an object")
        expression = payload.get("expr")
        params = payload.get("params")
        if not isinstance(expression, str) or len(expression) > 2048:
            raise RateLawError("expression must be a string of at most 2048 characters")
        if not isinstance(params, list) or not 1 <= len(params) <= 8:
            raise RateLawError("proposal must have one to eight parameters")
        for spec in params:
            if not isinstance(spec, dict):
                raise RateLawError("parameter must be an object")
            if any(isinstance(spec.get(k), bool) for k in ("low", "high")):
                raise RateLawError("boolean parameter bounds are invalid")
        try:
            law = RateLaw.from_payload(payload)
            tree = law.parsed_expression()
        except (RecursionError, TypeError, OverflowError) as exc:
            raise RateLawError("malformed or excessive expression") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > 96:
            raise RateLawError("expression exceeds 96 AST nodes")
        used = {n.id for n in nodes if isinstance(n, ast.Name)}
        if any(spec.name not in used for spec in law.parameters):
            raise RateLawError("every declared parameter must be used")
        key = hashlib.sha256(repr(law.canonical_key).encode()).hexdigest()
        if key not in self._laws and len(self._laws) >= self.max_laws:
            raise RateLawError("law pool capacity exceeded")
        self._laws.setdefault(key, (law, tree, len(nodes)))
        return key

    def snapshot(self, *, history_inputs, observations, designs, targets, sigma):
        if not self._laws:
            raise ValueError("cannot build an empty model pool")
        history = _points(history_inputs, "history", allow_empty=True)
        designs = _points(designs, "designs")
        targets = _points(targets, "targets")
        observations = np.asarray(observations, dtype=float)
        if observations.shape != (len(history),) or not np.isfinite(observations).all():
            raise ValueError("one finite raw observation is required per history row")
        if (
            isinstance(sigma, bool)
            or not np.isscalar(sigma)
            or not np.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("sigma must be finite and positive")
        points = np.concatenate((history, designs, targets))
        n = self.particles_per_law
        cells = n * len(points)
        work = cells * sum(item[2] for item in self._laws.values())
        if work > self.max_scalar_nodes:
            raise RuntimeError("scalar-node evaluation cap exceeded")
        # Conservative allowance for all expression intermediates, output and copies.
        workspace = (
            8
            * cells
            * (sum(item[2] for item in self._laws.values()) + 16 * len(self._laws))
        )
        if workspace > self.max_workspace_bytes:
            raise RuntimeError("expression workspace cap exceeded")
        predictions, parameter_rows, particle_laws = [], [], []
        for key, (law, tree, _) in sorted(self._laws.items()):
            draw_seed = int.from_bytes(
                hashlib.sha256(f"{self.seed}:{key}".encode()).digest()[:8], "big"
            )
            rng = np.random.default_rng(draw_seed)
            namespace = {
                name: points[None, :, col] for col, name in enumerate(INPUT_NAMES)
            }
            parameters = np.empty((n, len(law.parameters)))
            for col, spec in enumerate(law.parameters):
                lower, upper = spec.lower, spec.upper
                if spec.transform == "log":
                    lower, upper = np.log(lower), np.log(upper)
                draws = rng.uniform(lower, upper, size=n)
                if spec.transform == "log":
                    draws = np.exp(draws)
                parameters[:, col] = draws
                namespace[spec.name] = draws[:, None]
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    rates = np.broadcast_to(
                        _evaluate(tree, namespace), (n, len(points))
                    )
                    if np.any(rates < 0):
                        raise RateLawError("negative rate on inference inputs")
                    predictions.append(np.log1p(rates))
            except (FloatingPointError, OverflowError, ValueError) as exc:
                raise RateLawError(f"law {key} failed inference evaluation") from exc
            parameter_rows.extend(tuple(float(x) for x in row) for row in parameters)
            particle_laws.extend([key] * n)
        rates = np.concatenate(predictions)
        count = len(rates)
        prior_logs = np.full(count, -np.log(count))
        # Full-history replay is deliberate: conditioning an already fitted state
        # again would double count old observations when proposals are refreshed.
        h = len(history)
        with np.errstate(over="raise", invalid="raise"):
            log_likelihood = (
                -0.5 * ((rates[:, :h] - observations) / sigma) ** 2
                - np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
            ).sum(axis=1)
        unnormalized = prior_logs + log_likelihood
        maximum = float(np.max(unnormalized))
        evidence = maximum + float(np.log(np.exp(unnormalized - maximum).sum()))
        state = unnormalized - maximum
        state -= np.log(np.exp(state).sum())
        model = GaussianParticleModel(
            rates[:, h : h + len(designs)],
            sigma,
            rates[:, h + len(designs) :],
            np.full(count, 1 / count),
        )
        model._logs(tuple(state))
        history_key = json.dumps(
            {"inputs": history.tolist(), "observations": observations.tolist()},
            sort_keys=True,
            allow_nan=False,
        )
        return ExecutableSnapshot(
            model,
            tuple(state),
            tuple(sorted(self._laws)),
            tuple(particle_laws),
            tuple(parameter_rows),
            hashlib.sha256(history_key.encode()).hexdigest(),
            float(evidence),
            work,
        )
