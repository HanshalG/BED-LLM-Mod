"""Bounded rational genetic-programming proposal control, with no source access.

GP chooses structures on real history. Its fitted ephemeral constants become
uncertain parameters under a shared declared prior before downstream inference.
This is a practical control, not globally optimal symbolic regression or a
selection-corrected posterior. The private program-tree interface is versioned.
"""

from dataclasses import dataclass
import math
from time import monotonic

import numpy as np

from .executable_belief import ExecutableBeliefPool, _points, _positive_integer
from .ir import INPUT_NAMES, RateLawError


def export_program(program, bounds, *, parameter_bounds=(0.01, 10.0)):
    """Export an exact tree structure; certify division on the whole input box.

    Interval propagation includes the full parameter box, so protected division
    is inactive for every downstream particle, not just for training samples.
    """
    nodes = program.program
    if not 1 <= len(nodes) <= 31:
        raise RateLawError("GP tree exceeds export node cap")
    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (7, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("invalid input box")
    low, high = parameter_bounds
    if not 0 < low < high or not math.isfinite(high):
        raise ValueError("invalid shared parameter bounds")
    parameters = []
    offset = 0

    def visit():
        nonlocal offset
        if offset == len(nodes):
            raise RateLawError("incomplete GP tree")
        node = nodes[offset]
        offset += 1
        if isinstance(node, (int, np.integer)) and not isinstance(node, bool):
            if not 0 <= node < len(INPUT_NAMES):
                raise RateLawError("unknown GP input")
            return INPUT_NAMES[node], tuple(bounds[node])
        if isinstance(node, (float, np.floating)):
            if not math.isfinite(node) or not low <= node <= high:
                raise RateLawError("GP constant outside shared parameter prior")
            name = f"k{len(parameters)}"
            parameters.append(
                {"name": name, "low": low, "high": high, "transform": "log"}
            )
            return name, (low, high)
        if getattr(node, "arity", None) != 2 or getattr(node, "name", None) not in {
            "add",
            "sub",
            "mul",
            "div",
        }:
            raise RateLawError("unsupported GP operator")
        left, a = visit()
        right, b = visit()
        if node.name == "add":
            interval, symbol = (a[0] + b[0], a[1] + b[1]), "+"
        elif node.name == "sub":
            interval, symbol = (a[0] - b[1], a[1] - b[0]), "-"
        elif node.name == "mul":
            values = [x * y for x in a for y in b]
            interval, symbol = (min(values), max(values)), "*"
        else:
            if not (b[0] > 0.001 or b[1] < -0.001):
                raise RateLawError("protected division may activate in the public box")
            values = [x / y for x in a for y in b]
            interval, symbol = (min(values), max(values)), "/"
        if not np.isfinite(interval).all():
            raise RateLawError("nonfinite interval bound")
        interval = (
            np.nextafter(interval[0], -np.inf),
            np.nextafter(interval[1], np.inf),
        )
        return f"({left} {symbol} {right})", interval

    expression, interval = visit()
    if offset != len(nodes):
        raise RateLawError("extra GP tree tokens")
    if interval[0] < 0:
        raise RateLawError("rate positivity not certified over input and parameter box")
    if not parameters:
        if not low <= 1 <= high:
            raise RateLawError(
                "amplitude prior must include the unit-amplitude program"
            )
        expression = f"k0 * ({expression})"
        parameters = [{"name": "k0", "low": low, "high": high, "transform": "log"}]
    payload = {"name": "rational_gp", "expr": expression, "params": parameters}
    ExecutableBeliefPool().add(payload)
    return payload


def _log_rate_mse(y, prediction, sample_weight):
    if not np.isfinite(prediction).all() or np.any(prediction < 0):
        return 1e100
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(
            np.average((np.log1p(prediction) - y) ** 2, weights=sample_weight)
        )
    return value if math.isfinite(value) else 1e100


@dataclass(frozen=True)
class SymbolicProposals:
    payloads: tuple[dict, ...]
    attempted_programs: int
    rejected_exports: tuple[str, ...]
    exportable_unique: int
    search_generations: int
    approximate_training_scalar_nodes: int
    elapsed_seconds: float
    final_population_audit: tuple[dict, ...]


def propose_from_history(
    history_inputs,
    observations,
    *,
    input_bounds,
    seed=0,
    population_size=64,
    generations=3,
    proposal_count=4,
    parameter_bounds=(0.01, 10.0),
):
    """Return proposals only; no held-out inputs/outcomes are accepted.

    Fixed caps bound the search population and generations. There is no claimed
    hard wall-clock interrupt; elapsed time is reported. A subprocess deadline
    is required by any future experiment runner with a wall-clock budget.
    """
    from importlib.metadata import version
    from gplearn.fitness import make_fitness
    from gplearn.genetic import SymbolicRegressor

    if version("gplearn") != "0.4.3":
        raise RuntimeError("GP tree adapter requires gplearn 0.4.3")
    x = _points(history_inputs, "history")
    y = np.asarray(observations, dtype=float)
    bounds = np.asarray(input_bounds, dtype=float)
    if (
        bounds.shape != (7, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("invalid input box")
    if len(x) < 2 or len(x) > 128 or y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("search requires 2 to 128 real observations")
    if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError("history outside declared input box")
    for value, name, cap in [
        (population_size, "population", 512),
        (generations, "generations", 10),
        (proposal_count, "proposal_count", 16),
    ]:
        if _positive_integer(value, name) > cap:
            raise ValueError(f"{name} exceeds fixed search cap")
    if population_size < 8:
        raise ValueError("population must contain at least eight programs")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("invalid search seed")
    low, high = parameter_bounds
    if not 0 < low < high or not math.isfinite(high):
        raise ValueError("invalid shared parameter prior")
    started = monotonic()
    search = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        tournament_size=min(10, population_size),
        init_depth=(1, 3),
        function_set=("add", "sub", "mul", "div"),
        const_range=(low, high),
        metric=make_fitness(function=_log_rate_mse, greater_is_better=False),
        parsimony_coefficient=0.001,
        max_samples=1.0,
        n_jobs=1,
        random_state=seed,
        low_memory=True,
    )
    search.fit(x, y)
    programs = search._programs[-1]
    if len(programs) != population_size or any(p is None for p in programs):
        raise RuntimeError("GP final-population interface changed")
    unique, rejected, audit = {}, [], []
    canonical = ExecutableBeliefPool(max_laws=population_size)
    for program in sorted(programs, key=lambda p: (p.fitness_, p.length_)):
        record = {
            "tokens": [
                {"input": int(node)}
                if isinstance(node, (int, np.integer))
                else {"constant": float(node)}
                if isinstance(node, (float, np.floating))
                else {"operator": node.name, "arity": node.arity}
                for node in program.program
            ],
            "raw_training_log_rate_mse": float(program.raw_fitness_),
            "penalized_training_fitness": float(program.fitness_),
        }
        try:
            payload = export_program(program, bounds, parameter_bounds=parameter_bounds)
            key = canonical.add(payload)
        except RateLawError as exc:
            rejected.append(str(exc))
            record.update(status="export_rejected", reason=str(exc))
            audit.append(record)
            continue
        record.update(
            status="duplicate" if key in unique else "exportable", law_key=key
        )
        audit.append(record)
        unique.setdefault(key, payload)
    details = search.run_details_
    scalar_nodes = int(round(sum(details["average_length"]) * population_size * len(x)))
    return SymbolicProposals(
        tuple(list(unique.values())[:proposal_count]),
        population_size * len(details["generation"]),
        tuple(rejected),
        len(unique),
        len(details["generation"]),
        scalar_nodes,
        monotonic() - started,
        tuple(audit),
    )
