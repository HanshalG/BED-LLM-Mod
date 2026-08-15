"""Adaptive-tempered parameter SMC for executable ChemBench structures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from environments.chembench_mopen.source import LINEAR_PARAMETER_NAMES


ArrayLogLikelihood = Callable[[np.ndarray], np.ndarray]

_SOBOL_POLYNOMIALS = (1, 3, 7, 11, 13, 19, 25, 37)
_SOBOL_INITIAL_DIRECTIONS = (
    (1,),
    (1,),
    (1, 3),
    (1, 3, 1),
    (1, 1, 1),
    (1, 1, 3, 3),
    (1, 3, 5, 13),
    (1, 1, 5, 5, 17),
)


def _scrambled_sobol_points(dimension: int, power: int, seed: int) -> np.ndarray:
    """Generate a digitally shifted Sobol prefix for up to eight dimensions."""

    if not 1 <= dimension <= len(_SOBOL_POLYNOMIALS) or power <= 0:
        raise ValueError("Sobol dimension or power is unsupported")
    bits = 52
    directions = np.zeros((dimension, bits), dtype=np.uint64)
    for bit in range(1, bits + 1):
        directions[0, bit - 1] = np.uint64(1 << (bits - bit))
    for dim in range(1, dimension):
        polynomial = _SOBOL_POLYNOMIALS[dim]
        degree = polynomial.bit_length() - 1
        coefficient_bits = (polynomial >> 1) & ((1 << (degree - 1)) - 1)
        initial = _SOBOL_INITIAL_DIRECTIONS[dim]
        for bit in range(1, degree + 1):
            directions[dim, bit - 1] = np.uint64(initial[bit - 1] << (bits - bit))
        for bit in range(degree + 1, bits + 1):
            value = directions[dim, bit - degree - 1]
            value ^= value >> np.uint64(degree)
            for offset in range(1, degree):
                if (coefficient_bits >> (degree - 1 - offset)) & 1:
                    value ^= directions[dim, bit - offset - 1]
            directions[dim, bit - 1] = value

    count = 1 << power
    integer_points = np.empty((count, dimension), dtype=np.uint64)
    state = np.zeros(dimension, dtype=np.uint64)
    for index in range(count):
        integer_points[index] = state
        trailing_ones = 0
        value = index
        while value & 1:
            trailing_ones += 1
            value >>= 1
        state ^= directions[:, trailing_ones]
    rng = np.random.default_rng(seed)
    digital_shift = rng.integers(0, 1 << bits, size=dimension, dtype=np.uint64)
    integer_points ^= digital_shift[None, :]
    return integer_points.astype(float) / float(1 << bits)


def _logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(array - maximum).sum()))


def _normalized_weights(log_values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(log_values, dtype=float)
    normalizer = _logsumexp(values)
    if not math.isfinite(normalizer):
        raise ValueError("log weights have no finite normalizer")
    weights = np.exp(values - normalizer)
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("normalized weights are invalid")
    return weights, normalizer


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TransformedParameterPrior:
    """Uniform parameter prior in identity/log transformed coordinates."""

    names: tuple[str, ...]
    transforms: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    ordered_pairs: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_parameter_states(
        cls,
        parameter_states: Sequence[Mapping[str, Any]],
        *,
        expansion_factor: float = 1.5,
        minimum_log_span: float = math.log(2.0),
        minimum_linear_span: float = 0.5,
        clamp_identity_positive: bool = True,
    ) -> "TransformedParameterPrior":
        if not parameter_states:
            raise ValueError("parameter states must be nonempty")
        if expansion_factor < 0 or minimum_log_span <= 0 or minimum_linear_span <= 0:
            raise ValueError("parameter prior span settings are invalid")
        names = tuple(sorted(parameter_states[0]))
        if not names or any(tuple(sorted(state)) != names for state in parameter_states):
            raise ValueError("parameter states must have identical nonempty keys")

        transforms: list[str] = []
        lower: list[float] = []
        upper: list[float] = []
        for name in names:
            values = np.asarray([float(state[name]) for state in parameter_states], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("parameter states must be finite")
            transform = "identity" if name in LINEAR_PARAMETER_NAMES else "log"
            if transform == "log" and np.any(values <= 0):
                raise ValueError(f"log-scale parameter {name} must be positive")
            transformed = values if transform == "identity" else np.log(values)
            observed_low = float(np.min(transformed))
            observed_high = float(np.max(transformed))
            minimum_span = minimum_linear_span if transform == "identity" else minimum_log_span
            padding = expansion_factor * max(observed_high - observed_low, minimum_span)
            transforms.append(transform)
            proposed_lower = observed_low - padding
            if transform == "identity" and clamp_identity_positive:
                proposed_lower = max(proposed_lower, np.finfo(float).eps)
            lower.append(proposed_lower)
            upper.append(observed_high + padding)

        ordered_pairs: list[tuple[int, int]] = []
        if "pKa1" in names and "pKa2" in names:
            ordered_pairs.append((names.index("pKa1"), names.index("pKa2")))
        return cls(
            names=names,
            transforms=tuple(transforms),
            lower=_readonly(np.asarray(lower)),
            upper=_readonly(np.asarray(upper)),
            ordered_pairs=tuple(ordered_pairs),
        )

    @property
    def dimension(self) -> int:
        return len(self.names)

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def valid_rows(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("coordinate matrix has the wrong shape")
        valid = np.isfinite(values).all(axis=1)
        valid &= np.all(values >= self.lower[None, :], axis=1)
        valid &= np.all(values <= self.upper[None, :], axis=1)
        for lower_index, upper_index in self.ordered_pairs:
            valid &= values[:, lower_index] < values[:, upper_index]
        return valid

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("sample count must be positive")
        result = np.empty((count, self.dimension), dtype=float)
        filled = 0
        attempts = 0
        while filled < count:
            attempts += 1
            if attempts > 1_000:
                raise RuntimeError("could not sample an ordered transformed prior")
            batch_size = max(32, 2 * (count - filled))
            batch = rng.uniform(self.lower, self.upper, size=(batch_size, self.dimension))
            batch = batch[self.valid_rows(batch)]
            take = min(len(batch), count - filled)
            if take:
                result[filled : filled + take] = batch[:take]
                filled += take
        return result

    def sample_sobol(self, seed: int, count: int) -> np.ndarray:
        """Draw a scrambled Sobol prefix from the conditional transformed prior."""

        if count <= 0:
            raise ValueError("sample count must be positive")
        base_power = int(math.ceil(math.log2(count)))
        extra_power = 2 if self.ordered_pairs else 0
        for power in range(base_power + extra_power, base_power + extra_power + 8):
            unit = _scrambled_sobol_points(self.dimension, power, seed)
            values = self.lower + unit * self.width
            values = values[self.valid_rows(values)]
            if len(values) >= count:
                return np.asarray(values[:count], dtype=float)
        raise RuntimeError("could not draw enough ordered Sobol prior particles")

    def encode(self, parameters: Mapping[str, Any]) -> np.ndarray:
        if set(parameters) != set(self.names):
            raise ValueError("parameter mapping has the wrong keys")
        result = np.empty(self.dimension, dtype=float)
        for index, (name, transform) in enumerate(zip(self.names, self.transforms, strict=True)):
            value = float(parameters[name])
            if not math.isfinite(value) or (transform == "log" and value <= 0):
                raise ValueError(f"parameter {name} is invalid")
            result[index] = value if transform == "identity" else math.log(value)
        return result

    def decode(self, coordinates: np.ndarray) -> dict[str, float]:
        values = np.asarray(coordinates, dtype=float)
        if values.shape != (self.dimension,) or not self.valid_rows(values[None, :])[0]:
            raise ValueError("transformed parameter vector is outside the prior")
        return {
            name: float(value if transform == "identity" else math.exp(value))
            for name, transform, value in zip(
                self.names, self.transforms, values, strict=True
            )
        }

    def decode_many(self, coordinates: np.ndarray) -> tuple[dict[str, float], ...]:
        values = np.asarray(coordinates, dtype=float)
        if not self.valid_rows(values).all():
            raise ValueError("one or more transformed parameter vectors are outside the prior")
        return tuple(self.decode(row) for row in values)

    def outside_coordinate_count(self, parameters: Mapping[str, Any]) -> int:
        values = self.encode(parameters)
        outside = (values < self.lower) | (values > self.upper)
        for lower_index, upper_index in self.ordered_pairs:
            if values[lower_index] >= values[upper_index]:
                outside[lower_index] = True
                outside[upper_index] = True
        return int(np.count_nonzero(outside))


@dataclass(frozen=True)
class ImportanceResult:
    particles: np.ndarray
    weights: np.ndarray
    log_likelihoods: np.ndarray
    log_evidence: float
    effective_sample_size: float


def static_importance_sample(
    prior: TransformedParameterPrior,
    log_likelihood: ArrayLogLikelihood,
    *,
    num_particles: int,
    seed: int,
) -> ImportanceResult:
    rng = np.random.default_rng(seed)
    particles = prior.sample(rng, num_particles)
    log_likelihoods = np.asarray(log_likelihood(particles), dtype=float)
    if log_likelihoods.shape != (num_particles,) or not np.isfinite(log_likelihoods).all():
        raise ValueError("importance log likelihoods are invalid")
    weights, normalizer = _normalized_weights(log_likelihoods)
    return ImportanceResult(
        particles=_readonly(particles),
        weights=_readonly(weights),
        log_likelihoods=_readonly(log_likelihoods),
        log_evidence=float(normalizer - math.log(num_particles)),
        effective_sample_size=float(1.0 / np.square(weights).sum()),
    )


@dataclass(frozen=True)
class SMCDiagnostics:
    temperatures: tuple[float, ...]
    ess_before_resampling: tuple[float, ...]
    acceptance_rates: tuple[float, ...]
    invalid_proposal_rates: tuple[float, ...]
    total_accepted: int
    total_proposals: int
    total_invalid_proposals: int

    @property
    def num_rungs(self) -> int:
        return len(self.temperatures)

    @property
    def aggregate_acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposals if self.total_proposals else 0.0

    @property
    def aggregate_invalid_proposal_rate(self) -> float:
        return (
            self.total_invalid_proposals / self.total_proposals
            if self.total_proposals
            else 0.0
        )


@dataclass(frozen=True)
class SMCResult:
    particles: np.ndarray
    weights: np.ndarray
    log_likelihoods: np.ndarray
    log_evidence: float
    diagnostics: SMCDiagnostics


def _effective_sample_size(log_weights: np.ndarray) -> float:
    weights, _ = _normalized_weights(log_weights)
    return float(1.0 / np.square(weights).sum())


def _tempering_increment(
    log_weights: np.ndarray,
    log_likelihoods: np.ndarray,
    remaining: float,
    target_ess: float,
) -> float:
    if _effective_sample_size(log_weights + remaining * log_likelihoods) >= target_ess:
        return remaining
    low = 0.0
    high = remaining
    for _ in range(64):
        midpoint = 0.5 * (low + high)
        ess = _effective_sample_size(log_weights + midpoint * log_likelihoods)
        if ess >= target_ess:
            low = midpoint
        else:
            high = midpoint
    if low <= np.finfo(float).eps:
        raise RuntimeError("adaptive tempering could not make positive progress")
    return low


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError("resampling weights are invalid")
    positions = (rng.random() + np.arange(len(values), dtype=float)) / len(values)
    cumulative = np.cumsum(values)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def adaptive_tempered_smc(
    prior: TransformedParameterPrior,
    log_likelihood: ArrayLogLikelihood,
    *,
    num_particles: int,
    seed: int,
    target_ess_fraction: float = 0.6,
    rejuvenation_moves: int = 3,
    max_tempering_rungs: int = 80,
    proposal_scale: float = 0.5,
    proposal_floor_fraction: float = 0.01,
    initialization: str = "random",
    proposal_geometry: str = "diagonal",
) -> SMCResult:
    if num_particles <= 1:
        raise ValueError("SMC requires at least two particles")
    if not 0 < target_ess_fraction < 1:
        raise ValueError("target ESS fraction must lie in (0, 1)")
    if rejuvenation_moves <= 0 or max_tempering_rungs <= 0:
        raise ValueError("SMC rung and rejuvenation settings must be positive")
    if proposal_scale <= 0 or proposal_floor_fraction <= 0:
        raise ValueError("SMC proposal scales must be positive")
    if initialization not in {"random", "sobol"}:
        raise ValueError("SMC initialization must be random or sobol")
    if proposal_geometry not in {"diagonal", "full"}:
        raise ValueError("SMC proposal geometry must be diagonal or full")

    rng = np.random.default_rng(seed)
    particles = (
        prior.sample(rng, num_particles)
        if initialization == "random"
        else prior.sample_sobol(seed, num_particles)
    )
    log_likelihoods = np.asarray(log_likelihood(particles), dtype=float)
    if log_likelihoods.shape != (num_particles,) or not np.isfinite(log_likelihoods).all():
        raise ValueError("initial SMC log likelihoods are invalid")

    log_weights = np.full(num_particles, -math.log(num_particles), dtype=float)
    temperature = 0.0
    log_evidence = 0.0
    temperatures: list[float] = []
    ess_values: list[float] = []
    acceptance_rates: list[float] = []
    invalid_rates: list[float] = []
    total_accepted = 0
    total_proposals = 0
    total_invalid = 0
    target_ess = target_ess_fraction * num_particles

    for _ in range(max_tempering_rungs):
        if temperature >= 1.0 - 1e-12:
            temperature = 1.0
            break
        remaining = 1.0 - temperature
        increment = _tempering_increment(
            log_weights,
            log_likelihoods,
            remaining,
            target_ess,
        )
        unnormalized = log_weights + increment * log_likelihoods
        weights, increment_normalizer = _normalized_weights(unnormalized)
        log_evidence += increment_normalizer
        temperature = min(1.0, temperature + increment)
        ess = float(1.0 / np.square(weights).sum())
        temperatures.append(temperature)
        ess_values.append(ess)

        indices = systematic_resample(weights, rng)
        particles = particles[indices].copy()
        log_likelihoods = log_likelihoods[indices].copy()
        log_weights.fill(-math.log(num_particles))

        if proposal_geometry == "diagonal":
            particle_spread = np.std(particles, axis=0, ddof=1)
            step_scale = np.maximum(
                proposal_scale * particle_spread,
                proposal_floor_fraction * prior.width,
            )
            proposal_cholesky = None
        else:
            covariance = np.atleast_2d(np.cov(particles, rowvar=False, ddof=1))
            floor = proposal_floor_fraction * prior.width
            proposal_covariance = proposal_scale**2 * covariance + np.diag(floor**2)
            proposal_cholesky = np.linalg.cholesky(proposal_covariance)
            step_scale = None
        accepted_this_rung = 0
        invalid_this_rung = 0
        proposed_this_rung = num_particles * rejuvenation_moves
        for _move in range(rejuvenation_moves):
            innovations = rng.normal(size=particles.shape)
            if proposal_cholesky is None:
                if step_scale is None:
                    raise AssertionError("diagonal proposal scale is absent")
                proposals = particles + innovations * step_scale[None, :]
            else:
                proposals = particles + innovations @ proposal_cholesky.T
            valid = prior.valid_rows(proposals)
            invalid_this_rung += int(np.count_nonzero(~valid))
            proposal_log_likelihoods = np.full(num_particles, -math.inf, dtype=float)
            if np.any(valid):
                evaluated = np.asarray(log_likelihood(proposals[valid]), dtype=float)
                if evaluated.shape != (int(np.count_nonzero(valid)),) or not np.isfinite(
                    evaluated
                ).all():
                    raise ValueError("proposal log likelihoods are invalid")
                proposal_log_likelihoods[valid] = evaluated
            log_acceptance = temperature * (proposal_log_likelihoods - log_likelihoods)
            accept = valid & (np.log(rng.random(num_particles)) < np.minimum(0.0, log_acceptance))
            accepted_this_rung += int(np.count_nonzero(accept))
            particles[accept] = proposals[accept]
            log_likelihoods[accept] = proposal_log_likelihoods[accept]

        total_accepted += accepted_this_rung
        total_invalid += invalid_this_rung
        total_proposals += proposed_this_rung
        acceptance_rates.append(accepted_this_rung / proposed_this_rung)
        invalid_rates.append(invalid_this_rung / proposed_this_rung)
        if temperature >= 1.0 - 1e-12:
            temperature = 1.0
            break
    else:
        raise RuntimeError("adaptive tempering exceeded its rung limit")

    if temperature != 1.0:
        raise RuntimeError("adaptive tempering did not reach temperature one")
    weights = np.full(num_particles, 1.0 / num_particles, dtype=float)
    diagnostics = SMCDiagnostics(
        temperatures=tuple(float(value) for value in temperatures),
        ess_before_resampling=tuple(float(value) for value in ess_values),
        acceptance_rates=tuple(float(value) for value in acceptance_rates),
        invalid_proposal_rates=tuple(float(value) for value in invalid_rates),
        total_accepted=total_accepted,
        total_proposals=total_proposals,
        total_invalid_proposals=total_invalid,
    )
    return SMCResult(
        particles=_readonly(particles),
        weights=_readonly(weights),
        log_likelihoods=_readonly(log_likelihoods),
        log_evidence=float(log_evidence),
        diagnostics=diagnostics,
    )
