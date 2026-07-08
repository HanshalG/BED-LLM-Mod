"""Paper-style total EIG bounds for location-finding policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from environments.location_finding.physics import sample_source_configs_from_prior, signal_intensities_from_distances
from environments.location_finding.types import LocationObservation


EIG_BOUND_METRIC_NAMES = (
    "total_eig_lower_bound",
    "total_eig_upper_bound",
    "total_eig_lower_bound_se",
    "total_eig_upper_bound_se",
)


@dataclass(frozen=True)
class EIGBoundEstimate:
    lower: float
    upper: float
    lower_se: float
    upper_se: float
    lower_values: tuple[float, ...]
    upper_values: tuple[float, ...]

    def as_metric_traces(self) -> dict[str, list[float]]:
        return {
            "total_eig_lower_bound": [self.lower],
            "total_eig_upper_bound": [self.upper],
            "total_eig_lower_bound_se": [self.lower_se],
            "total_eig_upper_bound_se": [self.upper_se],
        }


def sample_prior_source_configs(
    rng: np.random.Generator,
    *,
    count: int,
    num_sources: int,
    dim: int,
    source_prior: str = "normal",
    source_radius: float = 1.0,
) -> np.ndarray:
    """Draw contrastive source configurations from the location prior."""
    if count < 1:
        raise ValueError("count must be positive")
    return sample_source_configs_from_prior(
        rng,
        count=count,
        num_sources=num_sources,
        dim=dim,
        source_prior=source_prior,
        source_radius=source_radius,
    )


def history_log_likelihoods(
    theta_samples: np.ndarray,
    observations: Sequence[LocationObservation],
    *,
    noise_sd: float,
    chunk_size: int = 8192,
    signal_model: str = "inverse_square",
    signal_lengthscale: float = 0.75,
    signal_amplitude: float = 5.0,
) -> np.ndarray:
    """Return log p(history | theta) for each theta sample.

    This matches the repository's location observation model:
    ``log y ~ Normal(log signal(x; theta), noise_sd)`` with the observation
    Jacobian omitted because it cancels in posterior comparisons and EIG ratios.
    """
    theta = np.asarray(theta_samples, dtype=float)
    if theta.ndim != 3:
        raise ValueError("theta_samples must have shape (N, num_sources, dim)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not observations:
        return np.zeros(theta.shape[0], dtype=float)

    result = np.empty(theta.shape[0], dtype=float)
    constant = -math.log(noise_sd) - 0.5 * math.log(2.0 * math.pi)
    for start in range(0, theta.shape[0], chunk_size):
        stop = min(start + chunk_size, theta.shape[0])
        chunk = theta[start:stop]
        log_likelihood = np.zeros(stop - start, dtype=float)
        for observation in observations:
            if observation.value <= 0.0:
                log_likelihood.fill(float("-inf"))
                break
            query = np.asarray(observation.query, dtype=float)
            distances_sq = np.sum((chunk - query[None, None, :]) ** 2, axis=2)
            means = signal_intensities_from_distances(
                distances_sq,
                signal_model=signal_model,
                signal_lengthscale=signal_lengthscale,
                signal_amplitude=signal_amplitude,
            )
            z = (math.log(observation.value) - np.log(means)) / noise_sd
            log_likelihood += -0.5 * z * z + constant
        result[start:stop] = log_likelihood
    return result


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("logmeanexp requires at least one value")
    max_value = float(np.max(values))
    if not math.isfinite(max_value):
        return max_value
    return max_value + float(np.log(np.mean(np.exp(values - max_value))))


def eig_bound_values_for_history(
    true_theta: np.ndarray,
    observations: Sequence[LocationObservation],
    contrastive_thetas: np.ndarray,
    *,
    noise_sd: float,
    chunk_size: int = 8192,
    signal_model: str = "inverse_square",
    signal_lengthscale: float = 0.75,
    signal_amplitude: float = 5.0,
) -> tuple[float, float]:
    """Compute one sPCE lower value and one sNMC upper value for a history."""
    true_log_likelihood = float(
        history_log_likelihoods(
            np.asarray(true_theta, dtype=float)[None, :, :],
            observations,
            noise_sd=noise_sd,
            chunk_size=chunk_size,
            signal_model=signal_model,
            signal_lengthscale=signal_lengthscale,
            signal_amplitude=signal_amplitude,
        )[0]
    )
    contrastive_log_likelihoods = history_log_likelihoods(
        contrastive_thetas,
        observations,
        noise_sd=noise_sd,
        chunk_size=chunk_size,
        signal_model=signal_model,
        signal_lengthscale=signal_lengthscale,
        signal_amplitude=signal_amplitude,
    )
    lower_denom = logmeanexp(
        np.concatenate(([true_log_likelihood], contrastive_log_likelihoods))
    )
    upper_denom = logmeanexp(contrastive_log_likelihoods)
    return true_log_likelihood - lower_denom, true_log_likelihood - upper_denom


def estimate_eig_bounds_from_run_result(
    run_result: Any,
    config: Any,
    *,
    rng: np.random.Generator,
) -> EIGBoundEstimate:
    lower_values: list[float] = []
    upper_values: list[float] = []
    inner_samples = int(getattr(config, "location_eig_bounds_inner_samples", 5000))
    chunk_size = int(getattr(config, "location_eig_bounds_chunk_size", 8192))
    for trial in run_result.trials:
        observations = [round_result.observation for round_result in trial.rounds]
        contrastive_thetas = sample_prior_source_configs(
            rng,
            count=inner_samples,
            num_sources=int(getattr(config, "location_num_sources", 3)),
            dim=int(getattr(config, "location_dim", 2)),
            source_prior=str(getattr(config, "location_source_prior", "normal")),
            source_radius=float(getattr(config, "location_source_radius", 1.0)),
        )
        lower, upper = eig_bound_values_for_history(
            np.asarray(trial.hidden_state, dtype=float),
            observations,
            contrastive_thetas,
            noise_sd=float(getattr(config, "location_noise_sd", 0.5)),
            chunk_size=chunk_size,
            signal_model=str(getattr(config, "location_signal_model", "inverse_square")),
            signal_lengthscale=float(getattr(config, "location_signal_lengthscale", 0.75)),
            signal_amplitude=float(getattr(config, "location_signal_amplitude", 5.0)),
        )
        lower_values.append(float(lower))
        upper_values.append(float(upper))

    return _summarize_bound_values(lower_values, upper_values)


def _summarize_bound_values(
    lower_values: Sequence[float],
    upper_values: Sequence[float],
) -> EIGBoundEstimate:
    lower = np.asarray(lower_values, dtype=float)
    upper = np.asarray(upper_values, dtype=float)
    if lower.size == 0 or upper.size == 0:
        return EIGBoundEstimate(
            lower=float("nan"),
            upper=float("nan"),
            lower_se=float("nan"),
            upper_se=float("nan"),
            lower_values=(),
            upper_values=(),
        )
    lower_se = float(np.std(lower, ddof=1) / math.sqrt(lower.size)) if lower.size > 1 else 0.0
    upper_se = float(np.std(upper, ddof=1) / math.sqrt(upper.size)) if upper.size > 1 else 0.0
    return EIGBoundEstimate(
        lower=float(np.mean(lower)),
        upper=float(np.mean(upper)),
        lower_se=lower_se,
        upper_se=upper_se,
        lower_values=tuple(float(value) for value in lower),
        upper_values=tuple(float(value) for value in upper),
    )
