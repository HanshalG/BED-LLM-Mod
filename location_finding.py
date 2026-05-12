from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

from helpers import Config, print_and_log, write_to_log

if TYPE_CHECKING:
    from model import Model


SourceConfig = tuple[tuple[float, ...], ...]
Location = tuple[float, ...]


@dataclass(frozen=True)
class LocationObservation:
    query: Location
    value: float


@dataclass(frozen=True)
class LocationBeliefState:
    hypotheses: list[SourceConfig] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.hypotheses) != len(self.probabilities):
            raise ValueError("LocationBeliefState hypotheses and probabilities must have the same length")


@dataclass(frozen=True)
class LocationFindingMetrics:
    source_rmse: list[float]
    top_probability: list[float]
    selected_eig: list[float]


class LocationFindingEnv:
    def __init__(
        self,
        num_sources: int = 3,
        dim: int = 2,
        b: float = 0.1,
        m: float = 1e-4,
        alpha: float = 1.0,
        noise_sd: float = 0.5,
        true_theta: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.num_sources = num_sources
        self.dim = dim
        self.b = b
        self.m = m
        self.alpha = alpha
        self.noise_sd = noise_sd
        self.rng = rng or np.random.default_rng()
        self.observed_data: list[LocationObservation] = []
        self.true_theta = np.zeros((self.num_sources, self.dim), dtype=float)
        self.reset(true_theta=true_theta)

    def reset(self, true_theta: np.ndarray | None = None) -> None:
        self.observed_data = []
        if true_theta is None:
            self.true_theta = self.rng.normal(0.0, 1.0, size=(self.num_sources, self.dim))
        else:
            theta = np.asarray(true_theta, dtype=float)
            if theta.shape != (self.num_sources, self.dim):
                raise ValueError(
                    f"true_theta must have shape {(self.num_sources, self.dim)}, received {theta.shape}"
                )
            self.true_theta = theta

    def signal_intensity(self, query: Location | np.ndarray) -> float:
        query_arr = np.asarray(query, dtype=float)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"query must have shape {(self.dim,)}, received {query_arr.shape}")
        distances_squared = np.sum((self.true_theta - query_arr) ** 2, axis=1)
        return float(self.b + np.sum(self.alpha / (self.m + distances_squared)))

    def step(self, query: Location | np.ndarray) -> float:
        intensity = self.signal_intensity(query)
        return float(self.rng.normal(intensity, self.noise_sd))

    def run_experiment(self, query: Location | np.ndarray) -> LocationObservation:
        query_tuple = normalize_location(query, self.dim)
        observation = round(self.step(query_tuple), 2)
        result = LocationObservation(query=query_tuple, value=float(observation))
        self.observed_data.append(result)
        return result

    def sample_random_input(self, bounds: tuple[float, float] = (-2.0, 2.0)) -> Location:
        low, high = bounds
        return tuple(float(value) for value in self.rng.uniform(low, high, size=self.dim))


def normalize_location(raw_location: object, dim: int) -> Location:
    try:
        values = [float(value) for value in raw_location]  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Location must be an iterable of {dim} numeric values") from exc
    if len(values) != dim:
        raise ValueError(f"Location must have length {dim}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Location values must be finite")
    return tuple(values)


def normalize_source_config(raw_config: object, num_sources: int, dim: int) -> SourceConfig:
    try:
        source_rows = list(raw_config)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Source configuration must be an iterable of source coordinates") from exc
    if len(source_rows) != num_sources:
        raise ValueError(f"Source configuration must contain exactly {num_sources} sources")
    sources = [normalize_location(row, dim) for row in source_rows]
    return tuple(sorted(sources))


def _dedupe_source_configs(configs: list[SourceConfig]) -> list[SourceConfig]:
    deduped: list[SourceConfig] = []
    seen: set[tuple[tuple[float, ...], ...]] = set()
    for config in configs:
        key = tuple(tuple(round(value, 6) for value in source) for source in config)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(config)
    return deduped


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_value(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    start_idx: int | None = None
    opening = ""
    closing = ""
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(stripped):
        if start_idx is None:
            if char == "{":
                start_idx = idx
                opening = "{"
                closing = "}"
                depth = 1
            elif char == "[":
                start_idx = idx
                opening = "["
                closing = "]"
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return stripped[start_idx:idx + 1]

    return None


def _loads_json_value(text: str) -> object:
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError) as exc:
        extracted = _extract_first_json_value(text)
        if extracted is None:
            raise ValueError(f"Could not find JSON value in completion: {text!r}") from exc
        try:
            return json.loads(extracted)
        except (json.JSONDecodeError, TypeError) as nested_exc:
            raise ValueError(f"Invalid JSON in completion: {text!r}") from nested_exc


def parse_source_hypotheses(completion: str, num_sources: int, dim: int) -> list[SourceConfig]:
    payload = _loads_json_value(completion)
    if isinstance(payload, dict):
        for key in ("hypotheses", "source_configurations", "sources", "configs"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError("Source hypothesis completion must decode to a JSON list or object")

    configs: list[SourceConfig] = []
    for item in payload:
        raw_config = item.get("sources") if isinstance(item, dict) else item
        try:
            configs.append(normalize_source_config(raw_config, num_sources, dim))
        except ValueError:
            continue
    return _dedupe_source_configs(configs)


def parse_candidate_locations(completion: str, dim: int, bounds: tuple[float, float]) -> list[Location]:
    payload = _loads_json_value(completion)
    if isinstance(payload, dict):
        for key in ("locations", "candidates", "queries", "points"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError("Candidate location completion must decode to a JSON list or object")

    low, high = bounds
    locations: list[Location] = []
    seen: set[tuple[float, ...]] = set()
    for item in payload:
        raw_location = item.get("location") if isinstance(item, dict) else item
        try:
            location = normalize_location(raw_location, dim)
        except ValueError:
            continue
        if any(value < low or value > high for value in location):
            continue
        key = tuple(round(value, 6) for value in location)
        if key in seen:
            continue
        seen.add(key)
        locations.append(location)
    return locations


def _format_observations(observations: list[LocationObservation]) -> str:
    if not observations:
        return "[]"
    rows = [
        {"query": list(observation.query), "signal_strength": observation.value}
        for observation in observations
    ]
    return json.dumps(rows)


def _format_weighted_hypotheses(belief_state: LocationBeliefState, top_n: int = 10) -> str:
    entries = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )[:top_n]
    rows = [
        {"sources": [list(source) for source in hypothesis], "probability": probability}
        for hypothesis, probability in entries
    ]
    return json.dumps(rows)


def _belief_generation_messages(
    observations: list[LocationObservation],
    belief_state: LocationBeliefState | None,
    config: Config,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    current_context = "[]"
    if belief_state is not None and belief_state.hypotheses:
        current_context = _format_weighted_hypotheses(belief_state)

    system = (
        "You propose structured belief support for a source-localization task. "
        "Return only strict JSON, with no markdown or explanation."
    )
    user = (
        "The environment is BoxingGym-style Location Finding.\n"
        f"There are exactly {config.location_num_sources} hidden signal sources in {config.location_dim}D space.\n"
        "Each source has identical strength. Source coordinates are independently drawn from Normal(0, 1).\n"
        "For a query x, the noiseless signal is b + sum_k alpha / (m + ||theta_k - x||^2), "
        "with b=0.1, alpha=1, and m=0.0001.\n"
        f"Observed signals are noisy with known Gaussian noise_sd={config.location_noise_sd}.\n"
        f"Measurement candidates must stay inside [{bounds[0]}, {bounds[1]}] for every coordinate.\n"
        "The true source coordinates are unknown and are not provided.\n\n"
        f"Observation history: {_format_observations(observations)}\n"
        f"Current weighted hypotheses, if any: {current_context}\n\n"
        f"Generate up to {config.location_max_beliefs} plausible source configurations. "
        f"Each source configuration must contain exactly {config.location_num_sources} coordinate lists, "
        f"and each coordinate must have length {config.location_dim}.\n"
        "Return exactly a JSON object with key \"hypotheses\" whose value is a list of source configurations.\n"
        "Example shape: {\"hypotheses\": [[[0.1, -0.2], [1.0, 0.0], [-0.4, 0.7]]]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _candidate_generation_messages(
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = (
        "You propose measurement locations for active source localization. "
        "Return only strict JSON, with no markdown or explanation."
    )
    user = (
        "The environment is BoxingGym-style Location Finding in 2D with exactly 3 hidden identical-strength sources.\n"
        "The known observation model has Gaussian noise_sd="
        f"{config.location_noise_sd}. Use this known noise model when designing measurements.\n"
        f"Allowed coordinate bounds are [{bounds[0]}, {bounds[1]}] for both x1 and x2.\n"
        "The true source coordinates are unknown and are not provided.\n\n"
        f"Observation history: {_format_observations(observations)}\n"
        f"Current posterior source hypotheses: {_format_weighted_hypotheses(belief_state, top_n=15)}\n\n"
        f"Generate exactly {config.location_target_num_candidates} candidate measurement locations. "
        "Locations should be discriminative among the posterior source hypotheses and should not repeat previous queries.\n"
        "Return exactly a JSON object with key \"locations\" whose value is a list of [x1, x2] coordinates."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_location_hypotheses(
    questioner: "Model",
    observations: list[LocationObservation],
    belief_state: LocationBeliefState | None,
    config: Config,
) -> list[SourceConfig]:
    completion = questioner.chat_complete(
        _belief_generation_messages(observations, belief_state, config),
        temperature=config.generation_temperature_diverse,
    )[0]
    return parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)


def _generate_location_hypotheses_many(
    questioner: "Model",
    observations_many: list[list[LocationObservation]],
    belief_states: list[LocationBeliefState | None],
    config: Config,
) -> list[list[SourceConfig]]:
    if len(observations_many) != len(belief_states):
        raise ValueError("observations_many and belief_states must have the same length")
    if not observations_many:
        return []

    batch_messages = [
        _belief_generation_messages(observations, belief_state, config)
        for observations, belief_state in zip(observations_many, belief_states)
    ]
    if callable(getattr(questioner, "chat_complete_messages_batched", None)):
        completions = questioner.chat_complete_messages_batched(
            batch_messages=batch_messages,
            temperature=config.generation_temperature_diverse,
            block_size=config.batched_block_size,
        )
    else:
        completions = [
            questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
            for messages in batch_messages
        ]
    if len(completions) != len(batch_messages):
        raise ValueError(f"Expected {len(batch_messages)} hypothesis completions, received {len(completions)}")

    hypotheses_many: list[list[SourceConfig]] = []
    for completion in completions:
        try:
            hypotheses_many.append(
                parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)
            )
        except ValueError:
            hypotheses_many.append([])
    return hypotheses_many


def _default_source_hypotheses(config: Config) -> list[SourceConfig]:
    anchors = [
        [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)],
        [(-1.0, 1.0), (0.0, 0.0), (1.0, -1.0)],
        [(-1.5, 0.0), (0.0, 1.5), (1.5, 0.0)],
        [(0.0, -1.5), (-1.5, 0.0), (1.5, 0.0)],
    ]
    return [
        normalize_source_config(anchor, config.location_num_sources, config.location_dim)
        for anchor in anchors
    ]


def generate_location_candidates(
    questioner: "Model",
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[Location]:
    bounds = tuple(config.location_query_bounds)
    completion = questioner.chat_complete(
        _candidate_generation_messages(belief_state, observations, config),
        temperature=config.generation_temperature_diverse,
    )[0]
    try:
        candidates = parse_candidate_locations(completion, config.location_dim, bounds)
    except ValueError:
        candidates = []
    if len(candidates) < config.location_target_num_candidates:
        candidates.extend(
            location
            for location in default_candidate_locations(config.location_dim, bounds)
            if location not in candidates
        )
    return candidates[:config.location_target_num_candidates]


def default_candidate_locations(dim: int, bounds: tuple[float, float]) -> list[Location]:
    low, high = bounds
    if dim != 2:
        center = tuple(0.0 for _ in range(dim))
        corners = [tuple(value for _ in range(dim)) for value in (low, high)]
        return [center] + corners

    midpoint = (low + high) / 2.0
    quarter_low = low + (high - low) * 0.25
    quarter_high = low + (high - low) * 0.75
    values = [low, quarter_low, midpoint, quarter_high, high]
    ordered = [
        (midpoint, midpoint),
        (quarter_low, quarter_low),
        (quarter_low, quarter_high),
        (quarter_high, quarter_low),
        (quarter_high, quarter_high),
        (low, midpoint),
        (high, midpoint),
        (midpoint, low),
        (midpoint, high),
    ]
    ordered.extend((x, y) for x in values for y in values)
    deduped: list[Location] = []
    seen: set[Location] = set()
    for location in ordered:
        normalized = tuple(float(value) for value in location)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def signal_intensity_for_hypothesis(
    hypothesis: SourceConfig,
    query: Location,
    b: float = 0.1,
    m: float = 1e-4,
    alpha: float = 1.0,
) -> float:
    theta = np.asarray(hypothesis, dtype=float)
    query_arr = np.asarray(query, dtype=float)
    distances_squared = np.sum((theta - query_arr) ** 2, axis=1)
    return float(b + np.sum(alpha / (m + distances_squared)))


def _log_normal_pdf(value: float, mean: float, sd: float) -> float:
    z = (value - mean) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)


def _logsumexp(log_values: list[float] | np.ndarray) -> float:
    values = np.asarray(log_values, dtype=float)
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _hypothesis_log_prior(hypothesis: SourceConfig) -> float:
    theta = np.asarray(hypothesis, dtype=float)
    dimension_count = theta.size
    return float(-0.5 * np.sum(theta ** 2) - 0.5 * dimension_count * math.log(2.0 * math.pi))


def build_location_belief_state(
    hypotheses: list[SourceConfig],
    observations: list[LocationObservation],
    config: Config,
) -> LocationBeliefState:
    hypotheses = _dedupe_source_configs(hypotheses)
    if not hypotheses:
        return LocationBeliefState([], [])

    log_scores = []
    for hypothesis in hypotheses:
        log_score = _hypothesis_log_prior(hypothesis)
        for observation in observations:
            mean = signal_intensity_for_hypothesis(hypothesis, observation.query)
            log_score += _log_normal_pdf(observation.value, mean, config.location_noise_sd)
        log_scores.append(log_score)

    normalizer = _logsumexp(log_scores)
    probabilities = [math.exp(log_score - normalizer) for log_score in log_scores]
    state = LocationBeliefState(hypotheses, probabilities)
    state = sort_location_belief_state(state)
    return prune_location_beliefs(
        state,
        max_beliefs=config.location_max_beliefs,
        min_probability_mass=config.location_min_probability_mass,
    )


def sort_location_belief_state(belief_state: LocationBeliefState) -> LocationBeliefState:
    ordered = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )
    return LocationBeliefState(
        [hypothesis for hypothesis, _probability in ordered],
        [float(probability) for _hypothesis, probability in ordered],
    )


def prune_location_beliefs(
    belief_state: LocationBeliefState,
    max_beliefs: int,
    min_probability_mass: float,
) -> LocationBeliefState:
    if len(belief_state.hypotheses) <= max_beliefs:
        return belief_state

    ordered = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )
    filtered = [entry for entry in ordered if entry[1] >= min_probability_mass]
    if not filtered:
        filtered = ordered[:1]
    filtered = filtered[:max_beliefs]
    total = sum(probability for _hypothesis, probability in filtered)
    if total <= 0.0:
        probability = 1.0 / len(filtered)
        return LocationBeliefState([hypothesis for hypothesis, _probability in filtered], [probability] * len(filtered))
    return LocationBeliefState(
        [hypothesis for hypothesis, _probability in filtered],
        [float(probability / total) for _hypothesis, probability in filtered],
    )


def _posterior_after_observation(
    belief_state: LocationBeliefState,
    query: Location,
    value: float,
    noise_sd: float,
) -> LocationBeliefState:
    if not belief_state.hypotheses:
        return belief_state
    log_scores = []
    for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities):
        mean = signal_intensity_for_hypothesis(hypothesis, query)
        log_scores.append(math.log(max(probability, 1e-300)) + _log_normal_pdf(value, mean, noise_sd))
    normalizer = _logsumexp(log_scores)
    return sort_location_belief_state(
        LocationBeliefState(
            list(belief_state.hypotheses),
            [math.exp(log_score - normalizer) for log_score in log_scores],
        )
    )


def _quadrature_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes.astype(float), (weights.astype(float) / math.sqrt(math.pi))


def expected_information_gain(
    belief_state: LocationBeliefState,
    query: Location,
    noise_sd: float,
    quadrature_order: int,
) -> float:
    if len(belief_state.hypotheses) <= 1:
        return 0.0

    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    means = np.asarray(
        [signal_intensity_for_hypothesis(hypothesis, query) for hypothesis in belief_state.hypotheses],
        dtype=float,
    )
    nodes, weights = _quadrature_nodes(quadrature_order)
    return _expected_information_gain_from_means(probabilities, means, noise_sd, nodes, weights)


def _normal_logpdf_array(values: np.ndarray, means: np.ndarray, noise_sd: float) -> np.ndarray:
    z = (values - means) / noise_sd
    return -0.5 * z * z - math.log(noise_sd) - 0.5 * math.log(2.0 * math.pi)


def _logsumexp_array(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True)), axis=axis)


def _expected_information_gain_from_means(
    probabilities: np.ndarray,
    means: np.ndarray,
    noise_sd: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    if len(means) <= 1:
        return 0.0

    probabilities = np.asarray(probabilities, dtype=float)
    means = np.asarray(means, dtype=float)
    y_values = means[:, None] + math.sqrt(2.0) * noise_sd * nodes[None, :]
    component_log_likelihoods = _normal_logpdf_array(y_values, means[:, None], noise_sd)
    all_log_likelihoods = _normal_logpdf_array(y_values[:, :, None], means[None, None, :], noise_sd)
    mixture_log_likelihoods = _logsumexp_array(
        all_log_likelihoods + np.log(np.maximum(probabilities, 1e-300))[None, None, :],
        axis=2,
    )
    value = np.sum(probabilities[:, None] * weights[None, :] * (component_log_likelihoods - mixture_log_likelihoods))
    return max(0.0, float(value))


def _expected_information_gain_batch_from_means(
    probability_rows: np.ndarray,
    means: np.ndarray,
    noise_sd: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if len(means) <= 1:
        return np.zeros(probability_rows.shape[0], dtype=float)

    probability_rows = np.asarray(probability_rows, dtype=float)
    means = np.asarray(means, dtype=float)
    y_values = means[None, :, None] + math.sqrt(2.0) * noise_sd * nodes[None, None, :]
    component_log_likelihoods = _normal_logpdf_array(y_values, means[None, :, None], noise_sd)
    all_log_likelihoods = _normal_logpdf_array(y_values[:, :, :, None], means[None, None, None, :], noise_sd)
    mixture_log_likelihoods = _logsumexp_array(
        all_log_likelihoods + np.log(np.maximum(probability_rows, 1e-300))[:, None, None, :],
        axis=3,
    )
    values = np.sum(
        probability_rows[:, :, None]
        * weights[None, None, :]
        * (component_log_likelihoods - mixture_log_likelihoods),
        axis=(1, 2),
    )
    return np.maximum(values, 0.0)


def score_candidate_locations(
    belief_state: LocationBeliefState,
    candidates: list[Location],
    config: Config,
    questioner: "Model | None" = None,
    observations: list[LocationObservation] | None = None,
) -> list[float]:
    if not candidates:
        return []

    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    candidate_means = np.asarray(
        [
            [signal_intensity_for_hypothesis(hypothesis, candidate) for hypothesis in belief_state.hypotheses]
            for candidate in candidates
        ],
        dtype=float,
    )
    nodes, weights = _quadrature_nodes(config.location_eig_quadrature_order)
    immediate = [
        _expected_information_gain_from_means(
            probabilities,
            means,
            config.location_noise_sd,
            nodes,
            weights,
        )
        for means in candidate_means
    ]
    if len(belief_state.hypotheses) <= 1:
        return immediate
    if config.location_search_depth == 1 or len(candidates) == 0:
        return immediate
    if config.location_search_depth != 2:
        raise ValueError("location_search_depth must be 1 or 2")
    if questioner is None or observations is None:
        raise ValueError("location_search_depth=2 requires questioner and observations for full branch updates")

    totals = list(immediate)
    branch_candidate_indices: list[int] = []
    branch_weights: list[float] = []
    branch_observations: list[list[LocationObservation]] = []

    for candidate_idx, (candidate, means) in enumerate(zip(candidates, candidate_means)):
        for hypothesis_idx, mean in enumerate(means):
            hypothesis_probability = probabilities[hypothesis_idx]
            if hypothesis_probability == 0.0:
                continue
            for node, node_weight in zip(nodes, weights):
                branch_value = float(mean + math.sqrt(2.0) * config.location_noise_sd * node)
                branch_candidate_indices.append(candidate_idx)
                branch_weights.append(float(hypothesis_probability * node_weight))
                branch_observations.append(
                    list(observations) + [LocationObservation(query=candidate, value=branch_value)]
                )

    generated_hypotheses_many = _generate_location_hypotheses_many(
        questioner,
        branch_observations,
        [belief_state for _ in branch_observations],
        config,
    )
    for candidate_idx, branch_weight, branch_history, generated_hypotheses in zip(
        branch_candidate_indices,
        branch_weights,
        branch_observations,
        generated_hypotheses_many,
    ):
        future_hypotheses = _merge_hypotheses(belief_state, generated_hypotheses)
        future_state = build_location_belief_state(future_hypotheses, branch_history, config)
        future_values = [
            expected_information_gain(
                future_state,
                future_candidate,
                noise_sd=config.location_noise_sd,
                quadrature_order=config.location_eig_quadrature_order,
            )
            for future_candidate in candidates
        ]
        if future_values:
            totals[candidate_idx] += branch_weight * max(future_values)
    return totals


def _posterior_probabilities_after_values(
    probabilities: np.ndarray,
    means: np.ndarray,
    values: np.ndarray,
    noise_sd: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    log_scores = (
        np.log(np.maximum(probabilities, 1e-300))[None, :]
        + _normal_logpdf_array(values[:, None], means[None, :], noise_sd)
    )
    normalizers = _logsumexp_array(log_scores, axis=1)
    return np.exp(log_scores - normalizers[:, None])


def source_rmse(predicted: SourceConfig, true_sources: np.ndarray) -> float:
    predicted_arr = np.asarray(predicted, dtype=float)
    if predicted_arr.shape != true_sources.shape:
        raise ValueError("predicted sources and true sources must have matching shape")
    best_mse = min(
        float(np.mean((np.asarray(permutation, dtype=float) - true_sources) ** 2))
        for permutation in permutations(predicted_arr)
    )
    return math.sqrt(best_mse)


def _top_source_rmse(belief_state: LocationBeliefState, true_sources: np.ndarray) -> float:
    if not belief_state.hypotheses:
        return float("inf")
    return source_rmse(belief_state.hypotheses[0], true_sources)


def _merge_hypotheses(previous: LocationBeliefState, generated: list[SourceConfig]) -> list[SourceConfig]:
    return _dedupe_source_configs(list(previous.hypotheses) + list(generated))


def _write_to_log_if_configured(message: str, config: Config) -> None:
    if config.log_path is not None:
        write_to_log(message, config)


def run_location_finding(questioner: "Model", config: Config, rng: np.random.Generator | None = None) -> LocationFindingMetrics:
    if config.location_num_sources != 3 or config.location_dim != 2:
        raise ValueError("The initial Location Finding implementation supports exactly 3 sources in 2D")
    if config.location_noise_sd != 0.5:
        raise ValueError("The initial Location Finding implementation requires known noise_sd=0.5")

    rng = rng or np.random.default_rng()
    rmse_totals = np.zeros(config.location_num_rounds, dtype=float)
    top_probability_totals = np.zeros(config.location_num_rounds, dtype=float)
    selected_eig_totals = np.zeros(config.location_num_rounds, dtype=float)

    print(f"[location] Running {config.location_num_trials} Location Finding trial(s)")
    for trial_idx in range(config.location_num_trials):
        env = LocationFindingEnv(
            num_sources=config.location_num_sources,
            dim=config.location_dim,
            noise_sd=config.location_noise_sd,
            rng=rng,
        )
        observations: list[LocationObservation] = []
        initial_hypotheses = generate_location_hypotheses(questioner, observations, None, config)
        if not initial_hypotheses:
            print_and_log("[location] No valid initial LLM hypotheses; using deterministic fallback support", config)
            initial_hypotheses = _default_source_hypotheses(config)
        belief_state = build_location_belief_state(initial_hypotheses, observations, config)

        for round_idx in range(config.location_num_rounds):
            _write_to_log_if_configured(f"\nLocation Finding trial {trial_idx + 1}: Round {round_idx + 1}\n", config)
            print(
                f"[location] trial {trial_idx + 1}/{config.location_num_trials}, "
                f"round {round_idx + 1}/{config.location_num_rounds}, "
                f"{len(belief_state.hypotheses)} belief(s)"
            )
            candidates = generate_location_candidates(questioner, belief_state, observations, config)
            scores = score_candidate_locations(
                belief_state,
                candidates,
                config,
                questioner=questioner,
                observations=observations,
            )
            best_idx = int(np.argmax(scores)) if scores else 0
            best_location = candidates[best_idx]
            best_score = float(scores[best_idx]) if scores else 0.0
            observation = env.run_experiment(best_location)
            observations.append(observation)
            print_and_log(
                f"[location] Selected query {list(best_location)} with score {best_score:.6f}; "
                f"observed {observation.value:.2f}",
                config,
            )

            generated_hypotheses = generate_location_hypotheses(questioner, observations, belief_state, config)
            merged_hypotheses = _merge_hypotheses(belief_state, generated_hypotheses)
            belief_state = build_location_belief_state(merged_hypotheses, observations, config)
            current_rmse = _top_source_rmse(belief_state, env.true_theta)
            top_probability = belief_state.probabilities[0] if belief_state.probabilities else 0.0
            rmse_totals[round_idx] += current_rmse
            top_probability_totals[round_idx] += top_probability
            selected_eig_totals[round_idx] += best_score
            print_and_log(
                f"[location] Top source RMSE after round {round_idx + 1}: {current_rmse:.6f}; "
                f"top probability {top_probability:.6f}",
                config,
            )

    divisor = float(config.location_num_trials)
    return LocationFindingMetrics(
        source_rmse=(rmse_totals / divisor).tolist(),
        top_probability=(top_probability_totals / divisor).tolist(),
        selected_eig=(selected_eig_totals / divisor).tolist(),
    )
