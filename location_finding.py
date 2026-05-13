from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
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
    if len(set(sources)) != num_sources:
        raise ValueError("Source configuration must contain distinct source coordinates")
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


def _clean_json_completion(text: str) -> str:
    return re.sub(r"<eos>\s*$", "", _strip_code_fences(text).strip()).strip()


def _extract_first_json_value(text: str) -> str | None:
    stripped = _clean_json_completion(text)
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
    stripped = _clean_json_completion(text)
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


def _collect_source_configs_from_payload(payload: object, num_sources: int, dim: int) -> list[SourceConfig]:
    configs: list[SourceConfig] = []
    if isinstance(payload, dict):
        for key in ("hypotheses", "source_configurations", "sources", "configs"):
            if key in payload:
                configs.extend(_collect_source_configs_from_payload(payload[key], num_sources, dim))
                return configs
        for value in payload.values():
            configs.extend(_collect_source_configs_from_payload(value, num_sources, dim))
        return configs

    if isinstance(payload, list):
        try:
            configs.append(normalize_source_config(payload, num_sources, dim))
            return configs
        except ValueError:
            pass
        for item in payload:
            configs.extend(_collect_source_configs_from_payload(item, num_sources, dim))
    return configs


def _extract_partial_source_configs(text: str, num_sources: int, dim: int) -> list[SourceConfig]:
    cleaned = _clean_json_completion(text)
    decoder = json.JSONDecoder()
    configs: list[SourceConfig] = []
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _end_idx = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        configs.extend(_collect_source_configs_from_payload(payload, num_sources, dim))
    return _dedupe_source_configs(configs)


def parse_source_hypotheses(completion: str, num_sources: int, dim: int) -> list[SourceConfig]:
    try:
        payload = _loads_json_value(completion)
    except ValueError:
        configs = _extract_partial_source_configs(completion, num_sources, dim)
        if configs:
            return configs
        raise

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
    if configs:
        return _dedupe_source_configs(configs)
    return _extract_partial_source_configs(completion, num_sources, dim)


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


def _format_location(location: Location) -> str:
    return "[" + ", ".join(f"{value:.3g}" for value in location) + "]"


def _format_probability(probability: float) -> str:
    if probability >= 0.001:
        return f"{probability:.4f}"
    return f"{probability:.2e}"


def _summarize_belief_state(belief_state: LocationBeliefState, top_n: int = 3) -> str:
    if not belief_state.hypotheses:
        return "0 belief(s)"
    top_entries = list(zip(belief_state.hypotheses, belief_state.probabilities))[:top_n]
    summaries = []
    for hypothesis, probability in top_entries:
        source_text = "[" + ", ".join(_format_location(source) for source in hypothesis) + "]"
        summaries.append(f"p={_format_probability(probability)} sources={source_text}")
    suffix = "" if len(belief_state.hypotheses) <= top_n else f"; +{len(belief_state.hypotheses) - top_n} more"
    return f"{len(belief_state.hypotheses)} belief(s): " + "; ".join(summaries) + suffix


def _summarize_candidates(candidates: list[Location]) -> str:
    if not candidates:
        return "[]"
    return "[" + ", ".join(_format_location(candidate) for candidate in candidates) + "]"


def _format_source_array(sources: np.ndarray) -> str:
    return "[" + ", ".join(_format_location(tuple(float(value) for value in source)) for source in sources) + "]"


def _log_location(message: str, config: Config) -> None:
    print_and_log(f"[location] {message}", config)


def _belief_system_prompt(config: Config, *, update: bool) -> str:
    role = (
        "You maintain and refresh a finite Bayesian belief support for a 2D source-localization problem."
        if update
        else "You maintain a finite Bayesian belief support for a 2D source-localization problem."
    )
    return (
        f"{role}\n\n"
        "There are exactly 3 hidden signal sources. A source configuration is a set of 3 distinct 2D coordinates:\n"
        "[[x1,y1],[x2,y2],[x3,y3]]\n\n"
        "The source order is irrelevant. Two configurations that differ only by source order are the same hypothesis.\n\n"
        "Prior:\n"
        "Each source coordinate is independently drawn from Normal(0,1). Prior-plausible coordinates are usually "
        "near the origin, but the data can justify separated sources.\n\n"
        "Measurement model:\n"
        "A query is a 2D coordinate x = [x1,x2].\n"
        "The noiseless signal at x is:\n"
        "signal(x; theta) = b + sum_k alpha / (m + ||theta_k - x||^2)\n"
        "with b=0.1, alpha=1.0, m=0.0001.\n"
        f"The observed scalar signal is y ~ Normal(signal(x; theta), noise_sd={config.location_noise_sd}).\n\n"
        "Interpretation:\n"
        "- Very high observations indicate at least one source is probably close to the queried coordinate.\n"
        "- Low or moderate observations make it unlikely that any source is extremely close to the queried coordinate.\n"
        "- Because signals add, one observation may be explained by different source configurations.\n"
        "- Your job is only to propose source configurations. Deterministic code will compute likelihoods and "
        "posterior probabilities."
    )


def _belief_output_contract(config: Config) -> str:
    return (
        "Return only this exact compact JSON shape:\n"
        "{\"hypotheses\":[[[x1,y1],[x2,y2],[x3,y3]],...]}\n\n"
        "Rules:\n"
        "- The final character must be }.\n"
        "- Do not include <eos>, markdown, comments, explanations, or trailing text.\n"
        f"- Generate up to {config.location_max_llm_prompt_beliefs} source configurations.\n"
        f"- Each hypothesis must contain exactly {config.location_num_sources} distinct "
        f"{config.location_dim}D source coordinates.\n"
        "- Do not repeat the same hypothesis with sources in a different order.\n"
        "- Source coordinates are hidden source locations, not measurement/query locations."
    )


def _belief_generation_messages(
    observations: list[LocationObservation],
    belief_state: LocationBeliefState | None,
    config: Config,
) -> list[dict[str, str]]:
    is_initial = not observations and (belief_state is None or not belief_state.hypotheses)
    output_contract = _belief_output_contract(config)

    if is_initial:
        system = _belief_system_prompt(config, update=False)
        user = (
            "Observation history: []\n\n"
            f"{output_contract}\n"
            "- Generate diverse prior-plausible source configurations from Normal(0,1).\n"
            "- Include configurations with different spatial patterns: compact near-origin, separated sources, "
            "asymmetric layouts, and multiple possible signs/quadrants."
        )
    else:
        current_context = "[]"
        if belief_state is not None and belief_state.hypotheses:
            current_context = _format_weighted_hypotheses(
                belief_state,
                top_n=config.location_max_llm_prompt_beliefs,
            )
        system = _belief_system_prompt(config, update=True)
        user = (
            f"Observation history: {_format_observations(observations)}\n"
            f"Current weighted hypotheses: {current_context}\n\n"
            "Generate updated source configurations conditioned on the full observation history and current "
            "weighted hypotheses.\n\n"
            f"{output_contract}\n"
            "- Include refinements of high-probability current hypotheses.\n"
            "- Include alternatives that fix large mismatches implied by the observations.\n"
            "- Include alternatives near high-signal query locations.\n"
            "- Include alternatives that avoid low-signal query locations.\n"
            "- Keep diverse alternatives so the belief support does not collapse."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _candidate_generation_messages(
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = (
        "You propose candidate measurement locations for an adaptive 2D source-localization experiment.\n\n"
        "There are exactly 3 hidden signal sources. The goal is to choose the next query coordinate x = [x1,x2] "
        "to learn the source locations as efficiently as possible.\n\n"
        "Measurement model:\n"
        "The noiseless signal at query x is:\n"
        "signal(x; theta) = b + sum_k alpha / (m + ||theta_k - x||^2)\n"
        "with b=0.1, alpha=1.0, m=0.0001.\n"
        f"The observed scalar signal is y ~ Normal(signal(x; theta), noise_sd={config.location_noise_sd}).\n\n"
        "Allowed query coordinates:\n"
        f"Each coordinate must be in [{bounds[0]}, {bounds[1]}].\n\n"
        "Design objective:\n"
        "Propose locations that are informative about the unknown sources. Good candidates should distinguish "
        "between plausible source configurations, test uncertain regions, and refine suspected source locations. "
        "Early in an experiment, broader exploration is useful. After strong signals appear, nearby follow-up "
        "measurements can refine source positions.\n\n"
        "Do not output source configurations. Output only measurement/query locations.\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"locations\":[[x1,y1],[x1,y1],...]}\n\n"
        "Rules:\n"
        "- The final character must be }.\n"
        "- Do not include <eos>, markdown, comments, explanations, or trailing text.\n"
        f"- Generate exactly {config.location_target_num_candidates} candidate measurement locations.\n"
        "- Every coordinate must be within the allowed query bounds.\n"
        "- Do not repeat previous query locations.\n"
        "- Include diverse candidates, not tiny variations of the same point."
    )
    user = (
        f"Observation history:\n{_format_observations(observations)}\n\n"
        f"Current weighted source hypotheses:\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_max_llm_prompt_beliefs)}\n\n"
        "Generate candidate measurement locations that are discriminative among these hypotheses and useful for "
        "the remaining experiment."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_location_hypotheses(
    questioner: "Model",
    observations: list[LocationObservation],
    belief_state: LocationBeliefState | None,
    config: Config,
    *,
    label: str = "belief generation",
) -> list[SourceConfig]:
    previous_count = 0 if belief_state is None else len(belief_state.hypotheses)
    _log_location(
        f"{label}: requesting source hypotheses "
        f"(observations={len(observations)}, previous_beliefs={previous_count}, "
        f"max_return={config.location_max_llm_prompt_beliefs})",
        config,
    )
    completion = questioner.chat_complete(
        _belief_generation_messages(observations, belief_state, config),
        temperature=config.generation_temperature_diverse,
    )[0]
    try:
        hypotheses = parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)
    except ValueError as exc:
        _log_location(f"{label}: could not parse source hypotheses ({exc}); using empty generated support", config)
        return []
    _log_location(f"{label}: parsed {len(hypotheses)} valid unique source configuration(s)", config)
    return hypotheses


def _generate_location_hypotheses_many(
    questioner: "Model",
    observations_many: list[list[LocationObservation]],
    belief_states: list[LocationBeliefState | None],
    config: Config,
    *,
    label: str = "batched belief generation",
) -> list[list[SourceConfig]]:
    if len(observations_many) != len(belief_states):
        raise ValueError("observations_many and belief_states must have the same length")
    if not observations_many:
        return []

    _log_location(
        f"{label}: requesting {len(observations_many)} hypothetical source-support refresh(es) "
        f"(block_size={config.batched_block_size})",
        config,
    )
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
    parse_failures = 0
    for completion in completions:
        try:
            hypotheses_many.append(
                parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)
            )
        except ValueError:
            parse_failures += 1
            hypotheses_many.append([])
    counts = [len(hypotheses) for hypotheses in hypotheses_many]
    nonempty_count = sum(1 for count in counts if count > 0)
    total_count = sum(counts)
    _log_location(
        f"{label}: parsed {total_count} generated hypothesis/hypotheses across "
        f"{nonempty_count}/{len(hypotheses_many)} nonempty refresh(es)"
        + (f"; parse_failures={parse_failures}" if parse_failures else ""),
        config,
    )
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
    _log_location(
        f"candidate generation: requesting {config.location_target_num_candidates} location(s) "
        f"(observations={len(observations)}, beliefs={len(belief_state.hypotheses)}, "
        f"bounds=[{bounds[0]}, {bounds[1]}])",
        config,
    )
    completion = questioner.chat_complete(
        _candidate_generation_messages(belief_state, observations, config),
        temperature=config.generation_temperature_diverse,
    )[0]
    try:
        candidates = parse_candidate_locations(completion, config.location_dim, bounds)
    except ValueError:
        _log_location("candidate generation: could not parse JSON candidates; using deterministic fallbacks", config)
        candidates = []
    parsed_count = len(candidates)
    if len(candidates) < config.location_target_num_candidates:
        fallback_locations = [
            location
            for location in default_candidate_locations(config.location_dim, bounds)
            if location not in candidates
        ]
        candidates.extend(fallback_locations)
    selected = candidates[:config.location_target_num_candidates]
    _log_location(
        f"candidate generation: parsed={parsed_count}, returned={len(selected)}, "
        f"locations={_summarize_candidates(selected)}",
        config,
    )
    return selected


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
    return prune_location_beliefs(state, max_beliefs=config.location_max_total_beliefs)


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
) -> LocationBeliefState:
    if len(belief_state.hypotheses) <= max_beliefs:
        return belief_state

    ordered = sorted(
        zip(belief_state.hypotheses, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )[:max_beliefs]
    total = sum(probability for _hypothesis, probability in ordered)
    if total <= 0.0:
        probability = 1.0 / len(ordered)
        return LocationBeliefState([hypothesis for hypothesis, _probability in ordered], [probability] * len(ordered))
    return LocationBeliefState(
        [hypothesis for hypothesis, _probability in ordered],
        [float(probability / total) for _hypothesis, probability in ordered],
    )


def prompt_location_belief_state(
    belief_state: LocationBeliefState,
    config: Config,
) -> LocationBeliefState:
    return prune_location_beliefs(belief_state, max_beliefs=config.location_max_llm_prompt_beliefs)


def _location_effective_sample_size(belief_state: LocationBeliefState) -> float:
    if not belief_state.probabilities:
        return 0.0
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    denominator = float(np.sum(probabilities ** 2))
    if denominator <= 0.0:
        return 0.0
    return 1.0 / denominator


def sample_location_eig_belief_state(
    belief_state: LocationBeliefState,
    config: Config,
    rng: np.random.Generator,
) -> tuple[LocationBeliefState, bool]:
    if len(belief_state.hypotheses) <= config.num_mc_samples:
        return belief_state, False

    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    sampled_indices = rng.choice(
        len(belief_state.hypotheses),
        size=config.num_mc_samples,
        replace=True,
        p=probabilities,
    )
    unique_indices, counts = np.unique(sampled_indices, return_counts=True)
    sample_collapsed = len(unique_indices) < min(2, len(belief_state.hypotheses))

    sampled_hypotheses = [belief_state.hypotheses[int(index)] for index in unique_indices]
    sampled_probabilities = [float(count / np.sum(counts)) for count in counts]
    sampled_state = sort_location_belief_state(LocationBeliefState(sampled_hypotheses, sampled_probabilities))
    return sampled_state, sample_collapsed


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
        _log_location("EIG scoring: no candidate locations to score", config)
        return []

    _log_location(
        f"EIG scoring: scoring {len(candidates)} candidate(s) with "
        f"{len(belief_state.hypotheses)} belief(s), depth={config.location_search_depth}, "
        f"quadrature_order={config.location_eig_quadrature_order}",
        config,
    )
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
    immediate_summary = sorted(
        zip(candidates, immediate),
        key=lambda entry: entry[1],
        reverse=True,
    )[: min(5, len(candidates))]
    _log_location(
        "EIG scoring: immediate top candidates: "
        + "; ".join(f"{_format_location(candidate)}={score:.6f}" for candidate, score in immediate_summary),
        config,
    )
    if len(belief_state.hypotheses) <= 1:
        _log_location("EIG scoring: only one belief remains; future value is zero", config)
        return immediate
    if config.location_search_depth == 1 or len(candidates) == 0:
        _log_location("EIG scoring: depth=1, using immediate EIG only", config)
        return immediate
    if config.location_search_depth != 2:
        raise ValueError("location_search_depth must be 1 or 2")
    if questioner is None or observations is None:
        raise ValueError("location_search_depth=2 requires questioner and observations for full branch updates")

    totals = list(immediate)
    future_contributions = [0.0 for _candidate in candidates]
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

    _log_location(
        f"EIG scoring: depth=2 expanding {len(branch_observations)} hypothetical observation branch(es) "
        f"({len(candidates)} candidates x {len(belief_state.hypotheses)} beliefs x "
        f"{len(nodes)} quadrature nodes, excluding zero-probability beliefs)",
        config,
    )
    generated_hypotheses_many = _generate_location_hypotheses_many(
        questioner,
        branch_observations,
        [belief_state for _ in branch_observations],
        config,
        label="EIG scoring future beliefs",
    )
    branch_future_maxima: list[float] = []
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
            branch_best = max(future_values)
            branch_future_maxima.append(branch_best)
            contribution = branch_weight * branch_best
            future_contributions[candidate_idx] += contribution
            totals[candidate_idx] += contribution
    if branch_future_maxima:
        _log_location(
            f"EIG scoring: depth=2 future best EIG range="
            f"[{min(branch_future_maxima):.6f}, {max(branch_future_maxima):.6f}]",
            config,
        )
    final_summary = sorted(
        zip(candidates, immediate, future_contributions, totals),
        key=lambda entry: entry[3],
        reverse=True,
    )[: min(5, len(candidates))]
    _log_location(
        "EIG scoring: final top candidates: "
        + "; ".join(
            f"{_format_location(candidate)} total={total:.6f} "
            f"(immediate={immediate_score:.6f}, future={future_score:.6f})"
            for candidate, immediate_score, future_score, total in final_summary
        ),
        config,
    )
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


def _signal_grid(
    env: LocationFindingEnv,
    extent: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    resolution: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = extent
    x_values = np.linspace(x_min, x_max, resolution)
    y_values = np.linspace(y_min, y_max, resolution)
    grid = np.empty((resolution, resolution), dtype=float)
    for row_idx, y_value in enumerate(y_values):
        for col_idx, x_value in enumerate(x_values):
            grid[row_idx, col_idx] = env.signal_intensity((x_value, y_value))
    return x_values, y_values, grid


def _plot_location_trial(
    env: LocationFindingEnv,
    observations: list[LocationObservation],
    belief_state: LocationBeliefState,
    trial_idx: int,
    final_rmse: float,
    final_top_probability: float,
    output_path: Path,
) -> None:
    try:
        import matplotlib
    except ImportError:
        _plot_location_trial_pillow(
            env,
            observations,
            belief_state,
            trial_idx,
            final_rmse,
            final_top_probability,
            output_path,
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = (-3.0, 3.0, -3.0, 3.0)
    _x_values, _y_values, signal = _signal_grid(env, extent=extent)
    signal_vmax = float(np.nanpercentile(signal, 99.0))
    if not math.isfinite(signal_vmax) or signal_vmax <= 0.0:
        signal_vmax = float(np.nanmax(signal))

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = ax.imshow(
        signal,
        origin="lower",
        extent=extent,
        cmap="Purples",
        alpha=0.72,
        vmin=0.0,
        vmax=signal_vmax,
        interpolation="bilinear",
    )
    signal_cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    signal_cbar.set_label("Signal intensity")

    true_sources = np.asarray(env.true_theta, dtype=float)
    ax.scatter(
        true_sources[:, 0],
        true_sources[:, 1],
        marker="*",
        s=180,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        label="True sources",
        zorder=4,
    )

    if belief_state.hypotheses:
        top_sources = np.asarray(belief_state.hypotheses[0], dtype=float)
        ax.scatter(
            top_sources[:, 0],
            top_sources[:, 1],
            marker="x",
            s=90,
            c="#1f77b4",
            linewidths=2.0,
            label="Top belief",
            zorder=4,
        )

    if observations:
        queries = np.asarray([observation.query for observation in observations], dtype=float)
        order = np.arange(1, len(observations) + 1)
        query_scatter = ax.scatter(
            queries[:, 0],
            queries[:, 1],
            c=order,
            cmap="YlOrRd",
            s=70,
            edgecolors="#333333",
            linewidths=0.8,
            label="Queries",
            zorder=5,
        )
        order_cbar = fig.colorbar(query_scatter, ax=ax, fraction=0.046, pad=0.10)
        order_cbar.set_label("Experiment order")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(color="#d9e0e3", linewidth=0.8, alpha=0.7)
    ax.set_title(
        f"Location Finding trial {trial_idx + 1}: "
        f"RMSE={final_rmse:.3f}, top p={final_top_probability:.3f}"
    )
    ax.legend(loc="upper right", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extent_to_pixel(
    x_value: float,
    y_value: float,
    extent: tuple[float, float, float, float],
    plot_size: int,
) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = extent
    x_pixel = int(round((x_value - x_min) / (x_max - x_min) * (plot_size - 1)))
    y_pixel = int(round((y_max - y_value) / (y_max - y_min) * (plot_size - 1)))
    return x_pixel, y_pixel


def _order_color(index: int, total: int) -> tuple[int, int, int]:
    fraction = 0.0 if total <= 1 else index / (total - 1)
    start = np.asarray([230, 61, 38], dtype=float)
    end = np.asarray([255, 245, 140], dtype=float)
    color = start * (1.0 - fraction) + end * fraction
    return tuple(int(round(value)) for value in color)


def _plot_location_trial_pillow(
    env: LocationFindingEnv,
    observations: list[LocationObservation],
    belief_state: LocationBeliefState,
    trial_idx: int,
    final_rmse: float,
    final_top_probability: float,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    extent = (-3.0, 3.0, -3.0, 3.0)
    plot_size = 720
    right_margin = 190
    bottom_margin = 115
    top_margin = 45
    left_margin = 60
    _x_values, _y_values, signal = _signal_grid(env, extent=extent, resolution=plot_size)
    signal_vmax = float(np.nanpercentile(signal, 99.0))
    if not math.isfinite(signal_vmax) or signal_vmax <= 0.0:
        signal_vmax = float(np.nanmax(signal))
    normalized = np.clip(signal / max(signal_vmax, 1e-12), 0.0, 1.0)
    low = np.asarray([238, 244, 246], dtype=float)
    high = np.asarray([65, 54, 160], dtype=float)
    rgb = (low[None, None, :] * (1.0 - normalized[:, :, None]) + high[None, None, :] * normalized[:, :, None])
    heatmap = Image.fromarray(np.flipud(rgb.astype(np.uint8)), mode="RGB")

    canvas = Image.new("RGB", (left_margin + plot_size + right_margin, top_margin + plot_size + bottom_margin), "white")
    canvas.paste(heatmap, (left_margin, top_margin))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for tick in range(-3, 4):
        x_pixel, _ = _extent_to_pixel(tick, 0.0, extent, plot_size)
        _, y_pixel = _extent_to_pixel(0.0, tick, extent, plot_size)
        x_abs = left_margin + x_pixel
        y_abs = top_margin + y_pixel
        draw.line((x_abs, top_margin, x_abs, top_margin + plot_size), fill=(218, 226, 230), width=1)
        draw.line((left_margin, y_abs, left_margin + plot_size, y_abs), fill=(218, 226, 230), width=1)
        draw.text((x_abs - 5, top_margin + plot_size + 8), str(tick), fill=(60, 60, 60), font=font)
        draw.text((left_margin - 28, y_abs - 6), str(tick), fill=(60, 60, 60), font=font)

    draw.rectangle(
        (left_margin, top_margin, left_margin + plot_size, top_margin + plot_size),
        outline=(45, 45, 45),
        width=2,
    )

    true_sources = np.asarray(env.true_theta, dtype=float)
    for x_value, y_value in true_sources:
        x_pixel, y_pixel = _extent_to_pixel(float(x_value), float(y_value), extent, plot_size)
        x_abs = left_margin + x_pixel
        y_abs = top_margin + y_pixel
        draw.line((x_abs - 9, y_abs, x_abs + 9, y_abs), fill="black", width=3)
        draw.line((x_abs, y_abs - 9, x_abs, y_abs + 9), fill="black", width=3)
        draw.line((x_abs - 6, y_abs - 6, x_abs + 6, y_abs + 6), fill="black", width=2)
        draw.line((x_abs - 6, y_abs + 6, x_abs + 6, y_abs - 6), fill="black", width=2)

    if belief_state.hypotheses:
        for x_value, y_value in np.asarray(belief_state.hypotheses[0], dtype=float):
            x_pixel, y_pixel = _extent_to_pixel(float(x_value), float(y_value), extent, plot_size)
            x_abs = left_margin + x_pixel
            y_abs = top_margin + y_pixel
            draw.line((x_abs - 9, y_abs - 9, x_abs + 9, y_abs + 9), fill=(31, 119, 180), width=3)
            draw.line((x_abs - 9, y_abs + 9, x_abs + 9, y_abs - 9), fill=(31, 119, 180), width=3)

    for observation_idx, observation in enumerate(observations):
        x_pixel, y_pixel = _extent_to_pixel(observation.query[0], observation.query[1], extent, plot_size)
        x_abs = left_margin + x_pixel
        y_abs = top_margin + y_pixel
        color = _order_color(observation_idx, len(observations))
        radius = 8
        draw.ellipse((x_abs - radius, y_abs - radius, x_abs + radius, y_abs + radius), fill=color, outline=(40, 40, 40), width=2)

    title = f"Location Finding trial {trial_idx + 1}: RMSE={final_rmse:.3f}, top p={final_top_probability:.3f}"
    draw.text((left_margin, 15), title, fill=(20, 20, 20), font=font)
    legend_x = left_margin + plot_size + 20
    legend_y = top_margin + 20
    draw.text((legend_x, legend_y), "true sources: black", fill=(20, 20, 20), font=font)
    draw.text((legend_x, legend_y + 18), "top belief: blue x", fill=(31, 119, 180), font=font)
    draw.text((legend_x, legend_y + 36), "queries: order color", fill=(20, 20, 20), font=font)

    bar_x = left_margin + plot_size + 35
    bar_y = top_margin + 95
    bar_width = 28
    bar_height = 180
    for offset in range(bar_height):
        frac = 1.0 - offset / max(bar_height - 1, 1)
        color = tuple(int(round(value)) for value in (low * (1.0 - frac) + high * frac))
        draw.line((bar_x, bar_y + offset, bar_x + bar_width, bar_y + offset), fill=color)
    draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height), outline=(80, 80, 80))
    draw.text((bar_x - 5, bar_y + bar_height + 8), "Signal", fill=(20, 20, 20), font=font)

    order_y = top_margin + plot_size + 45
    order_x = left_margin + 125
    order_width = 260
    order_height = 22
    for offset in range(order_width):
        color = _order_color(offset, order_width)
        draw.line((order_x + offset, order_y, order_x + offset, order_y + order_height), fill=color)
    draw.rectangle((order_x, order_y, order_x + order_width, order_y + order_height), outline=(80, 80, 80))
    draw.text((order_x - 95, order_y + 3), "Experiment order", fill=(20, 20, 20), font=font)
    if observations:
        draw.text((order_x - 5, order_y + order_height + 6), "1", fill=(20, 20, 20), font=font)
        draw.text((order_x + order_width - 14, order_y + order_height + 6), str(len(observations)), fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_to_log_if_configured(message: str, config: Config) -> None:
    if config.log_path is not None:
        write_to_log(message, config)


def run_location_finding(
    questioner: "Model",
    config: Config,
    rng: np.random.Generator | None = None,
    output_dir: Path | None = None,
) -> LocationFindingMetrics:
    if config.location_num_sources != 3 or config.location_dim != 2:
        raise ValueError("The initial Location Finding implementation supports exactly 3 sources in 2D")
    if config.location_noise_sd != 0.5:
        raise ValueError("The initial Location Finding implementation requires known noise_sd=0.5")

    rng = rng or np.random.default_rng()
    rmse_totals = np.zeros(config.location_num_rounds, dtype=float)
    top_probability_totals = np.zeros(config.location_num_rounds, dtype=float)
    selected_eig_totals = np.zeros(config.location_num_rounds, dtype=float)

    _log_location(
        f"Running {config.location_num_trials} Location Finding trial(s): "
        f"rounds={config.location_num_rounds}, sources={config.location_num_sources}, "
        f"dim={config.location_dim}, noise_sd={config.location_noise_sd}, "
        f"candidates={config.location_target_num_candidates}, depth={config.location_search_depth}, "
        f"quadrature_order={config.location_eig_quadrature_order}, "
        f"max_total_beliefs={config.location_max_total_beliefs}, "
        f"max_llm_prompt_beliefs={config.location_max_llm_prompt_beliefs}, "
        f"num_mc_samples={config.num_mc_samples}",
        config,
    )
    for trial_idx in range(config.location_num_trials):
        env = LocationFindingEnv(
            num_sources=config.location_num_sources,
            dim=config.location_dim,
            noise_sd=config.location_noise_sd,
            rng=rng,
        )
        observations: list[LocationObservation] = []
        _log_location(
            f"trial {trial_idx + 1}/{config.location_num_trials}: sampled hidden environment "
            f"with true_sources={_format_source_array(env.true_theta)}",
            config,
        )
        initial_hypotheses = generate_location_hypotheses(
            questioner,
            observations,
            None,
            config,
            label=f"trial {trial_idx + 1} initial belief generation",
        )
        if not initial_hypotheses:
            _log_location("No valid initial LLM hypotheses; using deterministic fallback support", config)
            initial_hypotheses = _default_source_hypotheses(config)
        belief_state = build_location_belief_state(initial_hypotheses, observations, config)
        prompt_belief_state = prompt_location_belief_state(belief_state, config)
        _log_location(
            f"trial {trial_idx + 1}: initial posterior {_summarize_belief_state(belief_state)}; "
            f"reservoir={len(belief_state.hypotheses)}, "
            f"prompt={len(prompt_belief_state.hypotheses)}, "
            f"ESS={_location_effective_sample_size(belief_state):.2f}",
            config,
        )

        for round_idx in range(config.location_num_rounds):
            prompt_belief_state = prompt_location_belief_state(belief_state, config)
            eig_belief_state, eig_sample_collapsed = sample_location_eig_belief_state(belief_state, config, rng)
            _write_to_log_if_configured(f"\nLocation Finding trial {trial_idx + 1}: Round {round_idx + 1}\n", config)
            _log_location(
                f"trial {trial_idx + 1}/{config.location_num_trials}, "
                f"round {round_idx + 1}/{config.location_num_rounds}, "
                f"posterior {_summarize_belief_state(belief_state)}; "
                f"reservoir={len(belief_state.hypotheses)}, "
                f"prompt={len(prompt_belief_state.hypotheses)}, "
                f"EIG_support={len(eig_belief_state.hypotheses)}, "
                f"ESS={_location_effective_sample_size(belief_state):.2f}",
                config,
            )
            if eig_sample_collapsed:
                _log_location("EIG posterior sampling produced one unique hypothesis; EIG support is collapsed", config)
            candidates = generate_location_candidates(questioner, prompt_belief_state, observations, config)
            scores = score_candidate_locations(
                eig_belief_state,
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

            generated_hypotheses = generate_location_hypotheses(
                questioner,
                observations,
                prompt_belief_state,
                config,
                label=f"trial {trial_idx + 1} round {round_idx + 1} belief update",
            )
            merged_hypotheses = _merge_hypotheses(belief_state, generated_hypotheses)
            reservoir_before_trim = len(merged_hypotheses)
            _log_location(
                f"round {round_idx + 1}: reservoir update merging previous={len(belief_state.hypotheses)} "
                f"with generated={len(generated_hypotheses)} -> unique={len(merged_hypotheses)}",
                config,
            )
            belief_state = build_location_belief_state(merged_hypotheses, observations, config)
            if reservoir_before_trim > len(belief_state.hypotheses):
                _log_location(
                    f"round {round_idx + 1}: reservoir trimmed {reservoir_before_trim} -> "
                    f"{len(belief_state.hypotheses)} by top posterior",
                    config,
                )
            prompt_belief_state = prompt_location_belief_state(belief_state, config)
            _log_location(
                f"round {round_idx + 1}: posterior after observation {_summarize_belief_state(belief_state)}; "
                f"reservoir={len(belief_state.hypotheses)}, "
                f"prompt={len(prompt_belief_state.hypotheses)}, "
                f"ESS={_location_effective_sample_size(belief_state):.2f}",
                config,
            )
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

        if config.location_plot_trials:
            if output_dir is None:
                _log_location("plotting requested but no output directory was provided; skipping trial plot", config)
            else:
                final_rmse = _top_source_rmse(belief_state, env.true_theta)
                final_top_probability = belief_state.probabilities[0] if belief_state.probabilities else 0.0
                plot_path = output_dir / f"location_trial_{trial_idx + 1:03d}.png"
                _plot_location_trial(
                    env,
                    observations,
                    belief_state,
                    trial_idx,
                    final_rmse,
                    final_top_probability,
                    plot_path,
                )
                _log_location(f"saved trial plot to {plot_path}", config)

    divisor = float(config.location_num_trials)
    return LocationFindingMetrics(
        source_rmse=(rmse_totals / divisor).tolist(),
        top_probability=(top_probability_totals / divisor).tolist(),
        selected_eig=(selected_eig_totals / divisor).tolist(),
    )
