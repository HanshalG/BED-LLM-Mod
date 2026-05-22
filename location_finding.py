from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from helpers import Config, _average_labeled_distributions_from_completions, print_and_log, write_to_log

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


@dataclass
class _LocationTrialState:
    trial_idx: int
    env: "LocationFindingEnv"
    observations: list[LocationObservation]
    rng: np.random.Generator
    belief_state: LocationBeliefState | None = None
    strategy_library: "LocationStrategyLibrary | None" = None
    final_estimate: SourceConfig | None = None
    final_rmse: float = float("inf")


@dataclass(frozen=True)
class LocationStrategyEntry:
    strategy: str
    mean_score: float
    score_variance: float
    root_query_fingerprint: str
    round_index: int
    root_query: Location | None = None


class LocationStrategyLibrary:
    def __init__(self) -> None:
        self.entries: list[LocationStrategyEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve_top_m(self, count: int) -> list[LocationStrategyEntry]:
        if count <= 0:
            return []
        ranked = sorted(
            self.entries,
            key=lambda entry: (entry.mean_score, -entry.score_variance),
            reverse=True,
        )
        return ranked[:count]

    def replace_entries(self, entries: list[LocationStrategyEntry]) -> None:
        """Replace all entries with the given list (keeps only the latest round)."""
        self.entries = list(entries)


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


def _extract_json_values(text: str) -> list[object]:
    cleaned = _clean_json_completion(text)
    decoder = json.JSONDecoder()
    values: list[object] = []
    for idx, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _end_idx = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        values.append(payload)
    return values


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
        for key in ("locations", "candidates", "queries", "points", "location"):
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


def parse_single_location(completion: str, dim: int, bounds: tuple[float, float]) -> Location:
    payload = _loads_json_value(completion)
    if isinstance(payload, dict):
        for key in ("location", "query", "point"):
            if key in payload:
                payload = payload[key]
                break
    location = normalize_location(payload, dim)
    low, high = bounds
    if any(value < low or value > high for value in location):
        raise ValueError("Location is outside query bounds")
    return location


def parse_single_location_from_completion(completion: str, dim: int, bounds: tuple[float, float]) -> Location:
    low, high = bounds
    for payload in reversed(_extract_json_values(completion)):
        raw_location: object = payload
        if isinstance(payload, dict):
            matched = False
            for key in ("location", "query", "point"):
                if key in payload:
                    raw_location = payload[key]
                    matched = True
                    break
            if not matched:
                continue
        try:
            location = normalize_location(raw_location, dim)
        except ValueError:
            continue
        if any(value < low or value > high for value in location):
            continue
        return location
    try:
        return parse_single_location(completion, dim, bounds)
    except ValueError:
        pass
    raise ValueError("No valid location JSON found in completion")


def parse_best_source_estimate_from_completion(completion: str, num_sources: int, dim: int) -> SourceConfig:
    for payload in reversed(_extract_json_values(completion)):
        configs = _collect_source_configs_from_payload(payload, num_sources, dim)
        if configs:
            return configs[-1]
    try:
        estimates = parse_source_hypotheses(completion, num_sources, dim)
    except ValueError:
        estimates = []
    if estimates:
        return estimates[-1]
    raise ValueError("No valid source estimate JSON found in completion")


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


def _location_posterior_labels(count: int) -> list[str]:
    return [f"h{idx}" for idx in range(count)]


def _source_count_text(count: int) -> str:
    return "1 hidden signal source" if count == 1 else f"{count} hidden signal sources"


def _source_config_schema_example(num_sources: int, dim: int) -> str:
    source_examples: list[str] = []
    for source_idx in range(1, num_sources + 1):
        if dim == 2:
            source_examples.append(f"[x{source_idx},y{source_idx}]")
        else:
            source_examples.append(
                "[" + ",".join(f"x{source_idx}_{coord_idx}" for coord_idx in range(1, dim + 1)) + "]"
            )
    return "[" + ",".join(source_examples) + "]"


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
        f"There are exactly {_source_count_text(config.location_num_sources)}. A source configuration is a set of "
        f"{config.location_num_sources} distinct {config.location_dim}D coordinates:\n"
        f"{_source_config_schema_example(config.location_num_sources, config.location_dim)}\n\n"
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
        f"{{\"hypotheses\":[{_source_config_schema_example(config.location_num_sources, config.location_dim)},...]}}\n\n"
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
            "- Include alternatives consistent with low-signal observations (no source extremely close to those queries).\n"
            "- Keep diverse alternatives so the belief support does not collapse."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _location_posterior_context_probabilities(
    hypotheses: list[SourceConfig],
    context_state: LocationBeliefState | None,
) -> list[float]:
    if not hypotheses:
        return []

    context_lookup: dict[SourceConfig, float] = {}
    if context_state is not None:
        context_lookup = {
            hypothesis: float(probability)
            for hypothesis, probability in zip(context_state.hypotheses, context_state.probabilities)
        }

    prior_log_scores = [_hypothesis_log_prior(hypothesis) for hypothesis in hypotheses]
    prior_normalizer = _logsumexp(prior_log_scores)
    prior_probabilities = [
        math.exp(log_score - prior_normalizer)
        for log_score in prior_log_scores
    ]
    weights = [
        max(context_lookup.get(hypothesis, prior_probability), 0.0)
        for hypothesis, prior_probability in zip(hypotheses, prior_probabilities)
    ]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / len(hypotheses)] * len(hypotheses)
    return [float(weight / total) for weight in weights]


def _location_posterior_distribution_messages(
    observations: list[LocationObservation],
    hypotheses: list[SourceConfig],
    context_probabilities: list[float],
    config: Config,
) -> list[dict[str, str]]:
    labels = _location_posterior_labels(len(hypotheses))
    hypothesis_rows = [
        {
            "id": label,
            "sources": [list(source) for source in hypothesis],
            "context_probability": probability,
        }
        for label, hypothesis, probability in zip(labels, hypotheses, context_probabilities)
    ]
    system = (
        "You estimate a posterior probability distribution over a finite support for a 2D "
        "source-localization problem.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. Each candidate hypothesis is a "
        f"set of {config.location_num_sources} distinct {config.location_dim}D source coordinates. "
        "The source order is irrelevant.\n\n"
        "Measurement model:\n"
        "A query is a 2D coordinate x = [x1,x2]. The noiseless signal is "
        "b + sum_k alpha / (m + ||theta_k - x||^2), with b=0.1, alpha=1.0, m=0.0001. "
        f"The observed scalar signal is y ~ Normal(signal(x; theta), noise_sd={config.location_noise_sd}).\n"
        "Interpretation: a very high observation near a coordinate means at least one source is probably very close "
        "to that coordinate; a low observation rules out any source being very close to that query location.\n\n"
        "context_probability is each hypothesis's prior weight from before the current observations; treat it as "
        "your starting point and update it based on how well each hypothesis explains the observation history. "
        "Assign posterior mass across only the listed candidate hypothesis ids. "
        "Do not invent new ids or source configurations. Prefer a sparse posterior: "
        "include only ids with meaningful positive mass and omit ids that are inconsistent with the observations."
    )
    user = (
        f"Observation history: {_format_observations(observations)}\n\n"
        f"Candidate source hypotheses: {json.dumps(hypothesis_rows)}\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"weights\":{\"h0\":w0,\"h7\":w7}}\n\n"
        "Rules:\n"
        "- The final character must be }.\n"
        "- Do not include <eos>, markdown, comments, explanations, or trailing text.\n"
        "- Weights do not need to sum to 1; deterministic code will normalize them.\n"
        "- Omit ids with zero or negligible weight instead of writing many zero entries.\n"
        "- Use only ids from the candidate list."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _permuted_location_observation_histories(
    observations: list[LocationObservation],
    num_samples: int,
) -> list[list[LocationObservation]]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if len(observations) <= 1:
        return [list(observations) for _ in range(num_samples)]
    histories: list[list[LocationObservation]] = []
    for _sample_idx in range(num_samples):
        permutation = np.random.permutation(len(observations))
        histories.append([observations[int(index)] for index in permutation])
    return histories


def _candidate_generation_messages(
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = (
        "You propose candidate measurement locations for an adaptive 2D source-localization experiment.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. "
        "The goal is to choose the next query coordinate x = [x1,x2] "
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
        "- Spread candidates across different regions and hypotheses — include locations that discriminate between "
        "competing hypotheses, not just refinements of the single most likely one."
    )
    user = (
        f"Observation history:\n{_format_observations(observations)}\n\n"
        f"Current weighted source hypotheses:\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_max_llm_prompt_beliefs)}\n\n"
        "Generate candidate measurement locations that are discriminative among these hypotheses and useful for "
        "the remaining experiment."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _naive_location_messages(
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = (
        "You choose the next measurement location for a 2D source-localization experiment.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. "
        "The hidden sources are fixed but unknown. A query is a 2D coordinate x = [x1,x2].\n\n"
        "Measurement model:\n"
        "The noiseless signal at query x is:\n"
        "signal(x; theta) = b + sum_k alpha / (m + ||theta_k - x||^2)\n"
        "with b=0.1, alpha=1.0, m=0.0001. The observed scalar signal is "
        f"Normal(signal(x; theta), noise_sd={config.location_noise_sd}).\n\n"
        f"Allowed query coordinates: each coordinate must be in [{bounds[0]}, {bounds[1]}].\n\n"
        "Use only the task description and the previous query/observation history.\n\n"
        "Return only this exact compact JSON shape as the final answer:\n"
        "{\"location\":[x1,y1]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        "- Output a measurement/query location only — do not output source hypotheses or source coordinates.\n"
        "- Do not repeat a previous query location.\n"
        "- If you reason internally, still end with exactly one JSON object in the required shape."
    )
    user = (
        f"Observation history:\n{_format_observations(observations)}\n\n"
        "Generate the single best next measurement location to help localize the hidden sources."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _naive_source_estimate_messages(
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = (
        "You output a JSON source-location estimate for a 2D source-localization game.\n"
        "Return only compact JSON in the required format. Do not include markdown or explanations outside the JSON.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. "
        "The hidden sources are fixed but unknown. A query is a 2D coordinate x = [x1,x2]. "
        "Use only the observation history below.\n\n"
        "Measurement model:\n"
        "signal(x; theta) = b + sum_k alpha / (m + ||theta_k - x||^2)\n"
        f"with b=0.1, alpha=1.0, m=0.0001, noise_sd={config.location_noise_sd}.\n\n"
        f"Source coordinates are expected to lie in approximately [{bounds[0]}, {bounds[1]}] per coordinate.\n\n"
        "Output exactly this compact JSON shape as the final answer:\n"
        f"{{\"sources\":{_source_config_schema_example(config.location_num_sources, config.location_dim)}}}\n"
    )
    user = (
        f"Observation history: {_format_observations(observations)}\n\n"
        "Reasoning guidance: for each high-signal observation, at least one source is likely near that query "
        "location; for low-signal observations, no source is very close to that query. Signals from multiple "
        "sources add, so a moderate signal may reflect contributions from several sources at moderate distances.\n\n"
        "Return the single best estimate now as JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _naive_source_estimate_repair_messages(
    completion: str,
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    system = (
        "Convert the previous answer into valid compact JSON only. "
        "Do not explain, do not use markdown, and do not include any text outside the JSON object."
    )
    user = (
        f"Observation history: {_format_observations(observations)}\n\n"
        f"Previous answer:\n{completion[-4000:]}\n\n"
        "Return exactly this shape with your best current source-location estimate:\n"
        f"{{\"sources\":{_source_config_schema_example(config.location_num_sources, config.location_dim)}}}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



@dataclass(frozen=True)
class _StrategyLocationRequest:
    strategy: str
    belief_state: LocationBeliefState
    observations: list[LocationObservation]


@dataclass(frozen=True)
class LocationStrategyCandidate:
    strategy: str
    root_query: Location | None = None


@dataclass
class _StrategyRollout:
    request_index: int
    strategy_index: int
    strategy: str
    truth: SourceConfig
    start_probability: float
    start_belief_state: LocationBeliefState
    belief_state: LocationBeliefState
    particle_support: list[SourceConfig] = field(default_factory=list)
    final_generated_hypotheses: list[SourceConfig] = field(default_factory=list)
    final_scoring_belief_state: LocationBeliefState | None = None
    simulated_observations: list[LocationObservation] = field(default_factory=list)
    simulated_belief_states: list[LocationBeliefState] = field(default_factory=list)
    root_query: Location | None = None


@dataclass(frozen=True)
class _StrategyEvaluationRequest:
    strategies: list[str]
    belief_state: LocationBeliefState
    observations: list[LocationObservation]
    rng: np.random.Generator
    root_queries: list[Location | None] | None = None


@dataclass(frozen=True)
class LocationStrategyEvaluation:
    strategy: str
    mean_score: float
    score_variance: float
    root_query_fingerprint: str
    rollout_scores: list[float]
    root_query: Location | None = None


_STRATEGY_MIN_LENGTH = 5


def _clean_strategy_text(raw_strategy: object) -> str | None:
    if not isinstance(raw_strategy, str):
        return None
    cleaned = re.sub(r"\s+", " ", raw_strategy.strip())
    cleaned = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", cleaned).strip()
    if len(cleaned) < _STRATEGY_MIN_LENGTH:
        return None
    return cleaned


def _collect_strategy_texts(payload: object) -> list[str]:
    if isinstance(payload, str):
        cleaned = _clean_strategy_text(payload)
        return [] if cleaned is None else [cleaned]
    if isinstance(payload, list):
        strategies: list[str] = []
        for item in payload:
            strategies.extend(_collect_strategy_texts(item))
        return strategies
    if isinstance(payload, dict):
        for key in ("strategies", "plans", "candidates"):
            if key in payload:
                return _collect_strategy_texts(payload[key])
        if "strategy" in payload:
            return _collect_strategy_texts(payload["strategy"])
        strategies = []
        for value in payload.values():
            strategies.extend(_collect_strategy_texts(value))
        return strategies
    return []


def _collect_strategy_root_candidates(
    payload: object,
    dim: int,
    bounds: tuple[float, float],
) -> list[LocationStrategyCandidate]:
    candidates: list[LocationStrategyCandidate] = []
    if isinstance(payload, list):
        for item in payload:
            candidates.extend(_collect_strategy_root_candidates(item, dim, bounds))
        return candidates
    if not isinstance(payload, dict):
        return candidates

    for key in ("strategies", "plans", "candidates"):
        if key in payload:
            return _collect_strategy_root_candidates(payload[key], dim, bounds)

    raw_strategy = payload.get("strategy")
    strategy = _clean_strategy_text(raw_strategy)
    if strategy is None:
        for value in payload.values():
            candidates.extend(_collect_strategy_root_candidates(value, dim, bounds))
        return candidates

    raw_root = None
    for key in ("root_query", "root_location", "first_query", "first_location", "query", "location"):
        if key in payload:
            raw_root = payload[key]
            break
    if raw_root is None:
        return candidates

    try:
        root_query = normalize_location(raw_root, dim)
    except ValueError:
        return candidates
    low, high = bounds
    if any(value < low or value > high for value in root_query):
        return candidates
    candidates.append(LocationStrategyCandidate(strategy=strategy, root_query=root_query))
    return candidates


def parse_location_strategies(completion: str) -> list[str]:
    payload = _loads_json_value(completion)
    strategies: list[str] = []
    seen: set[str] = set()
    for strategy in _collect_strategy_texts(payload):
        key = strategy.lower()
        if key in seen:
            continue
        seen.add(key)
        strategies.append(strategy)
    return strategies


def parse_location_strategy_roots(
    completion: str,
    dim: int,
    bounds: tuple[float, float],
) -> list[LocationStrategyCandidate]:
    payload = _loads_json_value(completion)
    candidates: list[LocationStrategyCandidate] = []
    seen: set[str] = set()
    for candidate in _collect_strategy_root_candidates(payload, dim, bounds):
        if candidate.root_query is None:
            continue
        key = _strategy_key(candidate.strategy)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def parse_strategy_location(completion: str, dim: int, bounds: tuple[float, float]) -> Location:
    payload = _loads_json_value(completion)
    raw_location = payload
    if isinstance(payload, dict):
        for key in ("location", "query", "point"):
            if key in payload:
                raw_location = payload[key]
                break
        else:
            candidate_locations = parse_candidate_locations(completion, dim, bounds)
            if candidate_locations:
                return candidate_locations[0]
            raise ValueError("Strategy location completion must include location, query, or point")

    location = normalize_location(raw_location, dim)
    low, high = bounds
    if any(value < low or value > high for value in location):
        raise ValueError("Strategy location is outside allowed query bounds")
    return location


def _format_strategy_entries(entries: list[LocationStrategyEntry]) -> str:
    rows = [
        {
            "strategy": entry.strategy,
            "mean_score": entry.mean_score,
            "score_variance": entry.score_variance,
            "root_query": entry.root_query_fingerprint,
            "round_index": entry.round_index,
        }
        for entry in entries
    ]
    return json.dumps(rows)


def _strategy_system_preamble(bounds: tuple[float, ...], num_strategies: int, task_instruction: str) -> str:
    return (
        "You propose natural-language adaptive strategies for a 2D source-localization experiment.\n\n"
        "A strategy is a few-sentence high-level plan for choosing future measurement locations. It should describe "
        "how to adapt after high, low, or ambiguous signal observations, not just name one coordinate.\n\n"
        "Measurement model:\n"
        "The noiseless signal at query x is: signal(x; theta) = b + sum_k alpha / (m + ||theta_k - x||^2), "
        "with b=0.1, alpha=1.0, m=0.0001. Noise is Gaussian.\n"
        "Interpretation: a very high observation means at least one source is probably very close to that query; "
        "a low observation rules out any source being very close to that query; signals from multiple sources add.\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"strategies\":[\"strategy text\",...]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        f"- Generate exactly {num_strategies} strategies.\n"
        f"- {task_instruction}\n"
        "- The strategies must differ substantively from one another.\n"
        f"- Every strategy must respect query bounds [{bounds[0]}, {bounds[1]}] for each coordinate."
    )


def _strategy_mutation_messages(
    retrieved_entries: list[LocationStrategyEntry],
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_mutation: int,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = _strategy_system_preamble(
        bounds, num_mutation,
        "Generate good perturbations of the retrieved strategies. Do not copy any retrieved strategy verbatim.",
    )
    user = (
        f"Observation history so far:\n{_format_observations(observations)}\n\n"
        f"Current belief summary (top {config.location_strategy_belief_summary_top_k} hypotheses with probabilities):\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_strategy_belief_summary_top_k)}\n\n"
        f"Retrieved elite strategies to perturb:\n{_format_strategy_entries(retrieved_entries)}\n\n"
        f"Generate {num_mutation} good perturbation(s) of the above strategies."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strategy_crossover_messages(
    retrieved_entries: list[LocationStrategyEntry],
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_crossover: int,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = _strategy_system_preamble(
        bounds, num_crossover,
        "Generate good crossovers of the retrieved strategies. "
        "Each result must be meaningfully different from any individual retrieved strategy.",
    )
    user = (
        f"Observation history so far:\n{_format_observations(observations)}\n\n"
        f"Current belief summary (top {config.location_strategy_belief_summary_top_k} hypotheses with probabilities):\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_strategy_belief_summary_top_k)}\n\n"
        f"Retrieved elite strategies to combine:\n{_format_strategy_entries(retrieved_entries)}\n\n"
        f"Generate {num_crossover} hybrid strategies that combine the best elements of the retrieved strategies."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strategy_diverse_messages(
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_diverse: int,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = _strategy_system_preamble(
        bounds, num_diverse,
        "Make their likely first measurement locations or first decision criteria different, "
        "so the options do not collapse to the same first move.",
    )
    user = (
        f"Observation history so far:\n{_format_observations(observations)}\n\n"
        f"Current belief summary (top {config.location_strategy_belief_summary_top_k} hypotheses with probabilities):\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_strategy_belief_summary_top_k)}\n\n"
        "Generate diverse strategies useful for the current posterior."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strategy_root_system_preamble(bounds: tuple[float, ...], num_strategies: int, task_instruction: str) -> str:
    return (
        "You propose adaptive strategies for a 2D source-localization experiment. Each strategy must include a fixed "
        "root measurement location that will be asked first whenever that strategy is evaluated or selected.\n\n"
        "A strategy is a few-sentence high-level plan for choosing future measurement locations after the fixed root "
        "measurement. The root_query is the first concrete query that commits the strategy to a distinctive opening "
        "measurement.\n\n"
        "Measurement model:\n"
        "The noiseless signal at query x is: signal(x; theta) = b + sum_k alpha / (m + ||theta_k - x||^2), "
        "with b=0.1, alpha=1.0, m=0.0001. Noise is Gaussian.\n"
        "Interpretation: a very high observation means at least one source is probably very close to that query; "
        "a low observation rules out any source being very close to that query; signals from multiple sources add.\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"strategies\":[{\"strategy\":\"strategy text\",\"root_query\":[x1,y1]},...]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        f"- Generate exactly {num_strategies} strategy/root_query pairs.\n"
        f"- {task_instruction}\n"
        f"- Every root_query coordinate must be in [{bounds[0]}, {bounds[1]}].\n"
        "- Do not repeat a previous query location."
    )


def _strategy_root_mutation_messages(
    retrieved_entries: list[LocationStrategyEntry],
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_mutation: int,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = _strategy_root_system_preamble(
        bounds, num_mutation,
        "Generate good perturbations of the retrieved strategies. Do not copy any retrieved strategy/root_query verbatim.",
    )
    user = (
        f"Observation history so far:\n{_format_observations(observations)}\n\n"
        f"Current belief summary (top {config.location_strategy_belief_summary_top_k} hypotheses with probabilities):\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_strategy_belief_summary_top_k)}\n\n"
        f"Retrieved elite strategies to perturb:\n{_format_strategy_entries(retrieved_entries)}\n\n"
        f"Generate {num_mutation} good perturbation(s) of the above strategy/root_query pairs."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strategy_root_crossover_messages(
    retrieved_entries: list[LocationStrategyEntry],
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_crossover: int,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = _strategy_root_system_preamble(
        bounds, num_crossover,
        "Generate good crossovers of the retrieved strategies. "
        "Each result must be meaningfully different from any individual retrieved strategy.",
    )
    user = (
        f"Observation history so far:\n{_format_observations(observations)}\n\n"
        f"Current belief summary (top {config.location_strategy_belief_summary_top_k} hypotheses with probabilities):\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_strategy_belief_summary_top_k)}\n\n"
        f"Retrieved elite strategies to combine:\n{_format_strategy_entries(retrieved_entries)}\n\n"
        f"Generate {num_crossover} hybrid strategy/root_query pairs that combine the best elements of the retrieved strategies."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strategy_root_diverse_messages(
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_diverse: int,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = _strategy_root_system_preamble(
        bounds, num_diverse,
        "The strategies and root_query locations must differ substantively from one another.",
    )
    user = (
        f"Observation history so far:\n{_format_observations(observations)}\n\n"
        f"Current belief summary (top {config.location_strategy_belief_summary_top_k} hypotheses with probabilities):\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_strategy_belief_summary_top_k)}\n\n"
        "Generate diverse strategy/root_query pairs useful for the current posterior."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strategy_location_messages(
    strategy: str,
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    bounds = tuple(config.location_query_bounds)
    system = (
        "You choose the next measurement location for a 2D source-localization experiment by following a supplied "
        "natural-language strategy.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. "
        "A query is a 2D coordinate x = [x1,x2]. The noiseless signal is "
        "b + sum_k alpha / (m + ||theta_k - x||^2), with b=0.1, alpha=1.0, m=0.0001. The observed scalar signal is "
        f"Normal(signal(x; theta), noise_sd={config.location_noise_sd}).\n\n"
        f"Allowed query coordinates: each coordinate must be in [{bounds[0]}, {bounds[1]}].\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"location\":[x1,y1]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        "- Output a measurement/query location, not a source configuration.\n"
        "- Do not repeat a previous query location.\n"
        "- If you reason internally, still end with exactly one JSON object in the required shape."
    )
    user = (
        f"Strategy to follow:\n{strategy}\n\n"
        f"Observation history:\n{_format_observations(observations)}\n\n"
        f"Current weighted source hypotheses:\n"
        f"{_format_weighted_hypotheses(belief_state, top_n=config.location_strategy_belief_summary_top_k)}\n\n"
        "Following the strategy, choose the single next measurement location."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strategy_key(strategy: str) -> str:
    return re.sub(r"\s+", " ", strategy.strip()).lower()


def _extend_unique_strategies(target: list[str], candidates: list[str], seen: set[str], limit: int) -> None:
    for strategy in candidates:
        cleaned = _clean_strategy_text(strategy)
        if cleaned is None:
            continue
        key = _strategy_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        target.append(cleaned)
        if len(target) >= limit:
            return


def _extend_unique_strategy_candidates(
    target: list[LocationStrategyCandidate],
    candidates: list[LocationStrategyCandidate],
    seen: set[str],
    limit: int,
    observations: list[LocationObservation],
) -> None:
    for candidate in candidates:
        cleaned = _clean_strategy_text(candidate.strategy)
        if cleaned is None or candidate.root_query is None:
            continue
        if _is_repeated_location(candidate.root_query, observations):
            continue
        key = _strategy_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        target.append(LocationStrategyCandidate(strategy=cleaned, root_query=candidate.root_query))
        if len(target) >= limit:
            return



def _strategy_phase_single(
    questioner: "Model",
    messages: list[dict[str, str]],
    strategies: list[str],
    seen: set[str],
    target_count: int,
    config: Config,
    phase_name: str,
) -> None:
    """Run one strategy-generation phase with up to 3 attempts, extending strategies in place."""
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            _extend_unique_strategies(strategies, parse_location_strategies(completion), seen, target_count)
            return
        except ValueError as exc:
            _log_location(
                f"strategy proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )


def _strategy_phase_batched(
    questioner: "Model",
    batch_messages: list[list[dict[str, str]]],
    pending_indices: list[int],
    prepared: list[dict[str, object]],
    target_count: int,
    config: Config,
    phase_name: str,
) -> None:
    """Run one batched strategy-generation phase (3 attempts, pending-indices retry)."""
    for attempt in range(3):
        if not pending_indices:
            break
        msgs = [batch_messages[i] for i in pending_indices]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=msgs,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(m, temperature=config.generation_temperature_diverse)[0]
                for m in msgs
            ]
        still_pending: list[int] = []
        for request_idx, completion in zip(pending_indices, completions):
            item = prepared[request_idx]
            try:
                _extend_unique_strategies(
                    item["strategies"],  # type: ignore[arg-type]
                    parse_location_strategies(completion),
                    item["seen"],  # type: ignore[arg-type]
                    target_count,
                )
            except ValueError as exc:
                _log_location(
                    f"strategy proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(request_idx)
        pending_indices[:] = still_pending


def generate_location_strategies(
    questioner: "Model",
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    library: LocationStrategyLibrary,
    config: Config,
) -> list[str]:
    target_count = config.location_strategy_num_candidates
    retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
    strategies: list[str] = []
    seen: set[str] = set()
    _log_location(
        f"strategy proposal: retrieved={len(retrieved_entries)}, mutation={config.location_strategy_num_mutation}, "
        f"crossover={config.location_strategy_num_crossover}, diverse={config.location_strategy_num_diverse}, "
        f"library_size={len(library)}",
        config,
    )

    # Phase R: retrieved (no LLM call)
    _extend_unique_strategies(strategies, [e.strategy for e in retrieved_entries], seen, target_count)

    # Phase M: mutation (falls back to diverse if library empty)
    if config.location_strategy_num_mutation > 0:
        msgs = (
            _strategy_mutation_messages(retrieved_entries, belief_state, observations, config,
                                        config.location_strategy_num_mutation)
            if retrieved_entries
            else _strategy_diverse_messages(belief_state, observations, config,
                                            config.location_strategy_num_mutation)
        )
        _strategy_phase_single(questioner, msgs, strategies, seen, target_count, config, "mutation")

    # Phase C: crossover (falls back to diverse if library empty)
    if config.location_strategy_num_crossover > 0:
        msgs = (
            _strategy_crossover_messages(retrieved_entries, belief_state, observations, config,
                                         config.location_strategy_num_crossover)
            if retrieved_entries
            else _strategy_diverse_messages(belief_state, observations, config,
                                            config.location_strategy_num_crossover)
        )
        _strategy_phase_single(questioner, msgs, strategies, seen, target_count, config, "crossover")

    # Phase D: diverse — fills any remaining slots (including unfilled M/C)
    diverse_needed = target_count - len(strategies)
    if diverse_needed > 0:
        msgs = _strategy_diverse_messages(belief_state, observations, config, diverse_needed)
        _strategy_phase_single(questioner, msgs, strategies, seen, target_count, config, "diverse")

    selected = strategies[:target_count]
    _log_location(f"strategy proposal: using {len(selected)} strategy/strategies", config)
    return selected


def generate_location_strategies_many(
    questioner: "Model",
    requests: list[tuple[LocationBeliefState, list[LocationObservation], LocationStrategyLibrary]],
    config: Config,
) -> list[list[str]]:
    if not requests:
        return []
    target_count = config.location_strategy_num_candidates
    prepared: list[dict[str, object]] = []
    for request_idx, (belief_state, observations, library) in enumerate(requests):
        retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
        strategies: list[str] = []
        seen: set[str] = set()
        _log_location(
            f"strategy proposal: retrieved={len(retrieved_entries)}, mutation={config.location_strategy_num_mutation}, "
            f"crossover={config.location_strategy_num_crossover}, diverse={config.location_strategy_num_diverse}, "
            f"library_size={len(library)}",
            config,
        )
        _extend_unique_strategies(strategies, [e.strategy for e in retrieved_entries], seen, target_count)
        prepared.append({
            "strategies": strategies,
            "seen": seen,
            "retrieved_entries": retrieved_entries,
            "observations": observations,
            "belief_state": belief_state,
        })

    # Phase M: mutation
    if config.location_strategy_num_mutation > 0:
        mutation_messages = [
            (
                _strategy_mutation_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_phase_batched(questioner, mutation_messages, pending, prepared, target_count, config, "mutation")

    # Phase C: crossover
    if config.location_strategy_num_crossover > 0:
        crossover_messages = [
            (
                _strategy_crossover_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_phase_batched(questioner, crossover_messages, pending, prepared, target_count, config, "crossover")

    # Phase D: diverse — fills any remaining slots
    diverse_messages = [
        _strategy_diverse_messages(
            prepared[i]["belief_state"],  # type: ignore[arg-type]
            prepared[i]["observations"],  # type: ignore[arg-type]
            config,
            target_count - len(prepared[i]["strategies"]),  # type: ignore[arg-type]
        )
        for i in range(len(prepared))
    ]
    pending = [i for i in range(len(prepared)) if len(prepared[i]["strategies"]) < target_count]  # type: ignore[arg-type]
    if pending:
        _strategy_phase_batched(questioner, diverse_messages, pending, prepared, target_count, config, "diverse")

    results: list[list[str]] = []
    for item in prepared:
        selected = (item["strategies"])[:target_count]  # type: ignore[index]
        _log_location(f"strategy proposal: using {len(selected)} strategy/strategies", config)
        results.append(selected)
    return results


def _strategy_root_phase_single(
    questioner: "Model",
    messages: list[dict[str, str]],
    candidates: list[LocationStrategyCandidate],
    seen: set[str],
    target_count: int,
    observations: list[LocationObservation],
    config: Config,
    phase_name: str,
) -> None:
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            parsed = parse_location_strategy_roots(
                completion, config.location_dim, tuple(config.location_query_bounds)
            )
            _extend_unique_strategy_candidates(candidates, parsed, seen, target_count, observations)
            return
        except ValueError as exc:
            _log_location(
                f"strategy+root proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )


def _strategy_root_phase_batched(
    questioner: "Model",
    batch_messages: list[list[dict[str, str]]],
    pending_indices: list[int],
    prepared: list[dict[str, object]],
    target_count: int,
    config: Config,
    phase_name: str,
) -> None:
    for attempt in range(3):
        if not pending_indices:
            break
        msgs = [batch_messages[i] for i in pending_indices]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=msgs,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(m, temperature=config.generation_temperature_diverse)[0]
                for m in msgs
            ]
        still_pending: list[int] = []
        for request_idx, completion in zip(pending_indices, completions):
            item = prepared[request_idx]
            item_observations: list[LocationObservation] = item["observations"]  # type: ignore[assignment]
            try:
                parsed = parse_location_strategy_roots(
                    completion, config.location_dim, tuple(config.location_query_bounds)
                )
                _extend_unique_strategy_candidates(
                    item["candidates"],  # type: ignore[arg-type]
                    parsed,
                    item["seen"],  # type: ignore[arg-type]
                    target_count,
                    item_observations,
                )
            except ValueError as exc:
                _log_location(
                    f"strategy+root proposal [{phase_name}]: attempt {attempt + 1}/3 could not parse ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(request_idx)
        pending_indices[:] = still_pending


def generate_location_strategy_roots(
    questioner: "Model",
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    library: LocationStrategyLibrary,
    config: Config,
) -> list[LocationStrategyCandidate]:
    target_count = config.location_strategy_num_candidates
    retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
    candidates: list[LocationStrategyCandidate] = []
    seen: set[str] = set()
    retrieved_candidates = [
        LocationStrategyCandidate(entry.strategy, entry.root_query)
        for entry in retrieved_entries
        if entry.root_query is not None
    ]
    _log_location(
        f"strategy+root proposal: retrieved={len(retrieved_candidates)}, "
        f"mutation={config.location_strategy_num_mutation}, crossover={config.location_strategy_num_crossover}, "
        f"diverse={config.location_strategy_num_diverse}, library_size={len(library)}",
        config,
    )

    # Phase R: retrieved (no LLM call)
    _extend_unique_strategy_candidates(candidates, retrieved_candidates, seen, target_count, observations)

    # Phase M: mutation (falls back to diverse if library empty)
    if config.location_strategy_num_mutation > 0:
        msgs = (
            _strategy_root_mutation_messages(retrieved_entries, belief_state, observations, config,
                                             config.location_strategy_num_mutation)
            if retrieved_entries
            else _strategy_root_diverse_messages(belief_state, observations, config,
                                                 config.location_strategy_num_mutation)
        )
        _strategy_root_phase_single(questioner, msgs, candidates, seen, target_count, observations, config, "mutation")

    # Phase C: crossover (falls back to diverse if library empty)
    if config.location_strategy_num_crossover > 0:
        msgs = (
            _strategy_root_crossover_messages(retrieved_entries, belief_state, observations, config,
                                              config.location_strategy_num_crossover)
            if retrieved_entries
            else _strategy_root_diverse_messages(belief_state, observations, config,
                                                 config.location_strategy_num_crossover)
        )
        _strategy_root_phase_single(questioner, msgs, candidates, seen, target_count, observations, config, "crossover")

    # Phase D: diverse — fills remaining slots
    diverse_needed = target_count - len(candidates)
    if diverse_needed > 0:
        msgs = _strategy_root_diverse_messages(belief_state, observations, config, diverse_needed)
        _strategy_root_phase_single(questioner, msgs, candidates, seen, target_count, observations, config, "diverse")

    selected = candidates[:target_count]
    _log_location(
        "strategy+root proposal: using "
        + "; ".join(f"{_format_location(candidate.root_query)} :: {candidate.strategy[:80]}"
                    for candidate in selected if candidate.root_query is not None),
        config,
    )
    return selected


def generate_location_strategy_roots_many(
    questioner: "Model",
    requests: list[tuple[LocationBeliefState, list[LocationObservation], LocationStrategyLibrary]],
    config: Config,
) -> list[list[LocationStrategyCandidate]]:
    if not requests:
        return []
    target_count = config.location_strategy_num_candidates
    prepared: list[dict[str, object]] = []
    for request_idx, (belief_state, observations, library) in enumerate(requests):
        retrieved_entries = library.retrieve_top_m(config.location_strategy_num_retrieved)
        candidates: list[LocationStrategyCandidate] = []
        seen: set[str] = set()
        retrieved_candidates = [
            LocationStrategyCandidate(entry.strategy, entry.root_query)
            for entry in retrieved_entries
            if entry.root_query is not None
        ]
        _log_location(
            f"strategy+root proposal: retrieved={len(retrieved_candidates)}, "
            f"mutation={config.location_strategy_num_mutation}, crossover={config.location_strategy_num_crossover}, "
            f"diverse={config.location_strategy_num_diverse}, library_size={len(library)}",
            config,
        )
        _extend_unique_strategy_candidates(candidates, retrieved_candidates, seen, target_count, observations)
        prepared.append({
            "candidates": candidates,
            "seen": seen,
            "retrieved_entries": retrieved_entries,
            "observations": observations,
            "belief_state": belief_state,
        })

    # Phase M: mutation
    if config.location_strategy_num_mutation > 0:
        mutation_messages = [
            (
                _strategy_root_mutation_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_root_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_mutation,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_root_phase_batched(questioner, mutation_messages, pending, prepared, target_count, config, "mutation")

    # Phase C: crossover
    if config.location_strategy_num_crossover > 0:
        crossover_messages = [
            (
                _strategy_root_crossover_messages(
                    prepared[i]["retrieved_entries"],  # type: ignore[arg-type]
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
                if prepared[i]["retrieved_entries"]
                else _strategy_root_diverse_messages(
                    prepared[i]["belief_state"],  # type: ignore[arg-type]
                    prepared[i]["observations"],  # type: ignore[arg-type]
                    config, config.location_strategy_num_crossover,
                )
            )
            for i in range(len(prepared))
        ]
        pending = list(range(len(prepared)))
        _strategy_root_phase_batched(questioner, crossover_messages, pending, prepared, target_count, config, "crossover")

    # Phase D: diverse — fills remaining slots
    diverse_messages = [
        _strategy_root_diverse_messages(
            prepared[i]["belief_state"],  # type: ignore[arg-type]
            prepared[i]["observations"],  # type: ignore[arg-type]
            config,
            target_count - len(prepared[i]["candidates"]),  # type: ignore[arg-type]
        )
        for i in range(len(prepared))
    ]
    pending = [i for i in range(len(prepared)) if len(prepared[i]["candidates"]) < target_count]  # type: ignore[arg-type]
    if pending:
        _strategy_root_phase_batched(questioner, diverse_messages, pending, prepared, target_count, config, "diverse")

    results: list[list[LocationStrategyCandidate]] = []
    for item in prepared:
        selected = (item["candidates"])[:target_count]  # type: ignore[index]
        _log_location(
            "strategy+root proposal: using "
            + "; ".join(f"{_format_location(candidate.root_query)} :: {candidate.strategy[:80]}"
                        for candidate in selected if candidate.root_query is not None),
            config,
        )
        results.append(selected)
    return results


def _location_key(location: Location) -> tuple[float, ...]:
    return tuple(round(value, 6) for value in location)


def _is_repeated_location(location: Location, observations: list[LocationObservation]) -> bool:
    key = _location_key(location)
    return any(_location_key(observation.query) == key for observation in observations)


def generate_strategy_locations_many(
    questioner: "Model",
    requests: list[_StrategyLocationRequest],
    config: Config,
) -> list[Location | None]:
    if not requests:
        return []

    bounds = tuple(config.location_query_bounds)
    batch_messages = [
        _strategy_location_messages(request.strategy, request.belief_state, request.observations, config)
        for request in requests
    ]
    results: list[Location | None] = [None] * len(requests)
    pending = list(range(len(requests)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_simple,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_simple)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} strategy-location completions, received {len(completions)}")
        still_pending: list[int] = []
        for request_idx, completion in zip(pending, completions):
            request = requests[request_idx]
            try:
                location = parse_strategy_location(completion, config.location_dim, bounds)
                if _is_repeated_location(location, request.observations):
                    raise ValueError("Location repeats a previous query")
                results[request_idx] = location
            except ValueError as exc:
                _log_location(
                    f"strategy location: attempt {attempt + 1}/3 failed ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(request_idx)
        pending = still_pending

    return results


def generate_strategy_location(
    questioner: "Model",
    strategy: str,
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> Location | None:
    return generate_strategy_locations_many(
        questioner,
        [_StrategyLocationRequest(strategy, belief_state, list(observations))],
        config,
    )[0]


def _sample_source_hypothesis(
    belief_state: LocationBeliefState,
    rng: np.random.Generator,
) -> tuple[SourceConfig, float]:
    if not belief_state.hypotheses:
        raise ValueError("Cannot sample a source hypothesis from an empty belief state")
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    sampled_index = int(rng.choice(len(belief_state.hypotheses), p=probabilities))
    return belief_state.hypotheses[sampled_index], float(probabilities[sampled_index])


def _hypothesis_probability(
    belief_state: LocationBeliefState,
    hypothesis: SourceConfig,
    probability_floor: float = 1e-300,
) -> float:
    for candidate, probability in zip(belief_state.hypotheses, belief_state.probabilities):
        if candidate == hypothesis:
            return max(float(probability), probability_floor)
    return probability_floor


def _full_rollout_observations(
    real_observations: list[LocationObservation],
    rollout: _StrategyRollout,
) -> list[LocationObservation]:
    return list(real_observations) + list(rollout.simulated_observations)


def _root_query_fingerprint(root_queries: list[Location | None]) -> str:
    formatted_queries = [
        _format_location(query)
        for query in root_queries
        if query is not None
    ]
    if not formatted_queries:
        return ""
    return Counter(formatted_queries).most_common(1)[0][0]


def _location_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _align_belief_state_to_support(
    belief_state: LocationBeliefState,
    support: list[SourceConfig],
) -> LocationBeliefState:
    probability_lookup = {
        hypothesis: float(probability)
        for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities)
    }
    probabilities = [max(probability_lookup.get(hypothesis, 0.0), 0.0) for hypothesis in support]
    total = sum(probabilities)
    if total <= 0.0 and support:
        probabilities = [1.0 / len(support)] * len(support)
    elif total > 0.0:
        probabilities = [float(probability / total) for probability in probabilities]
    return LocationBeliefState(list(support), probabilities)


def _rollout_entropy_reduction_score(
    rollout: _StrategyRollout,
    real_observations: list[LocationObservation],
    config: Config,
) -> float:
    support = _dedupe_source_configs(
        list(rollout.particle_support)
        + list(rollout.final_generated_hypotheses)
        + list(rollout.belief_state.hypotheses)
        + ([] if rollout.final_scoring_belief_state is None else list(rollout.final_scoring_belief_state.hypotheses))
        + [rollout.truth]
    )
    if len(support) <= 1:
        return 0.0
    start_state = _align_belief_state_to_support(rollout.start_belief_state, support)
    final_state = _align_belief_state_to_support(rollout.final_scoring_belief_state or rollout.belief_state, support)
    if config.location_strategy_discount_factor >= 1.0:
        return _location_entropy(start_state.probabilities) - _location_entropy(final_state.probabilities)
    if not rollout.simulated_observations:
        return _location_entropy(start_state.probabilities) - _location_entropy(final_state.probabilities)

    gamma = config.location_strategy_discount_factor
    score = 0.0
    previous_state = start_state
    full_history = list(real_observations)
    for step_idx, simulated_observation in enumerate(rollout.simulated_observations):
        full_history.append(simulated_observation)
        if step_idx == len(rollout.simulated_observations) - 1:
            next_state = final_state
        else:
            next_state = build_location_belief_state_unpruned(support, full_history, config)
        score += (gamma ** step_idx) * (
            _location_entropy(previous_state.probabilities) - _location_entropy(next_state.probabilities)
        )
        previous_state = next_state
    return score


def evaluate_location_strategies_by_rollout(
    questioner: "Model",
    strategies: list[str],
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    config: Config,
    rng: np.random.Generator,
    root_queries: list[Location | None] | None = None,
) -> list[LocationStrategyEvaluation]:
    if not strategies:
        return []
    if root_queries is None:
        root_queries = [None] * len(strategies)
    if len(root_queries) != len(strategies):
        raise ValueError("root_queries must have the same length as strategies")
    if not belief_state.hypotheses:
        return [
            LocationStrategyEvaluation(
                strategy,
                0.0,
                0.0,
                "" if root_query is None else _format_location(root_query),
                [0.0] * config.location_strategy_num_rollouts,
                root_query=root_query,
            )
            for strategy, root_query in zip(strategies, root_queries)
        ]

    rollouts: list[_StrategyRollout] = []
    for strategy_index, strategy in enumerate(strategies):
        for _rollout_idx in range(config.location_strategy_num_rollouts):
            truth, start_probability = _sample_source_hypothesis(belief_state, rng)
            rollouts.append(
                _StrategyRollout(
                    request_index=0,
                    strategy_index=strategy_index,
                    strategy=strategy,
                    truth=truth,
                    start_probability=start_probability,
                    start_belief_state=belief_state,
                    belief_state=belief_state,
                    particle_support=list(belief_state.hypotheses),
                    root_query=root_queries[strategy_index],
                )
            )

    for depth_idx in range(config.location_strategy_planning_depth):
        active_indices = [
            rollout_idx
            for rollout_idx, rollout in enumerate(rollouts)
            if rollout.belief_state.hypotheses
        ]
        if not active_indices:
            break

        fixed_root_indices = [
            rollout_idx
            for rollout_idx in active_indices
            if depth_idx == 0 and rollouts[rollout_idx].root_query is not None
        ]
        generated_indices = [rollout_idx for rollout_idx in active_indices if rollout_idx not in fixed_root_indices]
        locations_by_index: dict[int, Location] = {
            rollout_idx: rollouts[rollout_idx].root_query  # type: ignore[dict-item]
            for rollout_idx in fixed_root_indices
        }
        if generated_indices:
            location_requests = [
                _StrategyLocationRequest(
                    strategy=rollouts[rollout_idx].strategy,
                    belief_state=rollouts[rollout_idx].belief_state,
                    observations=_full_rollout_observations(observations, rollouts[rollout_idx]),
                )
                for rollout_idx in generated_indices
            ]
            locations = generate_strategy_locations_many(questioner, location_requests, config)
            locations_by_index.update(zip(generated_indices, locations))

        # Phase 1: add simulated observations for all rollouts that got a valid location
        stepped_indices: list[int] = []
        for rollout_idx in active_indices:
            location = locations_by_index[rollout_idx]
            rollout = rollouts[rollout_idx]
            if location is None:
                _log_location(
                    f"strategy rollout: depth {depth_idx + 1} location unavailable after retries; skipping rollout step",
                    config,
                )
                continue
            if depth_idx == 0 and rollout.root_query is None:
                rollout.root_query = location
            mean = signal_intensity_for_hypothesis(rollout.truth, location)
            observed_value = float(round(rng.normal(mean, config.location_noise_sd), 2))
            rollout.simulated_observations.append(
                LocationObservation(query=location, value=observed_value)
            )
            stepped_indices.append(rollout_idx)

        # Phase 2: batch-update belief states (respects location_posterior_mode)
        if stepped_indices:
            updated_states = build_location_posteriors_many(
                questioner,
                [rollouts[i].particle_support for i in stepped_indices],
                [_full_rollout_observations(observations, rollouts[i]) for i in stepped_indices],
                config,
                context_states=[rollouts[i].belief_state for i in stepped_indices],
                label=f"strategy rollout depth {depth_idx + 1} belief update",
                prune=False,
            )
            for rollout_idx, updated_state in zip(stepped_indices, updated_states):
                rollouts[rollout_idx].belief_state = updated_state
                rollouts[rollout_idx].simulated_belief_states.append(updated_state)

    final_refresh_indices = [
        rollout_idx
        for rollout_idx, rollout in enumerate(rollouts)
        if rollout.belief_state.hypotheses
    ]
    if final_refresh_indices:
        final_histories = [
            _full_rollout_observations(observations, rollouts[rollout_idx])
            for rollout_idx in final_refresh_indices
        ]
        final_generated_many = _generate_location_hypotheses_many(
            questioner,
            final_histories,
            [
                prompt_location_belief_state(rollouts[rollout_idx].belief_state, config)
                for rollout_idx in final_refresh_indices
            ],
            config,
            label="strategy rollout final belief refresh",
        )
        for rollout_idx, branch_history, generated_hypotheses in zip(
            final_refresh_indices,
            final_histories,
            final_generated_many,
        ):
            rollout = rollouts[rollout_idx]
            rollout.final_generated_hypotheses = generated_hypotheses
        final_supports = [
            _dedupe_source_configs(
                list(rollouts[rollout_idx].particle_support)
                + list(rollouts[rollout_idx].final_generated_hypotheses)
            )
            for rollout_idx in final_refresh_indices
        ]
        final_scoring_states = build_location_posteriors_many(
            questioner,
            final_supports,
            final_histories,
            config,
            context_states=[rollouts[rollout_idx].belief_state for rollout_idx in final_refresh_indices],
            label="strategy rollout final posterior scoring",
            prune=False,
        )
        for rollout_idx, final_scoring_state in zip(final_refresh_indices, final_scoring_states):
            rollouts[rollout_idx].final_scoring_belief_state = final_scoring_state
            rollouts[rollout_idx].belief_state = prune_location_beliefs(
                final_scoring_state,
                max_beliefs=config.location_max_total_beliefs,
            )

    scores_by_strategy: list[list[float]] = [[] for _strategy in strategies]
    root_queries_by_strategy: list[list[Location | None]] = [[] for _strategy in strategies]
    for rollout in rollouts:
        score = _rollout_entropy_reduction_score(rollout, observations, config)
        scores_by_strategy[rollout.strategy_index].append(float(score))
        root_queries_by_strategy[rollout.strategy_index].append(rollout.root_query)

    evaluations: list[LocationStrategyEvaluation] = []
    for strategy, scores, strategy_root_queries, fixed_root_query in zip(
        strategies,
        scores_by_strategy,
        root_queries_by_strategy,
        root_queries,
    ):
        if scores:
            mean_score = float(np.mean(scores))
            score_variance = float(np.var(scores))
        else:
            mean_score = 0.0
            score_variance = 0.0
        evaluations.append(
            LocationStrategyEvaluation(
                strategy=strategy,
                mean_score=mean_score,
                score_variance=score_variance,
                root_query_fingerprint=(
                    _format_location(fixed_root_query)
                    if fixed_root_query is not None
                    else _root_query_fingerprint(strategy_root_queries)
                ),
                rollout_scores=scores,
                root_query=fixed_root_query,
            )
        )
    return evaluations


def evaluate_location_strategies_by_rollout_many(
    questioner: "Model",
    requests: list[_StrategyEvaluationRequest],
    config: Config,
) -> list[list[LocationStrategyEvaluation]]:
    if not requests:
        return []

    evaluations_by_request: list[list[LocationStrategyEvaluation] | None] = [None] * len(requests)
    rollouts: list[_StrategyRollout] = []
    root_queries_by_request: list[list[Location | None]] = []
    for request_idx, request in enumerate(requests):
        root_queries = (
            [None] * len(request.strategies)
            if request.root_queries is None
            else list(request.root_queries)
        )
        if len(root_queries) != len(request.strategies):
            raise ValueError("root_queries must have the same length as strategies")
        root_queries_by_request.append(root_queries)
        if not request.strategies:
            evaluations_by_request[request_idx] = []
            continue
        if not request.belief_state.hypotheses:
            evaluations_by_request[request_idx] = [
                LocationStrategyEvaluation(
                    strategy,
                    0.0,
                    0.0,
                    "" if root_query is None else _format_location(root_query),
                    [0.0] * config.location_strategy_num_rollouts,
                    root_query=root_query,
                )
                for strategy, root_query in zip(request.strategies, root_queries)
            ]
            continue

        for strategy_index, strategy in enumerate(request.strategies):
            for _rollout_idx in range(config.location_strategy_num_rollouts):
                truth, start_probability = _sample_source_hypothesis(request.belief_state, request.rng)
                rollouts.append(
                    _StrategyRollout(
                        request_index=request_idx,
                        strategy_index=strategy_index,
                        strategy=strategy,
                        truth=truth,
                        start_probability=start_probability,
                        start_belief_state=request.belief_state,
                        belief_state=request.belief_state,
                        particle_support=list(request.belief_state.hypotheses),
                        root_query=root_queries[strategy_index],
                    )
                )

    for depth_idx in range(config.location_strategy_planning_depth):
        active_indices = [
            rollout_idx
            for rollout_idx, rollout in enumerate(rollouts)
            if rollout.belief_state.hypotheses
        ]
        if not active_indices:
            break

        fixed_root_indices = [
            rollout_idx
            for rollout_idx in active_indices
            if depth_idx == 0 and rollouts[rollout_idx].root_query is not None
        ]
        generated_indices = [rollout_idx for rollout_idx in active_indices if rollout_idx not in fixed_root_indices]
        locations_by_index: dict[int, Location] = {
            rollout_idx: rollouts[rollout_idx].root_query  # type: ignore[dict-item]
            for rollout_idx in fixed_root_indices
        }
        if generated_indices:
            location_requests = [
                _StrategyLocationRequest(
                    strategy=rollouts[rollout_idx].strategy,
                    belief_state=rollouts[rollout_idx].belief_state,
                    observations=_full_rollout_observations(
                        requests[rollouts[rollout_idx].request_index].observations,
                        rollouts[rollout_idx],
                    ),
                )
                for rollout_idx in generated_indices
            ]
            locations = generate_strategy_locations_many(questioner, location_requests, config)
            locations_by_index.update(zip(generated_indices, locations))

        # Phase 1: add simulated observations for all rollouts that got a valid location
        stepped_indices_many: list[int] = []
        for rollout_idx in active_indices:
            location = locations_by_index[rollout_idx]
            rollout = rollouts[rollout_idx]
            request = requests[rollout.request_index]
            if location is None:
                _log_location(
                    f"strategy rollout: depth {depth_idx + 1} location unavailable after retries; skipping rollout step",
                    config,
                )
                continue
            if depth_idx == 0 and rollout.root_query is None:
                rollout.root_query = location
            mean = signal_intensity_for_hypothesis(rollout.truth, location)
            observed_value = float(round(request.rng.normal(mean, config.location_noise_sd), 2))
            rollout.simulated_observations.append(
                LocationObservation(query=location, value=observed_value)
            )
            stepped_indices_many.append(rollout_idx)

        # Phase 2: batch-update belief states (respects location_posterior_mode)
        if stepped_indices_many:
            updated_states_many = build_location_posteriors_many(
                questioner,
                [rollouts[i].particle_support for i in stepped_indices_many],
                [
                    _full_rollout_observations(requests[rollouts[i].request_index].observations, rollouts[i])
                    for i in stepped_indices_many
                ],
                config,
                context_states=[rollouts[i].belief_state for i in stepped_indices_many],
                label=f"strategy rollout depth {depth_idx + 1} belief update",
                prune=False,
            )
            for rollout_idx, updated_state in zip(stepped_indices_many, updated_states_many):
                rollouts[rollout_idx].belief_state = updated_state
                rollouts[rollout_idx].simulated_belief_states.append(updated_state)

    final_refresh_indices = [
        rollout_idx
        for rollout_idx, rollout in enumerate(rollouts)
        if rollout.belief_state.hypotheses
    ]
    if final_refresh_indices:
        final_histories = [
            _full_rollout_observations(
                requests[rollouts[rollout_idx].request_index].observations,
                rollouts[rollout_idx],
            )
            for rollout_idx in final_refresh_indices
        ]
        final_generated_many = _generate_location_hypotheses_many(
            questioner,
            final_histories,
            [
                prompt_location_belief_state(rollouts[rollout_idx].belief_state, config)
                for rollout_idx in final_refresh_indices
            ],
            config,
            label="strategy rollout final belief refresh",
        )
        for rollout_idx, generated_hypotheses in zip(final_refresh_indices, final_generated_many):
            rollouts[rollout_idx].final_generated_hypotheses = generated_hypotheses
        final_supports = [
            _dedupe_source_configs(
                list(rollouts[rollout_idx].particle_support)
                + list(rollouts[rollout_idx].final_generated_hypotheses)
            )
            for rollout_idx in final_refresh_indices
        ]
        final_scoring_states = build_location_posteriors_many(
            questioner,
            final_supports,
            final_histories,
            config,
            context_states=[rollouts[rollout_idx].belief_state for rollout_idx in final_refresh_indices],
            label="strategy rollout final posterior scoring",
            prune=False,
        )
        for rollout_idx, final_scoring_state in zip(final_refresh_indices, final_scoring_states):
            rollouts[rollout_idx].final_scoring_belief_state = final_scoring_state
            rollouts[rollout_idx].belief_state = prune_location_beliefs(
                final_scoring_state,
                max_beliefs=config.location_max_total_beliefs,
            )

    scores_by_request: list[list[list[float]]] = [
        [[] for _strategy in request.strategies]
        for request in requests
    ]
    roots_by_request: list[list[list[Location | None]]] = [
        [[] for _strategy in request.strategies]
        for request in requests
    ]
    for rollout in rollouts:
        score = _rollout_entropy_reduction_score(
            rollout,
            requests[rollout.request_index].observations,
            config,
        )
        scores_by_request[rollout.request_index][rollout.strategy_index].append(float(score))
        roots_by_request[rollout.request_index][rollout.strategy_index].append(rollout.root_query)

    for request_idx, request in enumerate(requests):
        if evaluations_by_request[request_idx] is not None:
            continue
        evaluations: list[LocationStrategyEvaluation] = []
        root_queries = root_queries_by_request[request_idx]
        for strategy, scores, strategy_root_queries, fixed_root_query in zip(
            request.strategies,
            scores_by_request[request_idx],
            roots_by_request[request_idx],
            root_queries,
        ):
            if scores:
                mean_score = float(np.mean(scores))
                score_variance = float(np.var(scores))
            else:
                mean_score = 0.0
                score_variance = 0.0
            evaluations.append(
                LocationStrategyEvaluation(
                    strategy=strategy,
                    mean_score=mean_score,
                    score_variance=score_variance,
                    root_query_fingerprint=(
                        _format_location(fixed_root_query)
                        if fixed_root_query is not None
                        else _root_query_fingerprint(strategy_root_queries)
                    ),
                    rollout_scores=scores,
                    root_query=fixed_root_query,
                )
            )
        evaluations_by_request[request_idx] = evaluations
    return [evaluations or [] for evaluations in evaluations_by_request]


def _strategy_entries_from_evaluations(
    evaluations: list[LocationStrategyEvaluation],
    round_index: int,
) -> list[LocationStrategyEntry]:
    return [
        LocationStrategyEntry(
            strategy=evaluation.strategy,
            mean_score=evaluation.mean_score,
            score_variance=evaluation.score_variance,
            root_query_fingerprint=evaluation.root_query_fingerprint,
            round_index=round_index,
            root_query=evaluation.root_query,
        )
        for evaluation in evaluations
    ]


def choose_location_with_strategy_rollouts(
    questioner: "Model",
    belief_state: LocationBeliefState,
    observations: list[LocationObservation],
    library: LocationStrategyLibrary,
    config: Config,
    rng: np.random.Generator,
    round_index: int,
    *,
    fixed_root: bool = False,
) -> tuple[Location | None, float, LocationStrategyEvaluation | None]:
    if fixed_root:
        strategy_candidates = generate_location_strategy_roots(questioner, belief_state, observations, library, config)
        strategies = [candidate.strategy for candidate in strategy_candidates]
        root_queries = [candidate.root_query for candidate in strategy_candidates]
    else:
        strategies = generate_location_strategies(questioner, belief_state, observations, library, config)
        root_queries = None
    evaluations = evaluate_location_strategies_by_rollout(
        questioner,
        strategies,
        belief_state,
        observations,
        config,
        rng,
        root_queries=root_queries,
    )
    library.replace_entries(_strategy_entries_from_evaluations(evaluations, round_index))
    if not evaluations:
        return None, 0.0, None

    best_evaluation = max(evaluations, key=lambda evaluation: evaluation.mean_score)
    _log_location(
        "strategy rollout: best strategy "
        f"score={best_evaluation.mean_score:.6f}, variance={best_evaluation.score_variance:.6f}, "
        f"root={best_evaluation.root_query_fingerprint!r}",
        config,
    )
    if fixed_root and best_evaluation.root_query is not None:
        _log_location(f"strategy+root: asking fixed root query {_format_location(best_evaluation.root_query)}", config)
        location: Location | None = best_evaluation.root_query
    else:
        location = generate_strategy_location(
            questioner,
            best_evaluation.strategy,
            belief_state,
            observations,
            config,
        )
    return location, best_evaluation.mean_score, best_evaluation


def choose_locations_with_strategy_rollouts_many(
    questioner: "Model",
    states: list[_LocationTrialState],
    config: Config,
    round_index: int,
    *,
    fixed_root: bool = False,
) -> list[tuple[Location | None, float, LocationStrategyEvaluation | None]]:
    if not states:
        return []
    for state in states:
        if state.belief_state is None:
            raise ValueError("Strategy trial state is missing a belief state")
        if state.strategy_library is None:
            raise ValueError("Strategy trial state is missing a strategy library")

    strategy_requests = [
        (state.belief_state, state.observations, state.strategy_library)  # type: ignore[arg-type]
        for state in states
    ]
    if fixed_root:
        strategy_candidates_many = generate_location_strategy_roots_many(questioner, strategy_requests, config)
        strategies_many = [[candidate.strategy for candidate in candidates] for candidates in strategy_candidates_many]
        root_queries_many = [[candidate.root_query for candidate in candidates] for candidates in strategy_candidates_many]
    else:
        strategies_many = generate_location_strategies_many(questioner, strategy_requests, config)
        root_queries_many = [None for _state in states]

    evaluation_requests = [
        _StrategyEvaluationRequest(
            strategies=strategies,
            belief_state=state.belief_state,  # type: ignore[arg-type]
            observations=list(state.observations),
            rng=state.rng,
            root_queries=root_queries,
        )
        for state, strategies, root_queries in zip(states, strategies_many, root_queries_many)
    ]
    evaluations_many = evaluate_location_strategies_by_rollout_many(questioner, evaluation_requests, config)
    for state, evaluations in zip(states, evaluations_many):
        state.strategy_library.replace_entries(_strategy_entries_from_evaluations(evaluations, round_index))  # type: ignore[union-attr]

    results: list[tuple[Location | None, float, LocationStrategyEvaluation | None]] = []
    selected_location_requests: list[_StrategyLocationRequest] = []
    selected_location_indices: list[int] = []
    for idx, (state, evaluations) in enumerate(zip(states, evaluations_many)):
        if not evaluations:
            results.append((None, 0.0, None))
            continue

        best_evaluation = max(evaluations, key=lambda evaluation: evaluation.mean_score)
        _log_location(
            "strategy rollout: best strategy "
            f"score={best_evaluation.mean_score:.6f}, variance={best_evaluation.score_variance:.6f}, "
            f"root={best_evaluation.root_query_fingerprint!r}",
            config,
        )
        if fixed_root and best_evaluation.root_query is not None:
            _log_location(f"strategy+root: asking fixed root query {_format_location(best_evaluation.root_query)}", config)
            results.append((best_evaluation.root_query, best_evaluation.mean_score, best_evaluation))
        else:
            results.append((None, best_evaluation.mean_score, best_evaluation))
            selected_location_indices.append(idx)
            selected_location_requests.append(
                _StrategyLocationRequest(
                    best_evaluation.strategy,
                    state.belief_state,  # type: ignore[arg-type]
                    list(state.observations),
                )
            )

    if selected_location_requests:
        selected_locations = generate_strategy_locations_many(questioner, selected_location_requests, config)
        for result_idx, location in zip(selected_location_indices, selected_locations):
            _old_location, score, evaluation = results[result_idx]
            results[result_idx] = (location, score, evaluation)

    return results


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
    messages = _belief_generation_messages(observations, belief_state, config)
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            hypotheses = parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)
            _log_location(f"{label}: parsed {len(hypotheses)} valid unique source configuration(s)", config)
            return hypotheses
        except ValueError as exc:
            _log_location(
                f"{label}: attempt {attempt + 1}/3 could not parse source hypotheses ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )
    return []


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
    results: list[list[SourceConfig] | None] = [None] * len(observations_many)
    pending = list(range(len(observations_many)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} hypothesis completions, received {len(completions)}")
        still_pending: list[int] = []
        for idx, completion in zip(pending, completions):
            try:
                results[idx] = parse_source_hypotheses(completion, config.location_num_sources, config.location_dim)
            except ValueError as exc:
                _log_location(
                    f"{label}: attempt {attempt + 1}/3 item {idx} could not parse hypotheses ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(idx)
        pending = still_pending

    hypotheses_many = [result if result is not None else [] for result in results]
    counts = [len(hypotheses) for hypotheses in hypotheses_many]
    nonempty_count = sum(1 for count in counts if count > 0)
    total_count = sum(counts)
    _log_location(
        f"{label}: parsed {total_count} generated hypothesis/hypotheses across "
        f"{nonempty_count}/{len(hypotheses_many)} nonempty refresh(es)",
        config,
    )
    return hypotheses_many



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
    messages = _candidate_generation_messages(belief_state, observations, config)
    candidates: list[Location] = []
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            candidates = parse_candidate_locations(completion, config.location_dim, bounds)
            break
        except ValueError as exc:
            _log_location(
                f"candidate generation: attempt {attempt + 1}/3 could not parse candidates ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )
    selected = candidates[:config.location_target_num_candidates]
    _log_location(
        f"candidate generation: parsed={len(candidates)}, returned={len(selected)}, "
        f"locations={_summarize_candidates(selected)}",
        config,
    )
    return selected


def generate_location_candidates_many(
    questioner: "Model",
    belief_states: list[LocationBeliefState],
    observations_many: list[list[LocationObservation]],
    config: Config,
) -> list[list[Location]]:
    if len(belief_states) != len(observations_many):
        raise ValueError("belief_states and observations_many must have the same length")
    if not belief_states:
        return []
    bounds = tuple(config.location_query_bounds)
    _log_location(
        f"candidate generation: requesting candidates for {len(belief_states)} trial(s) "
        f"as a cross-trial batch (block_size={config.batched_block_size})",
        config,
    )
    batch_messages = [
        _candidate_generation_messages(belief_state, observations, config)
        for belief_state, observations in zip(belief_states, observations_many)
    ]
    results: list[list[Location] | None] = [None] * len(belief_states)
    pending = list(range(len(belief_states)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} candidate completions, received {len(completions)}")
        still_pending: list[int] = []
        for idx, completion in zip(pending, completions):
            try:
                results[idx] = parse_candidate_locations(completion, config.location_dim, bounds)
            except ValueError as exc:
                _log_location(
                    f"candidate generation: attempt {attempt + 1}/3 item {idx} could not parse candidates ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(idx)
        pending = still_pending

    candidates_many: list[list[Location]] = []
    for idx, raw in enumerate(results):
        candidates = raw if raw is not None else []
        selected = candidates[:config.location_target_num_candidates]
        _log_location(
            f"candidate generation: item {idx} parsed={len(candidates)}, returned={len(selected)}, "
            f"locations={_summarize_candidates(selected)}",
            config,
        )
        candidates_many.append(selected)
    return candidates_many


def choose_location_naive(
    questioner: "Model",
    observations: list[LocationObservation],
    config: Config,
) -> Location | None:
    bounds = tuple(config.location_query_bounds)
    _log_location(
        f"naive query generation: requesting one location "
        f"(observations={len(observations)}, bounds=[{bounds[0]}, {bounds[1]}])",
        config,
    )
    messages = _naive_location_messages(observations, config)
    for attempt in range(3):
        completion = questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
        try:
            location = parse_single_location_from_completion(completion, config.location_dim, bounds)
            if _is_repeated_location(location, observations):
                raise ValueError("Location repeats a previous query")
            _log_location(f"Naive selection: chose direct LLM query {list(location)}", config)
            return location
        except ValueError as exc:
            _log_location(
                f"naive query generation: attempt {attempt + 1}/3 failed ({exc})"
                + ("; retrying" if attempt < 2 else "; giving up"),
                config,
            )
    return None


def estimate_sources_naive(
    questioner: "Model",
    observations: list[LocationObservation],
    config: Config,
) -> SourceConfig:
    _log_location(
        f"naive source estimate: requesting one final source configuration "
        f"(observations={len(observations)})",
        config,
    )
    completion = questioner.chat_complete(
        _naive_source_estimate_messages(observations, config),
        temperature=config.generation_temperature_simple,
    )[0]
    try:
        estimate = parse_best_source_estimate_from_completion(
            completion,
            config.location_num_sources,
            config.location_dim,
        )
    except ValueError as exc:
        _log_location(f"naive source estimate: could not parse estimate ({exc}); retrying JSON repair", config)
        repair_completion = questioner.chat_complete(
            _naive_source_estimate_repair_messages(completion, observations, config),
            temperature=0.0,
        )[0]
        try:
            estimate = parse_best_source_estimate_from_completion(
                repair_completion,
                config.location_num_sources,
                config.location_dim,
            )
        except ValueError as repair_exc:
            _log_location(
                f"naive source estimate: repair failed ({repair_exc}); using center fallback",
                config,
            )
            estimate = tuple(
                tuple(0.0 for _coord_idx in range(config.location_dim))
                for _source_idx in range(config.location_num_sources)
            )
    _log_location(f"Naive estimate: chose sources {_format_source_array(np.asarray(estimate, dtype=float))}", config)
    return estimate


def choose_locations_naive_many(
    questioner: "Model",
    observations_many: list[list[LocationObservation]],
    config: Config,
) -> list[Location | None]:
    if not observations_many:
        return []
    bounds = tuple(config.location_query_bounds)
    _log_location(
        f"naive query generation: requesting {len(observations_many)} location(s) "
        f"as a cross-trial batch (block_size={config.batched_block_size})",
        config,
    )
    batch_messages = [_naive_location_messages(observations, config) for observations in observations_many]
    results: list[Location | None] = [None] * len(observations_many)
    pending = list(range(len(observations_many)))

    for attempt in range(3):
        if not pending:
            break
        pending_messages = [batch_messages[i] for i in pending]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=pending_messages,
                temperature=config.generation_temperature_diverse,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.generation_temperature_diverse)[0]
                for messages in pending_messages
            ]
        if len(completions) != len(pending):
            raise ValueError(f"Expected {len(pending)} naive query completions, received {len(completions)}")
        still_pending: list[int] = []
        for idx, completion in zip(pending, completions):
            item_observations = observations_many[idx]
            try:
                location = parse_single_location_from_completion(completion, config.location_dim, bounds)
                if _is_repeated_location(location, item_observations):
                    raise ValueError("Location repeats a previous query")
                _log_location(f"Naive selection: chose direct LLM query {list(location)}", config)
                results[idx] = location
            except ValueError as exc:
                _log_location(
                    f"naive query generation: attempt {attempt + 1}/3 item {idx} failed ({exc})"
                    + ("; retrying" if attempt < 2 else "; giving up"),
                    config,
                )
                still_pending.append(idx)
        pending = still_pending

    return results


def estimate_sources_naive_many(
    questioner: "Model",
    observations_many: list[list[LocationObservation]],
    config: Config,
) -> list[SourceConfig]:
    if not observations_many:
        return []
    _log_location(
        f"naive source estimate: requesting {len(observations_many)} source configuration(s) "
        f"as a cross-trial batch (block_size={config.batched_block_size})",
        config,
    )
    batch_messages = [_naive_source_estimate_messages(observations, config) for observations in observations_many]
    if callable(getattr(questioner, "chat_complete_messages_batched", None)):
        completions = questioner.chat_complete_messages_batched(
            batch_messages=batch_messages,
            temperature=config.generation_temperature_simple,
            block_size=config.batched_block_size,
            max_new_tokens=config.location_max_new_tokens,
        )
    else:
        completions = [
            questioner.chat_complete(messages, temperature=config.generation_temperature_simple)[0]
            for messages in batch_messages
        ]
    if len(completions) != len(batch_messages):
        raise ValueError(f"Expected {len(batch_messages)} naive estimate completions, received {len(completions)}")

    estimates: list[SourceConfig | None] = []
    failed_indices: list[int] = []
    for completion in completions:
        try:
            estimate = parse_best_source_estimate_from_completion(
                completion,
                config.location_num_sources,
                config.location_dim,
            )
        except ValueError as exc:
            _log_location(f"naive source estimate: could not parse estimate ({exc}); will retry JSON repair", config)
            estimate = None
            failed_indices.append(len(estimates))
        estimates.append(estimate)

    if failed_indices:
        repair_messages = [
            _naive_source_estimate_repair_messages(
                completions[idx],
                observations_many[idx],
                config,
            )
            for idx in failed_indices
        ]
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            repair_completions = questioner.chat_complete_messages_batched(
                batch_messages=repair_messages,
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.location_max_new_tokens,
            )
        else:
            repair_completions = [
                questioner.chat_complete(messages, temperature=0.0)[0]
                for messages in repair_messages
            ]
        if len(repair_completions) != len(repair_messages):
            raise ValueError(f"Expected {len(repair_messages)} naive repair completions, received {len(repair_completions)}")
        for idx, repair_completion in zip(failed_indices, repair_completions):
            try:
                estimates[idx] = parse_best_source_estimate_from_completion(
                    repair_completion,
                    config.location_num_sources,
                    config.location_dim,
                )
            except ValueError as repair_exc:
                _log_location(
                    f"naive source estimate: repair failed ({repair_exc}); using center fallback",
                    config,
                )
                estimates[idx] = tuple(
                    tuple(0.0 for _coord_idx in range(config.location_dim))
                    for _source_idx in range(config.location_num_sources)
                )

    final_estimates: list[SourceConfig] = []
    for estimate in estimates:
        if estimate is None:
            estimate = tuple(
                tuple(0.0 for _coord_idx in range(config.location_dim))
                for _source_idx in range(config.location_num_sources)
            )
        _log_location(f"Naive estimate: chose sources {_format_source_array(np.asarray(estimate, dtype=float))}", config)
        final_estimates.append(estimate)
    return final_estimates



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
    state = build_location_belief_state_unpruned(hypotheses, observations, config)
    return prune_location_beliefs(state, max_beliefs=config.location_max_total_beliefs)


def build_location_belief_state_unpruned(
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
    return sort_location_belief_state(state)


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


def build_location_posteriors_many(
    questioner: "Model" | None,
    hypotheses_many: list[list[SourceConfig]],
    observations_many: list[list[LocationObservation]],
    config: Config,
    *,
    context_states: list[LocationBeliefState | None] | None = None,
    label: str = "location posterior scoring",
    prune: bool = True,
) -> list[LocationBeliefState]:
    if len(hypotheses_many) != len(observations_many):
        raise ValueError("hypotheses_many and observations_many must have the same length")
    if context_states is None:
        context_states = [None] * len(hypotheses_many)
    if len(context_states) != len(hypotheses_many):
        raise ValueError("context_states and hypotheses_many must have the same length")

    if config.location_posterior_mode == "analytical_likelihood":
        states = [
            build_location_belief_state_unpruned(hypotheses, observations, config)
            for hypotheses, observations in zip(hypotheses_many, observations_many)
        ]
        return [
            prune_location_beliefs(state, max_beliefs=config.location_max_total_beliefs)
            if prune
            else state
            for state in states
        ]
    if config.location_posterior_mode != "llm_distribution":
        raise ValueError("location_posterior_mode must be one of: analytical_likelihood, llm_distribution")
    if questioner is None:
        raise ValueError("location_posterior_mode='llm_distribution' requires a questioner model")

    deduped_hypotheses_many = [_dedupe_source_configs(list(hypotheses)) for hypotheses in hypotheses_many]
    batch_messages: list[list[dict[str, str]]] = []
    branch_prompt_counts: list[int] = []
    branch_labels: list[list[str]] = []
    active_branch_indices: list[int] = []
    for branch_idx, (hypotheses, observations, context_state) in enumerate(
        zip(deduped_hypotheses_many, observations_many, context_states)
    ):
        labels = _location_posterior_labels(len(hypotheses))
        branch_labels.append(labels)
        if not hypotheses:
            branch_prompt_counts.append(0)
            continue
        active_branch_indices.append(branch_idx)
        context_probabilities = _location_posterior_context_probabilities(hypotheses, context_state)
        if config.belief_distribution_permute_history:
            histories = _permuted_location_observation_histories(
                observations,
                config.belief_distribution_num_calls,
            )
        else:
            histories = [list(observations) for _call_idx in range(config.belief_distribution_num_calls)]
        prompts = [
            _location_posterior_distribution_messages(history, hypotheses, context_probabilities, config)
            for history in histories
        ]
        branch_prompt_counts.append(len(prompts))
        batch_messages.extend(prompts)

    completions: list[str] = []
    if batch_messages:
        posterior_max_new_tokens = max(512, min(2048, 32 * max((len(labels) for labels in branch_labels), default=0) + 128))
        if callable(getattr(questioner, "chat_complete_messages_batched", None)):
            completions = questioner.chat_complete_messages_batched(
                batch_messages=batch_messages,
                temperature=config.belief_probability_temperature,
                block_size=config.batched_block_size,
                max_new_tokens=posterior_max_new_tokens,
            )
        else:
            completions = [
                questioner.chat_complete(messages, temperature=config.belief_probability_temperature)[0]
                for messages in batch_messages
            ]
        if len(completions) != len(batch_messages):
            raise ValueError(
                f"Expected {len(batch_messages)} location posterior completions, received {len(completions)}"
            )

    scored_states: list[LocationBeliefState] = []
    completion_offset = 0
    valid_total = 0
    prompt_total = 0
    for branch_idx, hypotheses in enumerate(deduped_hypotheses_many):
        labels = branch_labels[branch_idx]
        prompt_count = branch_prompt_counts[branch_idx]
        branch_completions = completions[completion_offset:completion_offset + prompt_count]
        completion_offset += prompt_count
        prompt_total += prompt_count
        if not hypotheses:
            scored_states.append(LocationBeliefState([], []))
            continue
        context_probabilities = _location_posterior_context_probabilities(
            hypotheses,
            context_states[branch_idx],
        )
        context_distribution = {
            label: probability
            for label, probability in zip(labels, context_probabilities)
        }
        distribution, valid_count = _average_labeled_distributions_from_completions(
            branch_completions,
            labels,
            fallback_to_uniform=config.probability_parse_fallback_to_uniform,
            fallback_distribution=context_distribution,
        )
        valid_total += valid_count
        scored_state = sort_location_belief_state(
            LocationBeliefState(
                hypotheses,
                [distribution[label] for label in labels],
            )
        )
        scored_states.append(
            prune_location_beliefs(scored_state, max_beliefs=config.location_max_total_beliefs)
            if prune
            else scored_state
        )

    detail = f"{valid_total}/{prompt_total} valid"
    if config.belief_distribution_permute_history:
        detail = f"permuted-history, {detail}"
    _log_location(
        f"{label}: scored LLM posterior distribution ({detail}) across "
        f"{len(active_branch_indices)}/{len(hypotheses_many)} nonempty support(s)",
        config,
    )
    return scored_states


def build_location_posterior(
    questioner: "Model" | None,
    hypotheses: list[SourceConfig],
    observations: list[LocationObservation],
    config: Config,
    *,
    context_state: LocationBeliefState | None = None,
    label: str = "location posterior scoring",
    prune: bool = True,
) -> LocationBeliefState:
    return build_location_posteriors_many(
        questioner,
        [hypotheses],
        [observations],
        config,
        context_states=[context_state],
        label=label,
        prune=prune,
    )[0]


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

    # Depth-2: branch per (candidate, hypothesis) using hypothesis mean as representative observation.
    # This mirrors the 20Q forward-search pattern: for each branch, update beliefs, generate fresh
    # candidates, score them at depth-1, and weight-accumulate into the total.
    branch_candidate_indices: list[int] = []
    branch_weights: list[float] = []
    branch_observations: list[list[LocationObservation]] = []

    for candidate_idx, (candidate, means) in enumerate(zip(candidates, candidate_means)):
        for hypothesis_idx, (mean, hypothesis_probability) in enumerate(zip(means, probabilities)):
            if hypothesis_probability == 0.0:
                continue
            # Use the hypothesis mean signal as representative branch observation value.
            branch_candidate_indices.append(candidate_idx)
            branch_weights.append(float(hypothesis_probability))
            branch_observations.append(
                list(observations) + [LocationObservation(query=candidate, value=float(mean))]
            )

    _log_location(
        f"EIG scoring: depth-2 expanding {len(branch_observations)} branch(es) "
        f"({len(candidates)} candidates × {len(belief_state.hypotheses)} hypotheses, "
        f"excluding zero-probability beliefs)",
        config,
    )

    totals = list(immediate)
    future_contributions = [0.0] * len(candidates)

    if branch_observations:
        # Step 1: generate fresh hypotheses for each branch (batched)
        generated_hypotheses_many = _generate_location_hypotheses_many(
            questioner,
            branch_observations,
            [belief_state for _ in branch_observations],
            config,
            label="EIG depth-2 future belief generation",
        )

        # Step 2: build future belief states respecting location_posterior_mode (batched)
        future_hypotheses_many = [
            _merge_hypotheses(belief_state, gen_hyps)
            for gen_hyps in generated_hypotheses_many
        ]
        future_states = build_location_posteriors_many(
            questioner,
            future_hypotheses_many,
            branch_observations,
            config,
            context_states=[belief_state for _ in branch_observations],
            label="EIG depth-2 future posterior scoring",
            prune=True,
        )

        # Step 3: generate fresh candidates conditioned on each future state (batched)
        prompt_future_states = [prompt_location_belief_state(s, config) for s in future_states]
        future_candidates_many = generate_location_candidates_many(
            questioner,
            prompt_future_states,
            branch_observations,
            config,
        )

        # Step 4: score fresh candidates at depth-1 and accumulate weighted best
        for branch_candidate_idx, branch_weight, future_state, future_candidates in zip(
            branch_candidate_indices, branch_weights, future_states, future_candidates_many
        ):
            if not future_candidates or not future_state.hypotheses:
                continue
            future_eig_values = [
                expected_information_gain(
                    future_state,
                    future_candidate,
                    noise_sd=config.location_noise_sd,
                    quadrature_order=config.location_eig_quadrature_order,
                )
                for future_candidate in future_candidates
            ]
            if future_eig_values:
                best_future = max(future_eig_values)
                future_contributions[branch_candidate_idx] += branch_weight * best_future
                totals[branch_candidate_idx] += branch_weight * best_future

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


def _location_trial_rng(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
) -> np.random.Generator:
    if config.location_seed is None:
        return fallback_rng
    seed_sequence = np.random.SeedSequence([config.location_seed, trial_idx])
    return np.random.default_rng(seed_sequence)


def _location_trial_planning_rng(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
) -> np.random.Generator:
    if config.location_seed is None:
        return np.random.default_rng(fallback_rng.integers(0, np.iinfo(np.uint32).max))
    seed_sequence = np.random.SeedSequence([config.location_seed, trial_idx, 1])
    return np.random.default_rng(seed_sequence)


def _make_location_trial_state(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
    method_name: str,
) -> _LocationTrialState:
    env_rng = _location_trial_rng(config, trial_idx, fallback_rng)
    planning_rng = _location_trial_planning_rng(config, trial_idx, fallback_rng)
    env = LocationFindingEnv(
        num_sources=config.location_num_sources,
        dim=config.location_dim,
        noise_sd=config.location_noise_sd,
        rng=env_rng,
    )
    return _LocationTrialState(
        trial_idx=trial_idx,
        env=env,
        observations=[],
        rng=planning_rng,
        strategy_library=LocationStrategyLibrary() if method_name in {"StrategyEIG", "StrategyEIG+root"} else None,
    )


def _plot_location_trial_state(
    state: _LocationTrialState,
    belief_state: LocationBeliefState,
    final_rmse: float,
    final_top_probability: float,
    config: Config,
    output_dir: Path | None,
) -> None:
    if not config.location_plot_trials:
        return
    if output_dir is None:
        _log_location("plotting requested but no output directory was provided; skipping trial plot", config)
        return
    plot_path = output_dir / f"location_trial_{state.trial_idx + 1:03d}.png"
    _plot_location_trial(
        state.env,
        state.observations,
        belief_state,
        state.trial_idx,
        final_rmse,
        final_top_probability,
        plot_path,
    )
    _log_location(f"saved trial plot to {plot_path}", config)


def _run_location_finding_batched(
    questioner: "Model",
    config: Config,
    rng: np.random.Generator | None = None,
    output_dir: Path | None = None,
    method_name: str = "EIG",
) -> LocationFindingMetrics:
    rng = rng or np.random.default_rng()
    rmse_totals = np.zeros(config.location_num_rounds, dtype=float)
    top_probability_totals = np.zeros(config.location_num_rounds, dtype=float)
    selected_eig_totals = np.zeros(config.location_num_rounds, dtype=float)

    _log_location(
        f"Running {config.location_num_trials} Location Finding trial(s) in cross-trial batches: "
        f"method={method_name}, batch_size={config.location_trial_batch_size}, "
        f"rounds={config.location_num_rounds}, sources={config.location_num_sources}, "
        f"dim={config.location_dim}, noise_sd={config.location_noise_sd}, "
        f"candidates={config.location_target_num_candidates}, depth={config.location_search_depth}, "
        f"quadrature_order={config.location_eig_quadrature_order}, "
        f"max_total_beliefs={config.location_max_total_beliefs}, "
        f"max_llm_prompt_beliefs={config.location_max_llm_prompt_beliefs}, "
        f"num_mc_samples={config.num_mc_samples}, "
        f"posterior_mode={config.location_posterior_mode}, "
        f"location_seed={config.location_seed}",
        config,
    )

    for batch_start in range(0, config.location_num_trials, config.location_trial_batch_size):
        batch_trial_indices = list(
            range(batch_start, min(config.location_num_trials, batch_start + config.location_trial_batch_size))
        )
        states = [
            _make_location_trial_state(config, trial_idx, rng, method_name)
            for trial_idx in batch_trial_indices
        ]
        for state in states:
            _log_location(
                f"trial {state.trial_idx + 1}/{config.location_num_trials}: sampled hidden environment "
                f"with true_sources={_format_source_array(state.env.true_theta)}",
                config,
            )

        if method_name == "Naive":
            for round_idx in range(config.location_num_rounds):
                for state in states:
                    _write_to_log_if_configured(
                        f"\nLocation Finding trial {state.trial_idx + 1}: Round {round_idx + 1}\n",
                        config,
                    )
                    _log_location(
                        f"trial {state.trial_idx + 1}/{config.location_num_trials}, "
                        f"round {round_idx + 1}/{config.location_num_rounds}, "
                        f"naive conversation observations={len(state.observations)}",
                        config,
                    )
                best_locations = choose_locations_naive_many(
                    questioner,
                    [state.observations for state in states],
                    config,
                )
                for state, best_location in zip(states, best_locations):
                    if best_location is None:
                        _log_location(
                            f"trial {state.trial_idx + 1}: no valid naive location after retries; "
                            f"skipping round {round_idx + 1}",
                            config,
                        )
                        continue
                    observation = state.env.run_experiment(best_location)
                    state.observations.append(observation)
                    print_and_log(
                        f"[location] Selected query {list(best_location)} with score 0.000000; "
                        f"observed {observation.value:.2f}",
                        config,
                    )

                estimates = estimate_sources_naive_many(
                    questioner,
                    [state.observations for state in states],
                    config,
                )
                for state, estimate in zip(states, estimates):
                    state.final_estimate = estimate
                    state.final_rmse = source_rmse(estimate, state.env.true_theta)
                    rmse_totals[round_idx] += state.final_rmse
                    top_probability_totals[round_idx] += 1.0
                    print_and_log(
                        f"[location] Naive source RMSE after round {round_idx + 1}: {state.final_rmse:.6f}",
                        config,
                    )

            for state in states:
                plot_state = LocationBeliefState(
                    hypotheses=[] if state.final_estimate is None else [state.final_estimate],
                    probabilities=[] if state.final_estimate is None else [1.0],
                )
                _plot_location_trial_state(
                    state,
                    plot_state,
                    state.final_rmse,
                    1.0 if state.final_estimate is not None else 0.0,
                    config,
                    output_dir,
                )
            continue

        initial_hypotheses_many = _generate_location_hypotheses_many(
            questioner,
            [state.observations for state in states],
            [None for _state in states],
            config,
            label="batched initial belief generation",
        )
        initial_belief_states = build_location_posteriors_many(
            questioner,
            initial_hypotheses_many,
            [state.observations for state in states],
            config,
            label="batched initial posterior scoring",
        )
        for state, belief_state in zip(states, initial_belief_states):
            state.belief_state = belief_state
            prompt_belief_state = prompt_location_belief_state(belief_state, config)
            _log_location(
                f"trial {state.trial_idx + 1}: initial posterior {_summarize_belief_state(belief_state)}; "
                f"reservoir={len(belief_state.hypotheses)}, "
                f"prompt={len(prompt_belief_state.hypotheses)}, "
                f"ESS={_location_effective_sample_size(belief_state):.2f}",
                config,
            )

        for round_idx in range(config.location_num_rounds):
            prompt_belief_states = [
                prompt_location_belief_state(state.belief_state, config)  # type: ignore[arg-type]
                for state in states
            ]
            eig_belief_states: list[LocationBeliefState] = []
            eig_sample_collapsed_flags: list[bool] = []
            support_label = "strategy_support"
            for state in states:
                if method_name == "EIG":
                    eig_belief_state, eig_sample_collapsed = sample_location_eig_belief_state(
                        state.belief_state,  # type: ignore[arg-type]
                        config,
                        state.rng,
                    )
                    support_label = "EIG_support"
                elif method_name in {"StrategyEIG", "StrategyEIG+root"}:
                    eig_belief_state = state.belief_state  # type: ignore[assignment]
                    eig_sample_collapsed = False
                    support_label = "strategy_root_support" if method_name == "StrategyEIG+root" else "strategy_support"
                else:
                    eig_belief_state = state.belief_state  # type: ignore[assignment]
                    eig_sample_collapsed = False
                    support_label = "naive_support"
                eig_belief_states.append(eig_belief_state)
                eig_sample_collapsed_flags.append(eig_sample_collapsed)

            for state, prompt_belief_state, eig_belief_state, eig_sample_collapsed in zip(
                states,
                prompt_belief_states,
                eig_belief_states,
                eig_sample_collapsed_flags,
            ):
                _write_to_log_if_configured(
                    f"\nLocation Finding trial {state.trial_idx + 1}: Round {round_idx + 1}\n",
                    config,
                )
                _log_location(
                    f"trial {state.trial_idx + 1}/{config.location_num_trials}, "
                    f"round {round_idx + 1}/{config.location_num_rounds}, "
                    f"posterior {_summarize_belief_state(state.belief_state)}; "
                    f"reservoir={len(state.belief_state.hypotheses)}, "
                    f"prompt={len(prompt_belief_state.hypotheses)}, "
                    f"{support_label}={len(eig_belief_state.hypotheses)}, "
                    f"ESS={_location_effective_sample_size(state.belief_state):.2f}",
                    config,
                )
                if eig_sample_collapsed:
                    _log_location("EIG posterior sampling produced one unique hypothesis; EIG support is collapsed", config)

            if method_name == "EIG":
                candidates_many = generate_location_candidates_many(
                    questioner,
                    prompt_belief_states,
                    [state.observations for state in states],
                    config,
                )
                best_locations: list[Location | None] = []
                best_scores: list[float] = []
                for state, eig_belief_state, candidates in zip(states, eig_belief_states, candidates_many):
                    if not candidates:
                        best_locations.append(None)
                        best_scores.append(0.0)
                        continue
                    scores = score_candidate_locations(
                        eig_belief_state,
                        candidates,
                        config,
                        questioner=questioner,
                        observations=state.observations,
                    )
                    best_idx = int(np.argmax(scores)) if scores else 0
                    best_locations.append(candidates[best_idx])
                    best_scores.append(float(scores[best_idx]) if scores else 0.0)
            else:
                strategy_results = choose_locations_with_strategy_rollouts_many(
                    questioner,
                    states,
                    config,
                    round_idx,
                    fixed_root=method_name == "StrategyEIG+root",
                )
                best_locations = [location for location, _score, _evaluation in strategy_results]
                best_scores = [score for _location, score, _evaluation in strategy_results]

            skipped: set[int] = set()
            for state_idx, (state, best_location, best_score) in enumerate(
                zip(states, best_locations, best_scores)
            ):
                if best_location is None:
                    _log_location(
                        f"trial {state.trial_idx + 1}: no valid location after retries; "
                        f"skipping round {round_idx + 1}",
                        config,
                    )
                    skipped.add(state_idx)
                    continue
                observation = state.env.run_experiment(best_location)
                state.observations.append(observation)
                print_and_log(
                    f"[location] Selected query {list(best_location)} with score {best_score:.6f}; "
                    f"observed {observation.value:.2f}",
                    config,
                )

            # Only update belief states for trials that received a new observation.
            # Skipped trials keep their current belief state (consistent with single-trial path).
            active_state_indices = [i for i in range(len(states)) if i not in skipped]
            if active_state_indices:
                active_states = [states[i] for i in active_state_indices]
                active_prompt_belief_states = [prompt_belief_states[i] for i in active_state_indices]
                generated_hypotheses_many = _generate_location_hypotheses_many(
                    questioner,
                    [state.observations for state in active_states],
                    active_prompt_belief_states,
                    config,
                    label=f"batched round {round_idx + 1} belief update",
                )
                active_previous_belief_states: list[LocationBeliefState] = []
                active_merged: list[list[SourceConfig]] = []
                active_before_trim: list[int] = []
                for state, generated_hypotheses in zip(active_states, generated_hypotheses_many):
                    previous_belief_state = state.belief_state  # type: ignore[assignment]
                    merged_hypotheses = _merge_hypotheses(previous_belief_state, generated_hypotheses)
                    active_merged.append(merged_hypotheses)
                    active_before_trim.append(len(merged_hypotheses))
                    active_previous_belief_states.append(previous_belief_state)
                    _log_location(
                        f"round {round_idx + 1}: reservoir update merging "
                        f"previous={len(previous_belief_state.hypotheses)} "
                        f"with generated={len(generated_hypotheses)} -> unique={len(merged_hypotheses)}",
                        config,
                    )
                updated_active_belief_states = build_location_posteriors_many(
                    questioner,
                    active_merged,
                    [state.observations for state in active_states],
                    config,
                    context_states=active_previous_belief_states,
                    label=f"batched round {round_idx + 1} posterior scoring",
                )
                for state, belief_state, before_trim in zip(
                    active_states, updated_active_belief_states, active_before_trim
                ):
                    state.belief_state = belief_state
                    if before_trim > len(belief_state.hypotheses):
                        _log_location(
                            f"round {round_idx + 1}: reservoir trimmed {before_trim} -> "
                            f"{len(belief_state.hypotheses)} by top posterior",
                            config,
                        )
                    prompt_belief_state = prompt_location_belief_state(belief_state, config)
                    _log_location(
                        f"round {round_idx + 1}: posterior after observation "
                        f"{_summarize_belief_state(belief_state)}; "
                        f"reservoir={len(belief_state.hypotheses)}, "
                        f"prompt={len(prompt_belief_state.hypotheses)}, "
                        f"ESS={_location_effective_sample_size(belief_state):.2f}",
                        config,
                    )

            for state_idx, (state, best_score) in enumerate(zip(states, best_scores)):
                current_rmse = _top_source_rmse(state.belief_state, state.env.true_theta)  # type: ignore[arg-type]
                top_probability = (
                    state.belief_state.probabilities[0]  # type: ignore[index]
                    if state.belief_state and state.belief_state.probabilities
                    else 0.0
                )
                rmse_totals[round_idx] += current_rmse
                top_probability_totals[round_idx] += top_probability
                selected_eig_totals[round_idx] += best_score
                print_and_log(
                    f"[location] Top source RMSE after round {round_idx + 1}: {current_rmse:.6f}; "
                    f"top probability {top_probability:.6f}",
                    config,
                )

        for state in states:
            final_rmse = _top_source_rmse(state.belief_state, state.env.true_theta)  # type: ignore[arg-type]
            final_top_probability = state.belief_state.probabilities[0] if state.belief_state and state.belief_state.probabilities else 0.0
            _plot_location_trial_state(
                state,
                state.belief_state,  # type: ignore[arg-type]
                final_rmse,
                final_top_probability,
                config,
                output_dir,
            )

    divisor = float(config.location_num_trials)
    return LocationFindingMetrics(
        source_rmse=(rmse_totals / divisor).tolist(),
        top_probability=(top_probability_totals / divisor).tolist(),
        selected_eig=(selected_eig_totals / divisor).tolist(),
    )


def run_location_finding(
    questioner: "Model",
    config: Config,
    rng: np.random.Generator | None = None,
    output_dir: Path | None = None,
    method_name: str = "EIG",
) -> LocationFindingMetrics:
    if config.location_dim != 2:
        raise ValueError("Location Finding currently supports 2D source locations")
    if config.location_noise_sd != 0.5:
        raise ValueError("The initial Location Finding implementation requires known noise_sd=0.5")
    if method_name not in {"EIG", "StrategyEIG", "StrategyEIG+root", "Naive"}:
        raise ValueError(
            "Location Finding currently supports method_name='EIG', 'StrategyEIG', 'StrategyEIG+root', or 'Naive'"
        )
    if config.location_strategy_num_retrieved > config.location_strategy_num_candidates:
        raise ValueError("location_strategy_num_retrieved must be less than or equal to location_strategy_num_candidates")
    if config.location_trial_batch_size > 1:
        return _run_location_finding_batched(
            questioner,
            config,
            rng=rng,
            output_dir=output_dir,
            method_name=method_name,
        )

    rng = rng or np.random.default_rng()
    rmse_totals = np.zeros(config.location_num_rounds, dtype=float)
    top_probability_totals = np.zeros(config.location_num_rounds, dtype=float)
    selected_eig_totals = np.zeros(config.location_num_rounds, dtype=float)

    _log_location(
        f"Running {config.location_num_trials} Location Finding trial(s): "
        f"method={method_name}, "
        f"rounds={config.location_num_rounds}, sources={config.location_num_sources}, "
        f"dim={config.location_dim}, noise_sd={config.location_noise_sd}, "
        f"candidates={config.location_target_num_candidates}, depth={config.location_search_depth}, "
        f"quadrature_order={config.location_eig_quadrature_order}, "
        f"max_total_beliefs={config.location_max_total_beliefs}, "
        f"max_llm_prompt_beliefs={config.location_max_llm_prompt_beliefs}, "
        f"num_mc_samples={config.num_mc_samples}, "
        f"posterior_mode={config.location_posterior_mode}, "
        f"location_seed={config.location_seed}",
        config,
    )
    for trial_idx in range(config.location_num_trials):
        env_rng = _location_trial_rng(config, trial_idx, rng)
        env = LocationFindingEnv(
            num_sources=config.location_num_sources,
            dim=config.location_dim,
            noise_sd=config.location_noise_sd,
            rng=env_rng,
        )
        observations: list[LocationObservation] = []
        strategy_library = LocationStrategyLibrary() if method_name in {"StrategyEIG", "StrategyEIG+root"} else None
        _log_location(
            f"trial {trial_idx + 1}/{config.location_num_trials}: sampled hidden environment "
            f"with true_sources={_format_source_array(env.true_theta)}",
            config,
        )
        if method_name == "Naive":
            final_estimate: SourceConfig | None = None
            final_rmse = float("inf")
            for round_idx in range(config.location_num_rounds):
                _write_to_log_if_configured(f"\nLocation Finding trial {trial_idx + 1}: Round {round_idx + 1}\n", config)
                _log_location(
                    f"trial {trial_idx + 1}/{config.location_num_trials}, "
                    f"round {round_idx + 1}/{config.location_num_rounds}, "
                    f"naive conversation observations={len(observations)}",
                    config,
                )
                best_location = choose_location_naive(questioner, observations, config)
                if best_location is None:
                    _log_location(
                        f"trial {trial_idx + 1}: no valid naive location after retries; "
                        f"skipping round {round_idx + 1}",
                        config,
                    )
                else:
                    observation = env.run_experiment(best_location)
                    observations.append(observation)
                    print_and_log(
                        f"[location] Selected query {list(best_location)} with score 0.000000; "
                        f"observed {observation.value:.2f}",
                        config,
                    )

                final_estimate = estimate_sources_naive(questioner, observations, config)
                final_rmse = source_rmse(final_estimate, env.true_theta)
                rmse_totals[round_idx] += final_rmse
                top_probability_totals[round_idx] += 1.0
                print_and_log(
                    f"[location] Naive source RMSE after round {round_idx + 1}: {final_rmse:.6f}",
                    config,
                )

            if config.location_plot_trials:
                if output_dir is None:
                    _log_location("plotting requested but no output directory was provided; skipping trial plot", config)
                else:
                    plot_path = output_dir / f"location_trial_{trial_idx + 1:03d}.png"
                    plot_state = LocationBeliefState(
                        hypotheses=[] if final_estimate is None else [final_estimate],
                        probabilities=[] if final_estimate is None else [1.0],
                    )
                    _plot_location_trial(
                        env,
                        observations,
                        plot_state,
                        trial_idx,
                        final_rmse,
                        1.0 if final_estimate is not None else 0.0,
                        plot_path,
                    )
                    _log_location(f"saved trial plot to {plot_path}", config)
            continue

        initial_hypotheses = generate_location_hypotheses(
            questioner,
            observations,
            None,
            config,
            label=f"trial {trial_idx + 1} initial belief generation",
        )
        belief_state = build_location_posterior(
            questioner,
            initial_hypotheses,
            observations,
            config,
            label=f"trial {trial_idx + 1} initial posterior scoring",
        )
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
            if method_name == "EIG":
                eig_belief_state, eig_sample_collapsed = sample_location_eig_belief_state(belief_state, config, rng)
                support_label = "EIG_support"
            elif method_name in {"StrategyEIG", "StrategyEIG+root"}:
                eig_belief_state = belief_state
                eig_sample_collapsed = False
                support_label = "strategy_root_support" if method_name == "StrategyEIG+root" else "strategy_support"
            else:
                eig_belief_state = belief_state
                eig_sample_collapsed = False
                support_label = "naive_support"
            _write_to_log_if_configured(f"\nLocation Finding trial {trial_idx + 1}: Round {round_idx + 1}\n", config)
            _log_location(
                f"trial {trial_idx + 1}/{config.location_num_trials}, "
                f"round {round_idx + 1}/{config.location_num_rounds}, "
                f"posterior {_summarize_belief_state(belief_state)}; "
                f"reservoir={len(belief_state.hypotheses)}, "
                f"prompt={len(prompt_belief_state.hypotheses)}, "
                f"{support_label}={len(eig_belief_state.hypotheses)}, "
                f"ESS={_location_effective_sample_size(belief_state):.2f}",
                config,
            )
            if eig_sample_collapsed:
                _log_location("EIG posterior sampling produced one unique hypothesis; EIG support is collapsed", config)
            if method_name == "EIG":
                candidates = generate_location_candidates(questioner, prompt_belief_state, observations, config)
                if not candidates:
                    best_location: Location | None = None
                    best_score = 0.0
                else:
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
            elif method_name == "Naive":
                best_location = choose_location_naive(questioner, observations, config)
                best_score = 0.0
            else:
                if strategy_library is None:
                    raise ValueError(f"{method_name} requires an in-memory strategy library")
                best_location, best_score, _best_strategy = choose_location_with_strategy_rollouts(
                    questioner,
                    belief_state,
                    observations,
                    strategy_library,
                    config,
                    rng,
                    round_idx,
                    fixed_root=method_name == "StrategyEIG+root",
                )
            if best_location is None:
                _log_location(
                    f"trial {trial_idx + 1}: no valid location after retries; skipping round {round_idx + 1}",
                    config,
                )
                current_rmse = _top_source_rmse(belief_state, env.true_theta)
                top_probability = belief_state.probabilities[0] if belief_state.probabilities else 0.0
                rmse_totals[round_idx] += current_rmse
                top_probability_totals[round_idx] += top_probability
                selected_eig_totals[round_idx] += best_score
                continue
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
            previous_belief_state = belief_state
            belief_state = build_location_posterior(
                questioner,
                merged_hypotheses,
                observations,
                config,
                context_state=previous_belief_state,
                label=f"trial {trial_idx + 1} round {round_idx + 1} posterior scoring",
            )
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
