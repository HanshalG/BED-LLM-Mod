from __future__ import annotations

import json

import numpy as np

from core import BeliefState
from helpers import Config, print_and_log
from .types import Location, LocationObservation, LocationStrategyEntry


def _format_observations(observations: list[LocationObservation]) -> str:
    if not observations:
        return "[]"
    rows = [
        {"query": list(observation.query), "signal_strength": observation.value}
        for observation in observations
    ]
    return json.dumps(rows)


def _format_weighted_hypotheses(belief_state: BeliefState, top_n: int = 10) -> str:
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


def _summarize_belief_state(belief_state: BeliefState, top_n: int = 3) -> str:
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


def _log_location(message: str, config: Config) -> None:
    print_and_log(f"[location] {message}", config)
