from __future__ import annotations

import json
import math

import numpy as np

from core import BeliefState
from helpers import Config
from .formatting import _format_observations, _format_strategy_entries, _format_weighted_hypotheses, _location_posterior_labels, _source_config_schema_example, _source_count_text
from .physics import _hypothesis_log_prior, _logsumexp
from .types import LocationObservation, SourceConfig


def _measurement_model_description(config: Config) -> str:
    return (
        "Measurement model:\n"
        "A query is a 2D coordinate x = [x1,x2].\n"
        "The noiseless signal at x is:\n"
        "signal(x; theta) = b + sum_k alpha / (m + ||theta_k - x||^2)\n"
        "with b=0.1, alpha=1.0, m=0.0001.\n"
        f"Observed readings are on the raw signal scale with multiplicative noise: "
        f"y = signal(x; theta) * exp(epsilon), epsilon ~ Normal(0, {config.location_noise_sd}). "
        f"Equivalently, log y ~ Normal(log signal(x; theta), {config.location_noise_sd}). "
        "A single reading carries roughly +/-50% multiplicative noise, so do not over-trust one value.\n"
        "Numeric anchors: readings are approximately 0.1 far from all sources, approximately 1 at distance 1 "
        "from one source, and approximately 100 within distance 0.1 of a source."
    )


def _belief_system_prompt(config: Config, *, update: bool) -> str:
    dim_label = f"{config.location_dim}D"
    role = (
        f"You maintain and refresh a finite Bayesian belief support for a {dim_label} source-localization problem."
        if update
        else f"You maintain a finite Bayesian belief support for a {dim_label} source-localization problem."
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
        f"{_measurement_model_description(config)}\n\n"
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
        f"- Generate up to {config.location_num_generated_hypotheses} source configurations.\n"
        f"- Each hypothesis must contain exactly {config.location_num_sources} distinct "
        f"{config.location_dim}D source coordinates.\n"
        "- Do not repeat the same hypothesis with sources in a different order.\n"
        "- Source coordinates are hidden source locations, not measurement/query locations."
    )


def _belief_generation_messages(
    observations: list[LocationObservation],
    belief_state: BeliefState | None,
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
    context_state: BeliefState | None,
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
        f"{_measurement_model_description(config)}\n"
        "Interpretation: a very high observation near a coordinate means at least one source is probably very close "
        "to that coordinate; a low observation rules out any source being very close to that query location.\n\n"
        "context_probability is your current belief in each hypothesis (the previous round's posterior, or the "
        "Normal(0,1) prior on the first update). Produce the full posterior implied by the entire observation "
        "history below; use context_probability only as a soft anchor, not as a factor to multiply by the "
        "likelihood of these same observations again. "
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
    rng: np.random.Generator | None = None,
) -> list[list[LocationObservation]]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if len(observations) <= 1:
        return [list(observations) for _ in range(num_samples)]
    local_rng = rng if rng is not None else np.random.default_rng()
    histories: list[list[LocationObservation]] = []
    for _sample_idx in range(num_samples):
        permutation = local_rng.permutation(len(observations))
        histories.append([observations[int(index)] for index in permutation])
    return histories


def _candidate_generation_messages(
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    system = (
        "You propose candidate measurement locations for an adaptive 2D source-localization experiment.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. "
        "The goal is to choose the next query coordinate x = [x1,x2] "
        "to learn the source locations as efficiently as possible.\n\n"
        f"{_measurement_model_description(config)}\n\n"
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
    belief_state: BeliefState | None = None,
) -> list[dict[str, str]]:
    system = (
        "You choose the next measurement location for a 2D source-localization experiment.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. "
        "The hidden sources are fixed but unknown. A query is a 2D coordinate x = [x1,x2].\n\n"
        f"{_measurement_model_description(config)}\n\n"
        "Use only the task description, the previous query/observation history, "
        "and any current belief summary provided by the user.\n\n"
        "Return only this exact compact JSON shape as the final answer:\n"
        "{\"location\":[x1,y1]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        "- Output a measurement/query location only — do not output source hypotheses or source coordinates.\n"
        "- If you reason internally, still end with exactly one JSON object in the required shape."
    )
    belief_context = ""
    if belief_state is not None:
        belief_context = (
            "\n\nCurrent belief summary (top weighted source hypotheses):\n"
            f"{_format_weighted_hypotheses(belief_state, top_n=config.location_max_llm_prompt_beliefs)}"
        )
    user = (
        f"Observation history:\n{_format_observations(observations)}"
        f"{belief_context}\n\n"
        "Generate the single best next measurement location to help localize the hidden sources."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _naive_source_estimate_messages(
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    system = (
        "You output a JSON source-location estimate for a 2D source-localization game.\n"
        "Return only compact JSON in the required format. Do not include markdown or explanations outside the JSON.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}. "
        "The hidden sources are fixed but unknown. A query is a 2D coordinate x = [x1,x2]. "
        "Use only the observation history below.\n\n"
        f"{_measurement_model_description(config)}\n\n"
        "Each source coordinate is drawn independently from Normal(0,1); plausible values cluster near the "
        "origin (typically within ~[-3,3]).\n\n"
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


def _strategy_system_preamble(config: Config, num_strategies: int, task_instruction: str) -> str:
    return (
        "You propose natural-language adaptive strategies for a 2D source-localization experiment.\n\n"
        "A strategy is a few-sentence high-level plan for choosing future measurement locations. It should describe "
        "how to adapt after high, low, or ambiguous signal observations, not just name one coordinate.\n\n"
        f"{_measurement_model_description(config)}\n"
        "Interpretation: a very high observation means at least one source is probably very close to that query; "
        "a low observation rules out any source being very close to that query; signals from multiple sources add.\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"strategies\":[\"strategy text\",...]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        f"- Generate exactly {num_strategies} strategies.\n"
        f"- {task_instruction}\n"
        "- The strategies must differ substantively from one another."
    )


def _strategy_mutation_messages(
    retrieved_entries: list[LocationStrategyEntry],
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_mutation: int,
) -> list[dict[str, str]]:
    system = _strategy_system_preamble(
        config,
        num_mutation,
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
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_crossover: int,
) -> list[dict[str, str]]:
    system = _strategy_system_preamble(
        config,
        num_crossover,
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
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_diverse: int,
) -> list[dict[str, str]]:
    system = _strategy_system_preamble(
        config,
        num_diverse,
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


def _strategy_root_system_preamble(config: Config, num_strategies: int, task_instruction: str) -> str:
    return (
        "You propose adaptive strategies for a 2D source-localization experiment. Each strategy must include a fixed "
        "root measurement location that will be asked first whenever that strategy is evaluated or selected.\n\n"
        "A strategy is a few-sentence high-level plan for choosing future measurement locations after the fixed root "
        "measurement. The root_query is the first concrete query that commits the strategy to a distinctive opening "
        "measurement.\n\n"
        f"{_measurement_model_description(config)}\n"
        "Interpretation: a very high observation means at least one source is probably very close to that query; "
        "a low observation rules out any source being very close to that query; signals from multiple sources add.\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"strategies\":[{\"strategy\":\"strategy text\",\"root_query\":[x1,y1]},...]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        f"- Generate exactly {num_strategies} strategy/root_query pairs.\n"
        f"- {task_instruction}"
    )


def _strategy_root_mutation_messages(
    retrieved_entries: list[LocationStrategyEntry],
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_mutation: int,
) -> list[dict[str, str]]:
    system = _strategy_root_system_preamble(
        config,
        num_mutation,
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
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_crossover: int,
) -> list[dict[str, str]]:
    system = _strategy_root_system_preamble(
        config,
        num_crossover,
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
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
    num_diverse: int,
) -> list[dict[str, str]]:
    system = _strategy_root_system_preamble(
        config,
        num_diverse,
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
    belief_state: BeliefState,
    observations: list[LocationObservation],
    config: Config,
) -> list[dict[str, str]]:
    system = (
        "You choose the next measurement location for a 2D source-localization experiment by following a supplied "
        "natural-language strategy.\n\n"
        f"There are exactly {_source_count_text(config.location_num_sources)}.\n\n"
        f"{_measurement_model_description(config)}\n\n"
        "Return only this exact compact JSON shape:\n"
        "{\"location\":[x1,y1]}\n\n"
        "Rules:\n"
        "- Do not include markdown, comments, explanations, or trailing text.\n"
        "- Output a measurement/query location, not a source configuration.\n"
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


belief_generation_messages = _belief_generation_messages
belief_output_contract = _belief_output_contract
belief_system_prompt = _belief_system_prompt
candidate_generation_messages = _candidate_generation_messages
location_posterior_distribution_messages = _location_posterior_distribution_messages
naive_location_messages = _naive_location_messages
naive_source_estimate_messages = _naive_source_estimate_messages
naive_source_estimate_repair_messages = _naive_source_estimate_repair_messages
permuted_location_observation_histories = _permuted_location_observation_histories
strategy_crossover_messages = _strategy_crossover_messages
strategy_diverse_messages = _strategy_diverse_messages
strategy_location_messages = _strategy_location_messages
strategy_mutation_messages = _strategy_mutation_messages
strategy_root_crossover_messages = _strategy_root_crossover_messages
strategy_root_diverse_messages = _strategy_root_diverse_messages
strategy_root_mutation_messages = _strategy_root_mutation_messages
def strategy_system_preamble(
    bounds: tuple[float, ...],
    num_strategies: int,
    task_instruction: str,
    config: Config | None = None,
) -> str:
    del bounds
    config = config or Config(task="location_finding", location_noise_sd=0.5)
    return _strategy_system_preamble(config, num_strategies, task_instruction)


def strategy_root_system_preamble(
    bounds: tuple[float, ...],
    num_strategies: int,
    task_instruction: str,
    config: Config | None = None,
) -> str:
    del bounds
    config = config or Config(task="location_finding", location_noise_sd=0.5)
    return _strategy_root_system_preamble(config, num_strategies, task_instruction)

__all__ = [
    "belief_generation_messages",
    "belief_output_contract",
    "belief_system_prompt",
    "candidate_generation_messages",
    "location_posterior_distribution_messages",
    "naive_location_messages",
    "naive_source_estimate_messages",
    "naive_source_estimate_repair_messages",
    "strategy_crossover_messages",
    "strategy_diverse_messages",
    "strategy_location_messages",
    "strategy_mutation_messages",
    "strategy_root_crossover_messages",
    "strategy_root_diverse_messages",
    "strategy_root_mutation_messages",
    "strategy_root_system_preamble",
    "strategy_system_preamble",
]
