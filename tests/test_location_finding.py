import json
import math

import numpy as np
import pytest

import model
from core import BeliefState
from helpers import Config
from core.experiment import run_from_config
from environments.location_finding.beliefs import (
    _location_effective_sample_size,
    build_location_belief_state,
    build_location_belief_state_unpruned,
    build_location_posterior,
    prompt_location_belief_state,
    prune_location_beliefs,
    sample_location_eig_belief_state,
)
from environments.location_finding.eig_bounds import (
    eig_bound_values_for_history,
    history_log_likelihoods,
    logmeanexp,
)
from environments.location_finding.env import LocationBEDEnvironment
from environments.location_finding.eig import expected_information_gain, score_candidate_locations
from environments.location_finding.parsing import (
    parse_best_source_estimate_from_completion,
    parse_candidate_locations,
    parse_location_strategies,
    parse_single_location_from_completion,
    parse_source_hypotheses,
    parse_strategy_location,
)
from environments.location_finding.physics import (
    observation_log_likelihood,
    round_positive_observation,
    sample_observation,
    signal_intensity_for_hypothesis,
    source_rmse,
)
from environments.location_finding.prompts import (
    belief_generation_messages as _belief_generation_messages,
    candidate_generation_messages as _candidate_generation_messages,
    location_posterior_distribution_messages as _location_posterior_distribution_messages,
    strategy_diverse_messages as _strategy_diverse_messages,
    strategy_location_messages as _strategy_location_messages,
    strategy_mutation_messages as _strategy_mutation_messages,
)
from environments.location_finding.strategy import (
    _StrategyEvaluationRequest,
    evaluate_location_strategies_by_rollout,
    evaluate_location_strategies_by_rollout_many,
    generate_strategy_locations_many,
    generate_location_strategies,
    _location_entropy,
    _rollout_entropy_reduction_score,
    _rollout_scoring_support,
)
from environments.location_finding.types import (
    LocationFindingEnv,
    LocationFindingMetrics,
    LocationObservation,
    LocationStrategyEntry,
    LocationStrategyLibrary,
    _StrategyLocationRequest,
    _StrategyRollout,
    normalize_source_config,
    project_location_to_step_radius,
)


def _run_location_config(model, config, rng=None, output_dir=None, method_name="EIG") -> LocationFindingMetrics:
    _run, summary = run_from_config(config, model, method_name=method_name, output_dir=output_dir)
    return LocationFindingMetrics(
        source_rmse=list(summary.metrics.get("source_rmse", [])),
        top_probability=list(summary.metrics.get("top_probability", [])),
        selected_eig=list(summary.metrics.get("selected_eig", [])),
        realized_entropy_drop=list(summary.metrics.get("realized_entropy_drop", [])),
    )


class FakeLocationModel:
    def __init__(self, completions: list[str]):
        self.completions = list(completions)
        self.calls: list[list[dict[str, str]]] = []
        self.batched_calls: list[list[list[dict[str, str]]]] = []

    def chat_complete(self, messages, temperature, num_responses=1):
        self.calls.append(messages)
        if not self.completions:
            raise AssertionError("No more completions configured")
        return [self.completions.pop(0)]

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        self.batched_calls.append(batch_messages)
        if len(self.completions) < len(batch_messages):
            raise AssertionError("No more completions configured")
        completions = self.completions[:len(batch_messages)]
        self.completions = self.completions[len(batch_messages):]
        return completions

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        raise AssertionError("chat_probabilities_messages_batched should not be used in Location Finding smoke test")


def test_location_step_radius_projection_clamps_to_previous_query():
    config = Config(
        task="location_finding",
        location_num_sources=1,
        location_dim=2,
        location_noise_sd=0.5,
        location_max_step_radius=0.5,
    )

    projected = project_location_to_step_radius((2.0, 0.0), (0.0, 0.0), config)

    assert projected == pytest.approx((0.5, 0.0))


def test_candidate_generation_projects_locations_to_step_radius():
    model = FakeLocationModel(['{"locations":[[2.0,0.0],[0.0,0.25]]}'])
    config = Config(
        task="location_finding",
        location_num_sources=1,
        location_dim=2,
        location_noise_sd=0.5,
        location_target_num_candidates=2,
        location_max_step_radius=0.5,
    )
    belief_state = BeliefState(hypotheses=[((0.0, 0.0),)], probabilities=[1.0])
    observations = [LocationObservation(query=(0.0, 0.0), value=1.0)]

    candidates = LocationBEDEnvironment(config).generate_candidate_actions(
        belief_state,
        [(observations[0].query, observations[0])],
        model,
        config,
    )

    assert candidates[0] == pytest.approx((0.5, 0.0))
    assert candidates[1] == pytest.approx((0.0, 0.25))


def test_location_environment_allows_lower_noise_for_constrained_oracle_variant():
    config = Config(
        task="location_finding",
        location_num_sources=1,
        location_dim=2,
        location_noise_sd=0.25,
        location_max_step_radius=0.5,
    )

    LocationBEDEnvironment(config).validate_config(config)


class RoutingLocationModel:
    def __init__(self, num_sources: int):
        self.num_sources = num_sources
        self.calls: list[list[dict[str, str]]] = []
        self.batched_calls: list[list[list[dict[str, str]]]] = []
        self._query_counter = 0

    def _completion_for_messages(self, messages) -> str:
        prompt = "\n".join(message["content"] for message in messages)
        if "root_query" in prompt and "strategy/root_query" in prompt:
            return json.dumps(
                {
                    "strategies": [
                        {
                            "strategy": "Ask the attached root query, then refine around whichever region remains plausible.",
                            "root_query": [0, 0],
                        }
                    ]
                }
            )
        if "adaptive strategies" in prompt and "{\"strategies\"" in prompt:
            return json.dumps(
                {
                    "strategies": [
                        "Start at the center, then refine around the strongest plausible source region."
                    ]
                }
            )
        if "candidate measurement locations" in prompt:
            return '{"locations": [[0, 0], [1, 1], [-1, -1]]}'
        if "{\"location\":[x1,y1]}" in prompt:
            x = 0.5 * (self._query_counter % 5 - 2)
            self._query_counter += 1
            return f'{{"location": [{x}, {x}]}}'
        if "best estimate of the hidden source locations" in prompt:
            return _source_hypotheses_json(self.num_sources, shifts=(0.0,))
        if "finite Bayesian belief support" in prompt:
            return _source_hypotheses_json(self.num_sources, shifts=(0.0, 0.2))
        return _source_hypotheses_json(self.num_sources, shifts=(0.0, 0.2))

    def chat_complete(self, messages, temperature, num_responses=1):
        self.calls.append(messages)
        return [self._completion_for_messages(messages)]

    def chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens=8192):
        self.batched_calls.append(batch_messages)
        return [self._completion_for_messages(messages) for messages in batch_messages]

    def chat_probabilities_messages_batched(self, messages, responses, temperature, block_size):
        raise AssertionError("chat_probabilities_messages_batched should not be used with analytical likelihood")


def _location_config(**overrides) -> Config:
    # Build defaults, then let caller overrides win.  All values are passed to
    # Config() in one shot so that __post_init__ (which resolves the
    # location_num_generated_hypotheses sentinel) sees the final field values.
    defaults: dict = dict(
        task="location_finding",
        location_num_rounds=1,
        location_num_trials=1,
        location_num_sources=3,
        location_dim=2,
        location_noise_sd=0.5,
        location_max_total_beliefs=1000,
        location_max_llm_prompt_beliefs=40,
        location_target_num_candidates=2,
        location_search_depth=1,
        location_eig_quadrature_order=5,
        generation_temperature_diverse=0.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _source_rows(num_sources: int, shift: float = 0.0) -> list[list[float]]:
    return [
        [
            round(-0.75 + shift + 0.5 * idx, 3),
            round(-0.35 + shift + 0.4 * (idx % 2), 3),
        ]
        for idx in range(num_sources)
    ]


def _source_hypotheses_json(num_sources: int, shifts: tuple[float, ...] = (0.0, 0.2)) -> str:
    return json.dumps({"hypotheses": [_source_rows(num_sources, shift) for shift in shifts]})


def test_location_env_matches_signal_defaults_and_records_noisy_observation():
    true_theta = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    env = LocationFindingEnv(true_theta=true_theta, rng=np.random.default_rng(0))

    assert env.true_theta.shape == (3, 2)
    assert env.noise_sd == pytest.approx(0.5)
    assert env.signal_intensity([1.0, 1.0]) == pytest.approx(0.1 + 1 / 2.0001 + 2 / 1.0001)

    observation = env.run_experiment([0.25, -0.5])

    assert observation.query == (0.25, -0.5)
    assert isinstance(observation.value, float)
    assert env.observed_data == [observation]


def test_location_env_supports_local_bump_signal_model():
    true_theta = np.array([[1.0, 0.0]])
    env = LocationFindingEnv(
        num_sources=1,
        true_theta=true_theta,
        signal_model="local_bump",
        signal_lengthscale=0.5,
        signal_amplitude=8.0,
        rng=np.random.default_rng(0),
    )

    on_source = env.signal_intensity([1.0, 0.0])
    far = env.signal_intensity([-2.0, 0.0])

    assert on_source == pytest.approx(8.1)
    assert far == pytest.approx(0.1, abs=1e-6)


def test_location_env_samples_branch_decoy_prior_near_endpoints():
    rng = np.random.default_rng(123)
    env = LocationFindingEnv(
        num_sources=1,
        source_prior="branch_decoy",
        source_radius=2.2,
        rng=rng,
    )
    endpoints = np.asarray([[-1.43, 0.0], [2.2, 1.76], [2.2, -1.76]])
    distances = np.linalg.norm(env.true_theta[0][None, :] - endpoints, axis=1)

    assert np.min(distances) < 0.5


def test_location_observation_model_is_lognormal_and_eig_prefers_informative_queries():
    rng = np.random.default_rng(123)
    mean = 2.5
    noise_sd = 0.5
    samples = np.asarray([sample_observation(mean, noise_sd, rng) for _idx in range(20_000)])
    log_residuals = np.log(samples) - math.log(mean)

    assert np.all(samples > 0.0)
    assert float(np.mean(log_residuals)) == pytest.approx(0.0, abs=0.02)
    assert float(np.var(log_residuals)) == pytest.approx(noise_sd**2, rel=0.08)
    assert observation_log_likelihood(mean, mean, noise_sd) > observation_log_likelihood(
        mean * math.exp(noise_sd),
        mean,
        noise_sd,
    )

    near = normalize_source_config([[0.0, 0.0]], 1, 2)
    far = normalize_source_config([[2.0, 2.0]], 1, 2)
    belief = BeliefState([near, far], [0.5, 0.5])
    on_source_eig = expected_information_gain(belief, (0.0, 0.0), noise_sd, 7)
    far_query_eig = expected_information_gain(belief, (3.0, 3.0), noise_sd, 7)

    assert on_source_eig >= 0.0
    assert far_query_eig >= 0.0
    assert on_source_eig > far_query_eig


def test_location_total_eig_bound_values_match_manual_likelihood_ratio():
    noise_sd = 0.5
    observations = [
        LocationObservation((0.0, 0.0), 1.2),
        LocationObservation((1.0, 0.0), 0.8),
    ]
    true_theta = np.asarray([[0.0, 0.0]], dtype=float)
    contrastive_thetas = np.asarray([[[1.0, 0.0]], [[2.0, 0.0]]], dtype=float)

    lower, upper = eig_bound_values_for_history(
        true_theta,
        observations,
        contrastive_thetas,
        noise_sd=noise_sd,
        chunk_size=1,
    )

    true_log = sum(
        observation_log_likelihood(
            observation.value,
            signal_intensity_for_hypothesis(((0.0, 0.0),), observation.query),
            noise_sd,
        )
        for observation in observations
    )
    contrastive_logs = np.asarray(
        [
            sum(
                observation_log_likelihood(
                    observation.value,
                    signal_intensity_for_hypothesis(tuple(tuple(row) for row in theta), observation.query),
                    noise_sd,
                )
                for observation in observations
            )
            for theta in contrastive_thetas
        ],
        dtype=float,
    )

    assert history_log_likelihoods(
        true_theta[None, :, :],
        observations,
        noise_sd=noise_sd,
    )[0] == pytest.approx(true_log)
    assert lower == pytest.approx(true_log - logmeanexp(np.concatenate(([true_log], contrastive_logs))))
    assert upper == pytest.approx(true_log - logmeanexp(contrastive_logs))


def test_location_round_metrics_include_realized_entropy_drop():
    config = _location_config(location_num_sources=1)
    env = LocationBEDEnvironment(config)
    truth = normalize_source_config([[0.0, 0.0]], 1, 2)
    other = normalize_source_config([[1.0, 0.0]], 1, 2)
    observation = LocationObservation(
        (0.0, 0.0),
        signal_intensity_for_hypothesis(truth, (0.0, 0.0)),
    )
    posterior = build_location_belief_state([truth, other], [observation], config)

    metrics = env.round_metrics(
        posterior,
        [((0.0, 0.0), observation)],
        np.asarray(truth, dtype=float),
    )

    support = list(posterior.hypotheses)
    previous = build_location_belief_state_unpruned(support, [], config)
    current = build_location_belief_state_unpruned(support, [observation], config)
    expected = _location_entropy(previous.probabilities) - _location_entropy(current.probabilities)
    assert metrics["realized_entropy_drop"] == pytest.approx(expected)
    assert metrics["realized_entropy_drop"] > 0.0


def test_rounded_lognormal_observations_preserve_positive_posterior_support():
    config = _location_config(location_num_sources=1)
    near_zero = round_positive_observation(0.004, 2)

    assert near_zero == pytest.approx(0.01)
    assert observation_log_likelihood(near_zero, 0.1, config.location_noise_sd) > float("-inf")

    hypotheses = [
        normalize_source_config([[3.0, 3.0]], 1, 2),
        normalize_source_config([[-3.0, -3.0]], 1, 2),
    ]
    state = build_location_belief_state(
        hypotheses,
        [LocationObservation((0.0, 0.0), near_zero)],
        config,
    )

    assert np.all(np.isfinite(state.probabilities))
    assert sum(state.probabilities) == pytest.approx(1.0)


def test_gemma_thought_only_strategy_location_is_forced_to_final_json():
    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return "Prompt"

        def __call__(self, text, add_special_tokens=False):
            return type("Tokenized", (), {"input_ids": list(range(len(text)))})()

        def parse_response(self, token_ids):
            return {"thinking": "draft"} if token_ids == [1] else {"content": "unused"}

    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                type("RequestOutput", (), {
                    "outputs": [
                        type("CompletionOutput", (), {
                            "text": "<|channel>thought\nchoose [0.25, -0.5]",
                            "token_ids": [1],
                            "finish_reason": "stop",
                        })()
                    ],
                    "prompt_token_ids": [11],
                })()
            ]
        return [
            type("RequestOutput", (), {
                "outputs": [
                    type("CompletionOutput", (), {
                        "text": 'Final Answer: {"location": [0.25, -0.5]}',
                        "token_ids": [2],
                        "finish_reason": "stop",
                    })()
                ],
                "prompt_token_ids": [22],
            })()
        ]

    adapter = model.GemmaVLLMAdapter.__new__(model.GemmaVLLMAdapter)
    adapter.model_name = "google/gemma-4-E4B-it"
    adapter.config = _location_config(
        location_num_sources=1,
        location_max_new_tokens=4096,
        batched_block_size=8,
    )
    adapter.tokenizer = _Tokenizer()
    adapter.max_model_len = 32768
    adapter.thinking = True
    adapter.thinking_max_new_tokens = 2048
    adapter.thinking_final_max_new_tokens = 512
    adapter.use_logprobs = False
    adapter.llm = type("LLM", (), {})()
    adapter.llm.generate = fake_generate

    belief_state = BeliefState([normalize_source_config([[0.0, 0.0]], 1, 2)], [1.0])
    locations = generate_strategy_locations_many(
        adapter,
        [_StrategyLocationRequest("Pick the next location.", belief_state, [])],
        adapter.config,
    )

    assert locations == [(0.25, -0.5)]
    assert len(calls) == 2
    assert calls[1][0] == ["Prompt<|channel>thought\nchoose [0.25, -0.5]\n<channel|>\nFinal Answer:"]


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_location_env_supports_configured_source_count_and_signal(num_sources):
    true_theta = np.asarray(_source_rows(num_sources), dtype=float)
    env = LocationFindingEnv(num_sources=num_sources, true_theta=true_theta, rng=np.random.default_rng(0))
    query = np.array([0.25, -0.5])

    distances_squared = np.sum((true_theta - query) ** 2, axis=1)
    expected_signal = 0.1 + float(np.sum(1.0 / (0.0001 + distances_squared)))

    assert env.true_theta.shape == (num_sources, 2)
    assert env.signal_intensity(query) == pytest.approx(expected_signal)


def test_parse_source_hypotheses_normalizes_order_dedupes_and_drops_invalid_entries():
    completion = """
    {
      "hypotheses": [
        [[1, 1], [-1, -1], [0, 0]],
        [[-1, -1], [0, 0], [1, 1]],
        [[0, 0], [1, 1]],
        {"sources": [[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]]}
      ]
    }
    """

    hypotheses = parse_source_hypotheses(completion, num_sources=3, dim=2)

    assert hypotheses == [
        ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)),
        ((-0.5, 0.0), (0.0, 0.5), (0.5, 0.0)),
    ]


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_parse_source_hypotheses_accepts_only_configured_source_count(num_sources):
    valid = _source_rows(num_sources)
    too_short = valid[:-1]
    too_long = valid + [[1.75, -1.25]]
    completion = json.dumps({"hypotheses": [valid, too_short, too_long]})

    hypotheses = parse_source_hypotheses(completion, num_sources=num_sources, dim=2)

    assert hypotheses == [normalize_source_config(valid, num_sources, 2)]


def test_parse_source_hypotheses_strips_eos_and_recovers_partial_complete_configs():
    completion = """
    {"hypotheses": [
      [[0, 0], [1, 1], [-1, -1]],
      [[0, 0], [1, 1], [-1, -1]],
      [[0, 0], [0, 0], [1, 1]],
      [[0.5, 0], [0, 0.5], [-0.5, 0]],
      [[1.2,
    <eos>
    """

    hypotheses = parse_source_hypotheses(completion, num_sources=3, dim=2)

    assert hypotheses == [
        ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)),
        ((-0.5, 0.0), (0.0, 0.5), (0.5, 0.0)),
    ]


def test_naive_parsers_prefer_final_json_after_reasoning_history_fragments():
    location_completion = """
    I considered previous query [0.0, 0.0] and signal 5.0.
    Final answer:
    {"location":[0.75,-0.25]}
    """
    source_completion = """
    The observations include [0.0, 0.0] and [1.0, 0.0].
    My current best source estimate is:
    {"sources":[[0.9,-0.2],[0.75,0.55]]}
    """

    assert parse_single_location_from_completion(location_completion, 2) == (0.75, -0.25)
    assert parse_best_source_estimate_from_completion(source_completion, 2, 2) == (
        (0.75, 0.55),
        (0.9, -0.2),
    )


def test_belief_generation_prompts_split_initial_and_update_modes():
    config = _location_config(location_max_llm_prompt_beliefs=7, location_num_generated_hypotheses=7)
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = BeliefState([hypothesis], [1.0])

    initial_messages = _belief_generation_messages([], None, config)
    update_messages = _belief_generation_messages(
        [LocationObservation((0.5, 0.5), 3.2)],
        belief_state,
        config,
    )

    initial_prompt = initial_messages[-1]["content"]
    initial_system = initial_messages[0]["content"]
    update_prompt = update_messages[-1]["content"]
    update_system = update_messages[0]["content"]
    assert "finite Bayesian belief support" in initial_system
    assert "Measurement model:" in initial_system
    assert "BoxingGym" not in initial_system
    assert "Generate diverse prior-plausible source configurations from Normal(0,1)" in initial_prompt
    assert "Current weighted hypotheses" not in initial_prompt
    assert "Generate up to 7 source configurations" in initial_prompt
    assert "finite Bayesian belief support" in update_system
    assert "Include refinements of high-probability current hypotheses" in update_prompt
    assert "Include alternatives near high-signal query locations" in update_prompt
    assert "Current weighted hypotheses:" in update_prompt


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_location_prompts_use_configured_source_count_without_stale_three_source_text(num_sources):
    config = _location_config(
        location_num_sources=num_sources,
        location_target_num_candidates=4,
        location_strategy_belief_summary_top_k=1,
    )
    hypothesis = normalize_source_config(_source_rows(num_sources), num_sources, 2)
    belief_state = BeliefState([hypothesis], [1.0])
    observations = [LocationObservation((0.5, 0.5), 3.2)]

    belief_messages = _belief_generation_messages(observations, belief_state, config)
    candidate_messages = _candidate_generation_messages(belief_state, observations, config)
    strategy_messages = _strategy_location_messages("Probe a high-disagreement location.", belief_state, observations, config)
    prompt_text = "\n".join(
        message["content"]
        for messages in (belief_messages, candidate_messages, strategy_messages)
        for message in messages
    )

    assert f"exactly {num_sources} hidden signal sources" in prompt_text
    assert f"[x{num_sources},y{num_sources}]" in prompt_text
    if num_sources != 3:
        assert "exactly 3 hidden signal sources" not in prompt_text
    if num_sources < 3:
        assert "[x3,y3]" not in prompt_text


def test_candidate_generation_prompt_separates_queries_from_source_configs():
    config = _location_config(location_target_num_candidates=5)
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = BeliefState([hypothesis], [1.0])

    messages = _candidate_generation_messages(
        belief_state,
        [LocationObservation((0.5, 0.5), 3.2)],
        config,
    )

    system_prompt = messages[0]["content"]
    user_prompt = messages[-1]["content"]
    assert "candidate measurement locations" in system_prompt
    assert "Do not output source configurations" in system_prompt
    assert "{\"locations\":[[x1,y1],[x1,y1],...]}" in system_prompt
    assert "Generate exactly 5 candidate measurement locations" in system_prompt
    assert "Observation history:" in user_prompt
    assert "Current weighted source hypotheses:" in user_prompt
    assert "BoxingGym" not in system_prompt


def test_support_grid_candidate_generation_uses_no_llm_and_respects_step_radius():
    config = _location_config(
        location_num_sources=1,
        location_target_num_candidates=4,
        location_candidate_generation_mode="support_grid",
        location_max_step_radius=0.5,
    )
    belief_state = BeliefState(
        [
            normalize_source_config([[2.0, 0.0]], 1, 2),
            normalize_source_config([[0.0, 1.0]], 1, 2),
        ],
        [0.8, 0.2],
    )
    history = [((0.0, 0.0), LocationObservation((0.0, 0.0), 0.2))]
    model = FakeLocationModel([])

    candidates = LocationBEDEnvironment(config).generate_candidate_actions_many(
        [belief_state],
        [history],
        model,
        config,
    )[0]

    assert len(candidates) == 4
    assert model.calls == []
    assert model.batched_calls == []
    assert candidates[0] == pytest.approx((0.5, 0.0))
    for candidate in candidates:
        assert math.dist(candidate, history[-1][0]) <= 0.5 + 1e-9


def test_parse_location_strategies_dedupes_json_wrapped_strategy_text():
    completion = """
    {"strategies": [
      "1. Start at the center, then move to high-disagreement quadrants.",
      "Start at the center, then move to high-disagreement quadrants.",
      {"strategy": "Probe suspected peaks with small offsets before broad exploration."}
    ]}
    """

    strategies = parse_location_strategies(completion)

    assert strategies == [
        "Start at the center, then move to high-disagreement quadrants.",
        "Probe suspected peaks with small offsets before broad exploration.",
    ]


def test_parse_strategy_location_accepts_single_location_json_without_bounds():
    assert parse_strategy_location('{"location": [0.25, -1]}', 2) == (0.25, -1.0)
    assert parse_strategy_location('{"location": [3, 0]}', 2) == (3.0, 0.0)


def test_strategy_library_retrieves_best_entries_and_new_library_is_empty():
    library = LocationStrategyLibrary()
    assert len(library) == 0

    library.replace_entries(
        [
            LocationStrategyEntry("weak", 0.1, 0.0, "[0, 0]", 0),
            LocationStrategyEntry("strong", 0.5, 0.2, "[1, 1]", 1),
            LocationStrategyEntry("steady", 0.5, 0.1, "[-1, 1]", 2),
        ]
    )

    retrieved = library.retrieve_top_m(2)

    assert [entry.strategy for entry in retrieved] == ["steady", "strong"]
    assert len(LocationStrategyLibrary()) == 0


def test_strategy_library_replace_entries_overwrites_previous_round():
    library = LocationStrategyLibrary()
    library.replace_entries([LocationStrategyEntry("round-1 strategy", 0.9, 0.0, "", 0)])
    assert len(library) == 1

    library.replace_entries([
        LocationStrategyEntry("round-2a", 0.3, 0.0, "", 1),
        LocationStrategyEntry("round-2b", 0.7, 0.0, "", 1),
    ])

    assert len(library) == 2
    assert [e.strategy for e in library.retrieve_top_m(2)] == ["round-2b", "round-2a"]


def test_strategy_prompts_include_history_beliefs_retrieved_examples_and_diversity_instructions():
    config = _location_config(location_strategy_belief_summary_top_k=1)
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = BeliefState([hypothesis], [1.0])
    observations = [LocationObservation((0.5, 0.5), 3.2)]
    retrieved = [LocationStrategyEntry("Start near the strongest current peak.", 1.2, 0.3, "[0, 0]", 0)]

    mutation_messages = _strategy_mutation_messages(retrieved, belief_state, observations, config, num_mutation=2)
    diverse_messages = _strategy_diverse_messages(belief_state, observations, config, num_diverse=2)
    location_messages = _strategy_location_messages(retrieved[0].strategy, belief_state, observations, config)

    mutation_system = mutation_messages[0]["content"]
    mutation_user = mutation_messages[-1]["content"]
    diverse_system = diverse_messages[0]["content"]
    location_system = location_messages[0]["content"]
    location_user = location_messages[-1]["content"]

    # Mutation prompt: shared preamble has "substantively"; user has retrieved entries
    assert "substantively" in mutation_system
    assert "Observation history so far" in mutation_user
    assert "Current belief summary (top 1 hypotheses" in mutation_user
    assert "Retrieved elite strategies" in mutation_user
    assert "Start near the strongest current peak." in mutation_user

    # Diverse prompt: preamble has "same first move" diversity instruction; user has NO retrieved context
    assert "same first move" in diverse_system
    assert "Retrieved elite strategies" not in diverse_messages[-1]["content"]

    # Strategy location execution prompt
    assert "{\"location\":[x1,y1]}" in location_system
    assert "Strategy to follow" in location_user
    assert "signal_strength" in location_user


def test_generate_location_strategies_four_phases():
    # num_candidates = retrieved(1) + mutation(1) + crossover(1) + diverse(1) = 4
    config = _location_config(
        location_strategy_num_retrieved=1,
        location_strategy_num_mutation=1,
        location_strategy_num_crossover=1,
        location_strategy_num_diverse=1,
    )
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = BeliefState([hypothesis], [1.0])
    library = LocationStrategyLibrary()
    library.replace_entries([LocationStrategyEntry("Elite plan.", 0.9, 0.1, "[0, 0]", 0)])
    model = FakeLocationModel([
        '{"strategies": ["Mutated plan."]}',
        '{"strategies": ["Crossover plan."]}',
        '{"strategies": ["Diverse plan."]}',
    ])

    strategies = generate_location_strategies(model, belief_state, [], library, config)

    assert strategies[0] == "Elite plan."       # Phase R: retrieved
    assert strategies[1] == "Mutated plan."     # Phase M: mutation
    assert strategies[2] == "Crossover plan."   # Phase C: crossover
    assert strategies[3] == "Diverse plan."     # Phase D: diverse
    assert len(strategies) == 4
    # Mutation prompt shows retrieved entries as parents
    assert "perturb" in model.calls[0][-1]["content"].lower()
    assert "Elite plan." in model.calls[0][-1]["content"]
    # Crossover prompt shows retrieved entries (now says "hybrid strategies that combine")
    assert "hybrid" in model.calls[1][-1]["content"].lower()
    # Diverse prompt has no retrieved section
    assert "Retrieved elite strategies" not in model.calls[2][-1]["content"]


def test_generate_location_strategies_empty_library_falls_back_to_diverse():
    # Round 1: library empty — mutation and crossover phases fall back to diverse prompts
    config = _location_config(
        location_strategy_num_retrieved=1,
        location_strategy_num_mutation=1,
        location_strategy_num_crossover=1,
        location_strategy_num_diverse=1,
    )
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = BeliefState([hypothesis], [1.0])
    library = LocationStrategyLibrary()  # empty
    model = FakeLocationModel([
        '{"strategies": ["Plan A."]}',
        '{"strategies": ["Plan B."]}',
        '{"strategies": ["Plan C."]}',
    ])

    strategies = generate_location_strategies(model, belief_state, [], library, config)

    # All 3 LLM calls used diverse prompts (no retrieved context)
    assert len(strategies) == 3
    assert all("Retrieved elite strategies" not in call[-1]["content"] for call in model.calls)


def test_parse_candidate_locations_filters_duplicates_and_invalid_entries_without_bounds():
    completion = '{"locations": [[0, 0], [3, 0], [0, 0], ["bad", 1], {"location": [1.5, -2]}]}'

    locations = parse_candidate_locations(completion, dim=2)

    assert locations == [(0.0, 0.0), (3.0, 0.0), (1.5, -2.0)]



def test_posterior_reweighting_prefers_hypothesis_matching_observation():
    config = _location_config()
    good = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    bad = normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2)
    query = (0.0, 0.0)
    observation = LocationObservation(query=query, value=signal_intensity_for_hypothesis(good, query))

    state = build_location_belief_state([bad, good], [observation], config)

    assert state.hypotheses[0] == good
    assert state.probabilities[0] > 0.999


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_posterior_reweighting_and_rmse_support_configured_source_count(num_sources):
    config = _location_config(location_num_sources=num_sources)
    good = normalize_source_config(_source_rows(num_sources), num_sources, 2)
    bad = normalize_source_config(_source_rows(num_sources, shift=0.4), num_sources, 2)
    query = good[0]
    observation = LocationObservation(query=query, value=signal_intensity_for_hypothesis(good, query))

    state = build_location_belief_state([bad, good], [observation], config)

    assert state.hypotheses[0] == good
    assert state.probabilities[0] > state.probabilities[1]
    assert source_rmse(good, np.asarray(good, dtype=float)) == pytest.approx(0.0)


def test_location_posterior_distribution_prompt_includes_support_and_contract():
    config = _location_config()
    hypotheses = [
        normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2),
        normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2),
    ]
    messages = _location_posterior_distribution_messages(
        [LocationObservation((0.5, 0.5), 3.2)],
        hypotheses,
        [0.7, 0.3],
        config,
    )

    prompt_text = "\n".join(message["content"] for message in messages)

    assert "Observation history" in prompt_text
    assert "signal_strength" in prompt_text
    assert "Candidate source hypotheses" in prompt_text
    assert "\"id\": \"h0\"" in prompt_text
    assert "\"id\": \"h1\"" in prompt_text
    assert "context_probability" in prompt_text
    assert "\"weights\"" in prompt_text


def test_build_location_posterior_llm_distribution_scores_current_support():
    config = _location_config(location_posterior_mode="llm_distribution")
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2)
    model = FakeLocationModel(['{"h0": 0.25, "h1": 0.75}'])

    state = build_location_posterior(
        model,
        [hypothesis_a, hypothesis_b],
        [LocationObservation((0.5, 0.5), 3.2)],
        config,
    )

    assert state.hypotheses == (hypothesis_b, hypothesis_a)
    assert state.probabilities == pytest.approx([0.75, 0.25])
    assert len(model.batched_calls) == 1
    assert len(model.batched_calls[0]) == 1


def test_build_location_posterior_llm_distribution_averages_permuted_history_samples():
    config = _location_config(
        location_posterior_mode="llm_distribution",
        belief_distribution_num_calls=3,
        belief_distribution_permute_history=True,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2)
    model = FakeLocationModel(
        [
            '{"h0": 1.0, "h1": 0.0}',
            'not json',
            '{"h0": 0.0, "h1": 1.0}',
        ]
    )

    state = build_location_posterior(
        model,
        [hypothesis_a, hypothesis_b],
        [LocationObservation((0.5, 0.5), 3.2), LocationObservation((-0.5, 0.5), 1.7)],
        config,
    )

    assert state.hypotheses == (hypothesis_a, hypothesis_b)
    assert state.probabilities == pytest.approx([0.5, 0.5])
    assert len(model.batched_calls) == 1
    assert len(model.batched_calls[0]) == 3
    assert all("\"h0\"" in messages[-1]["content"] for messages in model.batched_calls[0])
    assert all("\"h1\"" in messages[-1]["content"] for messages in model.batched_calls[0])
    assert all(messages[-1]["content"].count("signal_strength") == 2 for messages in model.batched_calls[0])


def test_build_location_posterior_llm_distribution_dedupes_and_falls_back_to_uniform():
    config = _location_config(
        location_posterior_mode="llm_distribution",
        belief_distribution_num_calls=2,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2)
    model = FakeLocationModel(["not json", "still not json"])

    state = build_location_posterior(model, [hypothesis_a, hypothesis_a, hypothesis_b], [], config)

    assert state.hypotheses == (hypothesis_a, hypothesis_b)
    assert state.probabilities == pytest.approx([0.5, 0.5])
    assert len(model.batched_calls[0]) == 2


def test_generated_refresh_hypotheses_compete_with_existing_support():
    config = _location_config()
    old = normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2)
    generated = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    query = (0.0, 0.0)
    observation = LocationObservation(query=query, value=signal_intensity_for_hypothesis(generated, query))

    state = build_location_belief_state([old, generated], [observation], config)

    assert state.hypotheses[0] == generated
    assert state.probabilities[0] > state.probabilities[1]


def test_deployed_belief_update_can_skip_llm_support_refresh():
    config = _location_config(location_belief_support_refresh_enabled=False)
    env = LocationBEDEnvironment(config)
    model = FakeLocationModel([])
    old = normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2)
    generated = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    query = (0.0, 0.0)
    observation = LocationObservation(query=query, value=signal_intensity_for_hypothesis(generated, query))
    belief_state = BeliefState([old], [1.0])

    states = env.update_belief_states(
        [belief_state],
        [[(query, observation)]],
        model,
        config,
    )

    assert len(model.calls) == 0
    assert len(model.batched_calls) == 0
    assert states[0].hypotheses == (old,)
    assert generated not in states[0].hypotheses


def test_truth_plus_sampled_rollout_scoring_support_includes_truth_and_caps_size():
    config = _location_config(
        location_num_sources=1,
        location_strategy_rollout_scoring_support_mode="truth_plus_sampled",
        location_strategy_rollout_scoring_support_size=2,
    )
    truth = normalize_source_config([[0, 0]], 1, 2)
    generated = [
        normalize_source_config([[1, 0]], 1, 2),
        normalize_source_config([[0, 1]], 1, 2),
        normalize_source_config([[2, 0]], 1, 2),
    ]
    belief_state = build_location_belief_state([truth, generated[0]], [], config)
    rollout = _StrategyRollout(
        request_index=0,
        strategy_index=0,
        strategy="test",
        truth=truth,
        start_probability=0.5,
        start_belief_state=belief_state,
        belief_state=belief_state,
        particle_support=list(belief_state.hypotheses),
        generated_hypotheses=list(generated),
        scoring_seed=123,
    )

    support = _rollout_scoring_support(rollout, config)

    assert support[0] == truth
    assert len(support) == 3
    assert set(support[1:]).issubset(set(generated))


def test_truth_plus_sampled_rollout_scoring_support_falls_back_to_union_without_generated_pool():
    config = _location_config(
        location_num_sources=1,
        location_strategy_rollout_scoring_support_mode="truth_plus_sampled",
        location_strategy_rollout_scoring_support_size=2,
    )
    truth = normalize_source_config([[0, 0]], 1, 2)
    other = normalize_source_config([[1, 0]], 1, 2)
    belief_state = build_location_belief_state([truth, other], [], config)
    rollout = _StrategyRollout(
        request_index=0,
        strategy_index=0,
        strategy="test",
        truth=truth,
        start_probability=0.5,
        start_belief_state=belief_state,
        belief_state=belief_state,
        particle_support=list(belief_state.hypotheses),
        scoring_seed=123,
    )

    assert _rollout_scoring_support(rollout, config) == [truth, other]


def test_truth_plus_sampled_entropy_score_recomputes_distributions_on_sampled_support():
    config = _location_config(
        location_num_sources=1,
        location_strategy_rollout_scoring_support_mode="truth_plus_sampled",
        location_strategy_rollout_scoring_support_size=2,
    )
    truth = normalize_source_config([[0, 0]], 1, 2)
    other_a = normalize_source_config([[1, 0]], 1, 2)
    other_b = normalize_source_config([[0, 1]], 1, 2)
    belief_state = build_location_belief_state([truth, other_a], [], config)
    observation = LocationObservation((0.0, 0.0), signal_intensity_for_hypothesis(truth, (0.0, 0.0)))
    rollout = _StrategyRollout(
        request_index=0,
        strategy_index=0,
        strategy="test",
        truth=truth,
        start_probability=0.5,
        start_belief_state=belief_state,
        belief_state=belief_state,
        particle_support=list(belief_state.hypotheses),
        generated_hypotheses=[other_a, other_b],
        simulated_observations=[observation],
        scoring_seed=123,
    )

    support = _rollout_scoring_support(rollout, config)
    expected_start = build_location_belief_state_unpruned(support, [], config)
    expected_final = build_location_belief_state_unpruned(support, [observation], config)

    assert _rollout_entropy_reduction_score(rollout, [], config) == pytest.approx(
        _location_entropy(expected_start.probabilities) - _location_entropy(expected_final.probabilities)
    )


def test_truth_start_end_rollout_scoring_support_uses_truth_start_and_end():
    config = _location_config(
        location_num_sources=1,
        location_strategy_rollout_scoring_support_mode="truth_start_end",
    )
    truth = normalize_source_config([[0, 0]], 1, 2)
    start_other = normalize_source_config([[1, 0]], 1, 2)
    end_other = normalize_source_config([[0, 1]], 1, 2)
    generated_only = normalize_source_config([[2, 0]], 1, 2)
    start_state = build_location_belief_state([truth, start_other], [], config)
    end_state = build_location_belief_state([truth, end_other], [], config)
    rollout = _StrategyRollout(
        request_index=0,
        strategy_index=0,
        strategy="test",
        truth=truth,
        start_probability=0.5,
        start_belief_state=start_state,
        belief_state=end_state,
        particle_support=[generated_only],
        generated_hypotheses=[generated_only],
        final_scoring_belief_state=end_state,
    )

    support = _rollout_scoring_support(rollout, config)

    assert support == [truth, start_other, end_other]


def test_fixed_common_rollout_scoring_support_uses_shared_support():
    config = _location_config(
        location_num_sources=1,
        location_strategy_rollout_scoring_support_mode="fixed_common",
    )
    truth = normalize_source_config([[0, 0]], 1, 2)
    start_other = normalize_source_config([[1, 0]], 1, 2)
    common_other = normalize_source_config([[0, 1]], 1, 2)
    generated_only = normalize_source_config([[2, 0]], 1, 2)
    start_state = build_location_belief_state([truth, start_other], [], config)
    rollout = _StrategyRollout(
        request_index=0,
        strategy_index=0,
        strategy="test",
        truth=truth,
        start_probability=0.5,
        start_belief_state=start_state,
        belief_state=start_state,
        particle_support=[generated_only],
        fixed_common_support=[truth, common_other],
        generated_hypotheses=[generated_only],
    )

    assert _rollout_scoring_support(rollout, config) == [truth, common_other]


def test_future_step_support_sum_scores_each_transition_on_next_support():
    config = _location_config(
        location_num_sources=1,
        location_strategy_rollout_score_mode="future_step_support_sum",
        location_strategy_discount_factor=1.0,
    )
    truth = normalize_source_config([[0, 0]], 1, 2)
    step1_other = normalize_source_config([[1, 0]], 1, 2)
    step2_other = normalize_source_config([[0, 1]], 1, 2)
    start_state = build_location_belief_state([truth, step1_other], [], config)
    observation_1 = LocationObservation(
        (0.0, 0.0),
        signal_intensity_for_hypothesis(truth, (0.0, 0.0)),
    )
    observation_2 = LocationObservation(
        (0.0, 1.0),
        signal_intensity_for_hypothesis(truth, (0.0, 1.0)),
    )
    step1_support = [truth, step1_other]
    step2_support = [truth, step1_other, step2_other]
    rollout = _StrategyRollout(
        request_index=0,
        strategy_index=0,
        strategy="test",
        truth=truth,
        start_probability=0.5,
        start_belief_state=start_state,
        belief_state=start_state,
        particle_support=list(step2_support),
        simulated_observations=[observation_1, observation_2],
        simulated_supports=[step1_support, step2_support],
    )

    expected_step_1 = (
        _location_entropy(build_location_belief_state_unpruned(step1_support, [], config).probabilities)
        - _location_entropy(build_location_belief_state_unpruned(step1_support, [observation_1], config).probabilities)
    )
    expected_step_2 = (
        _location_entropy(build_location_belief_state_unpruned(step2_support, [observation_1], config).probabilities)
        - _location_entropy(
            build_location_belief_state_unpruned(step2_support, [observation_1, observation_2], config).probabilities
        )
    )

    assert _rollout_entropy_reduction_score(rollout, [], config) == pytest.approx(
        expected_step_1 + expected_step_2
    )


def test_pruning_keeps_top_k_when_over_budget_and_renormalizes():
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(45)
    ]
    probabilities = [float(idx + 1) for idx in range(45)]
    state = BeliefState(hypotheses, probabilities)

    pruned = prune_location_beliefs(state, max_beliefs=40)

    assert len(pruned.hypotheses) == 40
    assert sum(pruned.probabilities) == pytest.approx(1.0)
    assert pruned.hypotheses[0] == hypotheses[-1]
    assert pruned.hypotheses[-1] == hypotheses[5]


def test_pruning_preserves_all_beliefs_when_within_budget():
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(4)
    ]
    state = BeliefState(hypotheses, [0.25] * 4)

    pruned = prune_location_beliefs(state, max_beliefs=40)

    assert pruned == state


def test_build_belief_state_trims_reservoir_to_max_total_beliefs():
    config = _location_config(location_max_total_beliefs=5)
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(10)
    ]

    state = build_location_belief_state(hypotheses, [], config)

    assert len(state.hypotheses) == 5
    assert sum(state.probabilities) == pytest.approx(1.0)


def test_prompt_belief_state_keeps_top_llm_prompt_beliefs():
    config = _location_config(location_max_llm_prompt_beliefs=3)
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(5)
    ]
    state = BeliefState(hypotheses, [0.05, 0.1, 0.2, 0.25, 0.4])

    prompt_state = prompt_location_belief_state(state, config)

    assert len(prompt_state.hypotheses) == 3
    assert prompt_state.hypotheses == (hypotheses[4], hypotheses[3], hypotheses[2])
    assert sum(prompt_state.probabilities) == pytest.approx(1.0)


def test_eig_belief_sampling_uses_num_mc_samples_and_renormalizes():
    config = _location_config(num_mc_samples=10)
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(50)
    ]
    state = BeliefState(hypotheses, [1 / 50] * 50)

    sampled_state, fallback_used = sample_location_eig_belief_state(state, config, np.random.default_rng(0))

    assert not fallback_used
    assert 1 < len(sampled_state.hypotheses) <= 10
    assert set(sampled_state.hypotheses).issubset(set(hypotheses))
    assert sum(sampled_state.probabilities) == pytest.approx(1.0)


def test_eig_belief_sampling_keeps_pure_posterior_sample_when_collapsed():
    config = _location_config(num_mc_samples=5)
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(6)
    ]
    state = BeliefState(hypotheses, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    sampled_state, sample_collapsed = sample_location_eig_belief_state(state, config, np.random.default_rng(0))

    assert sample_collapsed
    assert len(sampled_state.hypotheses) == 1
    assert sampled_state.hypotheses[0] == hypotheses[0]
    assert sum(sampled_state.probabilities) == pytest.approx(1.0)


def test_location_effective_sample_size_reports_weight_concentration():
    concentrated = BeliefState([((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))], [1.0])
    assert _location_effective_sample_size(concentrated) == pytest.approx(1.0)


def test_eig_is_zero_for_identical_predictions_and_positive_for_separated_predictions():
    config = _location_config(location_eig_quadrature_order=7)
    identical = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    separated = normalize_source_config([[2, 2], [1, 2], [2, 1]], 3, 2)

    identical_state = BeliefState([identical, identical], [0.5, 0.5])
    separated_state = BeliefState([identical, separated], [0.5, 0.5])

    assert expected_information_gain(identical_state, (0.0, 0.0), 0.5, 7) == pytest.approx(0.0, abs=1e-9)
    assert expected_information_gain(separated_state, (0.0, 0.0), 0.5, 7) > 0.01


def test_depth_two_eig_generates_fresh_candidates_per_hypothesis_branch():
    config = _location_config(location_search_depth=2, location_eig_quadrature_order=5)
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[2, 2], [1, 2], [2, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    candidates = [(0.0, 0.0), (1.0, 1.0)]
    fresh_candidate_completion = '{"locations": [[-1, -1], [1, 1]]}'
    model = FakeLocationModel([fresh_candidate_completion for _ in range(4)])

    scores = score_candidate_locations(belief_state, candidates, config, questioner=model, observations=[])

    # Depth-2 forward search: analytical branch refresh, then fresh candidates per branch.
    assert len(model.calls) == 4
    assert "candidate measurement locations" in model.calls[0][-1]["content"]

    # Manual scores: per-hypothesis branch uses mean signal as representative observation,
    # builds future belief state analytically, scores fresh candidates at depth-1.
    fresh_candidates = [(-1.0, -1.0), (1.0, 1.0)]
    manual_scores = []
    for candidate in candidates:
        immediate = expected_information_gain(
            belief_state,
            candidate,
            config.location_noise_sd,
            config.location_eig_quadrature_order,
        )
        expected_future = 0.0
        for hypothesis, prob in zip(belief_state.hypotheses, belief_state.probabilities):
            mean = signal_intensity_for_hypothesis(hypothesis, candidate)
            future_state = build_location_belief_state(
                list(belief_state.hypotheses),
                [LocationObservation(candidate, float(mean))],
                config,
            )
            best_future = max(
                expected_information_gain(
                    future_state,
                    fc,
                    config.location_noise_sd,
                    config.location_eig_quadrature_order,
                )
                for fc in fresh_candidates
            )
            expected_future += prob * best_future
        manual_scores.append(immediate + expected_future)

    assert scores == pytest.approx(manual_scores)


def test_strategy_rollout_samples_gaussian_observations_refreshes_beliefs_and_records_fingerprint():
    config = _location_config(
        location_strategy_num_rollouts=2,
        location_strategy_planning_depth=1,
        location_strategy_belief_summary_top_k=2,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[2, 2], [1, 2], [2, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    model = FakeLocationModel(
        [
            '{"location": [0, 0]}',
            '{"location": [0, 0]}',
            '{"hypotheses": []}',
            '{"hypotheses": []}',
        ]
    )

    evaluations = evaluate_location_strategies_by_rollout(
        model,
        ["Probe the center first, then refine any high-signal region."],
        belief_state,
        [],
        config,
        np.random.default_rng(3),
    )

    assert len(evaluations) == 1
    assert len(evaluations[0].rollout_scores) == 2
    assert all(math.isfinite(score) for score in evaluations[0].rollout_scores)
    assert evaluations[0].score_variance >= 0.0
    assert evaluations[0].root_query_fingerprint == "[0, 0]"
    assert len(model.batched_calls) == 2
    assert len(model.batched_calls[0]) == 2
    assert len(model.batched_calls[1]) == 2
    assert "signal_strength" in model.batched_calls[1][0][-1]["content"]


def _weighted_hypotheses_from_strategy_prompt(prompt: str) -> list[dict]:
    marker = "Current weighted source hypotheses:\n"
    start_idx = prompt.index(marker) + len(marker)
    end_idx = prompt.index("\n\nFollowing the strategy", start_idx)
    return json.loads(prompt[start_idx:end_idx])


def test_strategy_rollout_uses_closed_form_steps_then_one_final_refresh():
    config = _location_config(
        location_num_rounds=3,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=3,
        location_strategy_belief_summary_top_k=2,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 0], [1, -1], [-1, 1]], 3, 2)
    hypothesis_c = normalize_source_config([[0.5, 0.5], [1.5, -0.5], [-1.5, 0.25]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    model = FakeLocationModel(
        [
            '{"location": [1, 1]}',
            '{"location": [0, 0]}',
            '{"location": [-1, -1]}',
            json.dumps({"hypotheses": [[list(source) for source in hypothesis_c]]}),
        ]
    )

    evaluations = evaluate_location_strategies_by_rollout(
        model,
        ["Probe a discriminative suspected peak, then refine with offsets."],
        belief_state,
        [],
        config,
        np.random.default_rng(4),
    )

    assert len(evaluations) == 1
    assert math.isfinite(evaluations[0].mean_score)
    assert evaluations[0].mean_score > 0.0
    assert len(model.batched_calls) == 4
    assert all(len(batch) == 1 for batch in model.batched_calls)
    assert "{\"location\":[x1,y1]}" in model.batched_calls[0][0][0]["content"]
    assert "{\"location\":[x1,y1]}" in model.batched_calls[1][0][0]["content"]
    assert "{\"location\":[x1,y1]}" in model.batched_calls[2][0][0]["content"]
    assert "{\"hypotheses\"" in model.batched_calls[3][0][-1]["content"]

    second_location_prompt = model.batched_calls[1][0][-1]["content"]
    third_location_prompt = model.batched_calls[2][0][-1]["content"]
    final_refresh_prompt = model.batched_calls[3][0][-1]["content"]
    assert second_location_prompt.count("signal_strength") == 1
    assert third_location_prompt.count("signal_strength") == 2
    assert final_refresh_prompt.count("signal_strength") == 3

    second_prompt_hypotheses = _weighted_hypotheses_from_strategy_prompt(second_location_prompt)
    second_prompt_probabilities = [row["probability"] for row in second_prompt_hypotheses]
    assert any(abs(probability - 0.5) > 1e-3 for probability in second_prompt_probabilities)


def test_strategy_rollout_can_disable_final_refresh_for_scoring():
    config = _location_config(
        location_num_rounds=3,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=3,
        location_strategy_belief_summary_top_k=2,
        location_strategy_rollout_final_refresh_enabled=False,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 0], [1, -1], [-1, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    model = FakeLocationModel(
        [
            '{"location": [1, 1]}',
            '{"location": [0, 0]}',
            '{"location": [-1, -1]}',
        ]
    )

    evaluations = evaluate_location_strategies_by_rollout(
        model,
        ["Probe a discriminative suspected peak, then refine with offsets."],
        belief_state,
        [],
        config,
        np.random.default_rng(4),
    )

    assert len(evaluations) == 1
    assert math.isfinite(evaluations[0].mean_score)
    assert len(model.batched_calls) == 3
    assert all("{\"location\":[x1,y1]}" in batch[0][0]["content"] for batch in model.batched_calls)
    assert all("{\"hypotheses\"" not in batch[0][-1]["content"] for batch in model.batched_calls)


def test_strategy_rollout_can_use_analytic_eig_future_queries_without_llm_calls():
    config = _location_config(
        location_num_rounds=3,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=3,
        location_strategy_belief_summary_top_k=2,
        location_strategy_rollout_final_refresh_enabled=False,
        location_strategy_rollout_query_mode="analytic_eig",
        location_signal_model="local_bump",
        location_signal_lengthscale=0.5,
        location_signal_amplitude=8.0,
        location_max_step_radius=0.75,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 0], [1, -1], [-1, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    model = FakeLocationModel([])

    evaluations = evaluate_location_strategies_by_rollout(
        model,
        ["Use the fixed root, then let the calibrated executor refine analytically."],
        belief_state,
        [],
        config,
        np.random.default_rng(4),
        root_queries=[(0.0, 0.0)],
    )

    assert len(evaluations) == 1
    assert evaluations[0].root_query == (0.0, 0.0)
    assert math.isfinite(evaluations[0].mean_score)
    assert evaluations[0].mean_score > 0.0
    assert model.batched_calls == []
    assert model.calls == []


def test_batched_strategy_rollout_can_use_analytic_eig_future_queries_without_llm_calls():
    config = _location_config(
        location_num_rounds=3,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=3,
        location_strategy_belief_summary_top_k=2,
        location_strategy_rollout_final_refresh_enabled=False,
        location_strategy_rollout_query_mode="analytic_eig",
        location_signal_model="local_bump",
        location_signal_lengthscale=0.5,
        location_signal_amplitude=8.0,
        location_max_step_radius=0.75,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 0], [1, -1], [-1, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    model = FakeLocationModel([])

    evaluations_many = evaluate_location_strategies_by_rollout_many(
        model,
        [
            _StrategyEvaluationRequest(
                strategies=["Use the fixed root, then let the calibrated executor refine analytically."],
                belief_state=belief_state,
                observations=[],
                rng=np.random.default_rng(4),
                root_queries=[(0.0, 0.0)],
            )
        ],
        config,
    )

    assert len(evaluations_many) == 1
    assert len(evaluations_many[0]) == 1
    assert evaluations_many[0][0].root_query == (0.0, 0.0)
    assert math.isfinite(evaluations_many[0][0].mean_score)
    assert evaluations_many[0][0].mean_score > 0.0
    assert model.batched_calls == []
    assert model.calls == []


def test_strategy_rollout_depth_is_capped_by_remaining_rounds():
    config = _location_config(
        location_num_rounds=2,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=3,
        location_strategy_belief_summary_top_k=2,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 0], [1, -1], [-1, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    existing_observation = LocationObservation((0.0, 0.0), 1.0)
    model = FakeLocationModel(
        [
            '{"location": [1, 1]}',
            '{"hypotheses": []}',
        ]
    )

    evaluations = evaluate_location_strategies_by_rollout(
        model,
        ["Probe a discriminative suspected peak, then refine with offsets."],
        belief_state,
        [existing_observation],
        config,
        np.random.default_rng(4),
    )

    assert len(evaluations) == 1
    assert math.isfinite(evaluations[0].mean_score)
    assert len(model.batched_calls) == 2
    assert "{\"location\":[x1,y1]}" in model.batched_calls[0][0][0]["content"]
    assert "{\"hypotheses\"" in model.batched_calls[1][0][-1]["content"]
    assert model.batched_calls[1][0][-1]["content"].count("signal_strength") == 2


def test_strategy_rollout_can_refresh_hypotheses_at_each_simulated_step():
    config = _location_config(
        location_num_rounds=2,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=2,
        location_strategy_belief_summary_top_k=3,
        location_strategy_rollout_refresh_hypotheses_each_step=True,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 0], [1, -1], [-1, 1]], 3, 2)
    generated_step_1 = normalize_source_config([[0.5, 0.5], [1.5, -0.5], [-1.5, 0.25]], 3, 2)
    generated_step_2 = normalize_source_config([[0.2, 0.2], [1.2, -0.2], [-1.2, 0.5]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    model = FakeLocationModel(
        [
            '{"location": [1, 1]}',
            json.dumps({"hypotheses": [[list(source) for source in generated_step_1], ["bad", 1]]}),
            '{"location": [0, 0]}',
            json.dumps({"hypotheses": [[list(source) for source in generated_step_2], [[0, 0]]]}),
            '{"hypotheses": []}',
        ]
    )

    evaluations = evaluate_location_strategies_by_rollout(
        model,
        ["Probe a suspected peak, then refine with offsets."],
        belief_state,
        [],
        config,
        np.random.default_rng(4),
    )

    assert len(evaluations) == 1
    assert math.isfinite(evaluations[0].mean_score)
    assert len(model.batched_calls) == 5
    assert "{\"location\":[x1,y1]}" in model.batched_calls[0][0][0]["content"]
    assert "{\"hypotheses\"" in model.batched_calls[1][0][-1]["content"]
    assert "{\"location\":[x1,y1]}" in model.batched_calls[2][0][0]["content"]
    assert "{\"hypotheses\"" in model.batched_calls[3][0][-1]["content"]
    assert "{\"hypotheses\"" in model.batched_calls[4][0][-1]["content"]

    second_location_prompt = model.batched_calls[2][0][-1]["content"]
    final_refresh_prompt = model.batched_calls[4][0][-1]["content"]
    assert second_location_prompt.count("signal_strength") == 1
    assert final_refresh_prompt.count("signal_strength") == 2
    assert "0.5" in second_location_prompt


def test_strategy_rollout_final_refresh_uses_llm_posterior_mode():
    config = _location_config(
        location_num_rounds=2,
        location_posterior_mode="llm_distribution",
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=2,
        location_strategy_belief_summary_top_k=2,
    )
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[0, 0], [1, -1], [-1, 1]], 3, 2)
    belief_state = BeliefState([hypothesis_a, hypothesis_b], [0.5, 0.5])
    model = FakeLocationModel(
        [
            '{"location": [1, 1]}',    # depth-0 location
            '{"h0": 0.6, "h1": 0.4}', # depth-0 belief update (llm_distribution)
            '{"location": [0, 0]}',    # depth-1 location
            '{"h0": 0.5, "h1": 0.5}', # depth-1 belief update (llm_distribution)
            '{"hypotheses": []}',      # final hypothesis refresh
            '{"h0": 0.8, "h1": 0.2}', # final posterior scoring
        ]
    )

    evaluations = evaluate_location_strategies_by_rollout(
        model,
        ["Probe one suspected peak, then refine using the new signal."],
        belief_state,
        [],
        config,
        np.random.default_rng(5),
    )

    assert len(evaluations) == 1
    assert math.isfinite(evaluations[0].mean_score)
    # 2 depths × (location + belief update) + hypothesis refresh + final scoring = 6 batched calls
    assert len(model.batched_calls) == 6
    assert "{\"location\":[x1,y1]}" in model.batched_calls[0][0][0]["content"]  # depth-0 location
    assert "\"weights\"" in model.batched_calls[1][0][-1]["content"]             # depth-0 belief update
    assert "{\"location\":[x1,y1]}" in model.batched_calls[2][0][0]["content"]  # depth-1 location
    assert "\"weights\"" in model.batched_calls[3][0][-1]["content"]             # depth-1 belief update
    assert "{\"hypotheses\"" in model.batched_calls[4][0][-1]["content"]         # hypothesis refresh
    assert "\"weights\"" in model.batched_calls[5][0][-1]["content"]             # final posterior scoring
    assert model.batched_calls[5][0][-1]["content"].count("signal_strength") == 2


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_run_location_one_round_with_fake_llm_smoke(tmp_path, num_sources):
    initial_hypotheses = _source_hypotheses_json(num_sources)
    candidate_locations = '{"locations": [[0, 0], [1, 1]]}'
    update_hypotheses = _source_hypotheses_json(num_sources, shifts=(0.0, 0.1))
    model = FakeLocationModel([initial_hypotheses, candidate_locations, update_hypotheses])
    config = _location_config(
        location_num_sources=num_sources,
        location_target_num_candidates=2,
        location_search_depth=1,
        location_plot_trials=True,
    )

    metrics = _run_location_config(model, config, rng=np.random.default_rng(1), output_dir=tmp_path)

    assert len(metrics.source_rmse) == 1
    assert len(metrics.top_probability) == 1
    assert len(metrics.selected_eig) == 1
    assert math.isfinite(metrics.source_rmse[0])
    assert 0.0 <= metrics.top_probability[0] <= 1.0
    assert metrics.selected_eig[0] >= 0.0
    assert len(model.calls) == 3
    assert f"exactly {num_sources} hidden signal sources" in model.calls[0][0]["content"]
    assert "y = signal(x; theta) * exp(epsilon)" in model.calls[0][0]["content"]
    assert "epsilon ~ Normal(0, 0.5)" in model.calls[0][0]["content"]
    assert "Do not output source configurations" in model.calls[1][0]["content"]
    plot_path = tmp_path / "location_trial_001.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_run_location_eig_llm_posterior_smoke(tmp_path):
    initial_hypotheses = _source_hypotheses_json(3)
    candidate_locations = '{"locations": [[0, 0], [1, 1]]}'
    update_hypotheses = _source_hypotheses_json(3, shifts=(0.0, 0.1))
    model = FakeLocationModel(
        [
            initial_hypotheses,
            '{"h0": 0.6, "h1": 0.4}',
            candidate_locations,
            update_hypotheses,
            '{"h0": 0.7, "h1": 0.2, "h2": 0.1}',
        ]
    )
    config = _location_config(
        location_posterior_mode="llm_distribution",
        location_target_num_candidates=2,
        location_search_depth=1,
        location_plot_trials=False,
    )

    metrics = _run_location_config(model, config, rng=np.random.default_rng(1), output_dir=tmp_path)

    assert len(metrics.source_rmse) == 1
    assert len(metrics.top_probability) == 1
    assert metrics.top_probability[0] == pytest.approx(0.7)
    assert len(model.calls) == 3
    assert len(model.batched_calls) == 2
    assert "\"weights\"" in model.batched_calls[0][0][-1]["content"]
    assert "\"weights\"" in model.batched_calls[1][0][-1]["content"]


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_run_location_strategy_eig_one_round_with_fake_llm_smoke(tmp_path, num_sources):
    initial_hypotheses = _source_hypotheses_json(num_sources)
    strategy_completion = """
    {"strategies": [
      "Start at the center to test whether any source is near the origin, then refine around high-signal offsets."
    ]}
    """
    update_hypotheses = _source_hypotheses_json(num_sources, shifts=(0.0, 0.1))
    model = FakeLocationModel(
        [
            initial_hypotheses,
            strategy_completion,
            '{"location": [0, 0]}',
            '{"hypotheses": []}',
            '{"location": [1, 1]}',
            update_hypotheses,
        ]
    )
    config = _location_config(
        location_num_sources=num_sources,
        location_strategy_num_mutation=0, location_strategy_num_crossover=0, location_strategy_num_diverse=0,
        location_strategy_num_retrieved=1,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=1,
        location_plot_trials=False,
    )

    metrics = _run_location_config(
        model,
        config,
        rng=np.random.default_rng(1),
        output_dir=tmp_path,
        method_name="StrategyEIG",
    )

    assert len(metrics.source_rmse) == 1
    assert len(metrics.top_probability) == 1
    assert len(metrics.selected_eig) == 1
    assert math.isfinite(metrics.selected_eig[0])
    assert len(model.calls) == 3
    assert len(model.batched_calls) == 3
    assert "Generate diverse strategies" in model.calls[1][-1]["content"]  # round 1: library empty, diverse prompt
    assert f"exactly {num_sources} hidden signal sources" in model.batched_calls[0][0][0]["content"]
    assert "{\"location\":[x1,y1]}" in model.batched_calls[0][0][0]["content"]

    per_trial_path = tmp_path / "location_per_trial_metrics.json"
    assert per_trial_path.exists()
    per_trial_payload = json.loads(per_trial_path.read_text())
    assert per_trial_payload["metric_names"] == [
        "ess",
        "realized_entropy_drop",
        "selected_eig",
        "source_rmse",
        "support_size",
        "top_probability",
    ]
    assert per_trial_payload["trials"][0]["metrics"]["realized_entropy_drop"] == pytest.approx(
        metrics.realized_entropy_drop
    )
    assert per_trial_payload["trials"][0]["metrics"]["source_rmse"] == pytest.approx(metrics.source_rmse)
    assert per_trial_payload["trials"][0]["metrics"]["selected_eig"] == pytest.approx(metrics.selected_eig)

    strategies_path = tmp_path / "location_selected_strategies.jsonl"
    assert strategies_path.exists()
    strategy_records = [json.loads(line) for line in strategies_path.read_text().splitlines()]
    assert len(strategy_records) == 1
    assert strategy_records[0]["trial_index"] == 0
    assert strategy_records[0]["round_index"] == 0
    assert strategy_records[0]["location"] == [1.0, 1.0]
    assert strategy_records[0]["strategy"].startswith("Start at the center")


def test_run_location_strategy_eig_llm_posterior_smoke(tmp_path):
    initial_hypotheses = _source_hypotheses_json(3)
    strategy_completion = """
    {"strategies": [
      "Start at the center to test whether any source is near the origin, then refine around high-signal offsets."
    ]}
    """
    update_hypotheses = _source_hypotheses_json(3, shifts=(0.0, 0.1))
    model = FakeLocationModel(
        [
            initial_hypotheses,              # calls[0]: initial hypothesis gen
            '{"h0": 0.6, "h1": 0.4}',       # batched_calls[0]: initial posterior
            strategy_completion,             # calls[1]: strategy proposal
            '{"location": [0, 0]}',          # batched_calls[1]: rollout depth-0 location
            '{"h0": 0.7, "h1": 0.3}',       # batched_calls[2]: rollout depth-0 belief update
            '{"hypotheses": []}',            # batched_calls[3]: final hypothesis refresh
            '{"h0": 0.8, "h1": 0.2}',       # batched_calls[4]: rollout final posterior
            '{"location": [1, 1]}',          # batched_calls[5]: final location selection
            update_hypotheses,               # calls[2]: update hypothesis gen
            '{"h0": 0.7, "h1": 0.2, "h2": 0.1}',  # batched_calls[6]: update posterior
        ]
    )
    config = _location_config(
        location_posterior_mode="llm_distribution",
        location_strategy_num_mutation=0, location_strategy_num_crossover=0, location_strategy_num_diverse=0,
        location_strategy_num_retrieved=1,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=1,
        location_plot_trials=False,
    )

    metrics = _run_location_config(
        model,
        config,
        rng=np.random.default_rng(1),
        output_dir=tmp_path,
        method_name="StrategyEIG",
    )

    assert len(metrics.source_rmse) == 1
    assert metrics.top_probability[0] == pytest.approx(0.7)
    assert math.isfinite(metrics.selected_eig[0])
    assert len(model.calls) == 3
    # initial posterior + (rollout: location + belief update + hypothesis refresh + final posterior) +
    # final location selection + update posterior = 7 batched calls
    assert len(model.batched_calls) == 7
    assert "\"weights\"" in model.batched_calls[0][0][-1]["content"]             # initial posterior
    assert "{\"location\":[x1,y1]}" in model.batched_calls[1][0][0]["content"]  # rollout depth-0 location
    assert "\"weights\"" in model.batched_calls[2][0][-1]["content"]             # rollout depth-0 belief update
    assert "{\"hypotheses\"" in model.batched_calls[3][0][-1]["content"]         # hypothesis refresh
    assert "\"weights\"" in model.batched_calls[4][0][-1]["content"]             # rollout final posterior
    assert "{\"location\":[x1,y1]}" in model.batched_calls[5][0][0]["content"]  # final location selection
    assert "\"weights\"" in model.batched_calls[6][0][-1]["content"]             # update posterior


def test_run_location_naive_batches_across_trials(tmp_path):
    model = RoutingLocationModel(num_sources=2)
    config = _location_config(
        location_num_sources=2,
        location_num_trials=3,
        location_num_rounds=2,
        location_trial_batch_size=3,
        location_plot_trials=True,
    )

    metrics = _run_location_config(model, config, rng=np.random.default_rng(1), output_dir=tmp_path, method_name="Naive")

    assert len(metrics.source_rmse) == 2
    assert len(model.calls) == 0
    # Pure naive does not need initial/posterior belief batches; each round only
    # requests a naive query and a naive source estimate.
    assert [len(batch) for batch in model.batched_calls] == [3, 3, 3, 3]
    assert len(list(tmp_path.glob("location_trial_*.png"))) == 3


def test_run_location_eig_batches_initial_candidates_and_updates_across_trials(tmp_path):
    model = RoutingLocationModel(num_sources=2)
    config = _location_config(
        location_num_sources=2,
        location_num_trials=3,
        location_num_rounds=1,
        location_trial_batch_size=3,
        location_target_num_candidates=2,
        location_search_depth=1,
    )

    metrics = _run_location_config(model, config, rng=np.random.default_rng(1), output_dir=tmp_path, method_name="EIG")

    assert len(metrics.source_rmse) == 1
    assert len(model.calls) == 0
    assert [len(batch) for batch in model.batched_calls] == [3, 3, 3]
    flattened = [messages for batch in model.batched_calls for messages in batch]
    assert any("finite Bayesian belief support" in call[0]["content"] for call in flattened)
    assert any("candidate measurement locations" in call[0]["content"] for call in flattened)


def test_run_location_eig_bounds_adds_separate_total_eig_metrics():
    model = RoutingLocationModel(num_sources=1)
    config = _location_config(
        location_num_sources=1,
        location_num_trials=2,
        location_num_rounds=1,
        location_trial_batch_size=2,
        location_target_num_candidates=2,
        location_search_depth=1,
        location_eig_bounds_enabled=True,
        location_eig_bounds_inner_samples=3,
        location_eig_bounds_seed=77,
        location_eig_bounds_chunk_size=2,
    )

    _run_result, summary = run_from_config(config, model, method_name="EIG")

    for metric_name in (
        "total_eig_lower_bound",
        "total_eig_upper_bound",
        "total_eig_lower_bound_se",
        "total_eig_upper_bound_se",
    ):
        assert metric_name in summary.metrics
        assert len(summary.metrics[metric_name]) == 1
        assert math.isfinite(summary.metrics[metric_name][0])
    assert len(summary.metrics["source_rmse"]) == 1
    assert len(summary.metrics["selected_eig"]) == 1
    assert len(model.calls) == 0
    assert [len(batch) for batch in model.batched_calls] == [2, 2, 2]


def test_run_location_strategy_root_batches_trials_and_rollouts(tmp_path):
    model = RoutingLocationModel(num_sources=2)
    config = _location_config(
        location_num_sources=2,
        location_num_trials=2,
        location_num_rounds=1,
        location_trial_batch_size=2,
        location_strategy_num_mutation=0, location_strategy_num_crossover=0, location_strategy_num_diverse=0,
        location_strategy_num_retrieved=1,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=1,
    )

    metrics = _run_location_config(
        model,
        config,
        rng=np.random.default_rng(1),
        output_dir=tmp_path,
        method_name="StrategyEIG+root",
    )

    assert len(metrics.source_rmse) == 1
    assert len(model.calls) == 0
    assert all(len(batch) == 2 for batch in model.batched_calls)
    flattened = [messages for batch in model.batched_calls for messages in batch]
    assert any("finite Bayesian belief support" in call[0]["content"] for call in flattened)
    assert any("strategy/root_query" in call[0]["content"] for call in flattened)
