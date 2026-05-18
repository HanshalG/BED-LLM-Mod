import json
import math

import numpy as np
import pytest

from helpers import Config
from location_finding import (
    LocationBeliefState,
    LocationFindingEnv,
    LocationObservation,
    LocationStrategyEntry,
    LocationStrategyLibrary,
    _belief_generation_messages,
    _candidate_generation_messages,
    _default_source_hypotheses,
    _location_effective_sample_size,
    _strategy_location_messages,
    _strategy_proposal_messages,
    build_location_belief_state,
    evaluate_location_strategies_by_rollout,
    expected_information_gain,
    generate_location_strategies,
    normalize_source_config,
    parse_candidate_locations,
    parse_location_strategies,
    parse_strategy_location,
    parse_source_hypotheses,
    prompt_location_belief_state,
    prune_location_beliefs,
    run_location_finding,
    sample_location_eig_belief_state,
    score_candidate_locations,
    signal_intensity_for_hypothesis,
    source_rmse,
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


def _location_config(**overrides) -> Config:
    config = Config(
        task="location_finding",
        location_num_rounds=1,
        location_num_trials=1,
        location_num_sources=3,
        location_dim=2,
        location_noise_sd=0.5,
        location_query_bounds=[-2.0, 2.0],
        location_max_total_beliefs=1000,
        location_max_llm_prompt_beliefs=40,
        location_target_num_candidates=2,
        location_search_depth=1,
        location_eig_quadrature_order=5,
        generation_temperature_diverse=0.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


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


def test_belief_generation_prompts_split_initial_and_update_modes():
    config = _location_config(location_max_llm_prompt_beliefs=7)
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = LocationBeliefState([hypothesis], [1.0])

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
    belief_state = LocationBeliefState([hypothesis], [1.0])
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
    belief_state = LocationBeliefState([hypothesis], [1.0])

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


def test_parse_strategy_location_accepts_single_location_json_and_checks_bounds():
    assert parse_strategy_location('{"location": [0.25, -1]}', 2, (-2.0, 2.0)) == (0.25, -1.0)

    with pytest.raises(ValueError, match="outside allowed query bounds"):
        parse_strategy_location('{"location": [3, 0]}', 2, (-2.0, 2.0))


def test_strategy_library_retrieves_best_entries_and_new_library_is_empty():
    library = LocationStrategyLibrary()
    assert len(library) == 0

    library.add_entries(
        [
            LocationStrategyEntry("weak", 0.1, 0.0, "[0, 0]", 0),
            LocationStrategyEntry("strong", 0.5, 0.2, "[1, 1]", 1),
            LocationStrategyEntry("steady", 0.5, 0.1, "[-1, 1]", 2),
        ]
    )

    retrieved = library.retrieve_top_m(2)

    assert [entry.strategy for entry in retrieved] == ["steady", "strong"]
    assert len(LocationStrategyLibrary()) == 0


def test_strategy_prompts_include_history_beliefs_retrieved_examples_and_diversity_instructions():
    config = _location_config(location_strategy_belief_summary_top_k=1)
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = LocationBeliefState([hypothesis], [1.0])
    observations = [LocationObservation((0.5, 0.5), 3.2)]
    retrieved = [LocationStrategyEntry("Start near the strongest current peak.", 1.2, 0.3, "[0, 0]", 0)]

    proposal_messages = _strategy_proposal_messages(belief_state, observations, retrieved, config, num_fresh=2)
    location_messages = _strategy_location_messages(retrieved[0].strategy, belief_state, observations, config)

    proposal_system = proposal_messages[0]["content"]
    proposal_user = proposal_messages[-1]["content"]
    location_system = location_messages[0]["content"]
    location_user = location_messages[-1]["content"]
    assert "substantively" in proposal_system
    assert "same first move" in proposal_system
    assert "Observation history so far" in proposal_user
    assert "Current belief summary (top 1 hypotheses" in proposal_user
    assert "Retrieved elite strategies" in proposal_user
    assert "Start near the strongest current peak." in proposal_user
    assert "{\"location\":[x1,y1]}" in location_system
    assert "Strategy to follow" in location_user
    assert "signal_strength" in location_user


def test_generate_location_strategies_retrieves_proposes_and_fills_defaults():
    config = _location_config(
        location_strategy_num_candidates=3,
        location_strategy_num_retrieved=1,
    )
    hypothesis = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    belief_state = LocationBeliefState([hypothesis], [1.0])
    library = LocationStrategyLibrary()
    library.add_entries([LocationStrategyEntry("Use the previous elite plan.", 0.9, 0.1, "[0, 0]", 0)])
    model = FakeLocationModel(['{"strategies": ["Fresh posterior-disagreement plan."]}'])

    strategies = generate_location_strategies(model, belief_state, [], library, config)

    assert strategies[0] == "Use the previous elite plan."
    assert strategies[1] == "Fresh posterior-disagreement plan."
    assert len(strategies) == 3
    assert "Retrieved elite strategies" in model.calls[0][-1]["content"]


def test_parse_candidate_locations_filters_bounds_duplicates_and_invalid_entries():
    completion = '{"locations": [[0, 0], [3, 0], [0, 0], ["bad", 1], {"location": [1.5, -2]}]}'

    locations = parse_candidate_locations(completion, dim=2, bounds=(-2.0, 2.0))

    assert locations == [(0.0, 0.0), (1.5, -2.0)]


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_default_source_hypotheses_are_valid_for_configured_source_count(num_sources):
    config = _location_config(location_num_sources=num_sources)

    hypotheses = _default_source_hypotheses(config)

    assert len(hypotheses) == 4
    for hypothesis in hypotheses:
        assert len(hypothesis) == num_sources
        assert len(set(hypothesis)) == num_sources
        for source in hypothesis:
            assert len(source) == 2


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


def test_generated_refresh_hypotheses_compete_with_existing_support():
    config = _location_config()
    old = normalize_source_config([[0, 1], [1, 0], [-1, -1]], 3, 2)
    generated = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    query = (0.0, 0.0)
    observation = LocationObservation(query=query, value=signal_intensity_for_hypothesis(generated, query))

    state = build_location_belief_state([old, generated], [observation], config)

    assert state.hypotheses[0] == generated
    assert state.probabilities[0] > state.probabilities[1]


def test_pruning_keeps_top_k_when_over_budget_and_renormalizes():
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(45)
    ]
    probabilities = [float(idx + 1) for idx in range(45)]
    state = LocationBeliefState(hypotheses, probabilities)

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
    state = LocationBeliefState(hypotheses, [0.25] * 4)

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
    state = LocationBeliefState(hypotheses, [0.05, 0.1, 0.2, 0.25, 0.4])

    prompt_state = prompt_location_belief_state(state, config)

    assert len(prompt_state.hypotheses) == 3
    assert prompt_state.hypotheses == [hypotheses[4], hypotheses[3], hypotheses[2]]
    assert sum(prompt_state.probabilities) == pytest.approx(1.0)


def test_eig_belief_sampling_uses_num_mc_samples_and_renormalizes():
    config = _location_config(num_mc_samples=10)
    hypotheses = [
        normalize_source_config([[idx / 10, -1], [0, 1], [1, 0]], 3, 2)
        for idx in range(50)
    ]
    state = LocationBeliefState(hypotheses, [1 / 50] * 50)

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
    state = LocationBeliefState(hypotheses, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    sampled_state, sample_collapsed = sample_location_eig_belief_state(state, config, np.random.default_rng(0))

    assert sample_collapsed
    assert len(sampled_state.hypotheses) == 1
    assert sampled_state.hypotheses[0] == hypotheses[0]
    assert sum(sampled_state.probabilities) == pytest.approx(1.0)


def test_location_effective_sample_size_reports_weight_concentration():
    concentrated = LocationBeliefState([((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))], [1.0])
    assert _location_effective_sample_size(concentrated) == pytest.approx(1.0)


def test_eig_is_zero_for_identical_predictions_and_positive_for_separated_predictions():
    config = _location_config(location_eig_quadrature_order=7)
    identical = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    separated = normalize_source_config([[2, 2], [1, 2], [2, 1]], 3, 2)

    identical_state = LocationBeliefState([identical, identical], [0.5, 0.5])
    separated_state = LocationBeliefState([identical, separated], [0.5, 0.5])

    assert expected_information_gain(identical_state, (0.0, 0.0), 0.5, 7) == pytest.approx(0.0, abs=1e-9)
    assert expected_information_gain(separated_state, (0.0, 0.0), 0.5, 7) > 0.01


def test_depth_two_future_eig_matches_explicit_noisy_quadrature():
    config = _location_config(location_search_depth=2, location_eig_quadrature_order=5)
    hypothesis_a = normalize_source_config([[0, 0], [1, 1], [-1, -1]], 3, 2)
    hypothesis_b = normalize_source_config([[2, 2], [1, 2], [2, 1]], 3, 2)
    belief_state = build_location_belief_state([hypothesis_a, hypothesis_b], [], config)
    candidates = [(0.0, 0.0), (1.0, 1.0)]
    model = FakeLocationModel(['{"hypotheses": []}' for _ in range(20)])

    scores = score_candidate_locations(belief_state, candidates, config, questioner=model, observations=[])

    nodes, weights = np.polynomial.hermite.hermgauss(config.location_eig_quadrature_order)
    weights = weights / math.sqrt(math.pi)
    manual_scores = []
    for candidate in candidates:
        immediate = expected_information_gain(
            belief_state,
            candidate,
            config.location_noise_sd,
            config.location_eig_quadrature_order,
        )
        means = [
            signal_intensity_for_hypothesis(hypothesis, candidate)
            for hypothesis in belief_state.hypotheses
        ]
        expected_future = 0.0
        for hypothesis_idx, mean in enumerate(means):
            for node, weight in zip(nodes, weights):
                y_value = mean + math.sqrt(2.0) * config.location_noise_sd * node
                future_state = build_location_belief_state(
                    list(belief_state.hypotheses),
                    [LocationObservation(candidate, y_value)],
                    config,
                )
                best_future = max(
                    expected_information_gain(
                        future_state,
                        future_candidate,
                        config.location_noise_sd,
                        config.location_eig_quadrature_order,
                    )
                    for future_candidate in candidates
                )
                expected_future += belief_state.probabilities[hypothesis_idx] * weight * best_future
        manual_scores.append(immediate + expected_future)

    assert scores == pytest.approx(manual_scores)
    assert len(model.batched_calls[0]) == 20
    first_branch_prompt = model.batched_calls[0][0][-1]["content"]
    assert "Observation history:" in first_branch_prompt
    assert "signal_strength" in first_branch_prompt
    assert "Current weighted hypotheses" in first_branch_prompt


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


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_run_location_finding_one_round_with_fake_llm_smoke(tmp_path, num_sources):
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

    metrics = run_location_finding(model, config, rng=np.random.default_rng(1), output_dir=tmp_path)

    assert len(metrics.source_rmse) == 1
    assert len(metrics.top_probability) == 1
    assert len(metrics.selected_eig) == 1
    assert math.isfinite(metrics.source_rmse[0])
    assert 0.0 <= metrics.top_probability[0] <= 1.0
    assert metrics.selected_eig[0] >= 0.0
    assert len(model.calls) == 3
    assert f"exactly {num_sources} hidden signal sources" in model.calls[0][0]["content"]
    assert "noise_sd=0.5" in model.calls[0][0]["content"]
    assert "Do not output source configurations" in model.calls[1][0]["content"]
    plot_path = tmp_path / "location_trial_001.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


@pytest.mark.parametrize("num_sources", [2, 3, 4])
def test_run_location_finding_strategy_eig_one_round_with_fake_llm_smoke(tmp_path, num_sources):
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
        location_strategy_num_candidates=1,
        location_strategy_num_retrieved=1,
        location_strategy_num_rollouts=1,
        location_strategy_planning_depth=1,
        location_plot_trials=False,
    )

    metrics = run_location_finding(
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
    assert "Retrieved elite strategies" in model.calls[1][-1]["content"]
    assert f"exactly {num_sources} hidden signal sources" in model.batched_calls[0][0][0]["content"]
    assert "{\"location\":[x1,y1]}" in model.batched_calls[0][0][0]["content"]
