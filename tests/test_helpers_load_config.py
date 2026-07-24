import sys
import types

import pytest


fake_model_module = types.ModuleType("model")


class _ModelBase:
    pass


fake_model_module.Model = _ModelBase
sys.modules.setdefault("model", fake_model_module)

from helpers import ModelPair, ModelSpec, load_config


def test_load_config_parses_model_pairs_with_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: 0
animals:
  - ["cat"]
model_pairs:
  - questioner:
      model: "Qwen/Qwen3.5-4B"
    answerer:
      model: "google/gemma-4-E4B-it"
      thinking: true
  - questioner:
      model: "openai/gpt-oss-20b"
      reasoning_effort: high
    answerer:
      model: "Qwen/Qwen2.5-7B-Instruct"
      thinking: false
method_names:
  - "EIG"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.model_pairs == [
        ModelPair(
            questioner=ModelSpec(model="Qwen/Qwen3.5-4B", thinking=False),
            answerer=ModelSpec(
                model="google/gemma-4-E4B-it",
                thinking=True,
                thinking_max_new_tokens=4096,
                thinking_final_max_new_tokens=512,
            ),
        ),
        ModelPair(
            questioner=ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="high"),
            answerer=ModelSpec(model="Qwen/Qwen2.5-7B-Instruct", thinking=False),
        ),
    ]
    assert config.belief_state_mode == "uniform"
    assert config.belief_probability_temperature == 0.0
    assert config.belief_distribution_num_calls == 1
    assert config.belief_distribution_permute_history is False
    assert config.belief_prior_mode == "none"
    assert config.belief_prior_exponential_rate == pytest.approx(0.0)
    assert config.belief_generation_enabled is True
    assert config.belief_filtering_enabled is True
    assert config.answerer_sample_from_prior is False
    assert config.answerer_prior_mode == "inherit"
    assert config.answerer_prior_exponential_rate is None
    assert config.answerer_randomize_prior_order_per_trial is False
    assert config.answerer_num_prior_trials is None
    assert config.answerer_prior_seed is None
    assert config.belief_guess_threshold == pytest.approx(0.99)


def test_load_config_parses_thinking_budgets_with_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs:
  - questioner:
      model: "Qwen/Qwen3.5-4B"
      thinking: true
      thinking_max_new_tokens: 1024
      thinking_final_max_new_tokens: 128
    answerer:
      model: "google/gemma-4-E4B-it"
      thinking: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.model_pairs[0].questioner == ModelSpec(
        model="Qwen/Qwen3.5-4B",
        thinking=True,
        thinking_max_new_tokens=1024,
        thinking_final_max_new_tokens=128,
    )
    assert config.model_pairs[0].answerer == ModelSpec(
        model="google/gemma-4-E4B-it",
        thinking=True,
        thinking_max_new_tokens=4096,
        thinking_final_max_new_tokens=512,
    )


def test_load_config_projects_nested_environment_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
task: location_finding
model_pairs: []
method_names:
  - EIG
environment:
  num_rounds: 3
  num_trials: 2
  trial_batch_size: 2
  source_prior: branch_decoy
  source_radius: 2.2
  num_sources: 2
  dim: 2
  noise_sd: 0.5
  signal_model: local_bump
  signal_lengthscale: 0.5
  signal_amplitude: 8.0
  max_step_radius: 0.75
  belief_support_refresh_enabled: false
  candidate_generation_mode: support_grid
  search_depth: 1
  eig_quadrature_order: 5
  eig_bounds_enabled: true
  eig_bounds_inner_samples: 11
  eig_bounds_seed: 456
  eig_bounds_chunk_size: 7
  strategy_rollout_refresh_hypotheses_each_step: true
  strategy_rollout_scoring_support_mode: truth_plus_sampled
  strategy_rollout_scoring_support_size: 9
  strategy_rollout_score_mode: future_step_support_sum
  strategy_rollout_final_refresh_enabled: false
  strategy_rollout_query_mode: analytic_eig
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.environment["num_rounds"] == 3
    assert config.location_num_rounds == 3
    assert config.location_num_trials == 2
    assert config.location_trial_batch_size == 2
    assert config.location_source_prior == "branch_decoy"
    assert config.location_source_radius == pytest.approx(2.2)
    assert config.location_signal_model == "local_bump"
    assert config.location_signal_lengthscale == pytest.approx(0.5)
    assert config.location_signal_amplitude == pytest.approx(8.0)
    assert config.location_max_step_radius == pytest.approx(0.75)
    assert config.location_belief_support_refresh_enabled is False
    assert config.location_candidate_generation_mode == "support_grid"
    assert config.location_search_depth == 1
    assert config.location_eig_bounds_enabled is True
    assert config.location_eig_bounds_inner_samples == 11
    assert config.location_eig_bounds_seed == 456
    assert config.location_eig_bounds_chunk_size == 7
    assert config.location_strategy_rollout_refresh_hypotheses_each_step is True
    assert config.location_strategy_rollout_scoring_support_mode == "truth_plus_sampled"
    assert config.location_strategy_rollout_scoring_support_size == 9
    assert config.location_strategy_rollout_score_mode == "future_step_support_sum"
    assert config.location_strategy_rollout_final_refresh_enabled is False
    assert config.location_strategy_rollout_query_mode == "analytic_eig"


def test_load_config_parses_categorical_belief_state_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
belief_state_mode: categorical
belief_probability_temperature: 0.25
belief_distribution_num_calls: 4
belief_distribution_permute_history: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.belief_state_mode == "categorical"
    assert config.belief_probability_temperature == 0.25
    assert config.belief_distribution_num_calls == 4
    assert config.belief_distribution_permute_history is True


def test_load_config_parses_prior_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
belief_prior_mode: exponential_rank
belief_prior_exponential_rate: 0.4
belief_generation_enabled: false
belief_filtering_enabled: false
answerer_sample_from_prior: true
answerer_prior_mode: uniform
answerer_prior_exponential_rate: 0.2
answerer_randomize_prior_order_per_trial: true
answerer_num_prior_trials: 7
answerer_prior_seed: 123
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.belief_prior_mode == "exponential_rank"
    assert config.belief_prior_exponential_rate == pytest.approx(0.4)
    assert config.belief_generation_enabled is False
    assert config.belief_filtering_enabled is False
    assert config.answerer_sample_from_prior is True
    assert config.answerer_prior_mode == "uniform"
    assert config.answerer_prior_exponential_rate == pytest.approx(0.2)
    assert config.answerer_randomize_prior_order_per_trial is True
    assert config.answerer_num_prior_trials == 7
    assert config.answerer_prior_seed == 123


def test_load_config_parses_uniform_questioner_prior(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
belief_prior_mode: uniform
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.belief_prior_mode == "uniform"


def test_load_config_parses_location_finding_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
task: location_finding
location_num_rounds: 7
location_num_trials: 2
location_source_prior: branch_decoy
location_source_radius: 2.2
location_num_sources: 3
location_dim: 2
location_noise_sd: 0.5
location_signal_model: local_bump
location_signal_lengthscale: 0.5
location_signal_amplitude: 8.0
location_max_total_beliefs: 123
location_max_llm_prompt_beliefs: 40
location_belief_support_refresh_enabled: false
location_candidate_generation_mode: support_grid
location_target_num_candidates: 15
location_search_depth: 2
location_eig_quadrature_order: 9
location_plot_trials: true
location_eig_bounds_enabled: true
location_eig_bounds_inner_samples: 13
location_eig_bounds_seed: 321
location_eig_bounds_chunk_size: 17
location_strategy_rollout_refresh_hypotheses_each_step: true
location_strategy_rollout_scoring_support_mode: truth_plus_sampled
location_strategy_rollout_scoring_support_size: 19
location_strategy_rollout_score_mode: future_step_support_sum
location_strategy_rollout_final_refresh_enabled: false
location_strategy_rollout_query_mode: analytic_eig
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.task == "location_finding"
    assert config.method_names == ["EIG"]
    assert config.location_num_rounds == 7
    assert config.location_num_trials == 2
    assert config.location_source_prior == "branch_decoy"
    assert config.location_source_radius == pytest.approx(2.2)
    assert config.location_num_sources == 3
    assert config.location_dim == 2
    assert config.location_noise_sd == pytest.approx(0.5)
    assert config.location_signal_model == "local_bump"
    assert config.location_signal_lengthscale == pytest.approx(0.5)
    assert config.location_signal_amplitude == pytest.approx(8.0)
    assert config.location_max_total_beliefs == 123
    assert config.location_max_llm_prompt_beliefs == 40
    assert config.location_belief_support_refresh_enabled is False
    assert config.location_candidate_generation_mode == "support_grid"
    assert config.location_target_num_candidates == 15
    assert config.location_search_depth == 2
    assert config.location_eig_quadrature_order == 9
    assert config.location_plot_trials is True
    assert config.location_eig_bounds_enabled is True
    assert config.location_eig_bounds_inner_samples == 13
    assert config.location_eig_bounds_seed == 321
    assert config.location_eig_bounds_chunk_size == 17
    assert config.location_strategy_rollout_refresh_hypotheses_each_step is True
    assert config.location_strategy_rollout_scoring_support_mode == "truth_plus_sampled"
    assert config.location_strategy_rollout_scoring_support_size == 19
    assert config.location_strategy_rollout_score_mode == "future_step_support_sum"
    assert config.location_strategy_rollout_final_refresh_enabled is False
    assert config.location_strategy_rollout_query_mode == "analytic_eig"
    # location_strategy_num_candidates is now a derived property: the sum of the four
    # evolutionary-phase counts (retrieved + mutation + crossover + diverse).
    # Defaults are 2 + 1 + 1 + 2 = 6.
    assert config.location_strategy_num_candidates == 6
    assert config.location_strategy_num_retrieved == 2
    assert config.location_strategy_num_rollouts == 8
    assert config.location_strategy_planning_depth == 8
    assert config.location_strategy_belief_summary_top_k == 5
    assert config.location_posterior_mode == "analytical_likelihood"


def test_load_config_parses_location_strategy_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
task: location_finding
location_strategy_num_retrieved: 3
location_strategy_num_mutation: 1
location_strategy_num_crossover: 1
location_strategy_num_diverse: 1
location_strategy_num_rollouts: 4
location_strategy_planning_depth: 2
location_strategy_belief_summary_top_k: 7
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    # location_strategy_num_candidates is derived: 3 + 1 + 1 + 1 = 6
    assert config.location_strategy_num_candidates == 6
    assert config.location_strategy_num_retrieved == 3
    assert config.location_strategy_num_mutation == 1
    assert config.location_strategy_num_crossover == 1
    assert config.location_strategy_num_diverse == 1
    assert config.location_strategy_num_rollouts == 4
    assert config.location_strategy_planning_depth == 2
    assert config.location_strategy_belief_summary_top_k == 7


def test_load_config_parses_location_posterior_mode_and_reuses_distribution_controls(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
task: location_finding
location_posterior_mode: llm_distribution
belief_distribution_num_calls: 4
belief_distribution_permute_history: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.location_posterior_mode == "llm_distribution"
    assert config.belief_distribution_num_calls == 4
    assert config.belief_distribution_permute_history is True


def test_load_config_rejects_unknown_location_posterior_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
task: location_finding
location_posterior_mode: vibes
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="location_posterior_mode"):
        load_config(str(config_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("location_eig_bounds_enabled", "yes"),
        ("location_eig_bounds_inner_samples", 0),
        ("location_eig_bounds_seed", "seed"),
        ("location_eig_bounds_chunk_size", 0),
    ],
)
def test_load_config_rejects_invalid_location_eig_bound_options(tmp_path, field, value):
    config_path = tmp_path / "config.yaml"
    rendered_value = f'"{value}"' if isinstance(value, str) else str(value).lower()
    config_path.write_text(
        f"""
model_pairs: []
task: location_finding
{field}: {rendered_value}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_config(str(config_path))


@pytest.mark.parametrize(
    ("yaml_body", "message"),
    [
        ("location_strategy_rollout_scoring_support_mode: vibes", "location_strategy_rollout_scoring_support_mode"),
        ("location_strategy_rollout_scoring_support_size: 0", "location_strategy_rollout_scoring_support_size"),
        ("location_strategy_rollout_score_mode: vibes", "location_strategy_rollout_score_mode"),
        ("location_strategy_rollout_query_mode: vibes", "location_strategy_rollout_query_mode"),
        ("location_candidate_generation_mode: vibes", "location_candidate_generation_mode"),
        ("location_max_step_radius: 0", "location_max_step_radius"),
        ("location_max_step_radius: false", "location_max_step_radius"),
        ("location_source_prior: maze", "location_source_prior"),
        ("location_source_radius: 0", "location_source_radius"),
        ("location_signal_model: laser", "location_signal_model"),
        ("location_signal_lengthscale: 0", "location_signal_lengthscale"),
        ("location_signal_amplitude: 0", "location_signal_amplitude"),
        (
            "location_posterior_mode: llm_distribution\n"
            "location_strategy_rollout_scoring_support_mode: truth_plus_sampled",
            "truth_plus_sampled",
        ),
        (
            "location_posterior_mode: llm_distribution\n"
            "location_strategy_rollout_scoring_support_mode: truth_start_end",
            "truth_start_end",
        ),
        (
            "location_posterior_mode: llm_distribution\n"
            "location_strategy_rollout_scoring_support_mode: fixed_common",
            "fixed_common",
        ),
        (
            "location_strategy_rollout_final_refresh_enabled: sometimes",
            "location_strategy_rollout_final_refresh_enabled",
        ),
        (
            "location_belief_support_refresh_enabled: sometimes",
            "location_belief_support_refresh_enabled",
        ),
        (
            "location_posterior_mode: llm_distribution\n"
            "location_strategy_rollout_refresh_hypotheses_each_step: true",
            "location_strategy_rollout_refresh_hypotheses_each_step",
        ),
    ],
)
def test_load_config_rejects_invalid_location_strategy_rollout_options(tmp_path, yaml_body, message):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
task: location_finding
{yaml_body}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(str(config_path))


def test_load_config_strategy_candidate_count_is_derived_from_phase_counts(tmp_path):
    # location_strategy_num_candidates is no longer a YAML-settable field; it is the
    # sum of the four evolutionary-phase counts.  Setting it directly in YAML has no
    # effect (the key is ignored), and the derived value always equals the phase sum.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
task: location_finding
location_strategy_num_candidates: 2
location_strategy_num_retrieved: 3
location_strategy_num_mutation: 0
location_strategy_num_crossover: 0
location_strategy_num_diverse: 1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    # The unused location_strategy_num_candidates YAML key is ignored; the derived
    # value is retrieved + mutation + crossover + diverse = 3 + 0 + 0 + 1 = 4.
    assert config.location_strategy_num_candidates == 4
    assert config.location_strategy_num_retrieved == 3


def test_load_config_defaults_location_trial_plotting_to_false(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
task: location_finding
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.location_plot_trials is False
    assert config.location_max_total_beliefs == 1000
    assert config.location_max_llm_prompt_beliefs == 40


def test_load_config_uses_flat_location_max_beliefs_for_prompt_limit(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
task: location_finding
location_max_beliefs: 17
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.location_max_total_beliefs == 1000
    assert config.location_max_llm_prompt_beliefs == 17


@pytest.mark.parametrize("search_depth", [1, 2, 3, 7])
def test_load_config_parses_search_depth(tmp_path, search_depth):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
search_depth: {search_depth}
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.search_depth == search_depth


@pytest.mark.parametrize("raw_value", ["0", "-1", "true", "1.5", "'3'"])
def test_load_config_rejects_invalid_search_depth(tmp_path, raw_value):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
search_depth: {raw_value}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="search_depth must be a positive integer"):
        load_config(str(config_path))


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("null", None),
        ("0.99", 0.99),
        ("0", 0.0),
        ("1", 1.0),
    ],
)
def test_load_config_parses_belief_guess_threshold(tmp_path, raw_value, expected):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
belief_guess_threshold: {raw_value}
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.belief_guess_threshold == expected


@pytest.mark.parametrize("raw_value", ["-0.1", "1.1", "true", "'0.99'"])
def test_load_config_rejects_invalid_belief_guess_threshold(tmp_path, raw_value):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
belief_guess_threshold: {raw_value}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belief_guess_threshold"):
        load_config(str(config_path))


def test_load_config_rejects_invalid_model_pairs_entries(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs:
  - {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_pairs\\[0\\]"):
        load_config(str(config_path))


def test_load_config_rejects_thinking_for_gpt_oss(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs:
  - questioner:
      model: "openai/gpt-oss-20b"
      thinking: true
    answerer:
      model: "Qwen/Qwen3.5-4B"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="thinking is not supported"):
        load_config(str(config_path))


@pytest.mark.parametrize("field", ["thinking_max_new_tokens", "thinking_final_max_new_tokens"])
def test_load_config_rejects_invalid_thinking_budget_values(tmp_path, field):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs:
  - questioner:
      model: "Qwen/Qwen3.5-4B"
      thinking: true
      {field}: 0
    answerer:
      model: "Qwen/Qwen3.5-4B"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_config(str(config_path))


def test_load_config_rejects_thinking_budgets_without_thinking_enabled(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs:
  - questioner:
      model: "Qwen/Qwen3.5-4B"
      thinking: false
      thinking_max_new_tokens: 1024
    answerer:
      model: "Qwen/Qwen3.5-4B"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="thinking budgets require thinking: true"):
        load_config(str(config_path))


def test_load_config_rejects_thinking_budgets_for_gpt_oss(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs:
  - questioner:
      model: "openai/gpt-oss-20b"
      thinking_max_new_tokens: 1024
    answerer:
      model: "Qwen/Qwen3.5-4B"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="thinking budgets are not supported"):
        load_config(str(config_path))


def test_load_config_rejects_thinking_budgets_for_plain_models(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs:
  - questioner:
      model: "meta/llama-3"
      thinking_max_new_tokens: 1024
    answerer:
      model: "Qwen/Qwen3.5-4B"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="thinking budgets are only supported"):
        load_config(str(config_path))


def test_load_config_rejects_invalid_belief_state_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
belief_state_mode: weighted
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belief_state_mode"):
        load_config(str(config_path))


@pytest.mark.parametrize("raw_value", [0, -1])
def test_load_config_rejects_non_positive_belief_generation_num_calls(tmp_path, raw_value):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
belief_generation_num_calls: {raw_value}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belief_generation_num_calls must be at least 1"):
        load_config(str(config_path))


def test_load_config_parses_belief_generation_strata(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
belief_generation_strata:
  - mammals
  - birds
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.belief_generation_strata == ["mammals", "birds"]


@pytest.mark.parametrize("raw_value", [0, -1])
def test_load_config_rejects_non_positive_belief_distribution_num_calls(tmp_path, raw_value):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
belief_distribution_num_calls: {raw_value}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belief_distribution_num_calls must be at least 1"):
        load_config(str(config_path))


def test_load_config_rejects_non_boolean_belief_distribution_permute_history(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
belief_distribution_permute_history: "sometimes"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belief_distribution_permute_history must be a boolean"):
        load_config(str(config_path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("belief_prior_mode", "ranked", "belief_prior_mode"),
        ("belief_prior_exponential_rate", -0.1, "belief_prior_exponential_rate"),
        ("belief_generation_enabled", "no", "belief_generation_enabled"),
        ("belief_filtering_enabled", "no", "belief_filtering_enabled"),
        ("answerer_sample_from_prior", "yes", "answerer_sample_from_prior"),
        ("answerer_prior_mode", "ranked", "answerer_prior_mode"),
        ("answerer_prior_exponential_rate", -0.1, "answerer_prior_exponential_rate"),
        ("answerer_randomize_prior_order_per_trial", "yes", "answerer_randomize_prior_order_per_trial"),
        ("answerer_num_prior_trials", 0, "answerer_num_prior_trials"),
        ("answerer_prior_seed", "seed", "answerer_prior_seed"),
    ],
)
def test_load_config_rejects_invalid_prior_options(tmp_path, field, value, message):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
{field}: {value!r}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(str(config_path))


def test_load_config_parses_vllm_runtime_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs: []
tensor_parallel_size: 2
gpu_memory_utilization: 0.95
max_model_len: 8192
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.tensor_parallel_size == 2
    assert config.gpu_memory_utilization == pytest.approx(0.95)
    assert config.max_model_len == 8192
    assert config.location_max_new_tokens == 8192


def test_load_config_derives_generation_budget_from_model_specs(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
task: location_finding
model_pairs:
  - questioner:
      model: "Qwen/Qwen3.6-35B-A3B"
      max_model_len: 16384
    answerer:
      model: "Qwen/Qwen3.6-35B-A3B"
      max_model_len: 8192
method_names:
  - naive
environment:
  num_rounds: 1
  num_trials: 1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.location_max_new_tokens == 16384


def test_load_config_preserves_explicit_location_generation_budget(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_pairs:
  - questioner:
      model: "google/gemma-4-26B-A4B-it"
      max_model_len: 32768
    answerer:
      model: "google/gemma-4-26B-A4B-it"
      max_model_len: 32768
location_max_new_tokens: 2048
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.location_max_new_tokens == 2048


@pytest.mark.parametrize("value", [0, -1, True, 32769])
def test_load_config_rejects_invalid_location_generation_budget(
    tmp_path, value
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
model_pairs: []
max_model_len: 32768
location_max_new_tokens: {value!r}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="location_max_new_tokens"):
        load_config(str(config_path))


def test_phase4_constrained_and_unconstrained_location_configs_load():
    constrained = load_config("configs/config_location_branch_decoy_local.yaml")
    unconstrained = load_config("configs/config_location_branch_decoy_local_unconstrained.yaml")
    final_constrained = load_config("configs/config_location_branch_decoy_local_final50.yaml")
    final_unconstrained = load_config("configs/config_location_branch_decoy_local_unconstrained_final50.yaml")
    final_constrained_26b = load_config("configs/config_location_branch_decoy_local_final50_26b_a4b.yaml")
    final_unconstrained_26b = load_config(
        "configs/config_location_branch_decoy_local_unconstrained_final50_26b_a4b.yaml"
    )
    rollout8 = load_config("configs/config_location_branch_decoy_local_final50_rollouts8.yaml")
    rollout32 = load_config("configs/config_location_branch_decoy_local_final50_rollouts32.yaml")

    assert constrained.task == "location_finding"
    assert unconstrained.task == "location_finding"
    assert constrained.location_source_prior == "branch_decoy"
    assert unconstrained.location_source_prior == "branch_decoy"
    assert constrained.location_signal_model == "local_bump"
    assert unconstrained.location_signal_model == "local_bump"
    assert constrained.location_max_step_radius == pytest.approx(0.5)
    assert unconstrained.location_max_step_radius is None
    assert constrained.location_num_trials == unconstrained.location_num_trials
    assert constrained.location_num_rounds == unconstrained.location_num_rounds
    assert constrained.location_strategy_num_rollouts == unconstrained.location_strategy_num_rollouts
    assert final_constrained.task == "location_finding"
    assert final_unconstrained.task == "location_finding"
    assert final_constrained.location_num_trials == 50
    assert final_unconstrained.location_num_trials == 50
    assert final_constrained.location_max_step_radius == pytest.approx(0.5)
    assert final_unconstrained.location_max_step_radius is None
    assert final_constrained.location_num_rounds == final_unconstrained.location_num_rounds
    assert final_constrained.location_strategy_num_rollouts == final_unconstrained.location_strategy_num_rollouts
    assert final_constrained_26b.location_num_trials == 50
    assert final_unconstrained_26b.location_num_trials == 50
    assert final_constrained_26b.location_max_step_radius == pytest.approx(0.5)
    assert final_unconstrained_26b.location_max_step_radius is None
    assert final_constrained_26b.model_pairs[0].questioner.model == "google/gemma-4-26B-A4B-it"
    assert final_unconstrained_26b.model_pairs[0].questioner.model == "google/gemma-4-26B-A4B-it"
    assert final_constrained_26b.location_strategy_num_rollouts == 16
    assert final_constrained_26b.location_strategy_planning_depth == 3
    assert rollout8.location_num_trials == 50
    assert rollout32.location_num_trials == 50
    assert rollout8.location_max_step_radius == pytest.approx(0.5)
    assert rollout32.location_max_step_radius == pytest.approx(0.5)
    assert [
        rollout8.location_strategy_num_rollouts,
        final_constrained.location_strategy_num_rollouts,
        rollout32.location_strategy_num_rollouts,
    ] == [8, 16, 32]
