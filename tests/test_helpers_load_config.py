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
            answerer=ModelSpec(model="google/gemma-4-E4B-it", thinking=True),
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
  num_sources: 2
  dim: 2
  noise_sd: 0.5
  query_bounds: [-1.0, 1.0]
  search_depth: 1
  eig_quadrature_order: 5
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.environment["num_rounds"] == 3
    assert config.location_num_rounds == 3
    assert config.location_num_trials == 2
    assert config.location_trial_batch_size == 2
    assert config.location_query_bounds == [-1.0, 1.0]
    assert config.location_search_depth == 1


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
location_num_sources: 3
location_dim: 2
location_noise_sd: 0.5
location_query_bounds: [-2, 2]
location_max_total_beliefs: 123
location_max_llm_prompt_beliefs: 40
location_target_num_candidates: 15
location_search_depth: 2
location_eig_quadrature_order: 9
location_plot_trials: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.task == "location_finding"
    assert config.method_names == ["EIG"]
    assert config.location_num_rounds == 7
    assert config.location_num_trials == 2
    assert config.location_num_sources == 3
    assert config.location_dim == 2
    assert config.location_noise_sd == pytest.approx(0.5)
    assert config.location_query_bounds == [-2.0, 2.0]
    assert config.location_max_total_beliefs == 123
    assert config.location_max_llm_prompt_beliefs == 40
    assert config.location_target_num_candidates == 15
    assert config.location_search_depth == 2
    assert config.location_eig_quadrature_order == 9
    assert config.location_plot_trials is True
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
