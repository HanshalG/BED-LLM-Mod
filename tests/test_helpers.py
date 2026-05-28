import sys
import types
from pathlib import Path

import pytest

fake_model_module = types.ModuleType("model")


class _ModelBase:
    pass


fake_model_module.Model = _ModelBase
sys.modules.setdefault("model", fake_model_module)

from core import BeliefState, deduped_belief_state, ensure_belief_state, uniform_deduped

from helpers import (
    Config,
    ModelPair,
    ModelSpec,
    build_exponential_rank_prior,
    build_uniform_prior,
    build_output_stem,
    clean_generated_belief_labels,
    format_categorical_belief_summary,
    format_belief_state,
    is_uniform_belief_state,
    load_config,
    normalize_belief_label,
    format_config_for_log,
    get_answerer_prior,
    get_questioner_prior,
    print_and_log,
    resolve_run_id,
    write_to_log,
)


def test_build_output_stem_prefixes_run_id_and_matches_artifact_stems():
    run_id = "20260409T153012"
    stem = build_output_stem(
        run_id=run_id,
        method_name="EIG",
        questioner=ModelSpec(model="Qwen/Qwen3.5-4B", thinking=True),
        answerer=ModelSpec(model="meta/llama-3"),
        version=0,
        belief_state_mode="uniform",
    )

    results_path = Path("results") / f"{stem}.npy"
    log_path = Path("logs") / f"{stem}.log"

    assert stem == "20260409T153012_EIG_Q:Qwen_Qwen3.5-4B__thinking-on,A:meta_llama-3_uniform_depth-1_0_animals"
    assert results_path.stem == log_path.stem == stem
    assert stem.startswith(f"{run_id}_")


def test_build_output_stem_includes_reasoning_effort():
    stem = build_output_stem(
        run_id="run123",
        method_name="naive",
        questioner=ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="high"),
        answerer=ModelSpec(model="google/gemma-4-E4B-it", thinking=False),
        version=1,
        belief_state_mode="categorical",
        search_depth=2,
    )

    assert stem == (
        "run123_naive_Q:openai_gpt-oss-20b__reasoning-high,"
        "A:google_gemma-4-E4B-it__thinking-off_categorical_depth-2_1_animals"
    )


def test_resolve_run_id_prefers_slurm_job_id(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")

    assert resolve_run_id() == "123456"


def test_resolve_run_id_falls_back_to_timestamp(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    run_id = resolve_run_id()

    assert len(run_id) == 15
    assert run_id[8] == "T"
    assert run_id.replace("T", "").isdigit()


def test_write_to_log_appends_to_configured_path(tmp_path):
    config = Config(log_path=tmp_path / "logs" / "run.log")

    write_to_log("first line\n", config)
    write_to_log("second line\n", config)

    assert config.log_path.read_text(encoding="utf-8") == "first line\nsecond line\n"


def test_write_to_log_requires_configured_log_path():
    with pytest.raises(ValueError, match="config.log_path must be set before logging"):
        write_to_log("message", Config())


def test_print_and_log_mirrors_messages_to_stdout_and_log_file(tmp_path, capsys):
    config = Config(log_path=tmp_path / "logs" / "run.log")

    print_and_log("[categorical] hello trace", config)

    captured = capsys.readouterr()
    assert "[categorical] hello trace" in captured.out
    assert config.log_path.read_text(encoding="utf-8") == "[categorical] hello trace\n"


def test_format_config_for_log_serializes_paths_and_nested_model_specs(tmp_path):
    config = Config(
        version=2,
        model_pairs=[
            ModelPair(
                questioner=ModelSpec(model="Qwen/Qwen3.5-4B", thinking=False),
                answerer=ModelSpec(model="google/gemma-4-E4B-it", thinking=True),
            )
        ],
        method_names=["EIG"],
        animals=[["cat"]],
        belief_state_mode="categorical",
        search_depth=2,
        run_id="run123",
        log_path=tmp_path / "logs" / "run.log",
    )

    rendered = format_config_for_log(config)

    assert '"belief_state_mode": "categorical"' in rendered
    assert '"model": "Qwen/Qwen3.5-4B"' in rendered
    assert '"search_depth": 2' in rendered
    assert f'"log_path": "{config.log_path}"' in rendered


def test_load_config_reads_probability_parse_fallback_flag(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("probability_parse_fallback_to_uniform: true\n", encoding="utf-8")

    config = load_config(str(config_path))

    assert config.probability_parse_fallback_to_uniform is True


def test_load_config_defaults_probability_parse_fallback_to_uniform(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_pairs: []\n", encoding="utf-8")

    config = load_config(str(config_path))

    assert config.probability_parse_fallback_to_uniform is True


def test_deduped_belief_state_merges_duplicates_and_normalizes():
    belief_state = deduped_belief_state(
        ["Cat", "dog", "cat"],
        [0.2, 0.3, 0.5],
        key=lambda label: label.lower(),
        normalize=lambda label: label.strip() or None,
    )

    assert belief_state.hypotheses == ("Cat", "dog")
    assert belief_state.probabilities == pytest.approx([0.7, 0.3])


def test_deduped_belief_state_falls_back_to_uniform_when_requested():
    belief_state = deduped_belief_state(
        ["cat", "dog"],
        [0.0, 0.0],
        key=lambda label: label.lower(),
        fallback_to_uniform=True,
    )

    assert belief_state.hypotheses == ("cat", "dog")
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])
    assert is_uniform_belief_state(belief_state)


def test_ensure_belief_state_converts_lists_to_uniform_state():
    belief_state = ensure_belief_state(["cat", "dog", "wolf"], key=lambda label: label.lower())

    assert belief_state.hypotheses == ("cat", "dog", "wolf")
    assert belief_state.probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_ensure_belief_state_dedupes_lists_without_frequency_weighting():
    belief_state = ensure_belief_state(["cat", "cat", "dog"], key=lambda label: label.lower())

    assert belief_state.hypotheses == ("cat", "dog")
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])


def test_format_belief_state_limits_entries_when_requested():
    belief_state = BeliefState(
        hypotheses=["cat", "dog", "wolf"],
        probabilities=[0.1, 0.7, 0.2],
    )

    assert format_belief_state(belief_state, top_n=2) == "[dog (0.700), wolf (0.200)]"


def test_format_categorical_belief_summary_includes_count_and_top_entries():
    belief_state = BeliefState(
        hypotheses=["dog", "wolf", "cat"],
        probabilities=[0.7, 0.2, 0.1],
    )

    assert format_categorical_belief_summary(belief_state, top_n=2) == (
        "3 belief(s): [dog (0.700), wolf (0.200)]"
    )


def test_format_categorical_belief_summary_defaults_to_all_entries():
    belief_state = BeliefState(
        hypotheses=["dog", "wolf", "cat"],
        probabilities=[0.7, 0.2, 0.1],
    )

    assert format_categorical_belief_summary(belief_state) == (
        "3 belief(s): [dog (0.700), wolf (0.200), cat (0.100)]"
    )


def test_uniform_deduped_dedupes_without_frequency_weighting():
    belief_state = uniform_deduped(["cat", "cat", "dog"], key=lambda label: label.lower())

    assert belief_state.hypotheses == ("cat", "dog")
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])
    assert is_uniform_belief_state(belief_state)


def test_build_uniform_prior_dedupes_and_assigns_equal_probabilities():
    belief_state = build_uniform_prior(["cat", "cat", "dog"])

    assert belief_state.hypotheses == ("cat", "dog",)
    assert belief_state.probabilities == pytest.approx([0.5, 0.5])


def test_build_exponential_rank_prior_with_zero_rate_is_uniform():
    belief_state = build_exponential_rank_prior(["cat", "dog", "wolf"], 0.0)

    assert belief_state.hypotheses == ("cat", "dog", "wolf",)
    assert belief_state.probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_build_exponential_rank_prior_positive_rate_is_monotonic_and_normalized():
    belief_state = build_exponential_rank_prior(["cat", "dog", "wolf"], 0.5)

    assert belief_state.hypotheses == ("cat", "dog", "wolf",)
    assert belief_state.probabilities[0] > belief_state.probabilities[1] > belief_state.probabilities[2]
    assert sum(belief_state.probabilities) == pytest.approx(1.0)


def test_get_questioner_prior_uses_uniform_mode():
    config = Config(
        animals=[["cat", "dog", "wolf"]],
        belief_prior_mode="uniform",
    )

    belief_state = get_questioner_prior(config)

    assert belief_state is not None
    assert belief_state.hypotheses == ("cat", "dog", "wolf",)
    assert belief_state.probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_get_answerer_prior_inherits_questioner_prior_by_default():
    config = Config(
        animals=[["cat", "dog", "wolf"]],
        belief_prior_mode="exponential_rank",
        belief_prior_exponential_rate=0.5,
    )

    questioner_prior = get_questioner_prior(config)
    answerer_prior = get_answerer_prior(config)

    assert answerer_prior == questioner_prior


def test_get_answerer_prior_can_differ_from_uniform_questioner_prior():
    config = Config(
        animals=[["cat", "dog", "wolf"]],
        belief_prior_mode="uniform",
        answerer_prior_mode="exponential_rank",
        answerer_prior_exponential_rate=0.5,
    )

    questioner_prior = get_questioner_prior(config)
    answerer_prior = get_answerer_prior(config)

    assert questioner_prior is not None
    assert answerer_prior is not None
    assert questioner_prior.probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert answerer_prior.probabilities[0] > answerer_prior.probabilities[1] > answerer_prior.probabilities[2]


def test_get_answerer_prior_uniform_is_independent_of_questioner_prior():
    config = Config(
        animals=[["cat", "dog", "wolf"]],
        belief_prior_mode="exponential_rank",
        belief_prior_exponential_rate=0.5,
        answerer_prior_mode="uniform",
    )

    answerer_prior = get_answerer_prior(config)

    assert answerer_prior is not None
    assert answerer_prior.hypotheses == ("cat", "dog", "wolf",)
    assert answerer_prior.probabilities == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_get_answerer_prior_none_returns_none():
    config = Config(
        animals=[["cat", "dog"]],
        belief_prior_mode="uniform",
        answerer_prior_mode="none",
    )

    assert get_answerer_prior(config) is None


def test_normalize_belief_label_keeps_clean_animal_names():
    assert normalize_belief_label("Cassowary") == "Cassowary"
    assert normalize_belief_label("  Southern cassowary  ") == "Southern cassowary"
    assert normalize_belief_label("Blue dragon sea slug.") == "Blue dragon sea slug"
    assert normalize_belief_label("Ground squirrel (specific African species)") == "Ground squirrel"
    assert normalize_belief_label("Tapir (already in list)") == "Tapir"


def test_normalize_belief_label_rejects_malformed_reasoning_strings():
    assert normalize_belief_label("Capybara (No, has fur) -> Green Iguana") is None
    assert normalize_belief_label(
        "Since the logic of the previous answers creates a contradiction, there are no naturally occurring primates."
    ) is None
    assert normalize_belief_label("Is it a cassowary") is None
    assert normalize_belief_label("year: new String();") is None
    assert normalize_belief_label("&getnow/parent.js") is None


def test_clean_generated_belief_labels_preserves_variants_and_drops_bad_lines():
    raw_beliefs = [
        "Cassowary",
        "Southern cassowary",
        "Ground squirrel (specific African species)",
        "Capybara (No, has fur) -> Green Iguana",
        " Blue dragon sea slug ",
        "Because the previous answers imply a contradiction",
        "year: new String();",
    ]

    assert clean_generated_belief_labels(raw_beliefs) == [
        "Cassowary",
        "Southern cassowary",
        "Ground squirrel",
        "Blue dragon sea slug",
    ]
