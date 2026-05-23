"""Tests for padded animals run metrics."""

from __future__ import annotations

from core.bed_runner import RoundResult, TrialResult
from environments.animals.game_metrics import (
    animals_num_rounds,
    padded_trial_metric_series,
    summarize_animals_run_metrics,
)
from helpers import Config


def _config(*, animals_num_rounds: int = 12) -> Config:
    return Config(
        animals_num_rounds=animals_num_rounds,
        belief_state_mode="uniform",
        log_path=None,
    )


def test_padded_early_correct_fills_to_animals_num_rounds():
    config = _config(animals_num_rounds=8)
    trial = TrialResult(
        trial_index=0,
        hidden_state="cat",
        rounds=(
            RoundResult(
                round_index=2,
                candidates=(),
                chosen=None,  # type: ignore[arg-type]
                observation="Correct!",
                metrics={"guess_correct": 1.0, "correct_belief_mass": 0.4},
            ),
        ),
        final_metrics={},
    )
    metrics = padded_trial_metric_series(
        trial,
        {"accuracy": [0.0, 0.0, 1.0], "correct_belief_mass": [0.1, 0.2, 0.4]},
        method_name="EIG",
        num_rounds=animals_num_rounds(config),
    )
    assert len(metrics["accuracy"]) == 8
    assert metrics["accuracy"] == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert metrics["correct_belief_mass"][2:] == [1.0] * 6


def test_summarize_animals_run_metrics_averages_padded_trials():
    config = _config(animals_num_rounds=4)
    trials = (
        TrialResult(
            trial_index=0,
            hidden_state="cat",
            rounds=(
                RoundResult(
                    round_index=0,
                    candidates=(),
                    chosen=None,  # type: ignore[arg-type]
                    observation="Correct!",
                    metrics={"guess_correct": 1.0, "correct_belief_mass": 0.5},
                ),
            ),
            final_metrics={},
        ),
        TrialResult(
            trial_index=1,
            hidden_state="dog",
            rounds=(
                RoundResult(
                    round_index=1,
                    candidates=(),
                    chosen=None,  # type: ignore[arg-type]
                    observation="No",
                    metrics={"guess_correct": 0.0, "correct_belief_mass": 0.2},
                ),
                RoundResult(
                    round_index=2,
                    candidates=(),
                    chosen=None,  # type: ignore[arg-type]
                    observation="Correct!",
                    metrics={"guess_correct": 1.0, "correct_belief_mass": 0.9},
                ),
            ),
            final_metrics={},
        ),
    )
    run_result = type("RR", (), {"trials": trials})()
    metrics = summarize_animals_run_metrics(
        run_result,
        {},
        method_name="EIG",
        config=config,
    )
    assert len(metrics["accuracy"]) == 4
    # trial0 correct@0 -> [1,1,1,1]; trial1 correct@2 -> [0,0,1,1]
    assert metrics["accuracy"] == [0.5, 0.5, 1.0, 1.0]
