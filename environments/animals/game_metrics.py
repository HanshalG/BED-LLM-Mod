"""Padded per-round animals metrics shared by animals runner surfaces."""

from __future__ import annotations

from typing import Any, Sequence


def animals_num_rounds(config: Any) -> int:
    return int(getattr(config, "animals_num_rounds", 20))


def pad_round_series(values: list[float], *, num_rounds: int, fill: float) -> list[float]:
    if len(values) >= num_rounds:
        return list(values[:num_rounds])
    if not values:
        return [fill] * num_rounds
    return list(values) + [fill] * (num_rounds - len(values))


def padded_trial_metric_series(
    trial: Any,
    raw_metrics: dict[str, list[float]],
    *,
    method_name: str,
    num_rounds: int,
) -> dict[str, list[float]]:
    """Expand per-round metrics to fixed ``num_rounds`` (flat GameMetrics semantics)."""
    accuracy = list(raw_metrics.get("accuracy", raw_metrics.get("guess_correct", [])))
    belief_mass = list(raw_metrics.get("correct_belief_mass", []))

    ended_correct = bool(trial.rounds and trial.rounds[-1].observation == "Correct!")
    if ended_correct:
        start = int(trial.rounds[-1].round_index)
        accuracy = [0.0] * num_rounds
        accuracy[start:] = [1.0] * (num_rounds - start)
        if method_name not in {"naive", "naive+belief"}:
            belief_mass = [0.0] * num_rounds
            belief_mass[start:] = [1.0] * (num_rounds - start)
        elif method_name == "naive":
            belief_mass = [0.0] * num_rounds
        elif len(trial.rounds) == 1:
            belief_mass = [1.0] * num_rounds
        elif belief_mass:
            belief_mass = [float(belief_mass[0])] + [1.0] * (num_rounds - 1)
        else:
            belief_mass = [1.0] * num_rounds
    else:
        accuracy = pad_round_series(accuracy, num_rounds=num_rounds, fill=0.0)
        belief_mass = pad_round_series(belief_mass, num_rounds=num_rounds, fill=0.0)

    return {"accuracy": accuracy, "correct_belief_mass": belief_mass}


def average_padded_series(series_many: Sequence[Sequence[float]]) -> list[float]:
    if not series_many:
        return []
    width = max(len(series) for series in series_many)
    totals = [0.0] * width
    for series in series_many:
        for index, value in enumerate(series):
            totals[index] += float(value)
    count = float(len(series_many))
    return [total / count for total in totals]


def summarize_animals_run_metrics(
    run_result: Any,
    raw_metrics: dict[str, list[float]],
    *,
    method_name: str,
    config: Any,
) -> dict[str, list[float]]:
    """Pad each trial then average (matches ``twenty_questions_animals`` aggregation)."""
    num_rounds = animals_num_rounds(config)
    if not run_result.trials:
        return {
            "accuracy": [0.0] * num_rounds,
            "correct_belief_mass": [0.0] * num_rounds,
        }

    if len(run_result.trials) == 1:
        return padded_trial_metric_series(
            run_result.trials[0],
            raw_metrics,
            method_name=method_name,
            num_rounds=num_rounds,
        )

    per_trial = [
        padded_trial_metric_series(
            trial,
            {
                "accuracy": [
                    float(round_result.metrics.get("guess_correct", round_result.metrics.get("accuracy", 0.0)))
                    for round_result in trial.rounds
                ],
                "correct_belief_mass": [
                    float(round_result.metrics.get("correct_belief_mass", 0.0))
                    for round_result in trial.rounds
                ],
            },
            method_name=method_name,
            num_rounds=num_rounds,
        )
        for trial in run_result.trials
    ]
    return {
        "accuracy": average_padded_series([trial["accuracy"] for trial in per_trial]),
        "correct_belief_mass": average_padded_series(
            [trial["correct_belief_mass"] for trial in per_trial]
        ),
    }
