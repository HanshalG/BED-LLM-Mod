"""Cross-trial batched execution for location finding."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from helpers import Config, print_and_log

from .runner import (
    Location,
    LocationBeliefState,
    LocationFindingMetrics,
    _format_source_array,
    _generate_location_hypotheses_many,
    _log_location,
    _make_location_trial_state,
    _merge_hypotheses,
    _plot_location_trial_state,
    _summarize_belief_state,
    _top_source_rmse,
    _write_to_log_if_configured,
    build_location_posteriors_many,
    choose_locations_naive_many,
    choose_locations_with_strategy_rollouts_many,
    estimate_sources_naive_many,
    generate_location_candidates_many,
    prompt_location_belief_state,
    sample_location_eig_belief_state,
    score_candidate_locations,
    source_rmse,
    _location_effective_sample_size,
)

if TYPE_CHECKING:
    from model import Model

def run_location_trials_batched(
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

        if method_name.lower() in {"naive", "naive+belief"}:
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
                if method_name.upper() == "EIG":
                    eig_belief_state, eig_sample_collapsed = sample_location_eig_belief_state(
                        state.belief_state,  # type: ignore[arg-type]
                        config,
                        state.rng,
                    )
                    support_label = "EIG_support"
                elif method_name.lower() in {"strategyeig", "strategyeig+root"}:
                    eig_belief_state = state.belief_state  # type: ignore[assignment]
                    eig_sample_collapsed = False
                    support_label = (
                        "strategy_root_support" if method_name.lower() == "strategyeig+root" else "strategy_support"
                    )
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

            if method_name.upper() == "EIG":
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
                    fixed_root=method_name.lower() == "strategyeig+root",
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


