"""Cross-trial batched execution for hyperbolic temporal discounting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from helpers import Config, print_and_log

from .runner import (
    HyperbolicBeliefState,
    HyperbolicDesign,
    HyperbolicFindingMetrics,
    HyperbolicParams,
    _format_params,
    _generate_hyperbolic_hypotheses_many,
    _hyperbolic_effective_sample_size,
    _log_hyperbolic,
    _make_hyperbolic_trial_state,
    _merge_hypotheses,
    _summarize_belief_state,
    _top_parameter_rmse,
    _write_to_log_if_configured,
    build_hyperbolic_posteriors_many,
    choose_design_naive_many,
    choose_strategy_designs_many,
    estimate_params_naive_many,
    eval_holdout_designs,
    generate_hyperbolic_candidates_many,
    implied_choice_accuracy,
    k_rmse,
    parameter_rmse,
    prompt_hyperbolic_belief_state,
    sample_hyperbolic_eig_belief_state,
    score_candidate_designs,
)

if TYPE_CHECKING:
    from model import Model


def run_hyperbolic_trials_batched(
    questioner: "Model",
    config: Config,
    rng: np.random.Generator | None = None,
    output_dir: Path | None = None,
    method_name: str = "EIG",
) -> HyperbolicFindingMetrics:
    del output_dir
    rng = rng or np.random.default_rng()
    holdout_designs = eval_holdout_designs(config)
    rmse_totals = np.zeros(config.htd_num_rounds, dtype=float)
    k_rmse_totals = np.zeros(config.htd_num_rounds, dtype=float)
    top_probability_totals = np.zeros(config.htd_num_rounds, dtype=float)
    selected_eig_totals = np.zeros(config.htd_num_rounds, dtype=float)
    implied_accuracy_totals = np.zeros(config.htd_num_rounds, dtype=float)

    _log_hyperbolic(
        f"Running {config.htd_num_trials} trial(s) in cross-trial batches: "
        f"method={method_name}, batch_size={config.htd_trial_batch_size}, "
        f"rounds={config.htd_num_rounds}, posterior_mode={config.htd_posterior_mode}",
        config,
    )

    for batch_start in range(0, config.htd_num_trials, config.htd_trial_batch_size):
        batch_indices = list(
            range(batch_start, min(config.htd_num_trials, batch_start + config.htd_trial_batch_size))
        )
        states = [_make_hyperbolic_trial_state(config, trial_idx, rng) for trial_idx in batch_indices]
        for state in states:
            _log_hyperbolic(
                f"trial {state.trial_idx + 1}/{config.htd_num_trials}: true_params="
                f"{_format_params(state.env.true_params)}",
                config,
            )

        if method_name.lower() in {"naive", "naive+belief"}:
            for round_idx in range(config.htd_num_rounds):
                best_designs = choose_design_naive_many(
                    questioner,
                    [state.observations for state in states],
                    config,
                )
                for state, design in zip(states, best_designs):
                    if design is None:
                        continue
                    observation = state.env.run_experiment(design)
                    state.observations.append(observation)
                    print_and_log(
                        f"[hyperbolic] trial {state.trial_idx + 1} naive design "
                        f"iR={design.immediate_reward:g}, dR={design.delayed_reward:g}, "
                        f"days={design.days}; observed {observation.value:.4f}",
                        config,
                    )
                estimates = estimate_params_naive_many(
                    questioner,
                    [state.observations for state in states],
                    config,
                )
                for state, estimate in zip(states, estimates):
                    state.final_estimate = estimate
                    state.final_rmse = parameter_rmse(estimate, state.env.true_params)
                    rmse_totals[round_idx] += state.final_rmse
                    k_rmse_totals[round_idx] += k_rmse(estimate, state.env.true_params)
                    top_probability_totals[round_idx] += 1.0
                    implied_accuracy_totals[round_idx] += implied_choice_accuracy(
                        estimate,
                        state.env.true_params,
                        holdout_designs,
                    )
            continue

        initial_hypotheses_many = _generate_hyperbolic_hypotheses_many(
            questioner,
            [state.observations for state in states],
            [None for _state in states],
            config,
            label="batched initial belief generation",
        )
        initial_beliefs = build_hyperbolic_posteriors_many(
            questioner,
            initial_hypotheses_many,
            [state.observations for state in states],
            config,
            label="batched initial posterior scoring",
        )
        for state, belief_state in zip(states, initial_beliefs):
            state.belief_state = belief_state
            prompt_state = prompt_hyperbolic_belief_state(belief_state, config)
            _log_hyperbolic(
                f"trial {state.trial_idx + 1}: initial posterior {_summarize_belief_state(belief_state)}; "
                f"prompt={len(prompt_state.hypotheses)}, "
                f"ESS={_hyperbolic_effective_sample_size(belief_state):.2f}",
                config,
            )

        for round_idx in range(config.htd_num_rounds):
            prompt_beliefs = [
                prompt_hyperbolic_belief_state(state.belief_state, config)  # type: ignore[arg-type]
                for state in states
            ]
            eig_beliefs: list[HyperbolicBeliefState] = []
            eig_collapsed_flags: list[bool] = []
            for state in states:
                if method_name.upper() == "EIG":
                    eig_belief, collapsed = sample_hyperbolic_eig_belief_state(
                        state.belief_state,  # type: ignore[arg-type]
                        config,
                        state.rng,
                    )
                else:
                    eig_belief = state.belief_state  # type: ignore[assignment]
                    collapsed = False
                eig_beliefs.append(eig_belief)
                eig_collapsed_flags.append(collapsed)

            for state, prompt_belief, eig_belief, collapsed in zip(
                states, prompt_beliefs, eig_beliefs, eig_collapsed_flags
            ):
                _write_to_log_if_configured(
                    f"\nHyperbolic trial {state.trial_idx + 1}: Round {round_idx + 1}\n",
                    config,
                )
                _log_hyperbolic(
                    f"trial {state.trial_idx + 1}, round {round_idx + 1}: "
                    f"{_summarize_belief_state(state.belief_state)}; "  # type: ignore[arg-type]
                    f"EIG_support={len(eig_belief.hypotheses)}",
                    config,
                )
                if collapsed:
                    _log_hyperbolic("EIG posterior sampling collapsed to one hypothesis", config)

            if method_name.upper() == "EIG":
                candidates_many = generate_hyperbolic_candidates_many(
                    questioner,
                    prompt_beliefs,
                    [state.observations for state in states],
                    config,
                )
                best_designs: list[HyperbolicDesign | None] = []
                best_scores: list[float] = []
                for state, eig_belief, candidates in zip(states, eig_beliefs, candidates_many):
                    if not candidates:
                        best_designs.append(None)
                        best_scores.append(0.0)
                        continue
                    scores = score_candidate_designs(
                        eig_belief,
                        candidates,
                        config,
                        questioner=questioner,
                        observations=state.observations,
                    )
                    best_idx = int(np.argmax(scores)) if scores else 0
                    best_designs.append(candidates[best_idx])
                    best_scores.append(float(scores[best_idx]) if scores else 0.0)
            else:
                strategy_results = choose_strategy_designs_many(
                    questioner,
                    [state.belief_state for state in states],  # type: ignore[misc]
                    [state.observations for state in states],
                    config,
                )
                best_designs = [design for design, _score in strategy_results]
                best_scores = [score for _design, score in strategy_results]

            skipped: set[int] = set()
            for state_idx, (state, design, score) in enumerate(zip(states, best_designs, best_scores)):
                if design is None:
                    skipped.add(state_idx)
                    continue
                observation = state.env.run_experiment(design)
                state.observations.append(observation)
                print_and_log(
                    f"[hyperbolic] Selected design iR={design.immediate_reward:g}, "
                    f"dR={design.delayed_reward:g}, days={design.days} "
                    f"with score {score:.6f}; observed {observation.value:.4f}",
                    config,
                )

            active_indices = [index for index in range(len(states)) if index not in skipped]
            if active_indices:
                active_states = [states[index] for index in active_indices]
                active_prompt_beliefs = [prompt_beliefs[index] for index in active_indices]
                generated_many = _generate_hyperbolic_hypotheses_many(
                    questioner,
                    [state.observations for state in active_states],
                    active_prompt_beliefs,
                    config,
                    label=f"batched round {round_idx + 1} belief update",
                )
                merged_many: list[list[HyperbolicParams]] = []
                previous_beliefs: list[HyperbolicBeliefState] = []
                for state, generated in zip(active_states, generated_many):
                    previous = state.belief_state  # type: ignore[assignment]
                    previous_beliefs.append(previous)
                    merged_many.append(_merge_hypotheses(previous, generated))
                updated_beliefs = build_hyperbolic_posteriors_many(
                    questioner,
                    merged_many,
                    [state.observations for state in active_states],
                    config,
                    context_states=previous_beliefs,
                    label=f"batched round {round_idx + 1} posterior scoring",
                )
                for state, belief_state in zip(active_states, updated_beliefs):
                    state.belief_state = belief_state
                    _log_hyperbolic(
                        f"round {round_idx + 1}: posterior {_summarize_belief_state(belief_state)}",
                        config,
                    )

            for state, best_score in zip(states, best_scores):
                belief_state = state.belief_state
                current_rmse = _top_parameter_rmse(belief_state, state.env.true_params) if belief_state else float("inf")
                top_probability = (
                    belief_state.probabilities[0]
                    if belief_state and belief_state.probabilities
                    else 0.0
                )
                estimate = belief_state.hypotheses[0] if belief_state and belief_state.hypotheses else None
                implied_acc = (
                    implied_choice_accuracy(estimate, state.env.true_params, holdout_designs)
                    if estimate is not None
                    else 0.0
                )
                rmse_totals[round_idx] += current_rmse
                k_rmse_totals[round_idx] += (
                    k_rmse(estimate, state.env.true_params) if estimate is not None else float("inf")
                )
                top_probability_totals[round_idx] += top_probability
                selected_eig_totals[round_idx] += best_score
                implied_accuracy_totals[round_idx] += implied_acc

    divisor = float(config.htd_num_trials)
    return HyperbolicFindingMetrics(
        parameter_rmse=(rmse_totals / divisor).tolist(),
        k_rmse=(k_rmse_totals / divisor).tolist(),
        top_probability=(top_probability_totals / divisor).tolist(),
        selected_eig=(selected_eig_totals / divisor).tolist(),
        implied_choice_accuracy=(implied_accuracy_totals / divisor).tolist(),
    )
