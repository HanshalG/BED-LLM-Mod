"""Independent trace replay for the Mushroom feature-acquisition qualification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mushroom_feature_acquisition import MushroomFeatureModel  # noqa: E402


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != "mushroom_feature_acquisition_depth_qualification":
        raise ValueError("unexpected Mushroom qualification stage")
    model = MushroomFeatureModel()
    replayed: dict[str, list[dict[str, float]]] = {}
    legality = True
    outcome_match = True
    metric_match = True
    for arm, traces in payload["traces"].items():
        replayed[arm] = []
        for trace in traces:
            state = model.initial_state
            belief = model.initial_belief.copy()
            entropy: list[float] = []
            truth_log: list[float] = []
            for step in trace["steps"]:
                action = str(step["action"])
                legality &= action in model.legal_actions(state)
                outcome = model.observation(int(trace["truth_index"]), action)
                outcome_match &= outcome == step["observation"]
                belief = model.posterior(belief, action, outcome)
                state = model.next_state(state, action)
                entropy.append(model.target_entropy(belief))
                truth_log.append(model.truth_log_probability(belief, int(trace["truth_index"])))
                metric_match &= math.isclose(entropy[-1], step["entropy"], abs_tol=1e-12)
                metric_match &= math.isclose(
                    truth_log[-1], step["truth_log_probability"], abs_tol=1e-12
                )
            entropy_auc = float(np.mean(entropy))
            truth_auc = float(np.mean(truth_log))
            metric_match &= math.isclose(entropy_auc, trace["entropy_auc"], abs_tol=1e-12)
            metric_match &= math.isclose(
                truth_auc, trace["truth_log_probability_auc"], abs_tol=1e-12
            )
            replayed[arm].append(
                {"entropy_auc": entropy_auc, "truth_log_probability_auc": truth_auc}
            )
    one = replayed["depth_one"]
    two = replayed["depth_two"]
    entropy_gain = np.asarray(
        [d1["entropy_auc"] - d2["entropy_auc"] for d1, d2 in zip(one, two)]
    )
    truth_gain = np.asarray(
        [
            d2["truth_log_probability_auc"] - d1["truth_log_probability_auc"]
            for d1, d2 in zip(one, two)
        ]
    )
    stored = payload["comparison"]
    comparison_match = math.isclose(
        float(np.mean(entropy_gain)), stored["entropy_auc_gain"]["mean"], abs_tol=1e-12
    ) and math.isclose(
        float(np.mean(truth_gain)),
        stored["truth_log_probability_auc_gain"]["mean"],
        abs_tol=1e-12,
    )
    entropy_ci = _bootstrap(entropy_gain, seed=24_125)
    truth_ci = _bootstrap(truth_gain, seed=24_126)
    mechanics = {
        "all_actions_legal": legality,
        "deterministic_outcomes_match": outcome_match,
        "posterior_metrics_match": metric_match,
        "paired_truths_match": all(
            one_trace["truth_index"] == two_trace["truth_index"]
            for one_trace, two_trace in zip(
                payload["traces"]["depth_one"], payload["traces"]["depth_two"]
            )
        ),
        "truths_sampled_without_replacement": len(
            {trace["truth_index"] for trace in payload["traces"]["depth_one"]}
        )
        == len(payload["traces"]["depth_one"]),
        "stored_comparison_matches_replay": comparison_match,
        "no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "mushroom_feature_acquisition_depth_qualification_audit",
        "source_config": payload["config"],
        "mechanics": mechanics,
        "entropy_auc_gain": {
            "mean": float(np.mean(entropy_gain)),
            "independent_ci95": entropy_ci,
        },
        "truth_log_probability_auc_gain": {
            "mean": float(np.mean(truth_gain)),
            "independent_ci95": truth_ci,
        },
        "passed": all(mechanics.values()) and entropy_ci[0] > 0.0 and truth_ci[0] > 0.0,
    }


def render(summary: dict[str, Any]) -> str:
    entropy = summary["entropy_auc_gain"]
    truth = summary["truth_log_probability_auc_gain"]
    return "\n".join(
        [
            "# Mushroom Feature Acquisition Qualification Audit",
            "",
            f"Audit passed: **{summary['passed']}**.",
            "",
            f"- Entropy-AUC gain: `{entropy['mean']:+.6f}`, independent 95% CI "
            f"`[{entropy['independent_ci95'][0]:+.6f}, {entropy['independent_ci95'][1]:+.6f}]`.",
            f"- Truth-log-AUC gain: `{truth['mean']:+.6f}`, independent 95% CI "
            f"`[{truth['independent_ci95'][0]:+.6f}, {truth['independent_ci95'][1]:+.6f}]`.",
            "- Every stored action, deterministic observation, posterior metric, paired truth, and aggregate comparison was replayed from the raw catalog.",
            "- The audit made zero LLM calls.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
