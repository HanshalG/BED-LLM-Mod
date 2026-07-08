import json

from scripts.extract_location_qualitative_examples import extract_qualitative_examples


def test_extract_qualitative_examples_writes_strategy_text_and_plot(tmp_path):
    run_dir = tmp_path / "runs/demo"
    run_dir.mkdir(parents=True)
    summary = {
        "per_trial": [
            {
                "trial_index": 0,
                "policy_label": "EIG",
                "policy_kind": "EIG",
                "hidden_state": [[1.0, 0.0]],
                "history": [
                    {"action": [0.0, 0.0], "observation": {"query": [0.0, 0.0], "value": 1.0}},
                    {"action": [0.5, 0.0], "observation": {"query": [0.5, 0.0], "value": 2.0}},
                ],
                "round_metrics": [{"source_rmse": 0.8}, {"source_rmse": 0.6, "truth_log_probability": -2.0}],
            },
            {
                "trial_index": 0,
                "policy_label": "StrategyEIG-d3",
                "policy_kind": "StrategyEIG",
                "hidden_state": [[1.0, 0.0]],
                "history": [
                    {"action": [0.0, 0.0], "observation": {"query": [0.0, 0.0], "value": 1.0}},
                    {"action": [1.0, 0.0], "observation": {"query": [1.0, 0.0], "value": 8.0}},
                ],
                "round_metrics": [{"source_rmse": 0.7}, {"source_rmse": 0.1, "truth_log_probability": -0.2}],
            },
        ]
    }
    (run_dir / "fixed_root_depth_sweep_metrics.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    decisions = [
        {
            "trial_index": 0,
            "policy_label": "StrategyEIG-d3",
            "round_index": 0,
            "selected_eig": 0.5,
            "selected_root_query": [0.0, 0.0],
            "selected_strategy": "Move toward the right branch, then exploit the strong signal.",
        }
    ]
    (run_dir / "fixed_root_depth_sweep_decisions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in decisions),
        encoding="utf-8",
    )

    payload = extract_qualitative_examples(
        summary_path=run_dir,
        output_dir=tmp_path / "results/location_qualitative",
        run_label="demo",
        num_examples=1,
    )

    assert payload["num_examples"] == 1
    example = payload["examples"][0]
    assert example["rmse_delta_vs_eig"] == -0.5
    assert example["selected_strategies"][0]["strategy"].startswith("Move toward")
    assert (tmp_path / "results/location_qualitative/demo_qualitative_examples.md").exists()
    assert (tmp_path / "results/location_qualitative/demo_qualitative_example_1.png").stat().st_size > 0
