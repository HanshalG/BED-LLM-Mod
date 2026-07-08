import json

from scripts.cost_vs_depth_table import markdown_table, row_from_path, write_cost_table


def test_row_from_fixed_root_summary_includes_per_trial_costs(tmp_path):
    summary_path = tmp_path / "fixed_root_depth_sweep_metrics.json"
    summary_path.write_text(
        json.dumps(
            {
                "max_depth": 5,
                "num_trials": 10,
                "num_rounds": 6,
                "location_strategy_num_candidates": 4,
                "location_strategy_num_rollouts": 8,
                "token_usage": {
                    "total": {
                        "calls": 4,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    }
                },
                "aggregate_by_strategy_depth": {},
            }
        ),
        encoding="utf-8",
    )

    row = row_from_path(summary_path)

    assert row["method"] == "StrategyEIG fixed-root sweep"
    assert row["depth"] == "1..5"
    assert row["total_tokens"] == 150
    assert row["tokens_per_trial"] == 15
    assert row["tokens_per_trial_round"] == 2.5
    assert row["brute_force_cost_proxy"][2] == {
        "depth": 3,
        "branching_factor": 4,
        "rollouts": 8,
        "deployed_decisions": 60,
        "strategy_root_sets": 60,
        "strategy_rollout_paths": 1920,
        "strategy_simulated_steps": 5760,
        "brute_force_candidate_sets": 1260,
        "brute_force_leaf_sequences": 3840,
        "brute_force_candidate_set_ratio_vs_strategy": 21.0,
    }


def test_write_cost_table_from_run_dir_and_log(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "run.log").write_text(
        "\n".join(
            [
                "progress",
                json.dumps(
                    {
                        "event": "llm_token_usage",
                        "model": "test/model",
                        "call_type": "chat",
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    json_path, md_path = write_cost_table([run_dir], output_dir, "demo")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    text = md_path.read_text(encoding="utf-8")

    assert data["rows"][0]["label"] == "run_a"
    assert data["rows"][0]["total_tokens"] == 7
    assert "| run_a | unknown |" in text


def test_markdown_table_notes_fixed_root_total_cost():
    text = markdown_table(
        [
            {
                "label": "demo",
                "method": "StrategyEIG fixed-root sweep",
                "depth": "1..5",
                "num_trials": 1,
                "num_rounds": 2,
                "calls": 1,
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "tokens_per_trial": 7,
                "tokens_per_trial_round": 3.5,
                "brute_force_cost_proxy": [
                    {
                        "depth": 2,
                        "branching_factor": 3,
                        "rollouts": 4,
                        "deployed_decisions": 2,
                        "strategy_root_sets": 2,
                        "strategy_rollout_paths": 24,
                        "strategy_simulated_steps": 48,
                        "brute_force_candidate_sets": 8,
                        "brute_force_leaf_sequences": 18,
                        "brute_force_candidate_set_ratio_vs_strategy": 4.0,
                    }
                ],
            }
        ]
    )

    assert "# LLM Cost vs Depth" in text
    assert "fixed-root depth sweeps report total run cost" in text
    assert "StrategyEIG vs brute-force n-step EIG proxy" in text
    assert "| demo | 2 | 3 | 4 | 2 | 2 | 24 | 48 | 8 | 18 | 4 |" in text
