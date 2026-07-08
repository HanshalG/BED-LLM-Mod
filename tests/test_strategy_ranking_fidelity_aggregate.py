import json

import pytest

from scripts.aggregate_strategy_ranking_fidelity import aggregate_strategy_ranking_fidelity


def _record(probe_index: int, trial_index: int, round_index: int, rho: float) -> dict:
    return {
        "probe_index": probe_index,
        "trial_index": trial_index,
        "round_index": round_index,
        "score_variant_metrics": {
            "configured": {
                "2": {
                    "n": 3,
                    "spearman_entropy": rho,
                    "pearson_entropy": rho,
                    "spearman_rmse": rho / 2,
                    "pearson_rmse": rho / 2,
                    "top1_regret_entropy": 1.0 - rho,
                    "score_var_between_strategies": 0.2,
                    "score_var_within_strategy": 0.1,
                    "snr_between_over_within": 2.0,
                }
            }
        },
    }


def test_aggregate_strategy_ranking_fidelity_merges_records_and_writes_report(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records = [_record(0, 0, 0, 0.25), _record(1, 1, 3, 0.75)]
    records_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config_path": "configs/config1431.yaml",
                "questioner_model": "gemma-test",
                "num_trials": 2,
                "num_probe_states": 2,
                "state_rounds": [0, 3],
                "depths": [2],
                "deployments": 4,
                "score_variants": ["configured"],
                "location_seed": 1304,
                "location_num_rounds": 8,
                "location_strategy_num_rollouts": 8,
                "records_path": str(records_path),
            }
        ),
        encoding="utf-8",
    )

    out_summary_path = aggregate_strategy_ranking_fidelity(
        [summary_path],
        output_dir=tmp_path / "out",
        run_name="merged",
        seed=0,
    )
    merged = json.loads(out_summary_path.read_text(encoding="utf-8"))

    assert merged["num_probe_states"] == 2
    assert merged["num_trials"] == 2
    assert merged["state_rounds"] == [0, 3]
    assert merged["aggregate"]["configured"]["2"]["spearman_entropy"]["mean"] == pytest.approx(0.5)
    assert (tmp_path / "out" / "merged_records.jsonl").exists()
    report = (tmp_path / "out" / "merged_REPORT.md").read_text(encoding="utf-8")
    assert "| configured | 2 |" in report
