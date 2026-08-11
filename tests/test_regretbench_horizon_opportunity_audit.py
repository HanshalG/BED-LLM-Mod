from __future__ import annotations

import json

from scripts import regretbench_horizon_opportunity_audit as audit


def test_regretbench_has_no_environment_grounded_depth_two_action_gap(tmp_path):
    result = audit.run_audit(output_dir=tmp_path)

    assert result["status"] == "structural_opportunity_null"
    assert result["authorizes"] == "close_regretbench_primary_nonmyopic_route"
    assert result["gates"]["all_pass"] is True
    assert result["population"] == {
        "test_file_count": 6286,
        "eligible_task_count": 2419,
        "strict_positive_depth_two_gain_count": 0,
        "maximum_depth_two_gain_nats": 0.0,
        "tolerance": 1e-12,
    }
    fresh = result["fresh_exact4_population"]
    assert fresh["task_count"] == 759
    assert fresh["strict_positive_depth_two_gain_count"] == 0
    assert fresh["maximum_depth_two_gain_nats"] == 0.0
    assert fresh["prior_cohort_counts"]["union"] == 528
    assert all(
        row["strict_positive_gain_count"] == 0
        and row["maximum_gain_nats"] == 0.0
        for row in result["strata"].values()
    )
    assert result["model_calls_made"] == 0
    assert result["cost_usd"] == 0.0
    assert result["policy_endpoint_opened"] is False

    saved = json.loads((tmp_path / "RESULT.json").read_text())
    assert saved == result
