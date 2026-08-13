from scripts import tau2_mms_split_partition_source_audit as audit


def test_split_partition_cohort_is_fresh_balanced_and_bound():
    episodes = audit.load_episodes()
    assert len(episodes) == 6
    assert [row["family"] for row in episodes] == ["mms_abroad"] * 3 + ["mms_home"] * 3
    assert len({audit.episode_hash(row) for row in episodes}) == 6


def test_split_partition_source_opportunity_passes():
    result = audit.audit()
    assert result["status"] == "source_pass"
    assert result["gates"]["all_source_gates_pass"] is True
    assert result["model_calls_made"] == 0
    assert result["repair_or_task_success_endpoints_opened"] is False
