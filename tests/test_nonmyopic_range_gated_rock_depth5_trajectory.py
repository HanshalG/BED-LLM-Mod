from scripts.audit_nonmyopic_range_gated_rock_depth5_trajectory import audit
from scripts.nonmyopic_range_gated_rock_depth5_goal_compiler import (
    Depth5GoalCompilerConfig,
    Depth5GoalCompilerProvider,
    DeterministicDepth5GoalModel,
    build_depth5_model,
)
from scripts.nonmyopic_range_gated_rock_depth5_trajectory import (
    CachedDepth5GoalProvider,
    FocusedDepth5TrajectoryConfig,
    run_confirmation,
)
from scripts.nonmyopic_range_gated_rock_depth5_trajectory_smoke import (
    run_smoke,
    trajectory_prefix_histories,
)


def _base_provider(seed: int) -> Depth5GoalCompilerProvider:
    return Depth5GoalCompilerProvider(
        DeterministicDepth5GoalModel(),
        Depth5GoalCompilerConfig(seed=seed),
    )


def test_trajectory_smoke_covers_twelve_registered_states() -> None:
    histories = trajectory_prefix_histories()

    assert len(histories) == 12
    assert len(set(histories)) == 12
    assert histories[0] == ()
    assert histories[3] == (
        ("move-NORTH", None),
        ("move-WEST", None),
        ("move-WEST", None),
    )


def test_cached_h5_provider_reuses_only_identical_prompts() -> None:
    model = build_depth5_model()
    provider = CachedDepth5GoalProvider(
        _base_provider(24_243),
        max_unique_cells=64,
    )
    kwargs = {
        "model": model,
        "position": model.map_spec.start_position,
        "belief": model.initial_belief.copy(),
        "history": (),
    }

    first = provider.propose(cell_index=0, **kwargs)
    second = provider.propose(cell_index=1, **kwargs)

    assert first.plans == second.plans
    assert len(provider.physical_requests) == 1
    assert len(provider.logical_requests) == 2
    assert not provider.logical_requests[0]["cache_hit"]
    assert provider.logical_requests[1]["cache_hit"]


def test_deterministic_h5_trajectory_smoke_passes() -> None:
    config = Depth5GoalCompilerConfig(seed=24_244)
    result = run_smoke(
        _base_provider(config.seed),
        strategy_config=config,
    )

    assert all(result["mechanics"].values())
    assert result["exact_h5_root_match_count"] == 12


def test_deterministic_cached_h5_trajectory_and_audit_pass() -> None:
    config = FocusedDepth5TrajectoryConfig(seed=24_243)
    provider = CachedDepth5GoalProvider(
        _base_provider(config.seed),
        max_unique_cells=config.max_unique_llm_cells,
    )

    result = run_confirmation(provider, config)
    result["gate"] = {
        "passed": all(result["mechanics"].values())
        and all(result["endpoint_gate"].values())
    }
    replay = audit(result, audit_bootstrap_seed=24_246)

    assert result["gate"]["passed"]
    assert result["llm_recovery_fraction_of_exact_h5_gain"] == 1.0
    assert result["llm_registered_route_rate"] == 1.0
    assert len(result["logical_requests"]) == 400
    assert len(result["candidate_requests"]) == 18
    assert replay["gate"]["passed"]
    assert all(replay["mechanics"].values())
