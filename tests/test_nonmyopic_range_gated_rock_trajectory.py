from __future__ import annotations

from scripts.audit_nonmyopic_range_gated_rock_trajectory import audit
from scripts.nonmyopic_range_gated_rock_fixed_tail import (
    DeterministicFixedTailModel,
    FixedRootTailProvider,
)
from scripts.nonmyopic_range_gated_rock_strategy import RangeGatedStrategyConfig
from scripts.nonmyopic_range_gated_rock_trajectory import (
    CachedFixedRootTailProvider,
    RangeGatedTrajectoryConfig,
    _model,
    run_confirmation,
)
from scripts.nonmyopic_range_gated_rock_trajectory_smoke import (
    run_smoke,
    trajectory_prefix_histories,
)


def _base_provider(seed: int) -> FixedRootTailProvider:
    return FixedRootTailProvider(
        DeterministicFixedTailModel(),
        RangeGatedStrategyConfig(seed=seed),
        include_successor_grounding=True,
        accept_json_prefix=True,
    )


def test_trajectory_prefix_smoke_covers_twelve_late_states() -> None:
    histories = trajectory_prefix_histories()

    assert len(histories) == 12
    assert len(set(histories)) == 12
    assert histories[0] == ()
    assert histories[2] == (
        ("move-SOUTH", None),
        ("move-SOUTH", None),
    )


def test_cached_provider_reuses_only_identical_prompts() -> None:
    model = _model(RangeGatedTrajectoryConfig())
    base = _base_provider(24_188)
    provider = CachedFixedRootTailProvider(base, max_unique_cells=64)
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


def test_deterministic_late_state_smoke_passes() -> None:
    config = RangeGatedStrategyConfig(seed=24_187)
    result = run_smoke(_base_provider(config.seed), strategy_config=config)

    assert all(result["mechanics"].values())
    assert result["exact_d3_root_match_rate"] >= 0.75


def test_deterministic_cached_trajectory_passes_registered_endpoints() -> None:
    config = RangeGatedTrajectoryConfig(seed=24_188)
    provider = CachedFixedRootTailProvider(
        _base_provider(config.seed),
        max_unique_cells=config.max_unique_llm_cells,
    )
    result = run_confirmation(provider, config)

    assert all(result["mechanics"].values())
    assert all(result["endpoint_gate"].values())
    assert result["llm_recovery_fraction_of_exact_d3_gain"] > 0.99
    assert result["llm_first_two_south_rate"] == 1.0
    assert result["llm_onsite_by_round_three_rate"] == 1.0
    assert len(result["logical_requests"]) == 300
    assert len(result["candidate_requests"]) == 17
    result["gate"] = {
        "passed": all(result["mechanics"].values())
        and all(result["endpoint_gate"].values())
    }
    replay = audit(result)
    assert replay["gate"]["passed"]
    assert all(replay["mechanics"].values())
