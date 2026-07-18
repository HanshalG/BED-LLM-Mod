import numpy as np

from scripts.copex_direct_proposal_thinking_probe import _initial_state
from scripts.nonmyopic_copex_direct_proposals import DirectProposalConfig


def test_thinking_probe_initial_state_replays_policy_seed_schedule() -> None:
    config = DirectProposalConfig(
        num_trials=2,
        num_rounds=2,
        num_particles=4,
        candidate_width=2,
        outer_rollouts=2,
        child_rollouts=2,
        grid_resolution=3,
        bootstrap_replicates=2,
        trial_concurrency=1,
        seed=77,
        one_step_scoring="quadrature",
    )
    particles_first, probabilities_first, position_first, truth_first = _initial_state(config, 1)
    particles_second, probabilities_second, position_second, truth_second = _initial_state(config, 1)
    assert np.array_equal(particles_first, particles_second)
    assert np.array_equal(probabilities_first, probabilities_second)
    assert np.array_equal(position_first, position_second)
    assert np.array_equal(truth_first, truth_second)
