from pathlib import Path

from scripts import number_game_atomic_particle_v2_mechanics as mechanics
from scripts import number_game_atomic_particle_v2_verify as verifier
from tests.test_number_game_atomic_particle_mechanics import SyntheticAdapter


def test_v2_full_bank_and_independent_replay(tmp_path: Path):
    result = mechanics.produce_bank(output_dir=tmp_path, adapter=SyntheticAdapter())
    assert len(result["raw"]["responses"]) == 6400
    assert result["raw"]["interface_version"] == "number-game-atomic-particle-depth3-2"
    assert result["raw"]["responses"][0]["seed"] == 202608160000
    assert result["raw"]["responses"][-1]["seed"] == 202608166399
    assert verifier.verify(tmp_path)["status"] == "verification_pass"
