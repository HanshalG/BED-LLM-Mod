from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_slurm_launchers_accept_named_config_paths():
    for relative_path in [
        "scripts/run_location_fixed_root_depth_sweep.sh",
        "scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh",
        "scripts/run_strategy_ranking_fidelity.sh",
        "scripts/run_strategy_ranking_fidelity_gh200_singularity.sh",
    ]:
        script = ROOT / relative_path
        text = script.read_text(encoding="utf-8")

        assert 'CONFIG_ARG="$1"' in text or 'CONFIG_ARG="${1:?' in text
        assert 'if [ -f "$CONFIG_ARG" ]; then' in text
        assert 'CONFIG_PATH="$CONFIG_ARG"' in text
        assert 'CONFIG_PATH="configs/config${CONFIG_ARG}.yaml"' in text
        assert '-c "$CONFIG_PATH"' in text


def test_slurm_launchers_are_shell_parseable():
    subprocess.run(
        [
            "bash",
            "-n",
            str(ROOT / "scripts/run_location_fixed_root_depth_sweep.sh"),
            str(ROOT / "scripts/run_location_fixed_root_depth_sweep_gh200_singularity.sh"),
            str(ROOT / "scripts/run_strategy_ranking_fidelity.sh"),
            str(ROOT / "scripts/run_strategy_ranking_fidelity_gh200_singularity.sh"),
        ],
        check=True,
    )
