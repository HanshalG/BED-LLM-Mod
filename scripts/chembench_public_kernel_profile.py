"""Profile frozen public-prior search without constructing hidden worlds."""

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pstats
from time import monotonic

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from environments.chembench_mopen.pilot_data import build_public_pilot, read_protocol
from scripts.chembench_mopen_nonmyopic_opportunity import load_source, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config, digest = read_protocol()
    binding = verify_source(args.source_root)
    if binding["commit"] != config["source_commit"]:
        raise ValueError("source mismatch")
    model = build_public_pilot(
        load_source(args.source_root), model_type=EnvelopeGaussianModel
    ).model
    profile = cProfile.Profile()
    started = monotonic()
    profile.enable()
    try:
        plan = plan_batched(
            model,
            model.initial_state,
            args.horizon,
            max_states=8_000_000,
            max_seconds=60,
            max_workspace_bytes=64 * 1024**2,
        )
        result = {"status": "public_plan_complete", "plan": asdict(plan)}
    except Exception as exc:
        result = {
            "status": "public_plan_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        profile.disable()
    stats = pstats.Stats(profile)
    result.update(
        {
            "protocol_sha256": digest,
            "source_binding": binding,
            "elapsed_seconds": monotonic() - started,
            "horizon": args.horizon,
            "prior_particles": model.num_particles,
            "hidden_worlds_opened": False,
            "model_calls": 0,
            "cost_usd": 0,
            "paid_calls_authorized": False,
            "profile": [
                {
                    "file": key[0],
                    "line": key[1],
                    "function": key[2],
                    "calls": value[1],
                    "self_seconds": value[2],
                    "cumulative_seconds": value[3],
                }
                for key, value in sorted(
                    stats.stats.items(), key=lambda pair: -pair[1][3]
                )[:25]
            ],
            "source_hashes": {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in [
                    "scripts/chembench_public_kernel_profile.py",
                    "environments/chembench_mopen/batch_horizon.py",
                    "environments/chembench_mopen/envelope_belief.py",
                ]
            },
        }
    )
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"profile", "source_binding", "source_hashes"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
