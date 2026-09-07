"""Descriptive engineering summary, requiring a verified complete pilot."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.run_dir / "RESULT.json").read_bytes()
    result, replay = json.loads(raw), json.loads(args.replay.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if (
        replay.get("status") != "replay_verified"
        or replay.get("result_sha256") != digest
    ):
        raise ValueError("matching verified replay required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    table, traces = [], []
    for group in ("core", "population_oracle"):
        for arm in [a["arm"] for a in result["records"][0][group]]:
            episodes = [
                next(a for a in w[group] if a["arm"] == arm) for w in result["records"]
            ]
            mse = np.array([[s["target_mse"] for s in a["history"]] for a in episodes])
            table.append(
                {
                    "group": group,
                    "arm": arm,
                    "mean_final_mse": float(mse[:, -1].mean()),
                    "sd_final_mse": float(mse[:, -1].std(ddof=1)),
                    "mean_mse_trace": mse.mean(axis=0).tolist(),
                    "mean_rmse_trace": np.sqrt(mse).mean(axis=0).tolist(),
                }
            )
            for episode in episodes:
                for row in episode["history"]:
                    traces.append(
                        {
                            "group": group,
                            "arm": arm,
                            "world": episode["world"],
                            "round": row["round"],
                            "action": row["choice"]["action"],
                            "mse": row["target_mse"],
                            "rmse": float(np.sqrt(row["target_mse"])),
                            "model_variance": row["model_risk_after"],
                        }
                    )
    pairs = []
    for w in result["records"]:
        arms = {a["arm"]: a for a in w["core"]}
        sequence = {
            a: [s["choice"]["action"] for s in arms[a]["history"]] for a in arms
        }
        pairs.append(
            {
                "world": w["world"],
                "h1_minus_h3_mse": arms["h1"]["final_mse"] - arms["h3"]["final_mse"],
                "same_final_query_set": set(sequence["h1"]) == set(sequence["h3"]),
                "sequences": sequence,
            }
        )
    total = sum(p["h1_minus_h3_mse"] for p in pairs)
    same = sum(p["h1_minus_h3_mse"] for p in pairs if p["same_final_query_set"])
    means = result["mean_final_mse"]
    initial_equal = (
        json.loads((args.run_dir / "public_root_h2.json").read_text())["action"]
        == json.loads((args.run_dir / "public_root_h3.json").read_text())["action"]
    )
    summary = {
        "status": "descriptive_engineering_summary",
        "result_sha256": digest,
        "replay_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        "table": table,
        "paired_worlds": pairs,
        "h3_vs_h1_fractional_gain": 1 - means["h3"] / means["h1"],
        "h3_vs_open_loop_fractional_gain": 1 - means["h3"] / means["open_loop_h3"],
        "same_set_share_of_aggregate_gain": same / total if total else None,
        "h2_h3_identical_trajectories": all(
            p["sequences"]["h2"] == p["sequences"]["h3"] for p in pairs
        ),
        "h2_h3_initial_action_equal": initial_equal,
        "interpretation": "Eight-world source engineering only; neither powered efficacy nor LLM evidence. Round-by-design CRNs give reordered queries different observations. Same-set gains need a coupling/expected-risk diagnostic. If h2/h3 choose the same initial action, the three-query budget makes their subsequent policies identical under this fixed updater, ruling out a strict depth-three gain on that fixed-prior setup.",
        "model_calls": 0,
        "cost_usd": 0,
        "paid_calls_authorized": False,
    }
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (args.output_dir / "per_world_traces.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    lines = [
        "# Eight-world source engineering pilot",
        "",
        "Lower MSE is better. SD is across eight worlds; the oracle is a separate truth-support reference.",
        "",
        "| Group | Policy | Mean Final MSE | SD |",
        "|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {r['group']} | {r['arm']} | {r['mean_final_mse']:.6f} | {r['sd_final_mse']:.6f} |"
        for r in table
    )
    lines.extend(
        [
            "",
            summary["interpretation"],
            "",
            "No paid calls are authorized by this summary.",
            "",
        ]
    )
    (args.output_dir / "TABLE.md").write_text("\n".join(lines))
    print(summary["status"])


if __name__ == "__main__":
    main()
