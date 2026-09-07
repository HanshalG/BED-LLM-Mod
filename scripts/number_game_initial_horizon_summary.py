"""Retrospective tie-mechanism analysis; no new policy/LLM endpoints or gates."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scripts.number_game_initial_horizon_audit import (
    ExactMembershipHorizon,
    digest,
    load_compiler,
)


def summarize(result, initial, compile_rule):
    if result["status"] not in {"initial_opportunity_pass", "initial_opportunity_null"}:
        raise ValueError("requires complete initial-horizon diagnostic")
    if [x["tree_seed"] for x in result["rows"]] != [x["tree_seed"] for x in initial]:
        raise ValueError("initial/result panel mismatch")
    rows = []
    for outcome, record in zip(result["rows"], initial, strict=True):
        extensions = list(
            dict.fromkeys(compile_rule(x["expression"]) for x in record["initial"])
        )
        solver = ExactMembershipHorizon(extensions)
        optimum, query = solver.plan(solver.full, 2)
        if query != outcome["first_queries"][1]:
            raise ValueError("h2 initial choice does not replay")
        q3 = outcome["first_queries"][2]
        h3_root_h2_value = (
            solver.action_value(solver.full, 2, q3) if q3 is not None else optimum
        )
        if h3_root_h2_value < optimum:
            raise ArithmeticError("invalid h2 minimum")
        values = list(map(Fraction, outcome["exact_full_budget_values"]))
        rows.append(
            {
                "tree_seed": record["tree_seed"],
                "h2_to_h3_improves": values[2] < values[1],
                "h3_root_is_also_h2_optimal": h3_root_h2_value == optimum,
                "h2_objective_sacrifice": str(h3_root_h2_value - optimum),
                "h2_to_h3_full_budget_gain": str(values[1] - values[2]),
            }
        )
    return {
        "status": "descriptive_mechanism_complete",
        "rows": rows,
        "h3_improvements_from_h2_tied_root": sum(
            r["h2_to_h3_improves"] and r["h3_root_is_also_h2_optimal"] for r in rows
        ),
        "h3_improvements_requiring_h2_objective_sacrifice": sum(
            r["h2_to_h3_improves"] and not r["h3_root_is_also_h2_optimal"] for r in rows
        ),
        "original_status_unchanged": result["status"],
        "model_calls": 0,
        "paid_calls_authorized": False,
        "new_gate_authority": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.run_dir / "RESULT.json"
    initial_path = args.run_dir / "INITIAL_ONLY.json"
    result = json.loads(result_path.read_text())
    compiler_path = "scripts/number_game_generator_aware_bed.py"
    if (
        digest(compiler_path) != result["bindings"]["compiler"]
        or digest("scripts/number_game_initial_horizon_audit.py")
        != result["bindings"]["runner"]
    ):
        raise ValueError("diagnostic implementation changed")
    report = summarize(
        result, json.loads(initial_path.read_text()), load_compiler(compiler_path)
    )
    report["bindings"] = {
        "result": digest(result_path),
        "initial_projection": digest(initial_path),
        "summary_script": digest(__file__),
    }
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(
        report["h3_improvements_from_h2_tied_root"],
        report["h3_improvements_requiring_h2_objective_sacrifice"],
    )


if __name__ == "__main__":
    main()
