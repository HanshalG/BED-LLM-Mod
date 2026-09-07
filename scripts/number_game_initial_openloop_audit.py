"""Descriptive fixed-sequence and receding open-loop controls on initial beliefs."""

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import monotonic

from scripts.number_game_initial_horizon_audit import (
    ExactMembershipHorizon,
    digest,
    load_compiler,
)

PROTOCOL = Path("results/nonmyopic/NUMBER_GAME_INITIAL_OPENLOOP_PROTOCOL_20260908.json")


class OpenLoopMembership:
    def __init__(self, solver, max_sets=500000):
        self.solver = solver
        self.max_sets = max_sets
        self.sets = 0
        self.plans = {}
        self.values = {}

    def sequence_value(self, mask, queries):
        groups = [mask]
        for query in queries:
            next_groups = []
            for group in groups:
                positive = group & self.solver.columns[query]
                next_groups.extend(x for x in (positive, group ^ positive) if x)
            groups = next_groups
        return sum(
            (
                Fraction(g.bit_count(), mask.bit_count()) * self.solver.risk(g)
                for g in groups
            ),
            Fraction(0),
        )

    def plan(self, mask, horizon):
        key = mask, horizon
        if key not in self.plans:
            self.solver.check()
            queries = [q for q, _, _ in self.solver.partitions(mask)]
            best = None
            # Order is immaterial for a committed deterministic query set.
            # Extra noiseless observations cannot raise coherent Bayes risk.
            for sequence in combinations(queries, min(horizon, len(queries))):
                self.sets += 1
                if self.sets > self.max_sets:
                    raise RuntimeError("open-loop query-set cap")
                self.solver.check()
                candidate = self.sequence_value(mask, sequence), sequence
                if best is None or candidate < best:
                    best = candidate
            self.plans[key] = best
        return self.plans[key]

    def receding(self, mask, budget):
        key = mask, budget
        if key not in self.values:
            _, sequence = self.plan(mask, budget)
            if not budget or not sequence:
                value = self.solver.risk(mask)
            else:
                positive = mask & self.solver.columns[sequence[0]]
                value = sum(
                    (
                        Fraction(g.bit_count(), mask.bit_count())
                        * self.receding(g, budget - 1)
                        for g in (positive, mask ^ positive)
                        if g
                    ),
                    Fraction(0),
                )
            self.values[key] = value
        return self.values[key]


def evaluate_record(record, outcome, compile_rule, *, max_seconds, max_sets):
    if record["tree_seed"] != outcome["tree_seed"]:
        raise ValueError("panel identity mismatch")
    extensions = []
    for rule in record["initial"]:
        extension = compile_rule(rule["expression"])
        if (
            len(extension) != 101
            or hashlib.sha256(bytes(extension)).hexdigest() != rule["extension_sha256"]
            or sum(extension) != rule["positive_count"]
        ):
            raise ValueError("initial extension replay failed")
        if extension not in extensions:
            extensions.append(extension)
    solver = ExactMembershipHorizon(extensions, max_seconds=max_seconds)
    control = OpenLoopMembership(solver, max_sets=max_sets)
    committed, sequence = control.plan(solver.full, 3)
    receding = control.receding(solver.full, 3)
    h3 = Fraction(outcome["exact_full_budget_values"][2])
    if not h3 <= receding <= committed:
        raise ArithmeticError("tree/receding/committed dominance violated")
    return {
        "tree_seed": record["tree_seed"],
        "exact_values": {
            "h3": str(h3),
            "receding_openloop_h3": str(receding),
            "committed_three": str(committed),
        },
        "first_committed_sequence": sequence,
        "receding_first_query": sequence[0] if sequence else None,
        "h3_first_query": outcome["first_queries"][2],
        "exact_anticipated_adaptivity_gain": str(receding - h3),
        "exact_full_adaptivity_gap": str(committed - h3),
        "query_sets_evaluated": control.sets,
        "elapsed_seconds": monotonic() - solver.started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    for name in ("result", "initial", "compiler", "solver"):
        if digest(config[name]) != config[name + "_sha256"]:
            raise ValueError(f"{name} binding changed")
    result = json.loads(Path(config["result"]).read_text())
    initial = json.loads(Path(config["initial"]).read_text())
    if (
        result["status"] != "initial_opportunity_null"
        or [r["tree_seed"] for r in initial] != config["seeds"]
        or [r["tree_seed"] for r in result["rows"]] != config["seeds"]
    ):
        raise ValueError("closed parent panel mismatch")
    compile_rule = load_compiler(config["compiler"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, started = [], monotonic()
    report = {
        "status": "incomplete",
        "original_status_unchanged": result["status"],
        "new_gate_authority": False,
        "paid_calls_authorized": False,
        "model_calls": 0,
        "cost_usd": 0,
        "rows": rows,
        "protocol_sha256": digest(PROTOCOL),
        "runner_sha256": digest(__file__),
    }
    try:
        for record, outcome in zip(initial, result["rows"], strict=True):
            remaining = config["max_panel_seconds"] - (monotonic() - started)
            row = evaluate_record(
                record,
                outcome,
                compile_rule,
                max_seconds=min(remaining, config["max_seconds_per_tree"]),
                max_sets=config["max_sets_per_tree"],
            )
            rows.append(row)
            (args.output_dir / f"tree_{record['tree_seed']}.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
        means = {
            name: sum((Fraction(r["exact_values"][name]) for r in rows), Fraction(0))
            / len(rows)
            for name in rows[0]["exact_values"]
        }
        report.update(
            status="descriptive_control_complete",
            exact_means={k: str(v) for k, v in means.items()},
            means={k: float(v) for k, v in means.items()},
            anticipated_adaptivity_wins=sum(
                Fraction(r["exact_anticipated_adaptivity_gain"]) > 0 for r in rows
            ),
            committed_adaptivity_wins=sum(
                Fraction(r["exact_full_adaptivity_gap"]) > 0 for r in rows
            ),
        )
    except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as exc:
        report.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = monotonic() - started
    (args.output_dir / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if report["status"] != "descriptive_control_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
