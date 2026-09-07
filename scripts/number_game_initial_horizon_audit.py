"""Exact full-budget horizons on saved initial Number Game supports only."""

import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from time import monotonic


PROTOCOL = Path("results/nonmyopic/NUMBER_GAME_INITIAL_HORIZON_PROTOCOL_20260908.json")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def initial_records(events):
    """Discard all non-initial events without building their nested objects."""
    records, current, item = [], None, None
    for prefix, event, value in events:
        if prefix == "trees.item" and event == "start_map":
            current = {"initial": []}
        elif prefix == "trees.item.tree_seed" and event == "number":
            current["tree_seed"] = int(value)
        elif prefix == "trees.item.initial.item" and event == "start_map":
            item = {}
        elif prefix.startswith("trees.item.initial.item.") and event in {
            "string",
            "number",
        }:
            key = prefix.rsplit(".", 1)[-1]
            if key in {"name", "expression", "extension_sha256", "positive_count"}:
                item[key] = value
        elif prefix == "trees.item.initial.item" and event == "end_map":
            current["initial"].append(item)
        elif prefix == "trees.item" and event == "end_map":
            records.append(current)
    return records


def load_compiler(path):
    allowed_functions = {
        "divisible",
        "is_square",
        "is_power_of_two",
        "is_prime",
        "digit_sum",
        "ends_with",
        "compile_expression",
    }
    allowed_constants = {"DOMAIN", "_HELPERS", "_ALLOWED_NODE_TYPES"}
    body = []
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in allowed_functions:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in allowed_constants for t in node.targets
        ):
            body.append(node)
    namespace = {"ast": ast, "math": math}
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]), "<banked-rule-compiler>", "exec"
        ),
        namespace,
    )
    return namespace["compile_expression"]


class ExactMembershipHorizon:
    def __init__(self, extensions, *, max_seconds=30, max_states=250000):
        if (
            not extensions
            or not extensions[0]
            or len(set(map(tuple, extensions))) != len(extensions)
        ):
            raise ValueError("extensions must be nonempty and unique")
        self.size = len(extensions[0])
        if any(
            len(e) != self.size or any(type(x) is not bool for x in e)
            for e in extensions
        ):
            raise ValueError("extensions must be equally sized boolean vectors")
        self.full = (1 << len(extensions)) - 1
        self.columns = [
            sum(int(e[q]) << i for i, e in enumerate(extensions))
            for q in range(self.size)
        ]
        self.risks, self.plans, self.greedy_values = {}, {}, {}
        self.started = monotonic()
        self.max_seconds, self.max_states = max_seconds, max_states

    def check(self):
        if monotonic() - self.started > self.max_seconds:
            raise TimeoutError("exact horizon time cap")
        if (
            len(self.risks) + len(self.plans) + len(self.greedy_values)
            >= self.max_states
        ):
            raise RuntimeError("exact horizon state cap")

    def risk(self, mask):
        if mask not in self.risks:
            self.check()
            n = mask.bit_count()
            if n == 0:
                raise ValueError("empty posterior support")
            counts = [(mask & col).bit_count() for col in self.columns]
            self.risks[mask] = Fraction(
                sum(k * (n - k) for k in counts), n * n * self.size
            )
        return self.risks[mask]

    def partitions(self, mask):
        seen = set()
        for query, col in enumerate(self.columns):
            left = mask & col
            right = mask ^ left
            if not left or not right:
                continue
            pair = tuple(sorted((left, right)))
            if pair not in seen:
                seen.add(pair)
                yield query, left, right

    def action_value(self, mask, horizon, query):
        left = mask & self.columns[query]
        right = mask ^ left
        return sum(
            (
                Fraction(child.bit_count(), mask.bit_count())
                * self.plan(child, horizon - 1)[0]
                for child in (left, right)
                if child
            ),
            Fraction(0),
        )

    def plan(self, mask, horizon):
        key = mask, horizon
        if key not in self.plans:
            self.check()
            best = self.risk(mask), None
            if horizon and mask.bit_count() > 1:
                candidates = [
                    (self.action_value(mask, horizon, query), query)
                    for query, _, _ in self.partitions(mask)
                ]
                if candidates:
                    best = min(candidates)
            self.plans[key] = best
        return self.plans[key]

    def greedy(self, mask, remaining):
        key = mask, remaining
        if key not in self.greedy_values:
            self.check()
            if remaining == 0 or mask.bit_count() == 1:
                value = self.risk(mask)
            else:
                query = self.plan(mask, 1)[1]
                left = mask & self.columns[query]
                right = mask ^ left
                value = sum(
                    (
                        Fraction(child.bit_count(), mask.bit_count())
                        * self.greedy(child, remaining - 1)
                        for child in (left, right)
                    ),
                    Fraction(0),
                )
            self.greedy_values[key] = value
        return self.greedy_values[key]

    def compare(self):
        h1 = self.greedy(self.full, 3)
        q2 = self.plan(self.full, 2)[1]
        h3, q3 = self.plan(self.full, 3)
        h2 = (
            self.action_value(self.full, 3, q2)
            if q2 is not None
            else self.risk(self.full)
        )
        values = (h1, h2, h3)
        return {
            "full_budget_values": [float(v) for v in values],
            "exact_full_budget_values": [str(v) for v in values],
            "first_queries": [self.plan(self.full, 1)[1], q2, q3],
            "initial_risk": float(self.risk(self.full)),
            "states": len(self.risks) + len(self.plans) + len(self.greedy_values),
            "elapsed_seconds": monotonic() - self.started,
        }


def main():
    import ijson

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    for key in ["bank", "compiler"]:
        if digest(config[key]) != config[key + "_sha256"]:
            raise ValueError(f"{key} provenance changed")
    with Path(config["bank"]).open("rb") as file:
        records = initial_records(ijson.parse(file))
    if [r["tree_seed"] for r in records] != config["seeds"]:
        raise ValueError("initial panel identity changed")
    compile_rule = load_compiler(config["compiler"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "INITIAL_ONLY.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    rows, started = [], monotonic()
    try:
        for record in records:
            support = []
            for rule in record["initial"]:
                extension = compile_rule(rule["expression"])
                if (
                    len(extension) != 101
                    or hashlib.sha256(bytes(extension)).hexdigest()
                    != rule["extension_sha256"]
                    or sum(extension) != rule["positive_count"]
                ):
                    raise ValueError("initial extension does not replay")
                if extension not in support:
                    support.append(extension)
            remaining = config["max_panel_seconds"] - (monotonic() - started)
            solver = ExactMembershipHorizon(
                support,
                max_seconds=min(remaining, config["max_seconds_per_tree"]),
                max_states=config["max_states_per_tree"],
            )
            row = {
                "tree_seed": record["tree_seed"],
                "unique_initial_rules": len(support),
                **solver.compare(),
            }
            rows.append(row)
            (args.output_dir / f"tree_{record['tree_seed']}.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
        aggregate = [
            sum(Fraction(row["exact_full_budget_values"][i]) for row in rows)
            / len(rows)
            for i in range(3)
        ]
        gains = [
            float(1 - aggregate[i + 1] / aggregate[i]) if aggregate[i] else 0.0
            for i in range(2)
        ]
        result = {
            "status": "initial_opportunity_pass"
            if min(gains) >= config["minimum_adjacent_gain"]
            else "initial_opportunity_null",
            "rows": rows,
            "aggregate_full_budget_values": [float(x) for x in aggregate],
            "exact_aggregate_values": list(map(str, aggregate)),
            "adjacent_fractional_gains": gains,
        }
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "completed_rows": rows,
        }
    result.update(
        model_calls=0,
        cost_usd=0,
        historical_endpoints_used=False,
        dynamic_policy_replayed=False,
        paid_calls_authorized=False,
        elapsed_seconds=monotonic() - started,
        bindings={
            "protocol": digest(PROTOCOL),
            "compiler": config["compiler_sha256"],
            "bank": config["bank_sha256"],
            "runner": digest(__file__),
        },
    )
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(result["status"])


if __name__ == "__main__":
    main()
