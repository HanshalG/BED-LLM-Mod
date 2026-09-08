"""Descriptive support-refresh decomposition on already banked first-step data."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import monotonic

import ijson
from ijson.common import ObjectBuilder

from scripts.number_game_initial_horizon_audit import digest, load_compiler


PROTOCOL = Path(
    "results/nonmyopic/NUMBER_GAME_FIRST_REFRESH_DIAGNOSTIC_PROTOCOL_20260908.json"
)


def selected_records(events, allowed):
    current, builder, selected_prefix, field = None, None, None, None
    for prefix, event, value in events:
        if builder is not None:
            builder.event(event, value)
            if prefix == selected_prefix and event in ("end_array", "end_map"):
                current[field] = builder.value
                builder = None
            continue
        if prefix == "trees.item" and event == "start_map":
            current = {}
        elif prefix == "trees.item" and event == "end_map":
            if set(current) != set(allowed):
                raise ValueError("missing selected fields")
            yield current
        elif prefix.startswith("trees.item."):
            key = prefix[len("trees.item.") :]
            if key not in allowed:
                continue
            if event in ("start_map", "start_array"):
                builder, selected_prefix, field = ObjectBuilder(), prefix, key
                builder.event(event, value)
            elif event in ("string", "number", "boolean", "null"):
                current[key] = value


def prediction(support):
    if not support:
        return (Fraction(1, 2),) * 101
    return tuple(Fraction(sum(e[q] for e in support), len(support)) for q in range(101))


def brier(pred, truth):
    return sum(((p - y) ** 2 for p, y in zip(pred, truth)), Fraction(0)) / 101


def evaluate(tree, compiler):
    def extensions(items):
        out = []
        for item in items:
            ext = tuple(compiler(item["expression"]))
            if len(ext) != 101 or any(type(y) is not bool for y in ext):
                raise ValueError("invalid rule extension")
            if (
                hashlib.sha256(bytes(ext)).hexdigest() != item["extension_sha256"]
                or sum(ext) != item["positive_count"]
            ):
                raise ValueError("banked rule hash/count mismatch")
            out.append(ext)
        return out

    initial = set(extensions(tree["initial"]))
    roots = tree["roots"]
    if (
        not roots
        or len(set(roots)) != len(roots)
        or any(type(q) is not int or not 0 <= q <= 100 for q in roots)
    ):
        raise ValueError("invalid root coverage")
    expected = {f"{q}:{int(y)}" for q in roots for y in (False, True)}
    generated, retained = {}, {}
    for field, output in (
        ("generated_first_branches", generated),
        ("first_branches", retained),
    ):
        if set(tree[field]) != expected:
            raise ValueError("branch coverage mismatch")
        for key, items in tree[field].items():
            q, y = map(int, key.split(":"))
            output[key] = set(extensions(items))
            if any(e[q] != bool(y) for e in output[key]):
                raise ValueError("branch contradicts supplied observation")
    for key in expected:
        q, y = map(int, key.split(":"))
        if retained[key] != generated[key] | {e for e in initial if e[q] == bool(y)}:
            raise ValueError("bank does not implement retained refresh")
    # This pool contains only prospective imagined-query proposals, no target data.
    shared = initial.union(*generated.values())
    targets = extensions(tree["targets"])
    if not targets:
        raise ValueError("no saved targets")
    rows = []
    for q in roots:
        supports = {
            y: {
                "initial_filtered": {e for e in initial if e[q] == y},
                "retained_branch": retained[f"{q}:{int(y)}"],
                "all_first_proposals_shared_then_filtered": {
                    e for e in shared if e[q] == y
                },
            }
            for y in (False, True)
        }
        predictions = {
            y: {a: prediction(s) for a, s in arms.items()}
            for y, arms in supports.items()
        }
        losses = {
            a: sum((brier(predictions[t[q]][a], t) for t in targets), Fraction(0))
            / len(targets)
            for a in supports[False]
        }
        rows.append(
            {
                "root": q,
                "exact_losses": {a: str(v) for a, v in losses.items()},
                "empty_target_cases": {
                    a: sum(not supports[t[q]][a] for t in targets) for a in losses
                },
                "support_sizes_by_answer": {
                    str(y): {a: len(s) for a, s in arms.items()}
                    for y, arms in supports.items()
                },
            }
        )
    means = {
        a: sum((Fraction(r["exact_losses"][a]) for r in rows), Fraction(0)) / len(rows)
        for a in rows[0]["exact_losses"]
    }
    return {
        "tree_seed": tree["tree_seed"],
        "initial_count": len(initial),
        "shared_count": len(shared),
        "target_entries": len(targets),
        "roots": rows,
        "exact_means": {a: str(v) for a, v in means.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())
    for name in ("bank", "compiler", "extractor"):
        if digest(config[name]) != config[name + "_sha256"]:
            raise ValueError(f"binding changed: {name}")
    if args.output.exists():
        raise FileExistsError(args.output)
    compiler = load_compiler(config["compiler"])
    report = {
        "status": "incomplete",
        "rows": [],
        "model_calls": 0,
        "cost_usd": 0,
        "gate_authority": False,
        "paid_calls_authorized": False,
        "protocol_sha256": digest(PROTOCOL),
        "runner_sha256": digest(__file__),
    }
    start = monotonic()
    try:
        with Path(config["bank"]).open("rb") as handle:
            for tree in selected_records(ijson.parse(handle), config["allowed_fields"]):
                if monotonic() - start > config["max_seconds"]:
                    raise TimeoutError("diagnostic time cap")
                report["rows"].append(evaluate(tree, compiler))
        if [r["tree_seed"] for r in report["rows"]] != config["seeds"]:
            raise ValueError("tree coverage mismatch")
        means = {
            a: sum((Fraction(r["exact_means"][a]) for r in report["rows"]), Fraction(0))
            / len(report["rows"])
            for a in config["arms"]
        }
        report.update(
            status="descriptive_complete",
            exact_means={a: str(v) for a, v in means.items()},
            means={a: float(v) for a, v in means.items()},
        )
    except (ValueError, RuntimeError, TimeoutError, KeyError) as exc:
        report.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = monotonic() - start
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if report["status"] != "descriptive_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
