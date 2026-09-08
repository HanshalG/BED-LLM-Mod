"""Exact retrospective direction/size decomposition, never a weight-fitting gate."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import ijson

from scripts.number_game_first_refresh_diagnostic import prediction, selected_records
from scripts.number_game_initial_horizon_audit import digest, load_compiler


PARENT = Path("results/nonmyopic/NUMBER_GAME_FIRST_REFRESH_DIAGNOSTIC_20260908.json")
CONFIG = Path(
    "results/nonmyopic/NUMBER_GAME_FIRST_REFRESH_DIAGNOSTIC_PROTOCOL_20260908.json"
)
PROTOCOL = Path("results/nonmyopic/NUMBER_GAME_REFRESH_DIRECTION_PROTOCOL_20260908.md")
PARENT_SHA = "f0704ed90bdceb7180239128b66567c027f15138b4aa499f6dd6d06ba99d7386"
EXTRACTOR_SHA = "3b1e5c2b945e2e4cf89ca4986defd8048df6c950a8d582f69a3654c96f0b6e24"


def terms(before, after, truth):
    if not len(before) == len(after) == len(truth) or not before:
        raise ValueError("prediction/truth length mismatch")
    delta = [q - p for p, q in zip(before, after)]
    n = len(before)
    linear = (
        2 * sum(((p - y) * d for p, y, d in zip(before, truth, delta)), Fraction(0)) / n
    )
    quadratic = sum((d * d for d in delta), Fraction(0)) / n
    return linear, quadratic


def decompose(tree, parent, compiler):
    if tree["tree_seed"] != parent["tree_seed"]:
        raise ValueError("tree identity mismatch")

    def extensions(items):
        result = []
        for item in items:
            ext = tuple(compiler(item["expression"]))
            if (
                len(ext) != 101
                or hashlib.sha256(bytes(ext)).hexdigest() != item["extension_sha256"]
                or sum(ext) != item["positive_count"]
            ):
                raise ValueError("extension binding mismatch")
            result.append(ext)
        return result

    initial = set(extensions(tree["initial"]))
    retained = {k: set(extensions(v)) for k, v in tree["first_branches"].items()}
    targets = extensions(tree["targets"])
    if tree["roots"] != [r["root"] for r in parent["roots"]]:
        raise ValueError("root coverage mismatch")
    rows = []
    for q, saved in zip(tree["roots"], parent["roots"]):
        preds = {}
        for y in (False, True):
            old = {e for e in initial if e[q] == y}
            new = retained[f"{q}:{int(y)}"]
            if not old or not new or not old <= new:
                raise ValueError("requires nonempty retained supports")
            preds[y] = prediction(old), prediction(new)
        values = [terms(*preds[t[q]], t) for t in targets]
        a = sum((v[0] for v in values), Fraction(0)) / len(values)
        b = sum((v[1] for v in values), Fraction(0)) / len(values)
        saved_delta = Fraction(saved["exact_losses"]["retained_branch"]) - Fraction(
            saved["exact_losses"]["initial_filtered"]
        )
        if a + b != saved_delta:
            raise ArithmeticError("decomposition does not replay parent loss change")
        rows.append(
            {
                "root": q,
                "linear": str(a),
                "quadratic": str(b),
                "parent_delta": str(saved_delta),
            }
        )
    a, b = (
        sum((Fraction(r[k]) for r in rows), Fraction(0)) / len(rows)
        for k in ("linear", "quadratic")
    )
    return {
        "tree_seed": tree["tree_seed"],
        "roots": rows,
        "linear": str(a),
        "quadratic": str(b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if (
        digest(PARENT) != PARENT_SHA
        or digest("scripts/number_game_first_refresh_diagnostic.py") != EXTRACTOR_SHA
    ):
        raise ValueError("parent binding mismatch")
    for name in ("bank", "compiler", "extractor"):
        if digest(config[name]) != config[name + "_sha256"]:
            raise ValueError("source binding mismatch")
    parent = json.loads(PARENT.read_text())
    if parent["status"] != "descriptive_complete":
        raise ValueError("parent incomplete")
    if args.output.exists():
        raise FileExistsError(args.output)
    compiler = load_compiler(config["compiler"])
    report = {
        "status": "incomplete",
        "model_calls": 0,
        "cost_usd": 0,
        "gate_authority": False,
        "parent_sha256": PARENT_SHA,
        "runner_sha256": digest(__file__),
        "protocol_sha256": digest(PROTOCOL),
        "rows": [],
    }
    fields = {"tree_seed", "initial", "roots", "first_branches", "targets"}
    try:
        with Path(config["bank"]).open("rb") as handle:
            for tree, prior in zip(
                selected_records(ijson.parse(handle), fields),
                parent["rows"],
                strict=True,
            ):
                report["rows"].append(decompose(tree, prior, compiler))
        if [r["tree_seed"] for r in report["rows"]] != config["seeds"]:
            raise ValueError("coverage mismatch")
        a, b = (
            sum((Fraction(r[k]) for r in report["rows"]), Fraction(0))
            / len(report["rows"])
            for k in ("linear", "quadratic")
        )
        report.update(
            status="descriptive_complete",
            linear=str(a),
            quadratic=str(b),
            means={"linear": float(a), "quadratic": float(b), "total": float(a + b)},
            trees_wrong_direction=sum(
                Fraction(r["linear"]) >= 0 for r in report["rows"]
            ),
            trees_useful_direction_overshot=sum(
                Fraction(r["linear"])
                < 0
                < Fraction(r["linear"]) + Fraction(r["quadratic"])
                for r in report["rows"]
            ),
        )
    except (ValueError, ArithmeticError, KeyError) as exc:
        report.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if report["status"] != "descriptive_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
