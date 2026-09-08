"""Retrospective validation-only fitting, sealed before target-field evaluation."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import ijson

from scripts.number_game_first_refresh_diagnostic import (
    prediction,
    selected_records,
    brier,
)
from scripts.number_game_initial_horizon_audit import digest, load_compiler
from scripts.number_game_refresh_direction_audit import (
    terms,
    PARENT,
    PARENT_SHA,
    CONFIG,
    EXTRACTOR_SHA,
)


PROTOCOL = Path("results/nonmyopic/NUMBER_GAME_VALIDATION_WEIGHT_PROTOCOL_20260908.md")


def fit_weight(a, b):
    if b < 0:
        raise ValueError("negative curvature")
    return max(Fraction(0), min(Fraction(1), -a / (2 * b))) if b else Fraction(0)


def decode(items, compiler):
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


def forecasts(tree, compiler):
    initial = set(decode(tree["initial"], compiler))
    result = {}
    for q in tree["roots"]:
        for y in (False, True):
            old = {e for e in initial if e[q] == y}
            new = set(decode(tree["first_branches"][f"{q}:{int(y)}"], compiler))
            if not old or not new or not old <= new or any(e[q] != y for e in new):
                raise ValueError("nonempty retained-support contract failed")
            result[q, y] = prediction(old), prediction(new)
    return result


def calibrate(tree, compiler, index):
    if (
        tree["validation_seeds"] != list(range(28500 + 8 * index, 28508 + 8 * index))
        or tree["target_seed"] in tree["validation_seeds"]
    ):
        raise ValueError("calibration/target seed identity mismatch")
    if len(tree["validation_supports"]) != 8:
        raise ValueError("all eight calibration draws required")
    preds = forecasts(tree, compiler)
    components = []
    for support in tree["validation_supports"]:
        truths = decode(support, compiler)
        if not truths:
            raise ValueError("empty calibration draw")
        per_case = [terms(*preds[q, t[q]], t) for q in tree["roots"] for t in truths]
        components.append(
            tuple(
                sum((v[k] for v in per_case), Fraction(0)) / len(per_case)
                for k in (0, 1)
            )
        )
    a, b = (sum((v[k] for v in components), Fraction(0)) / 8 for k in (0, 1))
    return {
        "tree_seed": tree["tree_seed"],
        "weight": str(fit_weight(a, b)),
        "validation_linear": str(a),
        "validation_quadratic": str(b),
    }


def score_targets(tree, weight, parent, compiler):
    if (
        tree["tree_seed"] != weight["tree_seed"]
        or tree["tree_seed"] != parent["tree_seed"]
    ):
        raise ValueError("tree identity mismatch")
    if tree["roots"] != [r["root"] for r in parent["roots"]]:
        raise ValueError("root coverage mismatch")
    preds = forecasts(tree, compiler)
    targets = decode(tree["targets"], compiler)
    if not targets:
        raise ValueError("empty target draw")
    w = Fraction(weight["weight"])
    rows = []
    for q, saved in zip(tree["roots"], parent["roots"]):
        loss = {
            a: Fraction(0)
            for a in ("initial_filtered", "retained_branch", "validation_weighted")
        }
        for t in targets:
            p0, p1 = preds[q, t[q]]
            pm = tuple(p + w * (r - p) for p, r in zip(p0, p1))
            for name, p in zip(loss, (p0, p1, pm)):
                loss[name] += brier(p, t) / len(targets)
        for name in ("initial_filtered", "retained_branch"):
            if loss[name] != Fraction(saved["exact_losses"][name]):
                raise ArithmeticError("target endpoint differs from parent")
        rows.append({"root": q, "losses": {a: str(v) for a, v in loss.items()}})
    means = {
        a: sum((Fraction(r["losses"][a]) for r in rows), Fraction(0)) / len(rows)
        for a in rows[0]["losses"]
    }
    return {
        "tree_seed": tree["tree_seed"],
        "weight": str(w),
        "roots": rows,
        "exact_means": {a: str(v) for a, v in means.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if (
        digest(PARENT) != PARENT_SHA
        or digest("scripts/number_game_first_refresh_diagnostic.py") != EXTRACTOR_SHA
        or digest("scripts/number_game_refresh_direction_audit.py")
        != "d4526d1d1e732f3d3569c26084e01fe7113e3fefed9560cd592d0e5319107004"
    ):
        raise ValueError("analysis binding mismatch")
    for name in ("bank", "compiler", "extractor"):
        if digest(config[name]) != config[name + "_sha256"]:
            raise ValueError("source binding mismatch")
    compiler = load_compiler(config["compiler"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
    base = {"tree_seed", "initial", "roots", "first_branches"}
    try:
        weights = []
        with Path(config["bank"]).open("rb") as handle:
            for i, tree in enumerate(
                selected_records(
                    ijson.parse(handle),
                    base | {"target_seed", "validation_seeds", "validation_supports"},
                )
            ):
                weights.append(calibrate(tree, compiler, i))
        if [w["tree_seed"] for w in weights] != config["seeds"]:
            raise ValueError("calibration coverage mismatch")
        weight_path = args.output_dir / "WEIGHTS.json"
        with weight_path.open("x") as handle:
            json.dump(weights, handle, indent=2)
            handle.write("\n")
        report["weights_sha256"] = digest(weight_path)
        parent = json.loads(PARENT.read_text())
        with Path(config["bank"]).open("rb") as handle:
            for tree, w, prior in zip(
                selected_records(ijson.parse(handle), base | {"targets"}),
                weights,
                parent["rows"],
                strict=True,
            ):
                report["rows"].append(score_targets(tree, w, prior, compiler))
        means = {
            a: sum((Fraction(r["exact_means"][a]) for r in report["rows"]), Fraction(0))
            / len(weights)
            for a in report["rows"][0]["exact_means"]
        }
        report.update(
            status="descriptive_complete",
            exact_means={a: str(v) for a, v in means.items()},
            means={a: float(v) for a, v in means.items()},
        )
    except (ValueError, ArithmeticError, KeyError) as exc:
        report.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
    with (args.output_dir / "RESULT.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if report["status"] != "descriptive_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
