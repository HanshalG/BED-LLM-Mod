"""Freeze outcome-blind geometry for the eight SciLaws development tasks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

METADATA_SHA = "46a3ffcfdcf7481eec8343e4c78437d128dbd7176caa267f1fb0bb913d041c95"
CONTRACT_SHA = "f6f5a18208857c62b3fe5a84532aecbce330ef17a0bba521c44da08f8a462226"


def radical_inverse(index, base):
    value, weight = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * weight
        weight /= base
    return value


def geometry(metadata, contract):
    if metadata["task_id"] != contract["task_id"]:
        raise ValueError("task identity mismatch")
    if contract["status"] != "source_contract_valid":
        raise ValueError("invalid source contract")
    names = contract["used_inputs"]
    public = {row["name"]: row for row in metadata["inputs"]}
    if not 1 <= len(names) <= 3 or len(set(names)) != len(names):
        raise ValueError("one to three unique inputs required")
    axes = []
    for name in names:
        intervals = [public[name]["public_train_range"], contract["bounds"][name]]
        if any(
            len(v) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
            or v[0] >= v[1]
            for v in intervals
        ):
            raise ValueError("invalid bounds")
        lo, hi = max(v[0] for v in intervals), min(v[1] for v in intervals)
        if lo >= hi:
            raise ValueError("empty public/runtime intersection")
        axes.append(
            dict(
                name=name,
                bounds=[lo, hi],
                transform=(
                    "log"
                    if lo > 0 and math.log(hi) - math.log(lo) >= math.log(100)
                    else "linear"
                ),
            )
        )

    def point(unit):
        result = {}
        for axis, u in zip(axes, unit, strict=True):
            lo, hi = axis["bounds"]
            result[axis["name"]] = (
                math.exp((1 - u) * math.log(lo) + u * math.log(hi))
                if axis["transform"] == "log"
                else (1 - u) * lo + u * hi
            )
        return result

    def halton(index):
        return [radical_inverse(index, base) for base in (2, 3, 5)[: len(axes)]]

    initial = [[0.5] * len(axes)]
    for dim in range(len(axes)):
        for coordinate in (0.25, 0.75):
            unit = [0.5] * len(axes)
            unit[dim] = coordinate
            initial.append(unit)
    total_rows = 2 * len(initial) + 4
    if contract["fetch_budget_rows"] < total_rows:
        raise ValueError("insufficient source budget")
    return dict(
        task_id=metadata["task_id"],
        axes=axes,
        initial_points=[point(u) for u in initial],
        initial_replicates=2,
        action_points=[point(halton(i)) for i in range(1, 9)],
        target_points=[point(halton(i)) for i in range(129, 193)],
        target_weights=[1 / 64] * 64,
        adaptive_rounds=4,
        adaptive_replicates=1,
        total_rows_per_arm=total_rows,
    )


def build(metadata_path, contract_path):
    def bound_read(path, expected):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("source binding mismatch")
        return json.loads(raw)

    metadata = bound_read(metadata_path, METADATA_SHA)
    contract = bound_read(contract_path, CONTRACT_SHA)
    rows = {t["task_id"]: t for t in metadata["tasks"]}
    return dict(
        schema_version=1,
        metadata_sha256=METADATA_SHA,
        contract_sha256=CONTRACT_SHA,
        status="measurement_geometry_frozen_not_execution_authorized",
        tasks=[geometry(rows[t["task_id"]], t) for t in contract["tasks"]],
        model_calls=0,
        measurements_generated=0,
        paid_authorization=False,
        policy_endpoint_authorization=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = build(args.metadata, args.contract)
    with Path(args.output).open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
