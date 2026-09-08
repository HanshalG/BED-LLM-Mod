"""Audit public SciLaws metadata without task measurements or grader state.

Selection is a feasibility screen only, not scientific opportunity. Published
test ranges, counts, target ranges, priors and formula fields are not retained.
"""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

import yaml

COMMIT = "9239f66b921cb89c7a9d14061f782fdce49dcfb5"
DATA_REVISION = "15b93258fabc8f1785c5a5ffd4d3e51fc15dd860"
INDEX_SHA = "57ca04e6bc6d45b58bae0e808e3966ed0caa7472f8de495c446df98aa8e199af"
BASE = f"https://huggingface.co/datasets/RealSR/SciLaws-Bench/resolve/{DATA_REVISION}/"


def metadata_path(row):
    task = row["task_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ValueError("invalid task ID")
    groups = {"single": "typeI", "multi": "typeII"}
    return f"tasks/{groups[row['group_structure']]}/{task}/metadata.yaml"


def read_metadata(path):
    if not re.fullmatch(r"tasks/type(?:I|II)/[A-Za-z0-9_-]+/metadata\.yaml", path):
        raise ValueError("only public metadata paths may be fetched")
    with urlopen(BASE + path, timeout=30) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("metadata exceeds size cap")
    return raw


def numeric_range(value):
    return (
        type(value) is list
        and len(value) == 2
        and all(type(v) in (float, int) and math.isfinite(v) for v in value)
        and value[0] < value[1]
    )


def project(row, raw):
    meta = yaml.safe_load(raw)
    if type(meta) is not dict or meta.get("task_id") != row["task_id"]:
        raise ValueError("task identity mismatch")
    inputs = meta.get("inputs")
    if type(inputs) is not list or not inputs:
        raise ValueError("nonempty public inputs required")
    names = [v.get("name") for v in inputs]
    if any(type(n) is not str or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("input names must be unique")
    projected_inputs = []
    reasons = []
    for value in inputs:
        ranges = value.get("range")
        # Some grouped metadata uses an aggregate list, not a train/test map.
        # Do not reinterpret that as verified single-group measurement support.
        bounds = ranges.get("train") if type(ranges) is dict else None
        valid = numeric_range(bounds)
        if not valid:
            reasons.append("non_numeric_or_degenerate_public_input_range")
        projected_inputs.append(
            {
                "name": value["name"],
                "unit": value.get("unit"),
                "description": value.get("description"),
                "public_train_range": bounds if valid else None,
            }
        )
    if row["group_structure"] != "single":
        reasons.append("group_specific_support_not_publicly_established")
    if len(inputs) > 3:
        reasons.append("more_than_three_public_inputs")
    if len(inputs) != int(row["n_inputs"]):
        reasons.append("index_input_count_mismatch")
    target = meta.get("target", {})
    return {
        "task_id": row["task_id"],
        "discipline": row["discipline"],
        "group_structure": row["group_structure"],
        "license": meta.get("license"),
        "index_license": row["license"],
        "context": meta.get("context"),
        "target": {k: target.get(k) for k in ("name", "description", "unit")},
        "inputs": projected_inputs,
        "metadata_path": metadata_path(row),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "feasibility_candidate": not reasons,
        "exclusion_reasons": sorted(set(reasons)),
        "measurement_support_verified": False,
        "license_review_complete": False,
    }


def audit(index, fetch=read_metadata):
    if hashlib.sha256(index).hexdigest() != INDEX_SHA:
        raise ValueError("index binding mismatch")
    rows = list(csv.DictReader(io.StringIO(index.decode())))
    if len(rows) != 118 or len({r["task_id"] for r in rows}) != 118:
        raise ValueError("complete unique 118-task index required")
    results = []
    for row in rows:
        results.append(project(row, fetch(metadata_path(row))))
        print(f"metadata {len(results)}/118", flush=True)
    return {
        "schema_version": 1,
        "source_commit": COMMIT,
        "dataset_revision": DATA_REVISION,
        "index_sha256": INDEX_SHA,
        "tasks": results,
        "feasibility_candidate_count": sum(r["feasibility_candidate"] for r in results),
        "exclusion_counts": dict(
            Counter(x for r in results for x in r["exclusion_reasons"])
        ),
        "task_outcomes_loaded": 0,
        "simulator_states_loaded": 0,
        "model_calls": 0,
        "paid_cost_usd": 0,
        "opportunity_pass": False,
        "paid_authorization": False,
        "limitations": [
            "Public train ranges are not yet verified simulator support.",
            "No task has passed semantic, predictive, or horizon opportunity gates.",
            "Feasibility inclusion is not permission to use task data; licensing remains per-task.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = subprocess.check_output(
        ["git", "-C", str(args.repo), "show", f"{COMMIT}:dataset/task_index.csv"]
    )
    result = audit(index)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
