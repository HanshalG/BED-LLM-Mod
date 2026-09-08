"""Deterministic source-family split before SciLaws state/outcome inspection."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

PUBLIC_SHA = "46a3ffcfdcf7481eec8343e4c78437d128dbd7176caa267f1fb0bb913d041c95"
SALT = "scilaws-bed-source-development-v1"


def family(task_id):
    if type(task_id) is not str or "__" not in task_id:
        raise ValueError("source task ID must have a target suffix")
    return task_id.rsplit("__", 1)[0]


def rank(value):
    return hashlib.sha256(f"{SALT}:{value}".encode()).hexdigest()


def split(candidates):
    if len({r["task_id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate task IDs")
    families = defaultdict(list)
    for row in candidates:
        families[family(row["task_id"])].append(row)
    if len(families) < 8:
        raise ValueError("at least eight source families required")
    domains = defaultdict(list)
    for key, rows in families.items():
        disciplines = {r["discipline"] for r in rows}
        if len(disciplines) != 1:
            raise ValueError("source family spans multiple discipline labels")
        domains[next(iter(disciplines))].append(key)
    if len(domains) != 6:
        raise ValueError("all six public disciplines required")
    chosen = {min(keys, key=rank) for keys in domains.values()}
    chosen.update(sorted(set(families) - chosen, key=rank)[:2])
    development, guarded, holdout = [], [], []
    for key in sorted(families):
        rows = sorted(families[key], key=lambda r: rank(r["task_id"]))
        if key in chosen:
            development.append(rows[0])
            guarded.extend(rows[1:])
        else:
            holdout.extend(rows)
    return {
        "development": sorted(development, key=lambda r: r["task_id"]),
        "guarded_siblings": sorted(guarded, key=lambda r: r["task_id"]),
        "holdout_candidates": sorted(holdout, key=lambda r: r["task_id"]),
    }


def build(raw):
    if hashlib.sha256(raw).hexdigest() != PUBLIC_SHA:
        raise ValueError("public metadata binding mismatch")
    public = json.loads(raw)
    candidates = [r for r in public["tasks"] if r["feasibility_candidate"]]
    if len(candidates) != 29:
        raise ValueError("exact 29-task feasibility universe required")
    assignments = split(candidates)
    projected = {
        key: [
            dict(
                task_id=r["task_id"],
                source_family=family(r["task_id"]),
                discipline=r["discipline"],
                metadata_sha256=r["metadata_sha256"],
                metadata_path=r["metadata_path"],
                license_declared=r["license"],
            )
            for r in rows
        ]
        for key, rows in assignments.items()
    }
    return dict(
        schema_version=1,
        selection_salt=SALT,
        public_metadata_sha256=PUBLIC_SHA,
        dataset_revision=public["dataset_revision"],
        **projected,
        paid_authorization=False,
        policy_endpoint_authorization=False,
        stage="development_source_inspection_only",
        source_family_limitation="Task-prefix grouping prevents shared-target siblings; it does not prove independence of underlying datasets or mechanisms.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build(args.public.read_bytes())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
