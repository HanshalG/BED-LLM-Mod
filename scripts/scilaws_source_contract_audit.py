"""Exercise pinned SciLaws runtime closures with synthetic state only.

Never import the upstream module, deserialize joblib, download task data, or
execute a benchmark formula. This is source-contract evidence, not efficacy.
"""

import argparse
import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

COMMIT = "9239f66b921cb89c7a9d14061f782fdce49dcfb5"
SOURCE_SHA = "d59f6d3e9405335000662e1d7a9ca850c06bbb65980ee05758ed25f11246c825"


def source_bytes(repo):
    blob = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{COMMIT}:harness/sim_runtime.py"]
    )
    if hashlib.sha256(blob).hexdigest() != SOURCE_SHA:
        raise ValueError("runtime source binding mismatch")
    return blob


def fixture_bundle(source, kind, budget=4):
    names = {
        "safe_float",
        "group_id_key",
        "group_id_value",
        "TypeISimulatorBundle",
        "TypeIISimulatorBundle",
        "_build_typeI_simulator",
        "_build_typeII_simulator",
    }
    tree = ast.parse(source)
    body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in body} != names:
        raise ValueError("required runtime definitions missing")

    # Explicit artificial world: constant signal 10 with equiprobable +/-1 noise.
    # No files, real task states, released targets or hidden source are loaded.
    bootstrap = SimpleNamespace(sample=lambda x, rng: rng.choice([-1.0, 1.0], len(x)))
    support = {"x": {"min": 0.0, "max": 1.0}}
    state = {
        "used_inputs": ["x"],
        "all_input_cols": ["x"],
        "target": "y",
        "support": support,
        "fetch_budget_rows": budget,
        "residual_space": "linear",
        "noise_scale": 1.0,
        "law_constants": {},
        "bootstrap": bootstrap,
        "real_rows": [{"x": 0.5}],
        "groups": {
            "0": {
                "support": support,
                "params": {},
                "bootstrap": bootstrap,
                "real_rows": [{"x": 0.5}],
                "n": 1,
            }
        },
    }
    formula = SimpleNamespace(predict=lambda x, **kw: np.full(len(x), 10.0))
    env = dict(
        dataclass=dataclass,
        lru_cache=lru_cache,
        Path=Path,
        Any=Any,
        Callable=Callable,
        Dict=Dict,
        Optional=Optional,
        math=math,
        np=np,
        pd=pd,
        joblib=SimpleNamespace(load=lambda path: state),
        _load_typeI_formula=lambda *args: formula,
        _load_typeII_form=lambda *args: formula,
        DEFAULT_MAX_ROWS=50,
        DEFAULT_OVERSAMPLE=20,
        HARD_LIMIT=5000,
        FLOOR_FOR_LOG=1e-12,
    )
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]), "<pinned-scilaws-fixture>", "exec"
        ),
        env,
    )
    return env[f"_build_type{kind}_simulator"](
        "/synthetic/task/simulator/fixture/_wrapper.py"
    )


def audit(repo):
    source = source_bytes(repo)
    cases = {}
    for kind in ("I", "II"):
        extra = {} if kind == "I" else {"group_id": 0}
        sim = fixture_bundle(source, kind)
        one = sim.fetch_data(x=[0.5], seed=7, **extra)
        two = sim.fetch_data(x=[0.5], seed=7, **extra)
        clipped = sim.fetch_data(x=[2.0], seed=8, **extra)
        sim.fetch_data(x=[0.5], seed=9, **extra)
        exhausted = sim.fetch_data(x=[0.5], seed=10, **extra)
        where_after = sim.fetch_where(query="y > 0", seed=11, **extra)
        selected = fixture_bundle(source, kind, budget=20).fetch_where(
            query="y > 10", seed=12, limit=20, **extra
        )
        cases[kind] = {
            "same_seed_same_point_replays": one["rows"] == two["rows"],
            "clipped_coordinate": clipped["rows"][0]["x"],
            "clipped_count": clipped["n_clipped"],
            "explicit_point_rejected_after_budget": "error" in exhausted,
            "where_rows_after_budget": where_after.get("n_returned", 0),
            "used_after_where": sim.budget_status()["used"],
            "target_filtered_rows": selected.get("n_returned"),
            "target_filtered_values": sorted(set(selected.get("y", []))),
            "law_metadata_method_present": callable(
                getattr(sim, "formula_info" if kind == "I" else "form_info")
            ),
        }
    return {
        "schema_version": 1,
        "source_commit": COMMIT,
        "source_sha256": SOURCE_SHA,
        "status": "requires_point_query_isolation_before_adoption",
        "fixtures": cases,
        "benchmark_task_states_loaded": 0,
        "benchmark_outcomes_opened": 0,
        "model_calls": 0,
        "paid_cost_usd": 0,
        "opportunity_gate_passed": False,
        "paid_authorization": False,
        "limitation": "Runtime fixture behavior only; not evidence that baseline agents used fetch_where or inspected hidden laws.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.repo)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
