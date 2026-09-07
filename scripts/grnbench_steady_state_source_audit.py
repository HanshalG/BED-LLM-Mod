"""Bounded source-only fixed-point diagnostic, not a discovery experiment."""

import argparse
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np


PROTOCOL = Path(
    "results/nonmyopic/GRNBENCH_STEADY_STATE_SOURCE_AUDIT_PROTOCOL_20260908.json"
)
FUNCTIONS = {
    "_hill_act",
    "_hill_rep",
    "_merge_params",
    "_build_params",
    "_simulate_chain",
    "_simulate_coherent_ffl",
    "_simulate_incoherent_ffl",
    "_simulate_negative_feedback",
    "_simulate_toggle",
}
CONSTANTS = {
    "_COMMON_BASE",
    "_DIFFICULTY_OVERRIDES",
    "_VERSION_OVERRIDES",
    "_DOMAIN_BASES",
    "_SIMULATORS",
}


def implementations(source, published):
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in CONSTANTS
        ):
            selected.append(node)
    variants = {}
    for label, override in [
        ("published", None),
        ("next", "increment"),
        ("long", 200),
        ("long_next", 201),
    ]:
        body = copy.deepcopy(selected)
        found = set()
        for node in body:
            if isinstance(node, ast.FunctionDef) and node.name in published:
                loops = [x for x in ast.walk(node) if isinstance(x, ast.For)]
                if len(loops) != 1:
                    raise ValueError("published fixed iteration loop changed")
                call = loops[0].iter
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "range"
                    and len(call.args) == 1
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == published[node.name]
                ):
                    raise ValueError("published iteration count changed")
                found.add(node.name)
                if override is not None:
                    call.args[0].value = (
                        published[node.name] + 1
                        if override == "increment"
                        else override
                    )
        if found != set(published):
            raise ValueError("missing published recurrence")
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        namespace = {"np": np}
        exec(compile(module, "<pinned-grn-diagnostic>", "exec"), namespace)
        variants[label] = namespace
    return variants


def audit(source, config):
    start = monotonic()
    variants = implementations(source, config["published_iterations"])
    points = list(itertools.product(*config["bounds"])) + [(1.0,) * 5]
    rows = []
    for family, difficulty, version in itertools.product(
        config["families"], config["difficulties"], config["versions"]
    ):
        params = variants["published"]["_build_params"](family, difficulty, version)
        for point_index, point in enumerate(points):
            if monotonic() - start > config["max_seconds"]:
                raise TimeoutError("source audit time cap")
            inputs = dict(zip(config["input_order"], point, strict=True))
            values = {}
            for label, namespace in variants.items():
                state = namespace["_SIMULATORS"][family](params, inputs)
                values[label] = np.array([state[k] for k in config["states"]])
                if not np.isfinite(values[label]).all():
                    raise ValueError("nonfinite source state")
            row = {
                "family": family,
                "difficulty": difficulty,
                "version": version,
                "point_index": point_index,
                "inputs": inputs,
                "states": {k: v.tolist() for k, v in values.items()},
            }
            for label, a, b in [
                ("published", "published", "next"),
                ("long", "long", "long_next"),
            ]:
                delta = np.abs(values[b] - values[a])
                tolerance = config["absolute_tolerance"] + config[
                    "relative_tolerance"
                ] * np.abs(values[a])
                row[label + "_fixed_point"] = bool(np.all(delta <= tolerance))
                row[label + "_maximum_absolute_residual"] = float(delta.max())
                c = config["states"].index("C")
                row[label + "_log_reporter_change"] = float(
                    abs(
                        np.log1p(params["reporter_scale"] * values[a][c])
                        - np.log1p(params["reporter_scale"] * values[b][c])
                    )
                )
            rows.append(row)
    if len(rows) != config["expected_cases"]:
        raise ValueError("source panel coverage mismatch")
    summary = {}
    for family in config["families"]:
        subset = [r for r in rows if r["family"] == family]
        summary[family] = {
            "cases": len(subset),
            "published_nonfixed": sum(not r["published_fixed_point"] for r in subset),
            "long_nonfixed": sum(not r["long_fixed_point"] for r in subset),
            "max_published_log_reporter_change": max(
                r["published_log_reporter_change"] for r in subset
            ),
            "max_long_log_reporter_change": max(
                r["long_log_reporter_change"] for r in subset
            ),
        }
    return {
        "status": "source_fixed_point_check_passed"
        if all(r["published_fixed_point"] for r in rows)
        else "source_steady_state_interpretation_failed",
        "summary": summary,
        "rows": rows,
        "elapsed_seconds": monotonic() - start,
        "policy_efficacy_tested": False,
        "model_calls": 0,
        "cost_usd": 0,
        "paid_calls_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(PROTOCOL.read_text())

    def git(*parts):
        return subprocess.check_output(
            ["git", "-C", str(args.source_root), *parts], text=True
        ).strip()

    if (
        git("rev-parse", "HEAD") != config["source_commit"]
        or git("remote", "get-url", "origin") != config["source_origin"]
    ):
        raise ValueError("source checkout identity changed")
    ref = config["source_commit"] + ":" + config["source_path"]
    if git("rev-parse", ref) != config["source_blob"]:
        raise ValueError("source blob changed")
    source = git("show", ref)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(source, config)
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    result["bindings"] = {
        "source_blob": config["source_blob"],
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(result["status"])


if __name__ == "__main__":
    main()
