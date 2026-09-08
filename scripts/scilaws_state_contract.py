"""Restricted development-state projection, never execution of hidden formulas."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

import numpy as np
from joblib.numpy_pickle import NumpyArrayWrapper, NumpyUnpickler
from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap

SPLIT_SHA = "e73e800f38c535390c5dccc3d94b1d21bd61444094a773bdf0046f43da4aedf6"
MAX_BYTES = 20 * 1024 * 1024


class ResidualState:
    """Inert replacement for serialized bootstrap attributes; no sample method."""


class NumericArray(NumpyArrayWrapper):
    def read(self, *args, **kwargs):
        if (
            self.dtype.hasobject
            or math.prod(self.shape) * self.dtype.itemsize > MAX_BYTES
        ):
            raise ValueError("object or oversized array forbidden")
        return super().read(*args, **kwargs)


class RestrictedState(NumpyUnpickler):
    def find_class(self, module, name):
        if name == "ResidualBootstrap" and module in {
            "data_aug.sim2.bootstrap",
            "harness.sim_runtime",
        }:
            return ResidualState
        if name == "ResidualState" and module == "scripts.scilaws_state_contract":
            return ResidualState
        if (module, name) == ("joblib.numpy_pickle", "NumpyArrayWrapper"):
            return NumericArray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if name == "scalar" and module in {
            "numpy.core.multiarray",
            "numpy._core.multiarray",
        }:
            return np.core.multiarray.scalar
        raise ValueError(f"unapproved serialized class: {module}.{name}")


def inspect_state(path):
    with Path(path).open("rb") as raw:
        with _validate_fileobject_and_memmap(raw, str(path)) as (stream, _):
            if isinstance(stream, str):
                raise ValueError("legacy filename-loading mode forbidden")
            state = RestrictedState(
                str(path), stream, ensure_native_byte_order=True
            ).load()
    if type(state) is not dict:
        raise ValueError("state must be a dictionary")
    inputs, support = state.get("used_inputs"), state.get("support")
    if (
        type(inputs) is not list
        or not inputs
        or len(inputs) > 3
        or len(set(inputs)) != len(inputs)
        or any(type(x) is not str for x in inputs)
    ):
        raise ValueError("invalid input contract")
    if type(support) is not dict:
        raise ValueError("support dictionary required")
    bounds = {}
    for name in inputs:
        interval = support.get(name, {})
        lo, hi = interval.get("min"), interval.get("max")
        if (
            any(type(x) not in (float, int) or not math.isfinite(x) for x in (lo, hi))
            or lo >= hi
        ):
            raise ValueError("invalid source support")
        bounds[name] = [lo, hi]
    noise = state.get("noise_scale")
    budget = state.get("fetch_budget_rows")
    if type(noise) not in (float, int) or not math.isfinite(noise) or noise < 0:
        raise ValueError("invalid noise scale")
    if type(budget) is not int or budget <= 0:
        raise ValueError("invalid row budget")
    space = state.get("residual_space")
    if space not in {"linear", "log"}:
        raise ValueError("unrecognized residual space")
    bootstrap = state.get("bootstrap")
    if not isinstance(bootstrap, ResidualState):
        raise ValueError("missing inert residual state")
    residuals = bootstrap.__dict__.get("_res")
    if (
        not isinstance(residuals, np.ndarray)
        or residuals.ndim != 1
        or not len(residuals)
        or not np.isfinite(residuals).all()
    ):
        raise ValueError("invalid residual array")
    return dict(
        status="source_contract_valid",
        used_inputs=inputs,
        bounds=bounds,
        residual_space=space,
        noise_scale_zero=noise == 0,
        fetch_budget_rows=budget,
        residual_count=len(residuals),
        formulas_executed=0,
        measurements_generated=0,
        excluded_fields=[
            "formula_source",
            "law_constants",
            "real_rows",
            "residual_values",
            "noise_scale_magnitude",
        ],
    )


def run(split_path, private_root, output):
    raw = Path(split_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_SHA:
        raise ValueError("split binding mismatch")
    split = json.loads(raw)
    private_root = Path(private_root).resolve()
    if private_root.is_relative_to(Path.cwd().resolve()):
        raise ValueError("private states must be outside repository")
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_root.chmod(0o700)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    result = dict(
        split_sha256=SPLIT_SHA,
        dataset_revision=split["dataset_revision"],
        tasks=[],
        paid_authorization=False,
        policy_endpoint_authorization=False,
        model_calls=0,
        paid_cost_usd=0,
        holdout_states_loaded=0,
    )
    for task in split["development"]:
        tid = task["task_id"]
        directory = task["metadata_path"].rsplit("/", 1)[0] + "/simulator"
        api = f"https://huggingface.co/api/datasets/RealSR/SciLaws-Bench/tree/{split['dataset_revision']}/{directory}"
        with urlopen(api, timeout=30) as response:
            files = json.load(response)
        entry = next(x for x in files if x["path"] == directory + "/state.joblib")
        size, sha = entry["size"], entry["lfs"]["oid"]
        if not 0 < size <= MAX_BYTES:
            raise ValueError("state exceeds frozen size cap")
        path = private_root / (tid + ".joblib")
        if not path.exists():
            url = f"https://huggingface.co/datasets/RealSR/SciLaws-Bench/resolve/{split['dataset_revision']}/{entry['path']}"
            with urlopen(url, timeout=30) as response:
                blob = response.read(MAX_BYTES + 1)
            if len(blob) != size or hashlib.sha256(blob).hexdigest() != sha:
                raise ValueError("download integrity mismatch")
            with path.open("xb") as stream:
                stream.write(blob)
            path.chmod(0o600)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("private state integrity mismatch")
        # Use a bounded child; it can only emit the projection, never raw state.
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--inspect", str(path)],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if process.returncode:
            contract = dict(
                status="source_contract_invalid",
                error="restricted state projection failed",
            )
        else:
            contract = json.loads(process.stdout)
        result["tasks"].append(
            dict(task_id=tid, state_sha256=sha, size_bytes=size, **contract)
        )
        print(tid, contract["status"], flush=True)
    result["complete_development_coverage"] = len(result["tasks"]) == 8
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect", type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--private-root", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.inspect:
        print(json.dumps(inspect_state(args.inspect)))
    else:
        run(args.split, args.private_root, args.output)


if __name__ == "__main__":
    main()
