"""Read-only deterministic contract checks, not a NewtonBench efficacy panel."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np


SOURCE = "acf160eb6c96897748dd92b152703b59b74efc05"
REPO = Path("external/LLM-AutoSciLab-horizon-reference")
PATHS = {
    "wrapper": "autoscilab/oracle/newtonbench.py",
    "noise": "newtonbench_vendor/utils/noise.py",
    "optics": "newtonbench_vendor/modules/m4_snell_law/laws.py",
    "wrap": "newtonbench_vendor/modules/m4_snell_law/physics.py",
    "oscillator": "newtonbench_vendor/modules/m6_underdamped_harmonic/laws.py",
}


def functions(source, names, namespace):
    selected = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise ValueError("missing source function")
    # Do not import benchmark evaluators, model adapters, registries or judges.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    environment = dict(namespace)
    exec(compile(module, "<pinned-source-contract>", "exec"), environment)
    return {name: environment[name] for name in names}


def audit(sources):
    scalar_scales = []
    array_scales = []

    def gauss(*, mu, sigma):
        scalar_scales.append(sigma)
        return sigma

    def normal(*, loc, scale, size):
        array_scales.append(np.asarray(scale).tolist())
        return np.asarray(scale)

    numpy_proxy = SimpleNamespace(
        ndarray=np.ndarray,
        maximum=np.maximum,
        abs=np.abs,
        random=SimpleNamespace(normal=normal),
    )
    inject = functions(
        sources["noise"],
        {"inject_noise"},
        {"np": numpy_proxy, "random": SimpleNamespace(gauss=gauss)},
    )["inject_noise"]
    cases = [(2.0, 0.1, 0.01), (0.0, 0.1, 0.01), (-2.0, 0.1, 0.01)]
    measured = [inject(*case) for case in cases]
    clean = inject(3.0, 0.0, 9.0)
    array = inject(np.array([2.0, 0.0, -2.0]), 0.1, 0.01)
    if (
        scalar_scales != [0.2, 0.01, 0.2]
        or clean != 3.0
        or measured != [2.2, 0.01, -1.8]
        or array_scales != [[0.2, 0.01, 0.2]]
        or not np.array_equal(array, measured)
    ):
        raise ValueError("source noise contract changed")
    optic = functions(
        sources["optics"],
        {"_ground_truth_law_easy_v1", "_ground_truth_law_medium_v0"},
        {"math": math},
    )
    wrap = functions(sources["wrap"], {"wrap_angle"}, {})["wrap_angle"]
    at_one = optic["_ground_truth_law_easy_v1"](1, 1, wrap(1.0))
    invalid_optic = optic["_ground_truth_law_medium_v0"](3, 1, wrap(1.0))
    oscillator = functions(
        sources["oscillator"],
        {"_ground_truth_law_easy_v0", "_ground_truth_law_easy_v2"},
        {"math": math},
    )
    invalid_oscillator = oscillator["_ground_truth_law_easy_v0"](1, 0.1, 5)
    signed_oscillator = oscillator["_ground_truth_law_easy_v2"](1, 0.1, 5)
    if not (
        math.isclose(at_one, 1.0, abs_tol=1e-12)
        and math.isnan(invalid_optic)
        and math.isnan(invalid_oscillator)
        and signed_oscillator == -615.0
    ):
        raise ValueError("source unit/domain contract changed")
    return {
        "status": "observation_contract_incompatible_with_current_chemistry_adapter",
        "source_commit": SOURCE,
        "source_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in sources.items()
        },
        "noise": {
            "scale": "max(abs(true_value * noise_level), absolute_floor)",
            "raw_additive_gaussian": True,
            "zero_noise_bypasses_floor": True,
            "scalar_test_scales": scalar_scales,
            "array_test_scales": array_scales[0],
            "random_sampling_performed": False,
        },
        "optics": {
            "unit": "degrees",
            "equal_indices_output_for_input_one": at_one,
            "radian_input_would_correspond_to_degrees": math.degrees(1),
            "medium_v0_at_3_1_1_is_nan": math.isnan(invalid_optic),
        },
        "oscillator": {
            "input_k_m_b": [1, 0.1, 5],
            "easy_v0_is_nan": math.isnan(invalid_oscillator),
            "easy_v2_signed_value": signed_oscillator,
        },
        "coverage": "Specific source-contract counterexamples, not whole-domain validity or planning opportunity",
        "model_calls": 0,
        "cost_usd": 0,
        "paid_calls_authorized": False,
        "old_endpoints_reopened": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {
        name: subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"{SOURCE}:{path}"], text=True
        )
        for name, path in PATHS.items()
    }
    report = audit(sources)
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
