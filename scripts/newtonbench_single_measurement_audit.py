"""Source-only law separation at one prespecified public geometric midpoint.

This is a classification-error upper bound, not a prediction-risk/depth gate.
No LLM-generated support, target endpoint, or policy rollout is opened.
"""

import argparse
import ast
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from scripts.newtonbench_observation_contract_audit import REPO, SOURCE, functions


def literal_assignments(source):
    result = {}
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        else:
            continue
        try:
            result[name] = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
    return result


def coefficient(mean_a, sigma_a, mean_b, sigma_b):
    if (
        not all(math.isfinite(v) for v in (mean_a, sigma_a, mean_b, sigma_b))
        or min(sigma_a, sigma_b) <= 0
    ):
        raise ValueError("requires finite normal distributions with positive scales")
    scale = max(sigma_a, sigma_b)
    a, b = sigma_a / scale, sigma_b / scale
    variance = a * a + b * b
    return math.exp(
        0.5 * math.log(2 * a * b / variance)
        - ((mean_a - mean_b) / scale) ** 2 / (4 * variance)
    )


def classification_bound(means, scales):
    if not means or len(means) != len(scales):
        raise ValueError("invalid distribution panel")
    pairs = [
        coefficient(means[i], scales[i], means[j], scales[j])
        for i, j in combinations(range(len(means)), 2)
    ]
    # Bayes error <= sum of pairwise min(p_i f_i, p_j f_j), and
    # min(a,b) <= sqrt(a*b). Each prior mass is exactly 1/K.
    return min(1 - 1 / len(means), sum(pairs) / len(means))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bindings = {}

    def read(path):
        text = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"{SOURCE}:{path}"], text=True
        )
        bindings[path] = hashlib.sha256(text.encode()).hexdigest()
        return text

    registry = literal_assignments(read("autoscilab/oracle/newtonbench.py"))[
        "DOMAIN_REGISTRY"
    ]
    expected = [f"m{i}" for i in range(12)]
    domains = sorted(registry, key=lambda name: int(name.split("_")[0][1:]))
    if [name.split("_")[0] for name in domains] != expected:
        raise ValueError("source domain panel changed")
    shared = literal_assignments(read("newtonbench_vendor/modules/common/types.py"))
    rows = []
    for domain in domains:
        config = registry[domain]
        base = f"newtonbench_vendor/modules/{domain}"
        core = ast.parse(read(f"{base}/core.py"))
        runner = next(
            n
            for n in core.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_experiment_for_module"
        )
        noise_calls = [
            n
            for n in ast.walk(runner)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "inject_noise"
        ]
        if len(noise_calls) != 1 or not isinstance(noise_calls[0].args[2], ast.Name):
            raise ValueError("ambiguous source vanilla noise floor")
        floor_name = noise_calls[0].args[2].id
        constants = dict(shared)
        constants.update(
            literal_assignments(read(f"{base}/{domain.split('_')[0]}_types.py"))
        )
        floor = constants[floor_name]
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("invalid source floor")
        law_source = read(f"{base}/laws.py")
        law_names = [
            f"_ground_truth_law_{difficulty}_v{version}"
            for difficulty in ("easy", "medium", "hard")
            for version in range(3)
        ]
        namespace = {"np": np, "math": math, **literal_assignments(law_source)}
        laws = functions(law_source, set(law_names), namespace)
        point = [
            math.sqrt(config["bounds"][name][0] * config["bounds"][name][1])
            for name in config["param_names"]
        ]
        means, invalid = {}, []
        for name in law_names:
            try:
                value = float(laws[name](*point))
                if not math.isfinite(value):
                    raise ValueError("nonfinite raw law value")
                means[name] = value
            except (ArithmeticError, ValueError, TypeError) as exc:
                invalid.append({"law": name, "error_type": type(exc).__name__})
        strata = {}
        for difficulty in ("easy", "medium", "hard", "pooled"):
            selected = (
                law_names
                if difficulty == "pooled"
                else [n for n in law_names if f"_{difficulty}_" in n]
            )
            if any(name not in means for name in selected):
                strata[difficulty] = {"status": "invalid_source_law_no_dropping"}
            else:
                mu = [means[n] for n in selected]
                sigmas = [max(abs(x) * 0.01, floor) for x in mu]
                strata[difficulty] = {
                    "status": "bound_complete",
                    "law_count": len(mu),
                    "uniform_prior_error": 1 - 1 / len(mu),
                    "one_measurement_error_upper_bound": classification_bound(
                        mu, sigmas
                    ),
                }
        rows.append(
            {
                "domain": domain,
                "point": point,
                "input_order": config["param_names"],
                "absolute_noise_floor": floor,
                "finite_raw_means": means,
                "invalid_laws": invalid,
                "strata": strata,
            }
        )
    report = {
        "status": "source_separation_diagnostic_complete",
        "source_commit": SOURCE,
        "design_rule": "One geometric midpoint of each unmodified wrapper box; no design search",
        "relative_noise": 0.01,
        "prior": "Uniform source versions within each difficulty; separately pooled nine",
        "interpretation": "Classification error upper bound only; not predictive-risk or depth efficacy",
        "source_sha256": bindings,
        "rows": rows,
        "model_calls": 0,
        "cost_usd": 0,
        "paid_calls_authorized": False,
        "new_gate_authority": False,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [{"domain": r["domain"], "strata": r["strata"]} for r in rows], indent=2
        )
    )


if __name__ == "__main__":
    main()
