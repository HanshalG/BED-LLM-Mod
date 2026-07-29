#!/usr/bin/env python3
"""Audit fixed Number Game policies on grammar-novel Gemini endpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_classical_grammar_irreducibility_audit as grammar


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-gemini-classical-grammar-audit-1"
EXPECTED_BANK_SHA256 = (
    "6e2a523d7d7c0c59b5df0494de37087525c7009d43a5eaba01390e005d85e0e0"
)
PRIOR_AUDIT_RESULT_PATH = (
    REPO_ROOT
    / "results/nonmyopic/number_game_classical_grammar_irreducibility_audit/"
    "number-game-classical-grammar-irreducibility-audit-20260729/RESULT.json"
)
PRIOR_AUDIT_RESULT_SHA256 = (
    "514945d268712227c128ecb9be9ee0a495cf1682a1c53c99ff5428b8da0bf9a7"
)
GEMINI_STUDIES = (
    {
        "name": "fixed_policy_fresh_endpoints",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_confirmation/"
            "number-game-crossfit-depth-three-confirmation-20260728/TREES.json"
        ),
        "trees_sha256": (
            "cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9"
        ),
        "result": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_endpoint_precision/"
            "number-game-crossfit-endpoint-precision-20260728/RESULT.json"
        ),
        "result_sha256": (
            "47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e"
        ),
        "endpoints": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_endpoint_precision/"
            "number-game-crossfit-endpoint-precision-20260728/ENDPOINTS.json"
        ),
        "endpoints_sha256": (
            "0e6bd789b28a77f2ca47a13015fc80d46686daf14f3a2540caebe8663eff17c0"
        ),
    },
    {
        "name": "fresh_tree_replication",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/"
            "number_game_crossfit_depth_three_fresh_replication/"
            "number-game-crossfit-depth-three-fresh-replication-20260728/"
            "TREES.json"
        ),
        "trees_sha256": (
            "197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802"
        ),
        "result": (
            REPO_ROOT
            / "results/nonmyopic/"
            "number_game_crossfit_depth_three_fresh_replication/"
            "number-game-crossfit-depth-three-fresh-replication-20260728/"
            "RESULT.json"
        ),
        "result_sha256": (
            "25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8"
        ),
        "endpoints": (
            REPO_ROOT
            / "results/nonmyopic/"
            "number_game_crossfit_depth_three_fresh_replication/"
            "number-game-crossfit-depth-three-fresh-replication-20260728/"
            "ENDPOINTS.json"
        ),
        "endpoints_sha256": (
            "103626763fe25987541a48adb93ac1dea600f53ecf64b95d1d4dbc2d66efc824"
        ),
    },
)


def verify_prior_audit() -> dict[str, Any]:
    if (
        grammar.sha256_file(PRIOR_AUDIT_RESULT_PATH)
        != PRIOR_AUDIT_RESULT_SHA256
    ):
        raise ValueError("prior classical-grammar audit hash changed")
    result = json.loads(PRIOR_AUDIT_RESULT_PATH.read_text(encoding="utf-8"))
    if result["bank"]["sha256"] != EXPECTED_BANK_SHA256:
        raise ValueError("prior classical grammar SHA changed")
    if not result["support_novelty"]["all_gates_pass"]:
        raise ValueError("prior generated-support novelty gate did not pass")
    return result


def load_frozen_evidence() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, str],
]:
    sources: dict[str, dict[str, Any]] = {}
    combined_endpoint_trees = []
    hashes = {"prior_audit_result": PRIOR_AUDIT_RESULT_SHA256}
    for study in GEMINI_STUDIES:
        for artifact in ("trees", "result", "endpoints"):
            observed = grammar.sha256_file(study[artifact])
            expected = study[f"{artifact}_sha256"]
            if observed != expected:
                raise ValueError(
                    f"{study['name']} {artifact} hash changed: {observed}"
                )
            hashes[f"{study['name']}_{artifact}"] = expected

        trees = json.loads(study["trees"].read_text(encoding="utf-8"))
        result = json.loads(study["result"].read_text(encoding="utf-8"))
        endpoints = json.loads(
            study["endpoints"].read_text(encoding="utf-8")
        )
        if not (
            len(trees["trees"])
            == len(result["trees"])
            == len(endpoints["trees"])
            == 32
        ):
            raise ValueError(f"{study['name']} must contain exactly 32 trees")
        sources[study["name"]] = {"trees": trees, "result": result}
        for index, endpoint_tree in enumerate(endpoints["trees"]):
            if int(endpoint_tree["tree_index"]) != index:
                raise ValueError(f"{study['name']} endpoint order changed")
            source_tree = trees["trees"][index]
            combined_endpoint_trees.append(
                {
                    **endpoint_tree,
                    "source_study": study["name"],
                    "tree_seed": int(source_tree["tree_seed"]),
                }
            )
    return (
        sources,
        {"trees": combined_endpoint_trees},
        hashes,
    )


def endpoint_novelty_and_efficacy(
    sources: dict[str, dict[str, Any]],
    endpoints: dict[str, Any],
    *,
    bank: set[int],
) -> dict[str, Any]:
    tree_rows = []
    novel_masks: set[int] = set()
    source_analyzable_counts: dict[str, int] = defaultdict(int)
    source_endpoint_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "novel": 0}
    )
    for endpoint_tree in endpoints["trees"]:
        source_name = str(endpoint_tree["source_study"])
        local_index = int(endpoint_tree["tree_index"])
        source = sources[source_name]
        row = grammar._score_tree(
            source_name=source_name,
            source_tree=source["trees"]["trees"][local_index],
            source_metrics=source["result"]["trees"][local_index],
            endpoint_tree=endpoint_tree,
            bank=bank,
        )
        tree_rows.append(row)
        source_endpoint_counts[source_name]["total"] += row[
            "endpoint_occurrences"
        ]
        source_endpoint_counts[source_name]["novel"] += row[
            "grammar_novel_endpoint_occurrences"
        ]
        source_analyzable_counts[source_name] += int(row["analyzable"])
        for support in endpoint_tree["supports"]:
            novel_masks.update(
                grammar.item_mask(item)
                for item in support
                if grammar.item_mask(item) not in bank
            )

    total_occurrences = sum(
        row["endpoint_occurrences"] for row in tree_rows
    )
    novel_occurrences = sum(
        row["grammar_novel_endpoint_occurrences"] for row in tree_rows
    )
    analyzable = [row for row in tree_rows if row["analyzable"]]
    endpoint_gates = {
        "endpoint_novel_fraction_at_least_five_percent": (
            novel_occurrences / total_occurrences
            >= grammar.MIN_ENDPOINT_NOVEL_FRACTION
        ),
        "at_least_512_novel_endpoint_occurrences": (
            novel_occurrences >= grammar.MIN_ENDPOINT_NOVEL_OCCURRENCES
        ),
        "at_least_48_analyzable_trees": (
            len(analyzable) >= grammar.MIN_ANALYZABLE_TREES
        ),
        "each_source_has_at_least_24_analyzable_trees": all(
            source_analyzable_counts[study["name"]]
            >= grammar.MIN_ANALYZABLE_TREES_PER_SOURCE
            for study in GEMINI_STUDIES
        ),
    }
    efficacy = grammar._aggregate_efficacy(analyzable)
    return {
        "endpoint_occurrences": total_occurrences,
        "grammar_novel_endpoint_occurrences": novel_occurrences,
        "grammar_novel_endpoint_occurrence_fraction": (
            novel_occurrences / total_occurrences
        ),
        "grammar_novel_unique_endpoint_extensions": len(novel_masks),
        "analyzable_tree_count": len(analyzable),
        "analyzable_tree_count_by_source": dict(source_analyzable_counts),
        "endpoint_occurrences_by_source": {
            key: dict(value)
            for key, value in source_endpoint_counts.items()
        },
        "endpoint_power_gates": endpoint_gates,
        "all_endpoint_power_gates_pass": all(endpoint_gates.values()),
        "efficacy": efficacy,
        "trees": tree_rows,
    }


def audit_status(
    *,
    bank_gate: bool,
    endpoint_gate: bool,
    efficacy_gate: bool,
) -> str:
    if not bank_gate:
        return "mechanics_failure"
    if not endpoint_gate:
        return "inconclusive"
    if efficacy_gate:
        return "positive_irreducibility_audit"
    return "negative"


def run_audit(*, output_dir: Path) -> dict[str, Any]:
    prior = verify_prior_audit()
    bank, bank_diagnostics = grammar.build_classical_grammar_bank()
    sources, endpoints, hashes = load_frozen_evidence()
    endpoint = endpoint_novelty_and_efficacy(
        sources,
        endpoints,
        bank=bank,
    )
    bank_gate = (
        bank_diagnostics["sha256"] == EXPECTED_BANK_SHA256
        and bank_diagnostics["unique_nonconstant_extension_count"]
        >= grammar.MIN_BANK_EXTENSIONS
    )
    status = audit_status(
        bank_gate=bank_gate,
        endpoint_gate=endpoint["all_endpoint_power_gates_pass"],
        efficacy_gate=endpoint["efficacy"]["all_gates_pass"],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "NUMBER_GAME_GEMINI_CLASSICAL_GRAMMAR_AUDIT_"
                "PREREGISTRATION.md"
            ),
            "model_calls": 0,
            "cost_usd": 0.0,
            "retrospective_fixed_policy_audit": True,
            "source_hashes": hashes,
        },
        "status": status,
        "bank": {
            **bank_diagnostics,
            "expected_sha256": EXPECTED_BANK_SHA256,
            "mechanics_gate_pass": bank_gate,
        },
        "bound_generated_support_novelty": {
            "source_result_sha256": PRIOR_AUDIT_RESULT_SHA256,
            "all_gates_pass": prior["support_novelty"]["all_gates_pass"],
            "generated_second": prior["support_novelty"]["stages"][
                "generated_second"
            ],
        },
        "endpoint_novelty_and_efficacy": endpoint,
        "all_positive_gates_pass": (
            status == "positive_irreducibility_audit"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/"
            "number_game_gemini_classical_grammar_audit/"
            "number-game-gemini-classical-grammar-audit-20260729"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "bank": result["bank"],
                "endpoint_power_gates": result[
                    "endpoint_novelty_and_efficacy"
                ]["endpoint_power_gates"],
                "efficacy": result["endpoint_novelty_and_efficacy"][
                    "efficacy"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
