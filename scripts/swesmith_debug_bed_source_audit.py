#!/usr/bin/env python3
"""Metadata-only source admission for DebugGym over SWE-smith composed bugs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "swesmith-debug-bed-source-v1"
DEBUG_GYM_COMMIT = "cc3fe3ef4ce08919e522eb00ea1bea5689f3b53e"
DEBUG_GYM_TREE = "54b04c0313b66d72c7f285dc255d2693164a7193"
DATA_REVISION = "699b53400d3855206a0fbf3ff4beaf1a52f4f232"
MIN_POPULATION = 40_000
MIN_COMPOSED = 2_000
MIN_COMPOSED_PER_TYPE = 500
SPLIT_SALT = "swesmith-debug-bed-v1|"
EXPECTED_FIELDS = (
    "instance_id",
    "repo",
    "patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "created_at",
    "image_name",
    "base_commit",
    "problem_statement",
)
BOUND_FILES = {
    "swe_smith_environment": ("debug_gym/gym/envs/swe_smith.py", "a3fe5cb78df49df5742611bf33e6587ff943dc64cd6dcc90caa6a256bf92e3e2"),
    "repository_environment": ("debug_gym/gym/envs/env.py", "705faa2e83dac1336c7a6f62aab839a3b633a77a722b4f09c50ee583c2740967"),
    "pdb_tool": ("debug_gym/gym/tools/pdb.py", "ffd1453415ee0588bf72c62c3adb8ea238f9d2c004352aa884d8b947e79bc60b"),
    "eval_tool": ("debug_gym/gym/tools/eval.py", "6012e0ec5fc5108924dc373fd5e5908b2fedf6c4e9fb6dae98bf26faaae7f2ff"),
    "view_tool": ("debug_gym/gym/tools/view.py", "f2948627292b1030f457ef64da437da01a8d23909ec426d0ed1832c6b3843ee1"),
    "edit_tool": ("debug_gym/gym/tools/edit.py", "5362752ad45671a2933037641a944a441099e531337ee89298e13e000e130434"),
    "submit_tool": ("debug_gym/gym/tools/submit.py", "a424d295b764269d7ff0a382b432664917b561c58c57c211bfde26efde1d9c50"),
    "split_config": ("debug_gym/gym/envs/configs/swe_smith.yaml", "b5f8e0e12f96adb46f79769a29c603c30dc92331dcdca9cb5c9d6feffb287b14"),
}
SHARDS = {
    "data/train-00000-of-00011.parquet": (7_581_187, "dfd117de4998d8ea1d4d86d5779420f81fbf1881428bede14fb2998b684f1048"),
    "data/train-00001-of-00011.parquet": (6_445_888, "3a4f7b5f5cb2e73a00d7a667c52e826befc8603d15ba025c13092032f065be84"),
    "data/train-00002-of-00011.parquet": (5_092_697, "c666846cf683b5037eb141668dbe7e819e02c3404c2e69d70cd6702a477e0c46"),
    "data/train-00003-of-00011.parquet": (4_595_151, "f882c50b1acd39c77af812b47a207b3d6d91b5d5e45c982be3cdd8d3a98fc3bc"),
    "data/train-00004-of-00011.parquet": (7_265_387, "35a3c007c2aa1d49fffd962bc74b86497ca06cc5b875413d5dd3abda1d166299"),
    "data/train-00005-of-00011.parquet": (66_780_564, "6e7c1848bc31ad5eae9d7af07170b7fd578916702bad76178555ec605ae94dd1"),
    "data/train-00006-of-00011.parquet": (12_775_180, "e6a054f9e4cecc1f58f1eb156b3eec5c439714b5694fea0bc56f93245d5fea6b"),
    "data/train-00007-of-00011.parquet": (13_716_391, "9089ccc9551af345c785369e4aa83649e410d22ac58be22c826a880ade860655"),
    "data/train-00008-of-00011.parquet": (7_814_261, "c27fba33643442c20ee588d0fe632972cf3e6907ec55aab92b65695db4fc689c"),
    "data/train-00009-of-00011.parquet": (116_236_859, "b40c9c12548aeaf0f0269b542cec821b814bebbd68b11e58f4addc650292924f"),
    "data/train-00010-of-00011.parquet": (5_274_728, "e8b2bfa0a6fe0566b92b7aa0b810b6eebb81c762a6f2526f46441fa17dac9eda"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def git_value(root: Path, expression: str) -> str:
    return subprocess.run(["git", "rev-parse", expression], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def parse_simple_yaml_lists(path: Path) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(": ["):
            current = line[:-3]
            lists[current] = []
            continue
        if current and line == "]":
            current = None
            continue
        if current:
            for part in line.split(","):
                value = part.strip().strip("'\"")
                if value:
                    lists[current].append(value)
    return lists


def ordered_hash(values: list[str]) -> str:
    return sha256_bytes(canonical_bytes(values))


def audit(debug_root: Path, data_root: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for metadata-only Parquet projection") from exc

    code_hashes = {name: sha256_file(debug_root / path) for name, (path, _) in BOUND_FILES.items()}
    expected_code_hashes = {name: expected for name, (_, expected) in BOUND_FILES.items()}
    shard_bindings = {
        rel: {"size": (data_root / rel).stat().st_size, "sha256": sha256_file(data_root / rel)}
        for rel in SHARDS
    }
    expected_shards = {rel: {"size": size, "sha256": oid} for rel, (size, oid) in SHARDS.items()}
    schemas: set[tuple[str, ...]] = set()
    ids: list[str] = []
    images: list[str] = []
    f2p_lengths: list[int] = []
    p2p_lengths: list[int] = []
    for rel in SHARDS:
        parquet = pq.ParquetFile(data_root / rel)
        schemas.add(tuple(parquet.schema_arrow.names))
        table = parquet.read(columns=["instance_id", "image_name", "FAIL_TO_PASS", "PASS_TO_PASS"])
        projected = table.to_pydict()
        ids.extend(projected["instance_id"])
        images.extend(projected["image_name"])
        f2p_lengths.extend(len(value or []) for value in projected["FAIL_TO_PASS"])
        p2p_lengths.extend(len(value or []) for value in projected["PASS_TO_PASS"])

    config_lists = parse_simple_yaml_lists(debug_root / BOUND_FILES["split_config"][0])
    excluded = set(config_lists.get("excluded", []))
    development = config_lists.get("train-789", [])
    confirmation = config_lists.get("test-125", [])
    id_set = set(ids)
    index = {value: idx for idx, value in enumerate(ids)}
    composition = {
        "combine_file": [value for value in ids if ".combine_file__" in value],
        "combine_module": [value for value in ids if ".combine_module__" in value],
    }
    composed = composition["combine_file"] + composition["combine_module"]
    eligible = [value for value in composed if value not in excluded and value not in set(development) and value not in set(confirmation)]
    ordered = sorted(eligible, key=lambda value: sha256_bytes((SPLIT_SALT + value).encode()))
    mechanics, opportunity, reserve = ordered[:8], ordered[8:72], ordered[72:]
    environment = (debug_root / BOUND_FILES["swe_smith_environment"][0]).read_text(encoding="utf-8")
    repo_env = (debug_root / BOUND_FILES["repository_environment"][0]).read_text(encoding="utf-8")
    pdb_tool = (debug_root / BOUND_FILES["pdb_tool"][0]).read_text(encoding="utf-8")
    released_contract = {
        "exact_dataset_revision_bound": DATA_REVISION in environment,
        "bug_patch_applied_to_fresh_environment": "Apply bug patch" in environment and "self.bug_patch" in environment,
        "persistent_pdb_experiments_supported": "persistent_breakpoints" in pdb_tool and "interact_with_pdb" in pdb_tool,
        "official_tests_executed": "self.entrypoint" in environment and "self.log_parser" in environment,
        "f2p_scored_only_if_all_p2p_pass": "score = f2p_score if p2p_score == len(self.pass_to_pass) else 0" in environment,
        "upstream_remote_removed": "git remote remove origin" in (debug_root / "debug_gym/gym/envs/swe_bench.py").read_text(encoding="utf-8"),
        "seedable_environment": "self.rng = np.random.RandomState(seed)" in repo_env,
    }
    composed_indices = [index[value] for value in composed]
    exact_schema = len(schemas) == 1 and next(iter(schemas), ()) == EXPECTED_FIELDS
    official_lists_valid = (
        len(development) == 789 and len(confirmation) == 125
        and len(set(development)) == len(development) and len(set(confirmation)) == len(confirmation)
        and not (set(development) & set(confirmation))
        and set(development) <= id_set and set(confirmation) <= id_set
        and not (excluded & set(development)) and not (excluded & set(confirmation))
    )
    gates = {
        "immutable_debuggym_binding": git_value(debug_root, "HEAD") == DEBUG_GYM_COMMIT and git_value(debug_root, "HEAD^{tree}") == DEBUG_GYM_TREE and code_hashes == expected_code_hashes,
        "immutable_swesmith_shards": shard_bindings == expected_shards,
        "single_exact_schema_and_large_population": exact_schema and len(ids) >= MIN_POPULATION,
        "unique_nonempty_ids_and_images": len(set(ids)) == len(ids) and all(str(value).strip() for value in ids) and all(str(value).strip() for value in images),
        "large_composed_population": len(composed) >= MIN_COMPOSED and all(len(values) >= MIN_COMPOSED_PER_TYPE for values in composition.values()),
        "composed_tasks_have_f2p_and_p2p": all(f2p_lengths[idx] >= 1 and p2p_lengths[idx] >= 1 for idx in composed_indices),
        "released_execution_contract": all(released_contract.values()),
        "official_development_confirmation_valid": official_lists_valid,
        "mechanics_opportunity_reserve_complete_disjoint": len(mechanics) == 8 and len(opportunity) == 64 and len(mechanics) + len(opportunity) + len(reserve) == len(eligible) and len(set(mechanics + opportunity + reserve)) == len(eligible),
    }
    passed = all(gates.values())
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "debuggym": {"repository": "https://github.com/microsoft/debug-gym", "commit": DEBUG_GYM_COMMIT, "tree": DEBUG_GYM_TREE, "bound_file_sha256": code_hashes},
        "swesmith": {"repository": "https://huggingface.co/datasets/SWE-bench/SWE-smith", "revision": DATA_REVISION, "shards": shard_bindings},
        "population": {"total": len(ids), "repositories": len({value.split(".", 1)[0] for value in ids}), "images": len(set(images)), "composition_counts": {name: len(values) for name, values in composition.items()}},
        "schema_fields": list(EXPECTED_FIELDS),
        "split_counts": {"mechanics": len(mechanics), "opportunity": len(opportunity), "development": len(development), "confirmation": len(confirmation), "reserve": len(reserve), "excluded": len(excluded)},
        "ordered_split_id_sha256": {"mechanics": ordered_hash(mechanics), "opportunity": ordered_hash(opportunity), "development": ordered_hash(development), "confirmation": ordered_hash(confirmation), "reserve": ordered_hash(reserve), "excluded": ordered_hash(sorted(excluded))},
        "privacy": {"individual_instance_ids_serialized": False, "problem_statements_opened": False, "patches_opened": False, "test_names_opened": False, "test_outputs_opened": False, "repository_source_opened": False, "endpoints_opened": False},
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "execution_mechanics_authorized" if passed else "close_exact_swesmith_debug_bed_source",
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
        "released_contract": released_contract,
        "gates": gates,
        "privacy": manifest["privacy"],
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
        "authorizes": "zero_model_call_mechanics_only" if passed else "nothing",
    }
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.debug_root.resolve(), args.data_root.resolve(), args.protocol.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
