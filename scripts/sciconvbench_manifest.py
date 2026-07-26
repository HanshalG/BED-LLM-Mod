#!/usr/bin/env python3
"""Build a value-blind SciConvBench split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "results/nonmyopic/sciconvbench_manifest/MANIFEST.json"
)
EXPECTED_COMMIT = "0a87f8e755968a57d4e0ce063a6063fbc18ec87c"
SEED = 24420
MIN_MISSING_COMPONENTS = 3
OPPORTUNITY_PER_DOMAIN = 8
DEVELOPMENT_PER_DOMAIN = 4
DOMAINS = ("fluids", "foam", "matToolUse", "solMech", "solToolUse")
EXPECTED_SOURCES = {
    "fluids": {
        "path": "fluids/disambiguation_fluids.json",
        "sha256": "d57c1551284b22c2ecdb5a21d4dee4cbbaef8e55d6213e97e6cec66d89e63b53",
        "rows": 51,
    },
    "foam": {
        "path": "foam/disambiguation_foam.json",
        "sha256": "1bfb9d183facce1cec8c4c3b7014a4b4ec652ab0fdf052967bdf02c1a02c8ed6",
        "rows": 100,
    },
    "matToolUse": {
        "path": "matToolUse/disambiguation_matToolUse.json",
        "sha256": "10f0fdf6788d08561c714d4bc4e9063e82e4f64612b39542d523d1438272faa1",
        "rows": 48,
    },
    "solMech": {
        "path": "solMech/disambiguation_solMech.json",
        "sha256": "f246a8c2d1519d95885977aac52088b1f353e2da590f2a7b6f09bcd987b8a7de",
        "rows": 143,
    },
    "solToolUse": {
        "path": "solToolUse/disambiguation_solToolUse.json",
        "sha256": "677f18ea54c7e6fa4f89cbcfd4d344f07d0115e47d0509faba217be6629f220d",
        "rows": 85,
    },
}
EXPECTED_SPLIT_HASHES = {
    "mechanics": "8c5117faf027e1a20d80b30e0aad832213fe73e40374dfe05694158d958a05fc",
    "opportunity": "841250d0690a8ee43922065a98cf67aeb1e109ad177a4b94519e06d2c7eed926",
    "development": "73528ea7d1b4832d3977337bd3b1a765fd9c74c97c288eeaa17dbfb783888dde",
    "holdout": "eda8815ca90ee5243e406d19555f52b42d21079b12b6f8bcab690cd71024dd49",
}
EXPECTED_COMBINED_HASH = (
    "4aa6af551369a43d9a29d71d1343627369e96def0df9cf4f2c137958a362791f"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_hash(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_row(row: Any, *, domain: str, index: int) -> tuple[str, int]:
    if not isinstance(row, dict):
        raise ValueError(f"{domain} row {index} is not an object")
    required = {
        "id",
        "incomplete_user_req",
        "complete_user_req",
        "missing_entities",
        "ontology_components",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"{domain} row {index} missing keys: {sorted(missing)}")
    row_id = row["id"]
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"{domain} row {index} has invalid id")
    if not isinstance(row["incomplete_user_req"], str) or not row["incomplete_user_req"]:
        raise ValueError(f"{domain} row {row_id} has invalid incomplete requirement")
    if not isinstance(row["complete_user_req"], str) or not row["complete_user_req"]:
        raise ValueError(f"{domain} row {row_id} has invalid complete requirement")
    missing_entities = row["missing_entities"]
    ontology = row["ontology_components"]
    if not isinstance(missing_entities, list) or not all(
        isinstance(value, str) and value for value in missing_entities
    ):
        raise ValueError(f"{domain} row {row_id} has invalid missing entities")
    if not isinstance(ontology, list) or not all(
        isinstance(value, str) and value for value in ontology
    ):
        raise ValueError(f"{domain} row {row_id} has invalid ontology components")
    if len(missing_entities) != len(ontology):
        raise ValueError(f"{domain} row {row_id} has misaligned ontology")
    return row_id, len(missing_entities)


def build_domain_split(domain: str, eligible_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(eligible_ids)
    required = 1 + OPPORTUNITY_PER_DOMAIN + DEVELOPMENT_PER_DOMAIN + 1
    if len(ordered) < required:
        raise ValueError(
            f"{domain} has {len(ordered)} eligible rows; need at least {required}"
        )
    mechanics = ordered[:1]
    remaining = ordered[1:]
    remaining.sort(
        key=lambda row_id: hashlib.sha256(
            f"{SEED}:{domain}:{row_id}".encode("utf-8")
        ).hexdigest()
    )
    opportunity = remaining[:OPPORTUNITY_PER_DOMAIN]
    development = remaining[
        OPPORTUNITY_PER_DOMAIN : OPPORTUNITY_PER_DOMAIN + DEVELOPMENT_PER_DOMAIN
    ]
    holdout = remaining[OPPORTUNITY_PER_DOMAIN + DEVELOPMENT_PER_DOMAIN :]
    return {
        "mechanics": mechanics,
        "opportunity": opportunity,
        "development": development,
        "holdout": holdout,
    }


def build_manifest(
    source_root: Path,
    *,
    expected_commit: str = EXPECTED_COMMIT,
    enforce_frozen_hashes: bool = True,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    commit = _git_commit(source_root)
    if commit != expected_commit:
        raise ValueError(f"SciConvBench commit mismatch: {commit}")

    sources: dict[str, Any] = {}
    domain_splits: dict[str, dict[str, list[str]]] = {}
    for domain in DOMAINS:
        spec = EXPECTED_SOURCES[domain]
        path = source_root / str(spec["path"])
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ValueError(f"{domain} source hash mismatch: {digest}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != spec["rows"]:
            raise ValueError(f"{domain} source row count mismatch")

        ids: list[str] = []
        eligible: list[str] = []
        component_counts: dict[int, int] = {}
        for index, row in enumerate(rows):
            row_id, component_count = _validate_row(
                row, domain=domain, index=index
            )
            ids.append(row_id)
            component_counts[component_count] = (
                component_counts.get(component_count, 0) + 1
            )
            if component_count >= MIN_MISSING_COMPONENTS:
                eligible.append(row_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{domain} contains duplicate ids")

        domain_splits[domain] = build_domain_split(domain, eligible)
        sources[domain] = {
            "path": str(spec["path"]),
            "sha256": digest,
            "rows": len(rows),
            "eligible_rows": len(eligible),
            "missing_component_count_histogram": {
                str(key): component_counts[key] for key in sorted(component_counts)
            },
        }

    split_ids: dict[str, list[str]] = {}
    split_hashes: dict[str, str] = {}
    for split in ("mechanics", "opportunity", "development", "holdout"):
        values = [
            f"{domain}:{row_id}"
            for domain in DOMAINS
            for row_id in domain_splits[domain][split]
        ]
        split_ids[split] = values
        split_hashes[split] = ordered_hash(values)

    combined = [
        f"{split}:{domain}:{row_id}"
        for split in ("mechanics", "opportunity", "development", "holdout")
        for domain in DOMAINS
        for row_id in domain_splits[domain][split]
    ]
    combined_hash = ordered_hash(combined)

    if enforce_frozen_hashes:
        if split_hashes != EXPECTED_SPLIT_HASHES:
            raise ValueError(f"split hash mismatch: {split_hashes}")
        if combined_hash != EXPECTED_COMBINED_HASH:
            raise ValueError(f"combined split hash mismatch: {combined_hash}")

    all_ids = [value.split(":", 1)[1] for value in combined]
    passed = (
        sum(source["eligible_rows"] for source in sources.values()) == 318
        and {key: len(value) for key, value in split_ids.items()}
        == {"mechanics": 5, "opportunity": 40, "development": 20, "holdout": 253}
        and len(combined) == 318
        and len(all_ids) == 318
    )
    return {
        "protocol": "sciconvbench-value-blind-manifest-1",
        "source": {
            "repository": "https://github.com/csml-rpi/SciConvBench",
            "commit": commit,
            "domains": sources,
        },
        "selection": {
            "seed": SEED,
            "minimum_missing_components": MIN_MISSING_COMPONENTS,
            "opportunity_per_domain": OPPORTUNITY_PER_DOMAIN,
            "development_per_domain": DEVELOPMENT_PER_DOMAIN,
            "domain_splits": domain_splits,
            "split_ids": split_ids,
            "split_hashes": split_hashes,
            "combined_hash": combined_hash,
        },
        "access": {
            "content_emitted": False,
            "incomplete_requirements_emitted": False,
            "complete_requirements_emitted": False,
            "missing_entities_emitted": False,
            "ontology_components_emitted": False,
            "conversation_outcomes_read": False,
            "model_scores_read": False,
        },
        "accounting": {
            "openrouter_calls": 0,
            "openrouter_cost_usd": 0.0,
            "oatml_jobs": 0,
        },
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.source_root)
    if not manifest["passed"]:
        raise SystemExit("SciConvBench manifest gates failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["selection"]["split_hashes"], sort_keys=True))
    print(f"combined_hash={manifest['selection']['combined_hash']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
