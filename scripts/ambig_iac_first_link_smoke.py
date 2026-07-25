#!/usr/bin/env python3
"""Audit regenerated-support depth-two EIG on Ambig-IaC."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter


INTERFACE_VERSION = "ambig-iac-first-link-smoke-1"
SOURCE_REPOSITORY = "https://github.com/agent4ops/ambig-iac"
SOURCE_COMMIT = "4b50c142ed4638a4caaee9ca0b92d8d0e5b8c8cb"
DATASET_TREE_SHA256 = (
    "1f2d20cadb147962fac104e037d75c53d68d1a9b092050a80513d3751c683f63"
)
MODEL_ID = "openai/gpt-5.4-mini"
SEED = 24_372
TASK_IDS = (272, 66, 156)
PARTICLE_COUNT = 5
MIN_VALID_PARTICLES = 4
ROOT_COUNT = 4
EXPECTED_REQUESTS = len(TASK_IDS) * PARTICLE_COUNT * (1 + 2 * ROOT_COUNT)
MAX_COST_USD = 1.50
ADDRESS_PATTERN = re.compile(
    r"^(?:data\.)?(?:aws|awscc)_[a-z0-9_]+\.[A-Za-z0-9_-]+$"
)
SPEC_KEYS = frozenset({"resources", "topology", "attributes"})


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class SmokeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True, order=True)
class Feature:
    dimension: str
    subject: str
    value: str

    def key(self) -> str:
        return json.dumps(
            [self.dimension, self.subject, self.value],
            separators=(",", ":"),
        )

    def question(self) -> str:
        if self.dimension == "resource":
            count = int(self.value)
            noun = "resource" if count == 1 else "resources"
            return (
                f"Should the target configuration include at least {count} "
                f"`{self.subject}` {noun}?"
            )
        if self.dimension == "topology":
            return (
                f"Should a `{self.subject}` resource depend on a "
                f"`{self.value}` resource?"
            )
        if self.dimension == "attribute":
            return (
                f"Should a `{self.subject}` resource explicitly set the "
                f"`{self.value}` attribute?"
            )
        raise ValueError(f"unknown feature dimension {self.dimension!r}")

    def as_dict(self) -> dict[str, str]:
        return {
            "dimension": self.dimension,
            "subject": self.subject,
            "value": self.value,
            "question": self.question(),
        }


def _resource_type(address: str) -> str:
    parts = address.split(".")
    if parts[0] == "data" and len(parts) >= 3:
        return f"data.{parts[1]}"
    return parts[0]


def parse_generated_spec(text: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != SPEC_KEYS:
        raise ValueError("spec response must have exactly the three spec keys")
    resources = payload["resources"]
    topology = payload["topology"]
    attributes = payload["attributes"]
    if not isinstance(resources, dict) or not resources:
        raise ValueError("resources must be a nonempty object")
    if not isinstance(topology, dict) or not isinstance(attributes, dict):
        raise ValueError("topology and attributes must be objects")
    parsed_resources: dict[str, str] = {}
    for label, address in resources.items():
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(address, str)
            or ADDRESS_PATTERN.fullmatch(address.strip()) is None
        ):
            raise ValueError("resource labels or addresses are invalid")
        parsed_resources[label.strip()] = address.strip()
    parsed_topology: dict[str, list[str]] = {}
    for source, dependencies in topology.items():
        if source not in parsed_resources:
            raise ValueError("topology references an unknown resource label")
        if not isinstance(dependencies, list):
            raise ValueError("topology dependency list is invalid")
        if any(
            not isinstance(dependency, str)
            or dependency not in parsed_resources
            for dependency in dependencies
        ):
            raise ValueError("topology references an unknown resource label")
        parsed_topology[source] = list(dict.fromkeys(dependencies))
    parsed_attributes: dict[str, dict[str, Any]] = {}
    for label, values in attributes.items():
        if label not in parsed_resources or not isinstance(values, dict):
            raise ValueError("attributes reference an unknown resource label")
        if any(not isinstance(key, str) or not key.strip() for key in values):
            raise ValueError("attribute names must be nonempty strings")
        parsed_attributes[label] = values
    return {
        "resources": parsed_resources,
        "topology": parsed_topology,
        "attributes": parsed_attributes,
    }


def spec_features(spec: dict[str, dict[str, Any]]) -> frozenset[Feature]:
    resources = spec["resources"]
    counts = Counter(_resource_type(address) for address in resources.values())
    features: set[Feature] = set()
    for resource_type, count in counts.items():
        for threshold in range(1, count + 1):
            features.add(Feature("resource", resource_type, str(threshold)))
    for source, dependencies in spec["topology"].items():
        source_type = _resource_type(resources[source])
        for dependency in dependencies:
            dependency_type = _resource_type(resources[dependency])
            features.add(Feature("topology", source_type, dependency_type))
    for label, values in spec["attributes"].items():
        resource_type = _resource_type(resources[label])
        for key in values:
            features.add(Feature("attribute", resource_type, key))
    return frozenset(features)


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log(1.0 - probability)


def feature_entropy(
    particles: Sequence[frozenset[Feature]],
    feature: Feature,
) -> float:
    if not particles:
        return 0.0
    probability = sum(feature in particle for particle in particles) / len(particles)
    return binary_entropy(probability)


def varying_features(
    particles: Sequence[frozenset[Feature]],
    *,
    exclude: Iterable[Feature] = (),
) -> list[tuple[Feature, float]]:
    excluded = set(exclude)
    universe = set().union(*particles) if particles else set()
    rows = [
        (feature, feature_entropy(particles, feature))
        for feature in universe
        if feature not in excluded
    ]
    return sorted(rows, key=lambda row: (-row[1], row[0].key()))


def select_root_features(
    particles: Sequence[frozenset[Feature]],
    count: int = ROOT_COUNT,
) -> list[Feature]:
    rows = [row for row in varying_features(particles) if row[1] > 0.0]
    selected: list[Feature] = []
    for dimension in ("resource", "topology", "attribute"):
        match = next((feature for feature, _ in rows if feature.dimension == dimension), None)
        if match is not None:
            selected.append(match)
    for feature, _ in rows:
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= count:
            break
    return selected[:count]


def best_followup(
    particles: Sequence[frozenset[Feature]],
    *,
    exclude: Iterable[Feature] = (),
) -> tuple[Feature | None, float]:
    for feature, entropy in varying_features(particles, exclude=exclude):
        if entropy > 0.0:
            return feature, entropy
    return None, 0.0


def fixed_depth_two_score(
    particles: Sequence[frozenset[Feature]],
    root: Feature,
) -> float:
    immediate = feature_entropy(particles, root)
    continuation = 0.0
    for answer in (False, True):
        branch = [particle for particle in particles if (root in particle) is answer]
        if not branch:
            continue
        probability = len(branch) / len(particles)
        _, entropy = best_followup(branch, exclude=(root,))
        continuation += probability * entropy
    return immediate + continuation


def regenerated_depth_two_score(
    particles: Sequence[frozenset[Feature]],
    root: Feature,
    branches: dict[bool, Sequence[frozenset[Feature]]],
) -> float:
    immediate = feature_entropy(particles, root)
    probability_true = sum(root in particle for particle in particles) / len(particles)
    continuation = 0.0
    for answer, probability in (
        (False, 1.0 - probability_true),
        (True, probability_true),
    ):
        _, entropy = best_followup(branches[answer], exclude=(root,))
        continuation += probability * entropy
    return immediate + continuation


def _f1(left: set[Feature], right: set[Feature]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return 2.0 * overlap / (len(left) + len(right))


def spec_similarity(
    left: frozenset[Feature],
    right: frozenset[Feature],
) -> dict[str, float]:
    by_dimension = {}
    for dimension in ("resource", "topology", "attribute"):
        left_values = {value for value in left if value.dimension == dimension}
        right_values = {value for value in right if value.dimension == dimension}
        by_dimension[dimension] = _f1(left_values, right_values)
    by_dimension["combined"] = statistics.fmean(by_dimension.values())
    return by_dimension


def medoid_index(particles: Sequence[frozenset[Feature]]) -> int | None:
    if not particles:
        return None
    scores = []
    for index, particle in enumerate(particles):
        mean_similarity = statistics.fmean(
            spec_similarity(particle, other)["combined"] for other in particles
        )
        scores.append((mean_similarity, -index))
    return -max(scores)[1]


def realized_endpoint(
    target: frozenset[Feature],
    root: Feature,
    branches: dict[bool, Sequence[frozenset[Feature]]],
) -> dict[str, Any]:
    root_answer = root in target
    branch = list(branches[root_answer])
    followup, _ = best_followup(branch, exclude=(root,))
    followup_answer = None if followup is None else followup in target
    posterior = (
        branch
        if followup is None
        else [
            particle
            for particle in branch
            if (followup in particle) is followup_answer
        ]
    )
    selected_index = medoid_index(posterior)
    selected = None if selected_index is None else posterior[selected_index]
    selected_scores = (
        {"resource": 0.0, "topology": 0.0, "attribute": 0.0, "combined": 0.0}
        if selected is None
        else spec_similarity(selected, target)
    )
    mean_score = (
        0.0
        if not posterior
        else statistics.fmean(
            spec_similarity(particle, target)["combined"] for particle in posterior
        )
    )
    return {
        "root_answer": root_answer,
        "followup": None if followup is None else followup.as_dict(),
        "followup_answer": followup_answer,
        "posterior_particle_count": len(posterior),
        "posterior_empty": not posterior,
        "selected_similarity": selected_scores,
        "mean_particle_similarity": mean_score,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    ranks_left = _average_ranks(left)
    ranks_right = _average_ranks(right)
    mean_left = statistics.fmean(ranks_left)
    mean_right = statistics.fmean(ranks_right)
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(ranks_left, ranks_right)
    )
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in ranks_left)
        * sum((b - mean_right) ** 2 for b in ranks_right)
    )
    return None if denominator <= 0.0 else numerator / denominator


def _particle_messages(
    *,
    task_id: int,
    prompt: str,
    sample_index: int,
    history: Sequence[tuple[Feature, bool]],
) -> list[dict[str, str]]:
    history_rows = [
        {
            "feature": feature.as_dict(),
            "question": feature.question(),
            "answer": "yes" if answer else "no",
        }
        for feature, answer in history
    ]
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_request": prompt,
        "clarification_history": history_rows,
        "interpretation_sample": sample_index,
        "required_output": {
            "resources": {"short_label": "terraform_resource_type.instance_name"},
            "topology": {"source_label": ["dependency_label"]},
            "attributes": {"resource_label": {"attribute_name": "value"}},
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate one concrete plausible AWS Terraform specification for "
                "an underspecified request. Treat clarification answers as binding. "
                "Different interpretation_sample values should explore materially "
                "different plausible resource, dependency, or attribute choices. "
                "Use real aws_* or awscc_* Terraform addresses. Return exactly one "
                "JSON object with resources, topology, and attributes and no prose."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _parse_population(
    responses: Sequence[str],
    *,
    history: Sequence[tuple[Feature, bool]] = (),
) -> tuple[list[dict[str, dict[str, Any]]], list[dict[str, str]]]:
    particles = []
    errors = []
    for index, response in enumerate(responses):
        try:
            spec = parse_generated_spec(response)
            features = spec_features(spec)
            if any((feature in features) is not answer for feature, answer in history):
                raise ValueError("generated spec contradicts clarification history")
            particles.append(spec)
        except Exception as exc:
            errors.append(
                {"sample_index": str(index), "error": f"{type(exc).__name__}: {exc}"}
            )
    if len(particles) < MIN_VALID_PARTICLES:
        raise ValueError(
            f"population has {len(particles)} valid particles; "
            f"{MIN_VALID_PARTICLES} required"
        )
    return particles, errors


def _configuration_resources(root_module: dict[str, Any]) -> list[dict[str, Any]]:
    resources = list(root_module.get("resources", []))
    for module_call in (root_module.get("module_calls") or {}).values():
        module = module_call.get("module") or {}
        resources.extend(_configuration_resources(module))
    return resources


def _references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "references" and isinstance(nested, list):
                yield from (item for item in nested if isinstance(item, str))
            else:
                yield from _references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _references(nested)


def _reference_address(reference: str) -> str:
    parts = reference.split(".")
    if parts and parts[0] == "data":
        return ".".join(parts[:3])
    return ".".join(parts[:2])


def target_spec_from_plan(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root_module = (payload.get("configuration") or {}).get("root_module") or {}
    rows = _configuration_resources(root_module)
    addresses = {
        str(row["address"]): str(row["address"])
        for row in rows
        if isinstance(row.get("address"), str)
        and ADDRESS_PATTERN.fullmatch(str(row["address"])) is not None
    }
    topology: dict[str, list[str]] = {}
    attributes: dict[str, dict[str, Any]] = {}
    for row in rows:
        address = str(row.get("address", ""))
        if address not in addresses:
            continue
        dependencies = set()
        for dependency in row.get("depends_on", []) or []:
            if dependency in addresses and dependency != address:
                dependencies.add(dependency)
        expressions = row.get("expressions") or {}
        for reference in _references(expressions):
            dependency = _reference_address(reference)
            if dependency in addresses and dependency != address:
                dependencies.add(dependency)
        if dependencies:
            topology[address] = sorted(dependencies)
        explicit_attributes = {
            key: True
            for key in expressions
            if isinstance(key, str) and key.strip()
        }
        if explicit_attributes:
            attributes[address] = explicit_attributes
    if not addresses:
        raise ValueError(f"target plan {path} has no supported resources")
    return {
        "resources": addresses,
        "topology": topology,
        "attributes": attributes,
    }


def _sha256_dataset_tree(dataset_root: Path) -> str:
    content_hashes = []
    for path in sorted(value for value in dataset_root.rglob("*") if value.is_file()):
        content_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256("".join(content_hashes).encode()).hexdigest()


def verify_source(source_root: Path) -> Path:
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Ambig-IaC commit is {commit}, expected {SOURCE_COMMIT}")
    dataset_root = source_root / "datasets" / "ambig-iac"
    if _sha256_dataset_tree(dataset_root) != DATASET_TREE_SHA256:
        raise ValueError("Ambig-IaC dataset tree hash does not match")
    if sorted(int(path.name) for path in dataset_root.iterdir() if path.is_dir()) != list(
        range(300)
    ):
        raise ValueError("Ambig-IaC task manifest does not match 0..299")
    return dataset_root


def _usage_snapshot(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "generator": snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def run_smoke(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    model_adapter: ChatModel,
) -> dict[str, Any]:
    dataset_root = verify_source(source_root)
    prompts = {
        task_id: (dataset_root / str(task_id) / "prompt.txt").read_text(
            encoding="utf-8"
        )
        for task_id in TASK_IDS
    }
    raw: dict[str, Any] = {"task_ids": list(TASK_IDS)}
    initial_messages = [
        _particle_messages(
            task_id=task_id,
            prompt=prompts[task_id],
            sample_index=sample_index,
            history=(),
        )
        for task_id in TASK_IDS
        for sample_index in range(PARTICLE_COUNT)
    ]
    try:
        initial_raw = model_adapter.chat_complete_messages_batched(
            initial_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        initial_specs = {}
        initial_errors = {}
        root_features = {}
        for task_index, task_id in enumerate(TASK_IDS):
            start = task_index * PARTICLE_COUNT
            specs, errors = _parse_population(
                initial_raw[start : start + PARTICLE_COUNT]
            )
            particles = [spec_features(spec) for spec in specs]
            roots = select_root_features(particles)
            if len(roots) != ROOT_COUNT:
                raise ValueError(
                    f"task {task_id} produced {len(roots)} roots; {ROOT_COUNT} required"
                )
            initial_specs[task_id] = specs
            initial_errors[task_id] = errors
            root_features[task_id] = roots

        branch_messages = []
        branch_manifest = []
        for task_id in TASK_IDS:
            for root_index, root in enumerate(root_features[task_id]):
                for answer in (False, True):
                    for sample_index in range(PARTICLE_COUNT):
                        branch_messages.append(
                            _particle_messages(
                                task_id=task_id,
                                prompt=prompts[task_id],
                                sample_index=sample_index,
                                history=((root, answer),),
                            )
                        )
                        branch_manifest.append(
                            (task_id, root_index, answer, sample_index)
                        )
        branch_raw = model_adapter.chat_complete_messages_batched(
            branch_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["branches"] = branch_raw
        raw["branch_manifest"] = [
            {
                "task_id": task_id,
                "root_index": root_index,
                "answer": answer,
                "sample_index": sample_index,
            }
            for task_id, root_index, answer, sample_index in branch_manifest
        ]
        grouped_raw: dict[tuple[int, int, bool], list[str]] = {}
        for manifest, response in zip(branch_manifest, branch_raw, strict=True):
            grouped_raw.setdefault(manifest[:3], []).append(response)
        branch_specs = {}
        branch_errors = {}
        for key, responses in grouped_raw.items():
            task_id, root_index, answer = key
            specs, errors = _parse_population(
                responses,
                history=((root_features[task_id][root_index], answer),),
            )
            branch_specs[key] = specs
            branch_errors[key] = errors
        _checkpoint(raw_path, raw)
        usage = _usage_snapshot(model_adapter)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model_adapter),
        ) from exc

    # All target-blind model calls and scores are frozen before target plans load.
    target_blind = {}
    for task_id in TASK_IDS:
        particles = [spec_features(spec) for spec in initial_specs[task_id]]
        roots = root_features[task_id]
        rows = []
        for root_index, root in enumerate(roots):
            branches = {
                answer: [
                    spec_features(spec)
                    for spec in branch_specs[(task_id, root_index, answer)]
                ]
                for answer in (False, True)
            }
            rows.append(
                {
                    "root": root,
                    "branches": branches,
                    "immediate_eig": feature_entropy(particles, root),
                    "fixed_depth_two_eig": fixed_depth_two_score(particles, root),
                    "regenerated_depth_two_eig": regenerated_depth_two_score(
                        particles, root, branches
                    ),
                }
            )
        target_blind[task_id] = rows

    task_records = []
    for task_id in TASK_IDS:
        target = spec_features(
            target_spec_from_plan(dataset_root / str(task_id) / "plan.json")
        )
        rows = target_blind[task_id]
        endpoints = [
            realized_endpoint(target, row["root"], row["branches"]) for row in rows
        ]
        immediate = [row["immediate_eig"] for row in rows]
        fixed = [row["fixed_depth_two_eig"] for row in rows]
        regenerated = [row["regenerated_depth_two_eig"] for row in rows]
        endpoint_scores = [
            endpoint["selected_similarity"]["combined"] for endpoint in endpoints
        ]
        selections = {
            "myopic": _argmax(immediate),
            "fixed_depth_two": _argmax(fixed),
            "regenerated_depth_two": _argmax(regenerated),
        }
        task_records.append(
            {
                "task_id": task_id,
                "valid_initial_particles": len(initial_specs[task_id]),
                "invalid_initial_particles": initial_errors[task_id],
                "root_dimension_count": len(
                    {row["root"].dimension for row in rows}
                ),
                "roots": [
                    {
                        "root_index": index,
                        "feature": row["root"].as_dict(),
                        "immediate_eig": row["immediate_eig"],
                        "fixed_depth_two_eig": row["fixed_depth_two_eig"],
                        "regenerated_depth_two_eig": row[
                            "regenerated_depth_two_eig"
                        ],
                        "branch_valid_particles": {
                            str(answer).lower(): len(
                                branch_specs[(task_id, index, answer)]
                            )
                            for answer in (False, True)
                        },
                        "branch_invalid_particles": {
                            str(answer).lower(): branch_errors[
                                (task_id, index, answer)
                            ]
                            for answer in (False, True)
                        },
                        "endpoint": endpoints[index],
                    }
                    for index, row in enumerate(rows)
                ],
                "selections": {
                    name: {
                        "root_index": index,
                        "endpoint": endpoint_scores[index],
                    }
                    for name, index in selections.items()
                },
                "endpoint_range": max(endpoint_scores) - min(endpoint_scores),
                "spearman": {
                    "myopic": spearman(immediate, endpoint_scores),
                    "fixed_depth_two": spearman(fixed, endpoint_scores),
                    "regenerated_depth_two": spearman(
                        regenerated, endpoint_scores
                    ),
                },
            }
        )

    regenerated_rhos = [
        row["spearman"]["regenerated_depth_two"]
        for row in task_records
        if row["spearman"]["regenerated_depth_two"] is not None
    ]
    myopic_rhos = [
        row["spearman"]["myopic"]
        for row in task_records
        if row["spearman"]["myopic"] is not None
    ]
    mean_regenerated_rho = (
        statistics.fmean(regenerated_rhos) if regenerated_rhos else None
    )
    mean_myopic_rho = statistics.fmean(myopic_rhos) if myopic_rhos else None
    mean_regenerated_endpoint = statistics.fmean(
        row["selections"]["regenerated_depth_two"]["endpoint"]
        for row in task_records
    )
    mean_myopic_endpoint = statistics.fmean(
        row["selections"]["myopic"]["endpoint"] for row in task_records
    )
    mean_fixed_endpoint = statistics.fmean(
        row["selections"]["fixed_depth_two"]["endpoint"]
        for row in task_records
    )
    root_change_count = sum(
        row["selections"]["regenerated_depth_two"]["root_index"]
        != row["selections"]["myopic"]["root_index"]
        for row in task_records
    )
    dynamic_range_count = sum(row["endpoint_range"] >= 0.05 for row in task_records)
    selected_empty_count = sum(
        row["roots"][row["selections"]["regenerated_depth_two"]["root_index"]][
            "endpoint"
        ]["posterior_empty"]
        for row in task_records
    )
    generator = usage["generator"]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": int(generator.get("http_attempts", -1))
        == EXPECTED_REQUESTS,
        "zero_transport_retries": int(generator.get("retry_count", -1)) == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_populations_have_at_least_four_valid_particles": all(
            root["branch_valid_particles"][str(answer).lower()]
            >= MIN_VALID_PARTICLES
            for task in task_records
            for root in task["roots"]
            for answer in (False, True)
        )
        and all(
            task["valid_initial_particles"] >= MIN_VALID_PARTICLES
            for task in task_records
        ),
        "at_least_two_root_dimensions_each": all(
            task["root_dimension_count"] >= 2 for task in task_records
        ),
        "endpoint_dynamic_range_at_least_two_tasks": dynamic_range_count >= 2,
        "regenerated_root_differs_from_myopic_at_least_two_tasks": (
            root_change_count >= 2
        ),
        "mean_regenerated_spearman_at_least_0_20": (
            mean_regenerated_rho is not None and mean_regenerated_rho >= 0.20
        ),
        "regenerated_spearman_gain_over_myopic_at_least_0_10": (
            mean_regenerated_rho is not None
            and mean_myopic_rho is not None
            and mean_regenerated_rho - mean_myopic_rho >= 0.10
        ),
        "regenerated_endpoint_gain_over_myopic_at_least_0_03": (
            mean_regenerated_endpoint - mean_myopic_endpoint >= 0.03
        ),
        "regenerated_endpoint_gain_over_fixed_at_least_0_02": (
            mean_regenerated_endpoint - mean_fixed_endpoint >= 0.02
        ),
        "no_regenerated_selected_posterior_empty": selected_empty_count == 0,
        "cost_at_most_1_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "dataset_tree_sha256": DATASET_TREE_SHA256,
            "model": MODEL_ID,
            "seed": SEED,
            "task_ids": list(TASK_IDS),
            "particle_count": PARTICLE_COUNT,
            "root_count": ROOT_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "target_plan_loaded_after_all_model_calls_and_scores": True,
            "exact_target_answers": True,
            "repairs_or_scientific_retries": 0,
        },
        "metrics": {
            "task_count": len(task_records),
            "dynamic_range_task_count": dynamic_range_count,
            "regenerated_root_change_count": root_change_count,
            "mean_regenerated_spearman": mean_regenerated_rho,
            "mean_myopic_spearman": mean_myopic_rho,
            "mean_regenerated_endpoint": mean_regenerated_endpoint,
            "mean_myopic_endpoint": mean_myopic_endpoint,
            "mean_fixed_endpoint": mean_fixed_endpoint,
            "mean_regenerated_endpoint_gain_over_myopic": (
                mean_regenerated_endpoint - mean_myopic_endpoint
            ),
            "mean_regenerated_endpoint_gain_over_fixed": (
                mean_regenerated_endpoint - mean_fixed_endpoint
            ),
            "selected_empty_posterior_count": selected_empty_count,
        },
        "gates": gates,
        "tasks": task_records,
        "usage": usage,
    }


class DeterministicFixtureModel:
    """Small local model for parser and orchestration checks."""

    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            index = int(request["interpretation_sample"])
            history = request["clarification_history"]
            use_vpc = index % 2 == 0
            if history and "aws_vpc" in history[0]["question"]:
                use_vpc = history[0]["answer"] == "yes"
            resources = {"store": "aws_s3_bucket.store"}
            topology: dict[str, list[str]] = {}
            attributes: dict[str, dict[str, Any]] = {
                "store": {"versioning": True if index % 3 else False}
            }
            if use_vpc:
                resources["network"] = "aws_vpc.network"
                resources["endpoint"] = "aws_vpc_endpoint.endpoint"
                topology["endpoint"] = ["network", "store"]
                attributes["network"] = {"cidr_block": "10.0.0.0/16"}
            elif index % 3 == 0:
                resources["logs"] = "aws_cloudwatch_log_group.logs"
                topology["logs"] = ["store"]
            if history:
                feature = Feature(
                    history[0]["feature"]["dimension"],
                    history[0]["feature"]["subject"],
                    history[0]["feature"]["value"],
                )
                answer = history[0]["answer"] == "yes"
                if feature.dimension == "resource":
                    matching = [
                        label
                        for label, address in resources.items()
                        if _resource_type(address) == feature.subject
                    ]
                    target_count = int(feature.value) if answer else int(feature.value) - 1
                    while len(matching) < target_count:
                        label = f"fixture_{len(resources)}"
                        resources[label] = f"{feature.subject}.{label}"
                        matching.append(label)
                    for label in matching[target_count:]:
                        resources.pop(label)
                        topology.pop(label, None)
                        attributes.pop(label, None)
                        for dependencies in topology.values():
                            while label in dependencies:
                                dependencies.remove(label)
                elif feature.dimension == "topology":
                    source = next(
                        (
                            label
                            for label, address in resources.items()
                            if _resource_type(address) == feature.subject
                        ),
                        None,
                    )
                    dependency = next(
                        (
                            label
                            for label, address in resources.items()
                            if _resource_type(address) == feature.value
                        ),
                        None,
                    )
                    if answer:
                        if source is None:
                            source = f"fixture_source_{len(resources)}"
                            resources[source] = f"{feature.subject}.{source}"
                        if dependency is None:
                            dependency = f"fixture_dependency_{len(resources)}"
                            resources[dependency] = f"{feature.value}.{dependency}"
                        topology.setdefault(source, []).append(dependency)
                    else:
                        for source_label, dependencies in topology.items():
                            if _resource_type(resources[source_label]) != feature.subject:
                                continue
                            topology[source_label] = [
                                dependency_label
                                for dependency_label in dependencies
                                if _resource_type(resources[dependency_label])
                                != feature.value
                            ]
                elif feature.dimension == "attribute":
                    matching = [
                        label
                        for label, address in resources.items()
                        if _resource_type(address) == feature.subject
                    ]
                    if answer and not matching:
                        label = f"fixture_attribute_{len(resources)}"
                        resources[label] = f"{feature.subject}.{label}"
                        matching.append(label)
                    for label in matching:
                        if answer:
                            attributes.setdefault(label, {})[feature.value] = True
                        else:
                            attributes.setdefault(label, {}).pop(feature.value, None)
            responses.append(
                json.dumps(
                    {
                        "resources": resources,
                        "topology": topology,
                        "attributes": attributes,
                    }
                )
            )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Ambig-IaC smoke config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.75
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_smoke(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            model_adapter=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, SmokeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SMOKE_FAILURE.json", failure)
        raise
    output = args.output_dir / "SMOKE.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
