#!/usr/bin/env python3
"""Independently replay the RegretBench SMC depth-two policy artifacts.

The verifier imports only the pre-existing independent RegretBench formula
replay. It does not import the SMC policy core or producer and makes no model
calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import regretbench_deepseek_result_verify as base
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-dynamic-depth2-verify-1"
PRODUCER_INTERFACE = "regretbench-deepseek-smc-dynamic-depth2-experiment-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PROTOCOL_SHA256 = (
    "97e9e7a582d38ae989042352755ee500ae4a42726dd6a431309a8e66102dd3cd"
)
PRODUCER_SHA256 = (
    "dd0bf57bcbf598721883add6900957ab4f7203765dd13d8be7b8a5c832bc16bd"
)
PARTICLES = 8
QUESTIONS = 4
BRANCH_DRAWS = 2
MIN_RETAINED = 2
MAX_RETAINED = 6
PLANNING_REQUESTS = 8_256
ANNOTATION_SEED_START = 202608300000
BRANCH_SEED_START = 202608310000
TRUTH_SEED_START = 202608320000
RANDOM_SEED_START = 202608330000
ACTUAL_FIRST_SEED_START = 202608340000
ACTUAL_FINAL_SEED_START = 202608350000
BOOTSTRAP_SEED = 202608360000
NAIVE_FIRST_SEED_START = 202608370000
NAIVE_SECOND_SEED_START = 202608380000
NAIVE_FIRST_SUPPORT_SEED_START = 202608390000
NAIVE_FINAL_SUPPORT_SEED_START = 202608400000
PROBABILITY_FLOOR = 1e-12


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _close(left: Any, right: Any, path: str, mismatches: list[str]) -> None:
    base._close(left, right, path, mismatches)


def _questions(values: Any) -> list[str]:
    if not isinstance(values, list) or len(values) != QUESTIONS:
        raise ValueError("support must contain four questions")
    output = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value.strip().endswith("?"):
            raise ValueError("support contains a non-question")
        text = value.strip()
        key = base.normalize_text(text)
        if not key or key in seen:
            raise ValueError("support contains duplicate questions")
        seen.add(key)
        output.append(text)
    return output


def _replies(values: Any) -> list[str]:
    if not isinstance(values, list) or len(values) != QUESTIONS:
        raise ValueError("particle must contain four predicted replies")
    output = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("particle contains an empty reply")
        output.append(value.strip())
    return output


def _parse_parent(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("parent response has wrong top-level fields")
    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != PARTICLES:
        raise ValueError("parent response must contain eight raw slots")
    rows = []
    for index, item in enumerate(hypotheses):
        if not isinstance(item, dict) or set(item) != {
            "interpretation",
            "final_answer",
            "prior_weight",
        }:
            raise ValueError("parent particle has wrong fields")
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            raise ValueError("parent particle has invalid values")
        rows.append(
            {
                "parent_index": index,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
                "probability": float(weight),
            }
        )
    total = sum(row["probability"] for row in rows)
    if total <= 0:
        raise ValueError("parent weights sum to zero")
    for row in rows:
        row["probability"] /= total
    return {
        "hypotheses": rows,
        "questions": _questions(value["questions"]),
        "raw_parent_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def _parse_annotation(raw: str, parent: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("annotation has wrong top-level fields")
    values = value["particles"]
    if not isinstance(values, list) or len(values) != PARTICLES:
        raise ValueError("annotation must contain eight records")
    by_index = {}
    for item in values:
        if not isinstance(item, dict) or set(item) != {
            "parent_index",
            "predicted_replies",
        }:
            raise ValueError("annotation record has wrong fields")
        index = item["parent_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(PARTICLES)
            or index in by_index
        ):
            raise ValueError("annotation lineage is not an exact permutation")
        by_index[index] = _replies(item["predicted_replies"])
    if sorted(by_index) != list(range(PARTICLES)):
        raise ValueError("annotation lineage is not an exact permutation")
    support = {
        "hypotheses": [
            {**row, "predicted_replies": by_index[index]}
            for index, row in enumerate(parent["hypotheses"])
        ],
        "questions": list(parent["questions"]),
        "diagnostic": {
            "codec_mode": "strict_json",
            "parent_index_permutation_exact": True,
            "particle_count": PARTICLES,
            "question_count": QUESTIONS,
            "parent_population_sha256": parent["raw_parent_sha256"],
            "initial_hypotheses_regenerated": False,
            "initial_questions_regenerated": False,
        },
    }
    return support


def _support_hash(support: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "hypotheses": support["hypotheses"],
                "questions": support["questions"],
            }
        ).encode()
    ).hexdigest()


def _public_parents(support: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = support["hypotheses"]
    output = []
    for index, row in enumerate(rows):
        value = {
            "parent_index": index,
            "interpretation": row["interpretation"],
            "final_answer": row["final_answer"],
            "probability": float(row["probability"]),
        }
        if "predicted_replies" in row:
            value["predicted_replies"] = list(row["predicted_replies"])
        output.append(value)
    total = sum(row["probability"] for row in output)
    if total <= 0:
        raise ValueError("support probabilities sum to zero")
    for row in output:
        row["probability"] /= total
    return output


def _parse_transition(raw: str, parent: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("transition has wrong top-level fields")
    values = value["hypotheses"]
    if not isinstance(values, list) or len(values) != PARTICLES:
        raise ValueError("transition must contain eight children")
    parents = _public_parents(parent)
    rows = []
    indexes = []
    seen = set()
    retained = 0
    for item in values:
        if not isinstance(item, dict) or set(item) != {
            "parent_index",
            "revision_type",
            "interpretation",
            "final_answer",
            "prior_weight",
            "predicted_replies",
        }:
            raise ValueError("transition child has wrong fields")
        index = item["parent_index"]
        revision = item["revision_type"]
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(PARTICLES)
            or revision not in {"retained", "revised"}
            or not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            raise ValueError("transition child has invalid values")
        unchanged = (
            base.normalize_text(interpretation)
            == base.normalize_text(parents[index]["interpretation"])
            and base.normalize_text(answer)
            == base.normalize_text(parents[index]["final_answer"])
        )
        if (revision == "retained") != unchanged:
            raise ValueError("transition revision label is false")
        retained += revision == "retained"
        key = (base.normalize_text(interpretation), base.normalize_text(answer))
        if key in seen:
            raise ValueError("transition contains duplicate children")
        seen.add(key)
        indexes.append(index)
        rows.append(
            {
                "parent_index": index,
                "revision_type": revision,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
                "probability": float(weight),
                "predicted_replies": _replies(item["predicted_replies"]),
            }
        )
    if sorted(indexes) != list(range(PARTICLES)):
        raise ValueError("transition lineage is not an exact permutation")
    if not MIN_RETAINED <= retained <= MAX_RETAINED:
        raise ValueError("transition retention count is outside the frozen range")
    total = sum(row["probability"] for row in rows)
    if total <= 0:
        raise ValueError("transition weights sum to zero")
    for row in rows:
        row["probability"] /= total
    return {
        "hypotheses": rows,
        "questions": _questions(value["questions"]),
        "diagnostic": {
            "codec_mode": "strict_json",
            "valid_unique_count": PARTICLES,
            "parent_index_permutation_exact": True,
            "retained_count": retained,
            "revised_count": PARTICLES - retained,
            "question_count": QUESTIONS,
            "parent_support_sha256": _support_hash(parent),
        },
    }


def _posterior(support: Mapping[str, Any], question: int, reply: str) -> tuple[dict[str, Any], bool]:
    rows = _public_parents(support)
    observed = base.normalize_text(reply)
    matched = [
        base.normalize_text(row["predicted_replies"][question]) == observed
        for row in rows
    ]
    mass = sum(row["probability"] for row, keep in zip(rows, matched) if keep)
    output = []
    for row, keep in zip(rows, matched):
        probability = row["probability"]
        if mass > 0:
            probability = row["probability"] / mass if keep else 0.0
        output.append({**row, "probability": probability})
    return {"hypotheses": output, "questions": list(support["questions"])}, mass > 0


def _choose(
    initial: Mapping[str, Any],
    conditioned: Sequence[Mapping[str, float]],
    blind: Sequence[Mapping[str, float]],
    refresh: Sequence[Mapping[str, float]],
    myopic: Sequence[Mapping[str, float]],
    fixed: Sequence[Mapping[str, float]],
    task: int,
) -> dict[str, int]:
    selected = {
        "smc_dynamic_depth2": min(range(4), key=lambda i: (conditioned[i]["brier"], i)),
        "smc_myopic_refresh_brier": min(range(4), key=lambda i: (refresh[i]["brier"], i)),
        "smc_myopic_brier": min(range(4), key=lambda i: (myopic[i]["brier"], i)),
        "smc_history_blind_depth2": min(range(4), key=lambda i: (blind[i]["brier"], i)),
        "smc_myopic_width": min(range(4), key=lambda i: (-base._question_eig(initial, i), i)),
        "smc_fixed_depth2": min(range(4), key=lambda i: (fixed[i]["brier"], i)),
        "random": random.Random(RANDOM_SEED_START + task).randrange(4),
    }
    return selected


def _scorer_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in tasks:
        output.append(
            {
                **task,
                "selected_roots": {
                    (name[4:] if name.startswith("smc_") else name): root
                    for name, root in task["selected_roots"].items()
                },
                "policies": {
                    (name[4:] if name.startswith("smc_") else name): row
                    for name, row in task["policies"].items()
                },
            }
        )
    return output


def _science(tasks: Sequence[Mapping[str, Any]], samples: int) -> dict[str, Any]:
    prior = base.BOOTSTRAP_SEED
    base.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    try:
        value = base._policy_science(_scorer_tasks(tasks), samples)
    finally:
        base.BOOTSTRAP_SEED = prior
    for field in ("comparisons", "root_disagreements"):
        value[field] = {
            (name if name == "random" else f"smc_{name}"): row
            for name, row in value[field].items()
        }
    return value


def _annotation_audit(cig: Any, parent: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "task_id": cig.cig_id,
        "prompt": cig.prompt,
        "dialogue": [],
        "parent_particles": _public_parents(parent),
        "questions": list(parent["questions"]),
        "parent_population_sha256": parent["raw_parent_sha256"],
        "parent_source": "verified_primary_raw_root",
    }
    return {
        "passed": True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "parent_population_sha256": parent["raw_parent_sha256"],
        "parent_source": "verified_primary_raw_root",
    }


def _transition_audit(cig: Any, dialogue: Sequence[Mapping[str, str]], parent: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "task_id": cig.cig_id,
        "prompt": cig.prompt,
        "dialogue": list(dialogue),
        "parent_particles": _public_parents(parent),
        "parent_questions": list(parent["questions"]),
        "parent_support_sha256": _support_hash(parent),
        "parent_source": "model_generated_semantic_particles",
    }
    return {
        "passed": True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "parent_support_sha256": _support_hash(parent),
        "parent_source": "model_generated_semantic_particles",
    }


def _naive_audit(cig: Any, dialogue: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    payload = {"task_id": cig.cig_id, "prompt": cig.prompt, "dialogue": list(dialogue)}
    return {
        "passed": True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
    }


def verify(run_dir: Path, *, primary_dir: Path) -> dict[str, Any]:
    result = _load(run_dir / "RESULT.json")
    initial_artifact = _load(run_dir / "private/RAW_INITIAL.json")
    branch_artifact = _load(run_dir / "private/RAW_BRANCHES.json")
    actual = _load(run_dir / "private/RAW_ACTUAL_PRIMARY.json")
    frozen = _load(run_dir / "private/FROZEN_SELECTIONS.json")
    controls = _load(run_dir / "private/CONTROLS_PRIMARY.json")
    privacy_artifact = _load(run_dir / "private/PRIVACY.json")
    parent_raw = _load(primary_dir / "private/RAW_RESPONSES.json")
    cigs = base._stage_cigs("development")
    raw_parents = parent_raw.get("root") or []
    if len(raw_parents) != 64:
        raise ValueError("primary parent cohort changed")
    parents = [_parse_parent(raw) for raw in raw_parents]
    raw_initial = initial_artifact.get("responses") or []
    if initial_artifact.get("seeds") != [ANNOTATION_SEED_START + i for i in range(64)]:
        raise ValueError("annotation seed schedule changed")
    initial = [
        _parse_annotation(raw, parents[index])
        for index, raw in enumerate(raw_initial)
    ]
    if len(initial) != 64:
        raise ValueError("annotation response count changed")
    primary_privacy = [
        _annotation_audit(cig, parent)
        for cig, parent in zip(cigs, parents, strict=True)
    ]

    manifests = branch_artifact.get("manifest") or []
    paired_seeds = branch_artifact.get("paired_seeds") or []
    raw_branches = branch_artifact.get("responses") or []
    if len(manifests) != 4_096 or len(raw_branches) != 8_192 or len(paired_seeds) != 8_192:
        raise ValueError("planning branch schedule changed")
    branches: list[list[dict[str, Any]]] = [[] for _ in range(64)]
    transitions = []
    branch_schedule = True
    for index, manifest in enumerate(manifests):
        task = int(manifest["task_index"])
        root = int(manifest["root_index"])
        particle = int(manifest["hypothesis_index"])
        draw = int(manifest["draw"])
        seed = BRANCH_SEED_START + task * 16 + particle * 2 + draw
        branch_schedule &= (
            int(manifest["seed"]) == seed
            and int(manifest["conditioned_dispatch_index"]) == 2 * index
            and int(manifest["blind_dispatch_index"]) == 2 * index + 1
            and paired_seeds[2 * index] == seed
            and paired_seeds[2 * index + 1] == seed
        )
        conditioned = _parse_transition(raw_branches[2 * index], initial[task])
        blind = _parse_transition(raw_branches[2 * index + 1], initial[task])
        transitions.extend([conditioned, blind])
        branches[task].append(
            {**manifest, "conditioned": conditioned, "blind": blind}
        )
        dialogue = [
            {"role": "assistant", "content": initial[task]["questions"][root]},
            {
                "role": "user",
                "content": initial[task]["hypotheses"][particle][
                    "predicted_replies"
                ][root],
            },
        ]
        primary_privacy.extend(
            [
                _transition_audit(cigs[task], dialogue, initial[task]),
                _transition_audit(cigs[task], [], initial[task]),
            ]
        )
    plans = []
    for task, support in enumerate(initial):
        conditioned = base._dynamic_risks(support, branches[task], "conditioned")
        by_draw = [
            base._dynamic_risks_for_draw(support, branches[task], "conditioned", draw)
            for draw in range(BRANCH_DRAWS)
        ]
        blind = base._dynamic_risks(support, branches[task], "blind")
        refresh = base._myopic_refresh_brier_risks(support, branches[task])
        myopic = base._myopic_brier_risks(support)
        fixed = base._fixed_risks(support)
        plans.append(
            {
                "conditioned": conditioned,
                "conditioned_by_draw": by_draw,
                "selection_stability": base._selection_stability(conditioned, by_draw),
                "blind": blind,
                "myopic_refresh_brier": refresh,
                "myopic_brier": myopic,
                "fixed": fixed,
                "root_eig": [base._question_eig(support, i) for i in range(4)],
                "selected": _choose(support, conditioned, blind, refresh, myopic, fixed, task),
            }
        )
    if frozen.get("selected_roots") != [row["selected"] for row in plans] or frozen.get("hidden_truth_accessed") is not False:
        raise ValueError("frozen SMC selections do not replay")

    truth_rows = []
    control_by_id = {row["task_id"]: row for row in controls.get("tasks", [])}
    for task, cig in enumerate(cigs):
        truth_index, truth = base._truth(cig, TRUTH_SEED_START + task)
        aliases = str((truth.slots or {})["answer_aliases"])
        control = control_by_id.get(cig.cig_id)
        if control is None or control.get("truth_index") != truth_index or control.get("aliases") != aliases:
            raise ValueError("private SMC truth control changed")
        truth_rows.append((truth_index, truth, aliases))
    first_paths = {}
    actual_supports = []
    first_manifest = actual.get("first_manifest") or []
    first_responses = actual.get("first_responses") or []
    for manifest, raw in zip(first_manifest, first_responses, strict=True):
        task = int(manifest["task_index"])
        root = int(manifest["root_index"])
        if int(manifest["seed"]) != ACTUAL_FIRST_SEED_START + task:
            raise ValueError("actual first seed changed")
        question = initial[task]["questions"][root]
        mapping = base._map(cigs[task], question, truth_rows[task][1])
        dialogue = [
            {"role": "assistant", "content": question},
            {"role": "user", "content": mapping["answer"]},
        ]
        if base._close(manifest.get("first_mapping"), mapping) or manifest.get("dialogue") != dialogue:
            raise ValueError("actual first mapping changed")
        support = _parse_transition(raw, initial[task])
        second = base._select_question(support)
        second_mapping = base._map(
            cigs[task], support["questions"][second], truth_rows[task][1]
        )
        full_dialogue = [
            *dialogue,
            {"role": "assistant", "content": support["questions"][second]},
            {"role": "user", "content": second_mapping["answer"]},
        ]
        posterior, represented = _posterior(support, second, second_mapping["answer"])
        primary_privacy.append(_transition_audit(cigs[task], dialogue, initial[task]))
        first_paths[(task, root)] = {
            "support": support,
            "posterior": posterior,
            "first_mapping": mapping,
            "second": second,
            "second_mapping": second_mapping,
            "represented": represented,
            "dialogue": full_dialogue,
        }
        actual_supports.append(support)
    final_paths = {}
    final_manifest = actual.get("final_manifest") or []
    final_responses = actual.get("final_responses") or []
    for manifest, raw in zip(final_manifest, final_responses, strict=True):
        task = int(manifest["task_index"])
        root = int(manifest["root_index"])
        key = (task, root)
        if (
            int(manifest["seed"]) != ACTUAL_FINAL_SEED_START + task
            or manifest.get("second_reply_represented")
            is not first_paths[key]["represented"]
        ):
            raise ValueError("actual final schedule changed")
        primary_privacy.append(
            _transition_audit(cigs[task], first_paths[key]["dialogue"], first_paths[key]["posterior"])
        )
        final_paths[key] = _parse_transition(raw, first_paths[key]["posterior"])
        actual_supports.append(final_paths[key])
    if set(first_paths) != set(final_paths):
        raise ValueError("actual first/final path keys differ")

    naive_available = (result.get("naive_baseline") or {}).get("status") == "available"
    naive_rows = []
    naive_first_privacy = []
    naive_second_privacy = []
    endpoint_first_privacy = []
    endpoint_final_privacy = []
    if naive_available:
        naive = _load(run_dir / "private/RAW_NAIVE.json")
        first_questions = [base._parse_naive(raw) for raw in naive["first_questions"]]
        second_questions = [base._parse_naive(raw) for raw in naive["second_questions"]]
        if naive.get("first_question_seeds") != [NAIVE_FIRST_SEED_START + i for i in range(64)] or naive.get("second_question_seeds") != [NAIVE_SECOND_SEED_START + i for i in range(64)]:
            raise ValueError("naive question schedule changed")
        if naive.get("first_support_seeds") != [NAIVE_FIRST_SUPPORT_SEED_START + i for i in range(64)] or naive.get("final_support_seeds") != [NAIVE_FINAL_SUPPORT_SEED_START + i for i in range(64)]:
            raise ValueError("naive endpoint schedule changed")
        first_supports = []
        final_supports = []
        first_dialogues = []
        for task in range(64):
            first_mapping = base._map(cigs[task], first_questions[task], truth_rows[task][1])
            dialogue = [
                {"role": "assistant", "content": first_questions[task]},
                {"role": "user", "content": first_mapping["answer"]},
            ]
            first = _parse_transition(naive["first_supports"][task], initial[task])
            second_mapping = base._map(cigs[task], second_questions[task], truth_rows[task][1])
            full = [
                *dialogue,
                {"role": "assistant", "content": second_questions[task]},
                {"role": "user", "content": second_mapping["answer"]},
            ]
            final = _parse_transition(naive["final_supports"][task], first)
            first_supports.append(first)
            final_supports.append(final)
            first_dialogues.append((dialogue, full))
            aliases = truth_rows[task][2]
            raw_first_mass = base._truth_mass(first, aliases)
            raw_final_mass = base._truth_mass(final, aliases)
            valid = base._distinct_actions(first_mapping, second_mapping)
            final_mass = raw_final_mass if valid else 0.0
            naive_rows.append(
                {
                    "endpoint_mode": "fresh_smc_regeneration_descriptive",
                    "root_index": None,
                    "second_question_index": None,
                    "first_supported": first_mapping["supported"],
                    "second_supported": second_mapping["supported"],
                    "second_action_novel": valid,
                    "valid_two_action_trajectory": valid,
                    "raw_truth_mass_after_first": raw_first_mass,
                    "truth_mass_after_first": raw_first_mass if first_mapping["supported"] else 0.0,
                    "raw_truth_mass_final": raw_final_mass,
                    "truth_mass_final": final_mass,
                    "brier": (1.0 - final_mass) ** 2,
                    "log_loss": -math.log(max(PROBABILITY_FLOOR, final_mass)),
                    "covered": final_mass > 0.0,
                    "raw_fresh_truth_mass_final": raw_final_mass,
                    "fresh_truth_mass_final": final_mass,
                    "fresh_brier": (1.0 - final_mass) ** 2,
                    "fresh_log_loss": -math.log(max(PROBABILITY_FLOOR, final_mass)),
                    "fresh_covered": final_mass > 0.0,
                }
            )
            naive_first_privacy.append(_naive_audit(cigs[task], []))
            naive_second_privacy.append(_naive_audit(cigs[task], dialogue))
            endpoint_first_privacy.append(
                _transition_audit(cigs[task], dialogue, initial[task])
            )
            endpoint_final_privacy.append(
                _transition_audit(cigs[task], full, first)
            )

    tasks = []
    selected_questions_ok = True
    for task, (cig, support, plan) in enumerate(zip(cigs, initial, plans, strict=True)):
        policies = {}
        selected_questions = {}
        for name, root in plan["selected"].items():
            path = first_paths[(task, root)]
            metrics = base._path_metrics(
                support,
                path["support"],
                final_paths[(task, root)],
                first_question=root,
                question=path["second"],
                observed=path["second_mapping"]["answer"],
                aliases=truth_rows[task][2],
                first_mapping=path["first_mapping"],
                second_mapping=path["second_mapping"],
            )
            policies[name] = {
                "endpoint_mode": "aligned_generated_likelihood",
                "root_index": root,
                "second_question_index": path["second"],
                "first_supported": path["first_mapping"]["supported"],
                "second_supported": path["second_mapping"]["supported"],
                "second_action_novel": base._distinct_actions(path["first_mapping"], path["second_mapping"]),
                "posterior_parent_update_applied": path["represented"],
                **metrics,
            }
            selected_questions[name] = {
                "first": support["questions"][root],
                "second": path["support"]["questions"][path["second"]],
            }
        if naive_available:
            policies["naive_thinking"] = naive_rows[task]
            selected_questions["naive_thinking"] = {
                "first": first_questions[task],
                "second": second_questions[task],
            }
        selected_questions_ok &= control_by_id[cig.cig_id]["selected_questions"] == selected_questions
        tasks.append(
            {
                "task_id": cig.cig_id,
                "selected_roots": plan["selected"],
                "root_eig": plan["root_eig"],
                "conditioned_root_risks": plan["conditioned"],
                "conditioned_draw_root_risks": plan["conditioned_by_draw"],
                "dynamic_selection_stability": plan["selection_stability"],
                "blind_root_risks": plan["blind"],
                "myopic_refresh_brier_root_risks": plan["myopic_refresh_brier"],
                "myopic_brier_root_risks": plan["myopic_brier"],
                "fixed_root_risks": plan["fixed"],
                "policies": policies,
            }
        )
    samples = 20_000
    comparisons = (result.get("science") or {}).get("comparisons") or {}
    if comparisons:
        samples = int(next(iter(comparisons.values()))["brier_dynamic_minus_baseline"].get("samples", samples))
    usage = result["usage"]["deepseek_primary"]
    expected_primary = PLANNING_REQUESTS + len(first_manifest) + len(final_manifest)
    policy_names = [name for name in tasks[0]["policies"] if name != "naive_thinking"]
    planning_all = (
        len(initial) == 64
        and len(transitions) == 8_192
        and branch_schedule
        and all(
            sum(base._question_eig(row, i) > 1e-12 for i in range(4)) >= 2
            for row in initial
        )
        and sum(any(base._question_eig(row, i) > 1e-12 for i in range(4)) for row in transitions) >= 0.9 * len(transitions)
    )
    mechanics = {
        "planning_tree_all_pass": planning_all,
        "exact_primary_request_count": usage["adapter_requests"] == expected_primary,
        "exact_primary_http_attempt_count": usage["http_attempts"] == expected_primary,
        "primary_requests_within_frozen_maximum": expected_primary <= 8_768,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_actual_transitions_exact_lineage_and_retention": all(MIN_RETAINED <= row["diagnostic"]["retained_count"] <= MAX_RETAINED for row in actual_supports),
        "selection_frozen_before_truth_access": frozen["hidden_truth_accessed"] is False,
        "realized_task_level_common_random_numbers_exact": all(int(row["seed"]) == ACTUAL_FIRST_SEED_START + int(row["task_index"]) for row in first_manifest) and all(int(row["seed"]) == ACTUAL_FINAL_SEED_START + int(row["task_index"]) for row in final_manifest),
        "all_primary_privacy_and_parent_provenance_audits_pass": privacy_artifact.get("primary") == primary_privacy,
        "within_development_budget": float(usage["run_cost_usd"]) <= 3.50 + 1e-12,
    }
    for name in policy_names:
        mechanics[f"{name}_at_least_48_supported_first_actions"] = sum(task["policies"][name]["first_supported"] for task in tasks) >= 48
        mechanics[f"{name}_at_least_40_valid_second_actions"] = sum(task["policies"][name]["valid_two_action_trajectory"] for task in tasks) >= 40
        mechanics[f"{name}_at_least_40_exact_second_replies_represented"] = sum(task["policies"][name]["second_reply_likelihood_matched"] for task in tasks) >= 40
    mechanics.update(
        {
            "primary_deepseek_requests_within_8768": int(usage["adapter_requests"]) <= 8_768,
            "deepseek_requests_with_naive_endpoints_within_8896": int(usage["adapter_requests"]) + int(result["usage"]["deepseek_naive_endpoint"]["adapter_requests"]) <= 8_896,
            "combined_requests_within_9024": int(result["usage"]["combined_requests"]) <= 9_024,
            "combined_spend_within_350": float(result["usage"]["combined_cost_usd"]) <= 3.50 + 1e-12,
            "naive_baseline_is_descriptive_only": result["naive_baseline"]["descriptive_only"] is True,
            "naive_baseline_cannot_gate_or_abort_primary": result["naive_baseline"]["can_gate_or_abort_primary"] is False,
        }
    )
    mechanics["all_pass"] = all(mechanics.values())
    science = _science(tasks, samples) if mechanics["all_pass"] else None
    stability = base._draw_stability_summary(_scorer_tasks(tasks))
    expected_status = "mechanics_failed" if not mechanics["all_pass"] else ("passed" if science and science["gates"]["all_pass"] else "gated_null")
    mismatches: list[str] = []
    _close(result.get("tasks"), tasks, "$.tasks", mismatches)
    _close(result.get("mechanics_gates"), mechanics, "$.mechanics_gates", mismatches)
    _close(result.get("science"), science, "$.science", mismatches)
    _close(result.get("draw_stability_diagnostic"), stability, "$.draw_stability_diagnostic", mismatches)
    _close(result.get("status"), expected_status, "$.status", mismatches)
    _close((result.get("protocol") or {}).get("protocol_sha256"), PROTOCOL_SHA256, "$.protocol.protocol_sha256", mismatches)
    _close((result.get("protocol") or {}).get("producer_core_sha256"), PRODUCER_SHA256, "$.protocol.producer_core_sha256", mismatches)
    if not selected_questions_ok:
        mismatches.append("$.private.CONTROLS_PRIMARY.selected_questions")
    if naive_available:
        if privacy_artifact.get("naive") != [
            *naive_first_privacy,
            *naive_second_privacy,
        ]:
            mismatches.append("$.private.PRIVACY.naive")
        if privacy_artifact.get("naive_endpoint") != [
            *endpoint_first_privacy,
            *endpoint_final_privacy,
        ]:
            mismatches.append("$.private.PRIVACY.naive_endpoint")
    artifacts = {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(run_dir.rglob("*.json"))
        if path.name != "VERIFICATION.json"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not mismatches else "verification_failed",
        "kind": "smc_dynamic_depth2_policy",
        "model_calls": 0,
        "cost_usd": 0.0,
        "result_status": result.get("status"),
        "checks": {
            "raw_parent_annotations_reparsed": True,
            "all_smc_lineages_reconstructed": True,
            "planning_selections_recomputed": True,
            "truth_controls_replayed": True,
            "realized_paths_recomputed": True,
            "naive_baseline_recomputed": naive_available,
            "privacy_and_provenance_recomputed": True,
            "scientific_endpoint_recomputed": True,
            "reported_result_matches_replay": not mismatches,
        },
        "mismatches": mismatches,
        "artifact_sha256": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve(), primary_dir=args.primary_dir.resolve())
    if args.output:
        checkpoint(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
