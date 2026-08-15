"""Residual-conditioned typed proposal atlas for factored ChemBench."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .factored import (
    FactoredModelBank,
    FactoredState,
    RegistryEdit,
    RegistrySignature,
    TypedRegistryOracleProposer,
)


PANEL_SEED = 2026083700
RANDOM_CONTROL_SEED = 202608372000
OUTCOME_LABELS = ("low", "mid", "high")
ACTION_GROUPS = ("C_A", "C_I", "C_B", "C_P", "T", "pH")
PROPOSAL_KEYS = frozenset(
    {
        "parent_model_id",
        "operation",
        "core_family",
        "modifiers",
        "residual_motif",
        "exposing_assay_group",
        "falsifying_assay_group",
    }
)
RESPONSE_KEYS = frozenset({"proposals"})

CORE_DESCRIPTIONS = {
    "allosteric_activation": (
        "C_I is an activator: r=kcat*Enz*C_A/(Km+C_A)*C_I/(Kact+C_I)"
    ),
    "anticooperative_hill": (
        "negative cooperativity: r=kcat*Enz*C_A^n/(K_half^n+C_A^n), 0<n<1"
    ),
    "cooperative_inhibition": (
        "sigmoidal inhibitor response: r=kcat*Enz*C_A/(Km+C_A)"
        "/(1+(C_I/Ki)^n_inh), n_inh>1"
    ),
    "fractal_kinetics": (
        "non-integer substrate scaling: r=kcat*Enz*C_A^alpha, 0<alpha<1"
    ),
    "hill": (
        "positive cooperative saturation: r=kcat*Enz*C_A^n/"
        "(K_half^n+C_A^n), n>1"
    ),
    "metal_activation": (
        "C_B is a required cofactor: r=kcat*Enz*C_A/(KmA+C_A)"
        "*C_B^n/(Km_met^n+C_B^n)"
    ),
    "michaelis_menten": (
        "single-substrate saturation: r=kcat*Enz*C_A/(Km+C_A)"
    ),
    "mixed_inhibition": (
        "C_I changes affinity and capacity: r=kcat*Enz*C_A/"
        "(Km*(1+C_I/Ki)+C_A*(1+C_I/Ki_prime))"
    ),
    "monotonic_ph": (
        "single-ionization alkaline activation: r=kcat*Enz*C_A/(Km+C_A)"
        "/(1+10^(pKa-pH))"
    ),
    "ordered_bi_bi": (
        "A then B binding: r=kcat*Enz*C_A*C_B/"
        "(KiA*KmB+KmB*C_A+KmA*C_B+C_A*C_B)"
    ),
    "pingpong": (
        "ping-pong bisubstrate: r=kcat*Enz*C_A*C_B/"
        "(KmA*C_B+KmB*C_A+C_A*C_B)"
    ),
    "product_activation": (
        "C_P increases rate: r=kcat*Enz*C_A/(Km+C_A)"
        "*(Kthresh+C_P)/Kthresh"
    ),
    "substrate_inhibition": (
        "high C_A suppresses rate: r=kcat*Enz*C_A/(Km+C_A+C_A^2/Ki_s)"
    ),
    "two_substrate_inhibition": (
        "independent C_I and C_P inhibition: r=kcat*Enz*C_A/"
        "((Km*(1+C_I/Ki)+C_A)*(1+C_P/Kp))"
    ),
}
MODIFIER_DESCRIPTIONS = {
    "arrhenius": "multiply kcat by exp(-Ea/R*(1/T-1/T_ref))",
    "competitive_inhibition": "replace Km by Km*(1+C_I/Ki)",
    "noncompetitive_inhibition": "multiply rate by 1/(1+C_I/Ki)",
    "ph_bell_curve": (
        "multiply rate by 1/(1+10^(pKa1-pH)+10^(pH-pKa2))"
    ),
    "product_feedback": "replace Km by Km*(1+C_P/Ki_inh)",
    "product_inhibition": "replace Km by Km*(1+C_P/Kp)",
    "uncompetitive_inhibition": (
        "replace the MM denominator Km+C_A by Km+C_A*(1+C_I/Ki)"
    ),
}


@dataclass(frozen=True)
class OpportunityRow:
    difficulty: str
    truth: int
    truth_name: str
    truth_signature: RegistrySignature
    history_length: int
    history: tuple[tuple[int, int], ...]
    state: FactoredState
    oracle_proposals: tuple[int, ...]
    selection_rank: int = -1


@dataclass(frozen=True)
class ParsedProposal:
    parent_model_id: str
    operation: str
    core_family: str
    modifiers: tuple[str, ...]
    residual_motif: str
    exposing_assay_group: str
    falsifying_assay_group: str


@dataclass(frozen=True)
class CompiledProposal:
    parsed: ParsedProposal
    candidate: int
    edit: RegistryEdit


@dataclass(frozen=True)
class ParsedResponse:
    schema_valid: bool
    proposals: tuple[ParsedProposal, ...]
    error: str | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    result = 0.0
    for probability, reference in zip(left, right, strict=True):
        probability = float(probability)
        if probability <= 0:
            continue
        result += probability * (
            math.log(max(probability, 1e-300)) - math.log(max(float(reference), 1e-300))
        )
    return result


def construct_opportunity_rows(
    bank: FactoredModelBank,
    difficulty: str,
    truth_indices: Sequence[int],
) -> tuple[OpportunityRow, ...]:
    """Construct earliest source-only residual histories for every recoverable truth."""

    proposer = TypedRegistryOracleProposer(bank)
    rows: list[OpportunityRow] = []
    for truth_value in truth_indices:
        truth = int(truth_value)
        state = bank.initial_state()
        used: set[int] = set()
        for history_length in range(1, 4):
            scored: list[tuple[float, int]] = []
            for action in range(bank.num_actions):
                if action in used:
                    continue
                divergence = _kl_divergence(
                    bank.likelihoods[truth, action], bank.represented_predictive(state, action)
                )
                scored.append((-divergence, action))
            if not scored:
                break
            _, action = min(scored)
            used.add(action)
            outcome = int(np.argmax(bank.likelihoods[truth, action]))
            diagnostic = bank.expansion_diagnostic(state, action, outcome)
            oracle = proposer.propose(state, action, outcome, PANEL_SEED)
            state = bank.transition(state, action, outcome, ())
            if diagnostic.triggered and truth in oracle:
                rows.append(
                    OpportunityRow(
                        difficulty=str(difficulty),
                        truth=truth,
                        truth_name=bank.model_names[truth],
                        truth_signature=bank.compiler.signatures[truth],
                        history_length=history_length,
                        history=state.history,
                        state=state,
                        oracle_proposals=tuple(oracle),
                    )
                )
                break
    return tuple(rows)


def select_panel(rows: Sequence[OpportunityRow]) -> tuple[OpportunityRow, ...]:
    """Apply the clarified scarce-stratum-first globally unique allocation."""

    strata = [(difficulty, length) for difficulty in ("easy", "medium", "hard") for length in (1, 2, 3)]
    grouped = {
        stratum: [
            row
            for row in rows
            if (row.difficulty, row.history_length) == stratum
        ]
        for stratum in strata
    }
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    allocation_order = sorted(
        strata,
        key=lambda item: (len(grouped[item]), difficulty_order[item[0]], item[1]),
    )
    selected_by_stratum: dict[tuple[str, int], list[OpportunityRow]] = defaultdict(list)
    globally_selected: set[str] = set()
    for difficulty, history_length in allocation_order:
        available = [
            row
            for row in grouped[(difficulty, history_length)]
            if row.truth_name not in globally_selected
        ]
        family_count: Counter[str] = Counter()
        for rank in range(4):
            if not available:
                raise ValueError(
                    f"cannot allocate four unique tasks for {difficulty}/{history_length}"
                )
            row = min(
                available,
                key=lambda item: (
                    family_count[item.truth_signature.core],
                    hashlib.sha256(
                        (
                            f"{PANEL_SEED}|{difficulty}|{history_length}|"
                            f"{item.truth_name}"
                        ).encode()
                    ).hexdigest(),
                    item.truth_name,
                ),
            )
            selected_by_stratum[(difficulty, history_length)].append(
                OpportunityRow(**{**row.__dict__, "selection_rank": rank})
            )
            globally_selected.add(row.truth_name)
            family_count[row.truth_signature.core] += 1
            available = [
                candidate
                for candidate in available
                if candidate.truth_name not in globally_selected
            ]
    selected = tuple(
        row
        for difficulty in ("easy", "medium", "hard")
        for history_length in (1, 2, 3)
        for row in selected_by_stratum[(difficulty, history_length)]
    )
    if len(selected) != 36 or len({row.truth_name for row in selected}) != 36:
        raise AssertionError("source panel does not contain 36 unique generators")
    if len({row.truth_signature.core for row in selected}) < 10:
        raise AssertionError("source panel does not span ten core families")
    return selected


def _public_history(bank: FactoredModelBank, history: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {
            "action_index": int(action),
            "action_name": bank.action_names[action],
            "action_group": bank.action_groups[action],
            "outcome": OUTCOME_LABELS[outcome],
        }
        for action, outcome in history
    ]


def public_task_record(
    bank: FactoredModelBank,
    row: OpportunityRow,
    *,
    task_position: int,
    remaining_budget: int,
) -> dict[str, Any]:
    task_id = f"chem-{row.difficulty}-h{row.history_length}-{row.selection_rank + 1}"
    report = bank.residual_report(row.state, remaining_budget=remaining_budget)
    public = {
        "task_id": task_id,
        "task_position": int(task_position),
        "split": "atlas" if row.selection_rank < 3 else "heldout",
        "difficulty": row.difficulty,
        "history_length": row.history_length,
        "remaining_budget": int(remaining_budget),
        "phase": "explore",
        "history": _public_history(bank, row.history),
        "residual_report": report,
    }
    serialized = _canonical_json(public).lower()
    if row.truth_name.lower() in serialized:
        raise AssertionError("hidden generator leaked into public task record")
    return public


def private_task_record(row: OpportunityRow, public: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": public["task_id"],
        "truth_model_index": row.truth,
        "truth_model_name": row.truth_name,
        "truth_signature": {
            "core_family": row.truth_signature.core,
            "modifiers": list(row.truth_signature.modifiers),
        },
        "oracle_proposal_indices": list(row.oracle_proposals),
    }


def legal_signature_index(bank: FactoredModelBank) -> dict[tuple[str, tuple[str, ...]], int]:
    result: dict[tuple[str, tuple[str, ...]], int] = {}
    for model, signature in enumerate(bank.compiler.signatures):
        key = (signature.core, signature.modifiers)
        if key in result:
            raise ValueError(f"non-unique registry signature: {key}")
        result[key] = model
    return result


def response_json_schema(bank: FactoredModelBank) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(PROPOSAL_KEYS),
                    "properties": {
                        "parent_model_id": {
                            "type": "string",
                            "enum": [bank.model_names[index] for index in bank.initial_support],
                        },
                        "operation": {
                            "type": "string",
                            "enum": sorted(bank.compiler.OPERATIONS),
                        },
                        "core_family": {
                            "type": "string",
                            "enum": sorted(CORE_DESCRIPTIONS),
                        },
                        "modifiers": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "enum": sorted(MODIFIER_DESCRIPTIONS),
                            },
                        },
                        "residual_motif": {"type": "string", "minLength": 1},
                        "exposing_assay_group": {
                            "type": "string",
                            "enum": list(ACTION_GROUPS),
                        },
                        "falsifying_assay_group": {
                            "type": "string",
                            "enum": list(ACTION_GROUPS),
                        },
                    },
                },
            }
        },
    }


def _grammar_text(bank: FactoredModelBank) -> str:
    cores = "\n".join(
        f"- {name}: {description}." for name, description in CORE_DESCRIPTIONS.items()
    )
    modifiers = "\n".join(
        f"- {name}: {description}." for name, description in MODIFIER_DESCRIPTIONS.items()
    )
    primitives = "\n".join(
        (
            f"- {bank.model_names[index]}: core={bank.compiler.signatures[index].core}; "
            f"modifiers={list(bank.compiler.signatures[index].modifiers)}"
        )
        for index in bank.initial_support
    )
    return (
        "LEGAL CORE FAMILIES\n"
        f"{cores}\n\nLEGAL OPTIONAL MODIFIERS\n{modifiers}\n\n"
        "CURRENT EXECUTABLE PRIMITIVES\n"
        f"{primitives}\n\n"
        "A candidate has exactly one core family and zero or more compatible modifiers. "
        "Compose only scientifically meaningful factors. Operations mean: add_factor adds "
        "one or more modifiers while retaining the parent core; remove_factor removes one "
        "or more parent modifiers; replace_factor changes modifiers while retaining the core; "
        "replace_core changes the core and may also change modifiers."
    )


def render_prompt(bank: FactoredModelBank, task: Mapping[str, Any], arm: str) -> dict[str, str]:
    if arm not in {"residual_aware", "history_blind"}:
        raise ValueError(f"unknown proposal arm: {arm}")
    common = (
        "An enzyme catalyses a reaction with initial rate r0. Controllable inputs are "
        "substrate C_A, inhibitor C_I, second substrate C_B, product C_P, enzyme loading "
        "Enz, temperature T, and pH. Propose four structurally different executable edits "
        "to the current mechanism pool. Do not invent offsets or softening terms.\n\n"
        f"{_grammar_text(bank)}\n\n"
        "ASSAY-GROUP SEMANTICS\n"
        "- C_A probes saturation, cooperativity, and substrate inhibition.\n"
        "- C_I probes inhibitor mechanism.\n"
        "- C_B probes second-substrate mechanism.\n"
        "- C_P probes product inhibition, feedback, or activation.\n"
        "- T probes temperature dependence.\n"
        "- pH probes pH dependence.\n\n"
        f"History length: {task['history_length']}. Remaining experiment budget: "
        f"{task['remaining_budget']}. Current phase: {task['phase']}.\n"
    )
    if arm == "residual_aware":
        evidence = (
            "\nPUBLIC OBSERVATION AND POOL-RESIDUAL STATE\n"
            f"{json.dumps({'history': task['history'], 'residual_report': task['residual_report']}, sort_keys=True)}\n"
        )
    else:
        evidence = (
            "\nNo actions, outcomes, residuals, likelihoods, or surprise values are available "
            "for this matched history-blind control.\n"
        )
    instruction = (
        "\nReturn JSON only, matching the supplied schema exactly. Parent IDs must be current "
        "primitive IDs. Return exactly four distinct candidates. Explain each residual motif "
        "briefly and name one exposing and one falsifying assay group."
    )
    prompt = {
        "system": "You are an enzyme kineticist proposing typed mechanism edits. Reply JSON only.",
        "user": common + evidence + instruction,
    }
    if len(prompt["system"]) + len(prompt["user"]) > 24_000:
        raise ValueError("serialized proposal prompt exceeds 24,000 characters")
    return prompt


def parse_response(raw: Any) -> ParsedResponse:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError) as exc:
        return ParsedResponse(False, (), f"invalid_json:{type(exc).__name__}")
    if not isinstance(value, dict) or frozenset(value) != RESPONSE_KEYS:
        return ParsedResponse(False, (), "invalid_response_shape")
    items = value.get("proposals")
    if not isinstance(items, list) or len(items) != 4:
        return ParsedResponse(False, (), "invalid_proposal_count")
    parsed: list[ParsedProposal] = []
    for item in items:
        if not isinstance(item, dict) or frozenset(item) != PROPOSAL_KEYS:
            return ParsedResponse(False, (), "invalid_proposal_shape")
        scalar_keys = PROPOSAL_KEYS - {"modifiers"}
        if any(not isinstance(item[key], str) or not item[key].strip() for key in scalar_keys):
            return ParsedResponse(False, (), "invalid_proposal_scalar")
        modifiers = item["modifiers"]
        if (
            not isinstance(modifiers, list)
            or any(not isinstance(modifier, str) for modifier in modifiers)
            or len(set(modifiers)) != len(modifiers)
        ):
            return ParsedResponse(False, (), "invalid_modifiers")
        parsed.append(
            ParsedProposal(
                parent_model_id=item["parent_model_id"],
                operation=item["operation"],
                core_family=item["core_family"],
                modifiers=tuple(modifiers),
                residual_motif=item["residual_motif"].strip(),
                exposing_assay_group=item["exposing_assay_group"],
                falsifying_assay_group=item["falsifying_assay_group"],
            )
        )
    return ParsedResponse(True, tuple(parsed))


def compile_proposal(
    bank: FactoredModelBank,
    state: FactoredState,
    proposal: ParsedProposal,
) -> CompiledProposal:
    if proposal.operation not in bank.compiler.OPERATIONS:
        raise ValueError("unknown proposal operation")
    if proposal.core_family not in CORE_DESCRIPTIONS:
        raise ValueError("unknown proposal core family")
    if any(modifier not in MODIFIER_DESCRIPTIONS for modifier in proposal.modifiers):
        raise ValueError("unknown proposal modifier")
    if proposal.exposing_assay_group not in ACTION_GROUPS or proposal.falsifying_assay_group not in ACTION_GROUPS:
        raise ValueError("unknown proposal assay group")
    name_to_index = {name: index for index, name in enumerate(bank.model_names)}
    if proposal.parent_model_id not in name_to_index:
        raise ValueError("proposal parent is unknown")
    parent = name_to_index[proposal.parent_model_id]
    if parent not in state.represented_models:
        raise ValueError("proposal parent is not represented")
    signature_key = (proposal.core_family, tuple(sorted(proposal.modifiers)))
    signatures = legal_signature_index(bank)
    if signature_key not in signatures:
        raise ValueError("proposal signature is not an executable registry mechanism")
    candidate = signatures[signature_key]
    if candidate in state.represented_models or candidate in state.tried:
        raise ValueError("proposal candidate is represented or previously tried")
    edit = bank.compiler.compile((parent,), candidate)
    if edit.parent != parent or edit.operation != proposal.operation:
        raise ValueError("declared typed operation does not compile from parent")
    return CompiledProposal(parsed=proposal, candidate=candidate, edit=edit)


def compile_response(
    bank: FactoredModelBank,
    state: FactoredState,
    parsed: ParsedResponse,
) -> tuple[CompiledProposal | None, ...]:
    if not parsed.schema_valid:
        return ()
    compiled: list[CompiledProposal | None] = []
    for proposal in parsed.proposals:
        try:
            compiled.append(compile_proposal(bank, state, proposal))
        except ValueError:
            compiled.append(None)
    return tuple(compiled)


def _modifier_f1(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    return 2.0 * overlap / (len(left_set) + len(right_set))


def score_response(
    bank: FactoredModelBank,
    state: FactoredState,
    raw: Any,
    truth: int,
) -> dict[str, Any]:
    parsed = parse_response(raw)
    compiled_slots = compile_response(bank, state, parsed)
    valid = [item for item in compiled_slots if item is not None]
    candidates = [item.candidate for item in valid]
    truth_signature = bank.compiler.signatures[int(truth)]
    truth_recall = int(truth in candidates)
    core_recall = int(
        any(bank.compiler.signatures[candidate].core == truth_signature.core for candidate in candidates)
    )
    modifier_f1 = max(
        (
            _modifier_f1(bank.compiler.signatures[candidate].modifiers, truth_signature.modifiers)
            for candidate in candidates
        ),
        default=0.0,
    )
    all_distinct = len(candidates) == len(set(candidates))
    response_executable = len(compiled_slots) == 4 and len(valid) == 4 and all_distinct
    return {
        "schema_valid": parsed.schema_valid,
        "parse_error": parsed.error,
        "item_compile_rate": len(valid) / 4.0,
        "response_executable": response_executable,
        "compiled_candidate_indices": candidates,
        "compiled_candidate_names": [bank.model_names[candidate] for candidate in candidates],
        "truth_recall_at_4": truth_recall,
        "core_family_recall_at_4": core_recall,
        "modifier_f1": modifier_f1,
        "semantic_score": 2.0 * truth_recall + core_recall + modifier_f1,
    }


def feature_vector(
    bank: FactoredModelBank,
    task: Mapping[str, Any],
) -> tuple[tuple[str, ...], np.ndarray]:
    report = task["residual_report"]
    model_rows = {row["model_name"]: row for row in report["models"]}
    latest = report["latest"] or {}
    names: list[str] = ["history_length"]
    values: list[float] = [float(task["history_length"])]
    for group in ACTION_GROUPS:
        names.append(f"latest_group:{group}")
        values.append(float(latest.get("action_group") == group))
    latest_outcome = task["history"][-1]["outcome"] if task["history"] else None
    for outcome in OUTCOME_LABELS:
        names.append(f"latest_outcome:{outcome}")
        values.append(float(latest_outcome == outcome))
    names.append("latest_surprise")
    values.append(float(latest.get("surprise") or 0.0))
    for model in bank.initial_support:
        model_name = bank.model_names[model]
        row = model_rows[model_name]
        names.extend((f"weight:{model_name}", f"nll:{model_name}"))
        values.extend((float(row["evidence_weight"]), float(row["categorical_nll"])))
    for model in bank.initial_support:
        model_name = bank.model_names[model]
        innovations = model_rows[model_name]["signed_innovation_by_group"]
        for group in ACTION_GROUPS:
            observed = group in innovations
            names.extend((f"innovation:{model_name}:{group}", f"mask:{model_name}:{group}"))
            values.extend((float(innovations.get(group, 0.0)), float(observed)))
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("proposal-atlas feature vector is non-finite")
    return tuple(names), result


def standardized_features(
    banks: Mapping[str, FactoredModelBank],
    atlas_tasks: Sequence[Mapping[str, Any]],
    target_tasks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw: dict[str, np.ndarray] = {}
    feature_names: tuple[str, ...] | None = None
    for task in (*atlas_tasks, *target_tasks):
        names, values = feature_vector(banks[str(task["difficulty"])], task)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("proposal-atlas feature names are inconsistent")
        raw[str(task["task_id"])] = values
    if not atlas_tasks or feature_names is None:
        raise ValueError("proposal atlas requires development tasks")
    development = np.stack([raw[str(task["task_id"])] for task in atlas_tasks])
    mean = np.mean(development, axis=0)
    scale = np.std(development, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = {task_id: (values - mean) / scale for task_id, values in raw.items()}
    metadata = {
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "development_task_ids": [task["task_id"] for task in atlas_tasks],
        "sha256": _payload_hash(
            {
                "feature_names": list(feature_names),
                "mean": mean.tolist(),
                "scale": scale.tolist(),
            }
        ),
    }
    return standardized, metadata


def retrieve_atlas_candidates(
    heldout_task: Mapping[str, Any],
    atlas_tasks: Sequence[Mapping[str, Any]],
    standardized: Mapping[str, np.ndarray],
    compiled_by_task: Mapping[str, Sequence[Sequence[int]]],
    *,
    neighbors: int = 3,
    limit: int = 4,
) -> dict[str, Any]:
    heldout_id = str(heldout_task["task_id"])
    distances = sorted(
        (
            float(np.linalg.norm(standardized[heldout_id] - standardized[str(task["task_id"])])),
            str(task["task_id"]),
        )
        for task in atlas_tasks
    )[:neighbors]
    scores: dict[int, float] = defaultdict(float)
    for distance, task_id in distances:
        weight = math.exp(-distance)
        for response_candidates in compiled_by_task.get(task_id, ()):
            for candidate in dict.fromkeys(int(item) for item in response_candidates):
                scores[candidate] += weight
    selected = sorted(scores, key=lambda candidate: (-scores[candidate], candidate))[:limit]
    return {
        "candidate_indices": selected,
        "neighbors": [
            {"task_id": task_id, "distance": distance} for distance, task_id in distances
        ],
        "candidate_scores": {str(candidate): scores[candidate] for candidate in selected},
    }


def proposal_induced_risk(
    bank: FactoredModelBank,
    history: Sequence[tuple[int, int]],
    truth: int,
    candidates: Sequence[int],
) -> dict[str, Any]:
    support = tuple(dict.fromkeys((*bank.initial_support, *(int(item) for item in candidates))))
    state = bank.state(history, support, tried=support)
    used = {int(action) for action, _ in history}
    action_risks: list[tuple[float, int]] = []
    for action in range(bank.num_actions):
        if action in used:
            continue
        expected = 0.0
        for outcome, probability in enumerate(bank.likelihoods[int(truth), action]):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            child = bank.transition(state, action, outcome, ())
            expected += probability * bank.truth_loss(child, int(truth))
        action_risks.append((expected, action))
    if not action_risks:
        return {"risk": bank.truth_loss(state, int(truth)), "action_index": -1, "action_name": None}
    risk, action = min(action_risks)
    return {"risk": float(risk), "action_index": action, "action_name": bank.action_names[action]}


def random_typed_candidates(
    bank: FactoredModelBank,
    state: FactoredState,
    *,
    task_position: int,
    count: int = 4,
) -> tuple[int, ...]:
    admissible = []
    for candidate in range(bank.num_models):
        if candidate in state.tried or candidate in state.represented_models:
            continue
        try:
            bank.compiler.compile(state.represented_models, candidate)
        except ValueError:
            continue
        admissible.append(candidate)
    rng = np.random.default_rng(RANDOM_CONTROL_SEED + int(task_position))
    if len(admissible) < count:
        raise ValueError("random typed control has insufficient candidates")
    return tuple(int(item) for item in rng.choice(admissible, size=count, replace=False))


def core_family_jaccard(
    bank: FactoredModelBank,
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    left_families = {bank.compiler.signatures[int(item)].core for item in left}
    right_families = {bank.compiler.signatures[int(item)].core for item in right}
    union = left_families | right_families
    return len(left_families & right_families) / len(union) if union else 1.0
