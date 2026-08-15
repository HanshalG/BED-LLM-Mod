#!/usr/bin/env python3
"""Independent label-free replay of an atomic-particle response bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.number_game_stratified_atomic_particle_v3_codec import (
    BELIEF_WIDTH,
    GENERATED_SLOTS,
    INITIAL_SLOTS,
    INTERFACE_VERSION,
    MODEL_REQUESTS,
    MODEL_SEEDS,
    ROOT_COUNT,
    TREE_SEEDS,
    RootPlan,
    anchors_for,
    best_query,
    candidate_roots,
    choose_plan,
    extension_hash,
    fixed_depth3_score,
    generated_depth3_score,
    parse_stratified,
    particle_signature,
    request_identity,
    refresh_belief,
    systematic_resample,
)
from scripts.number_game_generator_aware_bed import RuleHypothesis


MODEL_ID = "qwen/qwen3.7-plus"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def require_group(valid: Sequence[RuleHypothesis], diag: Mapping[str, Any], *, initial: bool = False) -> None:
    valid_floor, unique_floor = (48, 24) if initial else (24, 16)
    if len(valid) < valid_floor or int(diag["unique_extensions"]) < unique_floor or int(diag["unique_signatures"]) < 24:
        raise RuntimeError("replayed group misses frozen floor")


def parse_values(values: Sequence[str], history: Sequence[tuple[int, bool]], anchors: Sequence[int]):
    parsed = [
        parse_stratified(
            value,
            history,
            anchors,
            tuple(bit == "1" for bit in format(slot // (len(values) // 32), "05b")),
        )
        for slot, value in enumerate(values)
    ]
    slots = [row.hypothesis for row in parsed]
    valid = [row for row in slots if row is not None]
    rejected: dict[str, int] = {}
    for row in parsed:
        if row.rejection:
            rejected[row.rejection] = rejected.get(row.rejection, 0) + 1
    return valid, slots, {
        "slots": len(values),
        "valid_particles": len(valid),
        "unique_extensions": len({row.extension for row in valid}),
        "unique_signatures": len({particle_signature(row, anchors) for row in valid}),
        "rejections": rejected,
    }


class ReplayCursor:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self.responses = responses
        self.index = 0

    def take(
        self,
        *,
        stage: str,
        tree: int,
        history: Sequence[tuple[int, bool]],
        protected: Sequence[int],
        count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[str]:
        output: list[str] = []
        observations = [[number, answer] for number, answer in history]
        for slot in range(count):
            if self.index >= len(self.responses):
                raise RuntimeError("raw response bank ended early")
            row = self.responses[self.index]
            seed = MODEL_SEEDS[self.index]
            anchors, signature, messages = request_identity(history, protected=protected, slot=slot, count=count)
            expected = {
                "stage": stage,
                "tree": tree,
                "slot": slot,
                "seed": seed,
                "observations": observations,
                "anchors": list(anchors),
                "signature": "".join("1" if value else "0" for value in signature),
                **dict(metadata or {}),
                "prompt_sha256": hashlib.sha256(canonical_json(messages).encode()).hexdigest(),
            }
            if {key: row.get(key) for key in expected} != expected:
                raise RuntimeError(f"raw request identity changed at slot {self.index}")
            if set(row) != {*expected, "response"} or not isinstance(row["response"], str):
                raise RuntimeError("raw response shape changed")
            output.append(row["response"])
            self.index += 1
        return output


def filter_blind(pool: Sequence[RuleHypothesis], history: Sequence[tuple[int, bool]], anchors: Sequence[int]):
    valid = [row for row in pool if all(row.extension[number] is answer for number, answer in history)]
    diag = {
        "slots": len(pool),
        "valid_particles": len(valid),
        "unique_extensions": len({row.extension for row in valid}),
        "unique_signatures": len({particle_signature(row, anchors) for row in valid}),
        "rejections": {"history": len(pool) - len(valid)},
    }
    if len(valid) < 24 or diag["unique_extensions"] < 16 or diag["unique_signatures"] < 16:
        raise RuntimeError("replayed blind branch misses frozen floor")
    return valid, diag


def build_plans(tree: int, bank: Mapping[str, Any]) -> dict[str, dict[int, RootPlan]]:
    out = {"dynamic": {}, "blind": {}, "fixed": {}}
    blind_first = {False: bank["first_blind"], True: bank["first_blind"]}
    blind_second = {(a, b): bank["second_blind"] for a in (False, True) for b in (False, True)}
    for root in bank["roots"]:
        out["dynamic"][root] = generated_depth3_score(
            bank["initial"], root, bank["first_conditioned"][root], bank["second_conditioned"][root], seed=TREE_SEEDS[tree]
        )
        out["blind"][root] = generated_depth3_score(
            bank["initial"], root, blind_first, blind_second, seed=TREE_SEEDS[tree]
        )
        out["fixed"][root] = fixed_depth3_score(bank["initial"], root, seed=TREE_SEEDS[tree])
    return out


def slot_half(slots: Sequence[RuleHypothesis | None], parity: int) -> list[RuleHypothesis]:
    return [row for index, row in enumerate(slots) if index % 2 == parity and row is not None]


def half_stability(tree: int, bank: Mapping[str, Any]) -> dict[str, Any]:
    full = choose_plan(bank["plans"]["dynamic"])
    halves = []
    for parity in (0, 1):
        try:
            initial = systematic_resample(slot_half(bank["initial_slots"], parity), BELIEF_WIDTH // 2, TREE_SEEDS[tree])
            plans = {}
            for root in bank["roots"]:
                first = {answer: slot_half(bank["first_conditioned_slots"][root][answer], parity) for answer in (False, True)}
                second = {answers: slot_half(slots, parity) for answers, slots in bank["second_conditioned_slots"][root].items()}
                plans[root] = generated_depth3_score(
                    initial,
                    root,
                    first,
                    second,
                    seed=TREE_SEEDS[tree],
                    width=BELIEF_WIDTH // 2,
                    fixed_second_queries=bank["plans"]["dynamic"][root].second_queries,
                )
            selected = choose_plan(plans)
            halves.append({
                "parity": parity,
                "status": "valid",
                "selected_root": selected,
                "agrees_with_full": selected == full,
                "scores": {str(root): plan.score for root, plan in plans.items()},
            })
        except (KeyError, ValueError) as error:
            halves.append({"parity": parity, "status": "invalid", "selected_root": None, "agrees_with_full": False, "error": str(error)})
    return {"full_selected_root": full, "halves": halves, "both_halves_agree": all(row["agrees_with_full"] for row in halves)}


def replay(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(raw) != {"schema_version", "interface_version", "responses"} or raw["schema_version"] != 1 or raw["interface_version"] != INTERFACE_VERSION:
        raise RuntimeError("raw bank envelope changed")
    responses = raw["responses"]
    if not isinstance(responses, list) or len(responses) != MODEL_REQUESTS:
        raise RuntimeError("raw bank is incomplete")
    cursor = ReplayCursor(responses)
    banks: list[dict[str, Any]] = [{"diagnostics": {}} for _ in TREE_SEEDS]
    for tree, bank in enumerate(banks):
        values = cursor.take(stage="initial", tree=tree, history=(), protected=(), count=INITIAL_SLOTS)
        anchors = anchors_for(())
        valid, slots, diag = parse_values(values, (), anchors)
        require_group(valid, diag, initial=True)
        bank.update(initial_raw=valid, initial_slots=slots, initial=systematic_resample(valid, BELIEF_WIDTH, TREE_SEEDS[tree]))
        bank["diagnostics"]["initial"] = diag
        bank["initial_anchors"] = anchors
        bank["roots"] = candidate_roots(bank["initial"], seed=TREE_SEEDS[tree])
    for tree, bank in enumerate(banks):
        bank["first_conditioned"], bank["first_conditioned_slots"] = {}, {}
        for root in bank["roots"]:
            bank["first_conditioned"][root], bank["first_conditioned_slots"][root] = {}, {}
            for answer in (False, True):
                history = ((root, answer),)
                values = cursor.take(stage="first_conditioned", tree=tree, history=history, protected=(root,), count=GENERATED_SLOTS, metadata={"root": root, "first_answer": answer})
                anchors = anchors_for((root,))
                valid, slots, diag = parse_values(values, history, anchors)
                require_group(valid, diag)
                bank["first_conditioned"][root][answer] = valid
                bank["first_conditioned_slots"][root][answer] = slots
                bank.setdefault("first_conditioned_anchors", {}).setdefault(root, {})[answer] = anchors
                bank["diagnostics"][f"first_conditioned:{root}:{int(answer)}"] = diag
    for tree, bank in enumerate(banks):
        values = cursor.take(stage="first_blind", tree=tree, history=(), protected=bank["roots"], count=256)
        anchors = anchors_for(bank["roots"])
        pool, slots, diag = parse_values(values, (), anchors)
        bank["first_blind"], bank["first_blind_slots"] = pool, slots
        bank["first_blind_anchors"] = anchors
        bank["diagnostics"]["first_blind_pool"] = diag
        for root in bank["roots"]:
            for answer in (False, True):
                _, filtered = filter_blind(pool, ((root, answer),), anchors)
                bank["diagnostics"][f"first_blind:{root}:{int(answer)}"] = filtered
    for tree, bank in enumerate(banks):
        bank["second_conditioned"], bank["second_conditioned_slots"] = {}, {}
        for root in bank["roots"]:
            bank["second_conditioned"][root], bank["second_conditioned_slots"][root] = {}, {}
            for first_answer in (False, True):
                history1 = ((root, first_answer),)
                first = refresh_belief(bank["initial"], bank["first_conditioned"][root][first_answer], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(first_answer)))
                second, _ = best_query(first, (root,))
                for second_answer in (False, True):
                    history2 = (*history1, (second, second_answer))
                    metadata = {"root": root, "first_answer": first_answer, "second_query": second, "second_answer": second_answer}
                    values = cursor.take(stage="second_conditioned", tree=tree, history=history2, protected=(root, second), count=GENERATED_SLOTS, metadata=metadata)
                    anchors = anchors_for((root, second))
                    valid, slots, diag = parse_values(values, history2, anchors)
                    require_group(valid, diag)
                    bank["second_conditioned"][root][(first_answer, second_answer)] = valid
                    bank["second_conditioned_slots"][root][(first_answer, second_answer)] = slots
                    bank.setdefault("second_conditioned_anchors", {}).setdefault(root, {})[(first_answer, second_answer)] = anchors
                    bank["diagnostics"][f"second_conditioned:{root}:{int(first_answer)}:{second}:{int(second_answer)}"] = diag
    for tree, bank in enumerate(banks):
        protected = set(bank["roots"])
        second_query_sets = {}
        for root in bank["roots"]:
            for first_answer in (False, True):
                history1 = ((root, first_answer),)
                conditioned = refresh_belief(bank["initial"], bank["first_conditioned"][root][first_answer], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(first_answer)))
                blind = refresh_belief(bank["initial"], bank["first_blind"], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(first_answer)))
                queries = {best_query(conditioned, (root,))[0], best_query(blind, (root,))[0]}
                second_query_sets[(root, first_answer)] = queries
                protected.update(queries)
        values = cursor.take(stage="second_blind", tree=tree, history=(), protected=tuple(sorted(protected)), count=512)
        anchors = anchors_for(protected)
        pool, slots, diag = parse_values(values, (), anchors)
        bank["second_blind"], bank["second_blind_slots"] = pool, slots
        bank["second_blind_anchors"] = anchors
        bank["diagnostics"]["second_blind_pool"] = diag
        for root in bank["roots"]:
            for first_answer in (False, True):
                history1 = ((root, first_answer),)
                for query in second_query_sets[(root, first_answer)]:
                    for second_answer in (False, True):
                        _, filtered = filter_blind(pool, (*history1, (query, second_answer)), anchors)
                        bank["diagnostics"][f"second_blind:{root}:{int(first_answer)}:{query}:{int(second_answer)}"] = filtered
        bank["plans"] = build_plans(tree, bank)
        bank["half_bank_stability"] = half_stability(tree, bank)
    if cursor.index != MODEL_REQUESTS:
        raise RuntimeError("raw bank has an unconsumed suffix")
    return banks


def serialize_tree(tree: int, bank: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tree": tree,
        "tree_seed": TREE_SEEDS[tree],
        "roots": list(bank["roots"]),
        "initial_extension_hashes": [extension_hash(row) for row in bank["initial"]],
        "initial_anchors": list(bank["initial_anchors"]),
        "first_blind_anchors": list(bank["first_blind_anchors"]),
        "second_blind_anchors": list(bank["second_blind_anchors"]),
        "diagnostics": bank["diagnostics"],
        "half_bank_stability": bank["half_bank_stability"],
        "plans": {
            family: {
                str(root): {"score": plan.score, "second_queries": {"1" if answer else "0": query for answer, query in plan.second_queries.items()}}
                for root, plan in plans.items()
            }
            for family, plans in bank["plans"].items()
        },
    }


def verify(run_dir: Path, *, output: Path | None = None) -> dict[str, Any]:
    raw_path = run_dir / "private/RAW_RESPONSES.json"
    topology_path = run_dir / "private/TOPOLOGY.json"
    raw, topology = load(raw_path), load(topology_path)
    banks = replay(raw)
    transport = topology.get("transport") or {}
    records = transport.get("records") or []
    transport_gates = transport.get("gates") or {}
    record_by_seed = {int(row.get("seed", -1)): row for row in records}
    raw_by_seed = {int(row["seed"]): row for row in raw["responses"]}
    gates = {
        "raw_hash_matches": topology.get("raw_response_sha256") == digest(raw_path),
        "exact_request_count": topology.get("request_count") == MODEL_REQUESTS,
        "transport_claim_complete": len(records) == MODEL_REQUESTS and all(transport_gates.values()),
        "transport_seed_set_exact": {int(row.get("seed", -1)) for row in records} == set(MODEL_SEEDS),
        "transport_records_replay": (
            len(record_by_seed) == MODEL_REQUESTS
            and all(
                record_by_seed[seed].get("model_requested")
                == record_by_seed[seed].get("model_returned")
                == MODEL_ID
                and record_by_seed[seed].get("finish_reasons") == ["stop"]
                and record_by_seed[seed].get("prompt_sha256") == raw_by_seed[seed]["prompt_sha256"]
                for seed in MODEL_SEEDS
            )
        ),
        "topology_replays": topology.get("trees") == [serialize_tree(tree, bank) for tree, bank in enumerate(banks)],
        "outcomes_remain_closed": topology.get("canonical_targets_opened") is False and topology.get("classical_grammar_opened") is False,
    }
    result = {
        "schema_version": 1,
        "interface_version": "number-game-stratified-atomic-particle-v3-verifier-1",
        "status": "verification_pass" if all(gates.values()) else "verification_failed",
        "gates": gates,
        "model_calls_made": 0,
        "targets_opened": False,
        "classical_grammar_opened": False,
    }
    if output:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "verification_pass":
        raise RuntimeError("atomic particle replay verification failed")
    return result
