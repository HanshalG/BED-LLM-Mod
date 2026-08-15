#!/usr/bin/env python3
"""Produce and score atomic-particle Number Game mechanics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_stratified_atomic_particle_v3_codec import (
    BELIEF_WIDTH,
    GENERATED_SLOTS,
    INITIAL_SLOTS,
    INTERFACE_VERSION,
    MODEL_SEEDS,
    ROOT_COUNT,
    TREE_SEEDS,
    AtomicResponse,
    RootPlan,
    answer_probability,
    best_query,
    candidate_roots,
    choose_plan,
    extension_hash,
    fixed_depth3_score,
    generated_depth3_score,
    anchors_for,
    parse_stratified,
    particle_signature,
    request_identity,
    posterior_predictive_brier,
    predictive_probabilities,
    refresh_belief,
    response_format,
    spearman,
    systematic_resample,
)
from scripts.number_game_classical_grammar_irreducibility_audit import (
    build_classical_grammar_bank,
    extension_mask,
)
from scripts.number_game_external_canonical_replay import canonical_targets
from scripts.number_game_generator_aware_bed import RuleHypothesis


PROTOCOL = REPO_ROOT / "results/nonmyopic/NUMBER_GAME_STRATIFIED_ATOMIC_PARTICLE_V3_PROTOCOL_20260813.md"
PROTOCOL_SHA256 = "a1f119276aa00f9d7619c5a20b1de41df6b35b5dc87e087a0f1a97aa654a7244"
PARENT_PROTOCOL = REPO_ROOT / "results/nonmyopic/NUMBER_GAME_ATOMIC_PARTICLE_DEPTH3_MECHANICS_PROTOCOL_20260813.md"
PARENT_PROTOCOL_SHA256 = "e3141377bc9c65cb54550a4e7319c192e6a14b85d13181e0fdce79e77a2105ea"
MODEL_ID = "qwen/qwen3.7-plus"
MAX_TOKENS = 220
TEMPERATURE = .85
STAGE_CAP_USD = 3.50
BLOCK_SIZE = 256
GRAMMAR_SHA256 = "6e2a523d7d7c0c59b5df0494de37087525c7009d43a5eaba01390e005d85e0e0"


class Adapter(Protocol):
    def complete(self, messages, seeds, *, response_format, max_tokens): ...
    def usage_snapshot(self): ...
    def records(self): ...


@dataclass
class SeedCursor:
    index: int = 0

    def take(self, count: int) -> tuple[int, ...]:
        values = MODEL_SEEDS[self.index : self.index + count]
        if len(values) != count:
            raise ValueError("atomic particle seed schedule exhausted")
        self.index += count
        return values


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool_key(value: bool) -> str:
    return "1" if value else "0"


def _history_key(history: Sequence[tuple[int, bool]]) -> str:
    return "|".join(f"{number}:{int(answer)}" for number, answer in history)


def _call_blocks(
    *,
    adapter: Adapter,
    requests: Sequence[dict[str, Any]],
    raw: dict[str, Any],
    raw_path: Path,
    block_authorizer: Callable[[Sequence[dict[str, Any]]], None] | None = None,
) -> list[str]:
    responses: list[str] = []
    for start in range(0, len(requests), BLOCK_SIZE):
        block = requests[start : start + BLOCK_SIZE]
        if block_authorizer is not None:
            block_authorizer(block)
        values = adapter.complete(
            [row["messages"] for row in block],
            [row["seed"] for row in block],
            response_format=response_format(),
            max_tokens=MAX_TOKENS,
        )
        if len(values) != len(block):
            raise ValueError("atomic adapter omitted responses")
        for request, response in zip(block, values, strict=True):
            raw["responses"].append({
                **{key: value for key, value in request.items() if key != "messages"},
                "prompt_sha256": hashlib.sha256(json.dumps(request["messages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "response": response,
            })
            responses.append(response)
        checkpoint(raw_path, raw)
    return responses


def _requests(
    cursor: SeedCursor,
    *,
    stage: str,
    tree: int,
    observations: Sequence[tuple[int, bool]],
    protected: Sequence[int],
    count: int,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    seeds = cursor.take(count)
    requests = []
    for slot, seed in enumerate(seeds):
        anchors, signature, messages = request_identity(
            observations,
            protected=protected,
            slot=slot,
            count=count,
        )
        requests.append({
            "stage": stage,
            "tree": tree,
            "slot": slot,
            "seed": seed,
            "observations": [[number, answer] for number, answer in observations],
            "anchors": list(anchors),
            "signature": "".join("1" if value else "0" for value in signature),
            **dict(metadata or {}),
            "messages": messages,
        })
    return requests


def _parse(
    values: Sequence[str],
    history: Sequence[tuple[int, bool]],
    anchors: Sequence[int],
) -> tuple[list[RuleHypothesis], list[RuleHypothesis | None], dict[str, Any]]:
    rows: list[AtomicResponse] = [
        parse_stratified(
            value,
            history,
            anchors,
            tuple(bit == "1" for bit in format(slot // (len(values) // 32), "05b")),
        )
        for slot, value in enumerate(values)
    ]
    slots = [row.hypothesis for row in rows]
    valid = [row for row in slots if row is not None]
    rejected: dict[str, int] = {}
    for row in rows:
        if row.rejection:
            rejected[row.rejection] = rejected.get(row.rejection, 0) + 1
    return valid, slots, {
        "slots": len(values),
        "valid_particles": len(valid),
        "unique_extensions": len({row.extension for row in valid}),
        "unique_signatures": len({particle_signature(row, anchors) for row in valid}),
        "rejections": rejected,
    }


def _slot_half(slots: Sequence[RuleHypothesis | None], parity: int) -> list[RuleHypothesis]:
    return [row for index, row in enumerate(slots) if index % 2 == parity and row is not None]


def _require_group(valid: Sequence[RuleHypothesis], diagnostic: Mapping[str, Any], *, initial: bool = False) -> None:
    minimum_valid, minimum_unique = (48, 24) if initial else (24, 16)
    minimum_signatures = 24
    if (
        len(valid) < minimum_valid
        or int(diagnostic["unique_extensions"]) < minimum_unique
        or int(diagnostic["unique_signatures"]) < minimum_signatures
    ):
        raise ValueError("atomic particle group misses valid or unique floor")


def _filter_blind(
    pool: Sequence[RuleHypothesis],
    history: Sequence[tuple[int, bool]],
    anchors: Sequence[int],
) -> tuple[list[RuleHypothesis], dict[str, Any]]:
    valid = [row for row in pool if all(row.extension[number] is answer for number, answer in history)]
    diagnostic = {
        "slots": len(pool),
        "valid_particles": len(valid),
        "unique_extensions": len({row.extension for row in valid}),
        "unique_signatures": len({particle_signature(row, anchors) for row in valid}),
        "rejections": {"history": len(pool) - len(valid)},
    }
    if (
        len(valid) < 24
        or diagnostic["unique_extensions"] < 16
        or diagnostic["unique_signatures"] < 16
    ):
        raise ValueError("blind particle branch misses valid, unique, or signature floor")
    return valid, diagnostic


def transport_summary(adapter: Adapter) -> dict[str, Any]:
    snapshot = adapter.usage_snapshot()
    records = list(adapter.records())
    totals = {
        "accepted_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retries": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "forced_final_requests": int(snapshot.get("forced_final_requests", 0)),
        "cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
    }
    gates = {
        "exact_6400_accepted_and_http": totals["accepted_requests"] == totals["http_attempts"] == len(MODEL_SEEDS),
        "zero_retries_reasoning_forced": (
            totals["retries"]
            == totals["reasoning_tokens"]
            == totals["forced_exits"]
            == totals["forced_final_requests"]
            == 0
        ),
        "all_clean_stops": len(records) == len(MODEL_SEEDS) and all(row.get("finish_reasons") == ["stop"] for row in records),
        "exact_model_and_seeds": (
            len(records) == len(MODEL_SEEDS)
            and {int(row.get("seed", -1)) for row in records} == set(MODEL_SEEDS)
            and all(row.get("model_requested") == row.get("model_returned") == MODEL_ID for row in records)
        ),
        "within_stage_cap": totals["cost_usd"] <= STAGE_CAP_USD + 1e-12,
        "usage_values_finite_nonnegative": all(math.isfinite(value) and value >= 0 for value in totals.values()),
    }
    return {"totals": totals, "records": records, "gates": gates}


def _build_plans(
    *,
    tree: int,
    initial: Sequence[RuleHypothesis],
    roots: Sequence[int],
    first_conditioned: Mapping[int, Mapping[bool, Sequence[RuleHypothesis]]],
    first_blind: Sequence[RuleHypothesis],
    second_conditioned: Mapping[int, Mapping[tuple[bool, bool], Sequence[RuleHypothesis]]],
    second_blind: Sequence[RuleHypothesis],
    width: int = BELIEF_WIDTH,
) -> dict[str, dict[int, RootPlan]]:
    dynamic: dict[int, RootPlan] = {}
    blind: dict[int, RootPlan] = {}
    fixed: dict[int, RootPlan] = {}
    seed = TREE_SEEDS[tree]
    for root in roots:
        dynamic[root] = generated_depth3_score(
            initial,
            root,
            first_conditioned[root],
            second_conditioned[root],
            seed=seed,
            width=width,
        )
        first_blind_map = {False: first_blind, True: first_blind}
        second_blind_map = {(a, b): second_blind for a in (False, True) for b in (False, True)}
        blind[root] = generated_depth3_score(initial, root, first_blind_map, second_blind_map, seed=seed, width=width)
        fixed[root] = fixed_depth3_score(initial, root, seed=seed)
    return {"dynamic": dynamic, "blind": blind, "fixed": fixed}


def _half_bank_stability(*, tree: int, bank: Mapping[str, Any]) -> dict[str, Any]:
    full_root = choose_plan(bank["plans"]["dynamic"])
    halves: list[dict[str, Any]] = []
    for parity in (0, 1):
        try:
            initial_raw = _slot_half(bank["initial_slots"], parity)
            initial = systematic_resample(initial_raw, BELIEF_WIDTH // 2, TREE_SEEDS[tree])
            plans: dict[int, RootPlan] = {}
            for root in bank["roots"]:
                first = {
                    answer: _slot_half(bank["first_conditioned_slots"][root][answer], parity)
                    for answer in (False, True)
                }
                second = {
                    answers: _slot_half(bank["second_conditioned_slots"][root][answers], parity)
                    for answers in bank["second_conditioned_slots"][root]
                }
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
                "agrees_with_full": selected == full_root,
                "scores": {str(root): plan.score for root, plan in plans.items()},
            })
        except (KeyError, ValueError) as error:
            halves.append({
                "parity": parity,
                "status": "invalid",
                "selected_root": None,
                "agrees_with_full": False,
                "error": str(error),
            })
    return {
        "full_selected_root": full_root,
        "halves": halves,
        "both_halves_agree": all(row["agrees_with_full"] for row in halves),
    }


def produce_bank(
    *,
    output_dir: Path,
    adapter: Adapter,
    block_authorizer: Callable[[Sequence[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    if PROTOCOL_SHA256 and digest(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("atomic particle protocol changed")
    if digest(PARENT_PROTOCOL) != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("atomic particle parent protocol changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    raw_path = private / "RAW_RESPONSES.json"
    topology_path = private / "TOPOLOGY.json"
    raw: dict[str, Any] = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "responses": []}
    cursor = SeedCursor()
    banks: list[dict[str, Any]] = [{"diagnostics": {}} for _ in TREE_SEEDS]

    for tree in range(len(TREE_SEEDS)):
        request = _requests(cursor, stage="initial", tree=tree, observations=(), protected=(), count=INITIAL_SLOTS)
        values = _call_blocks(adapter=adapter, requests=request, raw=raw, raw_path=raw_path, block_authorizer=block_authorizer)
        initial_anchors = tuple(request[0]["anchors"])
        particles, slots, diagnostic = _parse(values, (), initial_anchors)
        _require_group(particles, diagnostic, initial=True)
        banks[tree]["initial_raw"] = particles
        banks[tree]["initial_slots"] = slots
        banks[tree]["initial"] = systematic_resample(particles, BELIEF_WIDTH, TREE_SEEDS[tree])
        banks[tree]["diagnostics"]["initial"] = diagnostic
        banks[tree]["initial_anchors"] = initial_anchors
        banks[tree]["roots"] = candidate_roots(banks[tree]["initial"], seed=TREE_SEEDS[tree])

    for tree, bank in enumerate(banks):
        first_conditioned: dict[int, dict[bool, list[RuleHypothesis]]] = {}
        for root in bank["roots"]:
            first_conditioned[root] = {}
            for answer in (False, True):
                history = ((root, answer),)
                requests = _requests(cursor, stage="first_conditioned", tree=tree, observations=history, protected=(root,), count=GENERATED_SLOTS, metadata={"root": root, "first_answer": answer})
                values = _call_blocks(adapter=adapter, requests=requests, raw=raw, raw_path=raw_path, block_authorizer=block_authorizer)
                anchors = tuple(requests[0]["anchors"])
                particles, slots, diagnostic = _parse(values, history, anchors)
                _require_group(particles, diagnostic)
                first_conditioned[root][answer] = particles
                bank.setdefault("first_conditioned_slots", {}).setdefault(root, {})[answer] = slots
                bank.setdefault("first_conditioned_anchors", {}).setdefault(root, {})[answer] = anchors
                bank["diagnostics"][f"first_conditioned:{root}:{int(answer)}"] = diagnostic
        bank["first_conditioned"] = first_conditioned

    for tree, bank in enumerate(banks):
        requests = _requests(cursor, stage="first_blind", tree=tree, observations=(), protected=bank["roots"], count=256)
        values = _call_blocks(adapter=adapter, requests=requests, raw=raw, raw_path=raw_path, block_authorizer=block_authorizer)
        anchors = tuple(requests[0]["anchors"])
        pool, slots, diagnostic = _parse(values, (), anchors)
        bank["first_blind"] = pool
        bank["first_blind_slots"] = slots
        bank["first_blind_anchors"] = anchors
        bank["diagnostics"]["first_blind_pool"] = diagnostic
        for root in bank["roots"]:
            for answer in (False, True):
                _, filtered = _filter_blind(pool, ((root, answer),), anchors)
                bank["diagnostics"][f"first_blind:{root}:{int(answer)}"] = filtered

    for tree, bank in enumerate(banks):
        second_conditioned: dict[int, dict[tuple[bool, bool], list[RuleHypothesis]]] = {}
        for root in bank["roots"]:
            second_conditioned[root] = {}
            for first_answer in (False, True):
                history1 = ((root, first_answer),)
                first = refresh_belief(bank["initial"], bank["first_conditioned"][root][first_answer], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(first_answer)))
                second_query, _ = best_query(first, (root,))
                for second_answer in (False, True):
                    history2 = (*history1, (second_query, second_answer))
                    requests = _requests(cursor, stage="second_conditioned", tree=tree, observations=history2, protected=(root, second_query), count=GENERATED_SLOTS, metadata={"root": root, "first_answer": first_answer, "second_query": second_query, "second_answer": second_answer})
                    values = _call_blocks(adapter=adapter, requests=requests, raw=raw, raw_path=raw_path, block_authorizer=block_authorizer)
                    anchors = tuple(requests[0]["anchors"])
                    particles, slots, diagnostic = _parse(values, history2, anchors)
                    _require_group(particles, diagnostic)
                    second_conditioned[root][(first_answer, second_answer)] = particles
                    bank.setdefault("second_conditioned_slots", {}).setdefault(root, {})[(first_answer, second_answer)] = slots
                    bank.setdefault("second_conditioned_anchors", {}).setdefault(root, {})[(first_answer, second_answer)] = anchors
                    bank["diagnostics"][f"second_conditioned:{root}:{int(first_answer)}:{second_query}:{int(second_answer)}"] = diagnostic
        bank["second_conditioned"] = second_conditioned

    for tree, bank in enumerate(banks):
        protected = set(bank["roots"])
        second_query_sets: dict[tuple[int, bool], set[int]] = {}
        for root in bank["roots"]:
            for first_answer in (False, True):
                history1 = ((root, first_answer),)
                conditioned_first = refresh_belief(bank["initial"], bank["first_conditioned"][root][first_answer], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(first_answer)))
                blind_first = refresh_belief(bank["initial"], bank["first_blind"], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(first_answer)))
                queries = {best_query(conditioned_first, (root,))[0], best_query(blind_first, (root,))[0]}
                second_query_sets[(root, first_answer)] = queries
                protected.update(queries)
        requests = _requests(cursor, stage="second_blind", tree=tree, observations=(), protected=tuple(sorted(protected)), count=512)
        values = _call_blocks(adapter=adapter, requests=requests, raw=raw, raw_path=raw_path, block_authorizer=block_authorizer)
        anchors = tuple(requests[0]["anchors"])
        pool, slots, diagnostic = _parse(values, (), anchors)
        bank["second_blind"] = pool
        bank["second_blind_slots"] = slots
        bank["second_blind_anchors"] = anchors
        bank["diagnostics"]["second_blind_pool"] = diagnostic
        for root in bank["roots"]:
            for first_answer in (False, True):
                history1 = ((root, first_answer),)
                for query in second_query_sets[(root, first_answer)]:
                    for second_answer in (False, True):
                        _, filtered = _filter_blind(pool, (*history1, (query, second_answer)), anchors)
                        bank["diagnostics"][f"second_blind:{root}:{int(first_answer)}:{query}:{int(second_answer)}"] = filtered
        bank["plans"] = _build_plans(
            tree=tree,
            initial=bank["initial"],
            roots=bank["roots"],
            first_conditioned=bank["first_conditioned"],
            first_blind=bank["first_blind"],
            second_conditioned=bank["second_conditioned"],
            second_blind=bank["second_blind"],
        )
        bank["half_bank_stability"] = _half_bank_stability(tree=tree, bank=bank)

    if cursor.index != len(MODEL_SEEDS):
        raise RuntimeError(f"atomic request schedule incomplete: {cursor.index}")
    transport = transport_summary(adapter)
    if not all(transport["gates"].values()):
        raise RuntimeError("atomic particle transport gate failed")
    topology = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "request_count": cursor.index,
        "raw_response_sha256": digest(raw_path),
        "transport": transport,
        "trees": [
            {
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
                    family: {str(root): {"score": plan.score, "second_queries": {_bool_key(answer): query for answer, query in plan.second_queries.items()}} for root, plan in plans.items()}
                    for family, plans in bank["plans"].items()
                },
            }
            for tree, bank in enumerate(banks)
        ],
        "canonical_targets_opened": False,
        "classical_grammar_opened": False,
    }
    checkpoint(topology_path, topology)
    return {"raw": raw, "topology": topology, "banks": banks}


def _canonical_belief(targets: Sequence[RuleHypothesis], history: Sequence[tuple[int, bool]]) -> list[RuleHypothesis]:
    return [target for target in targets if all(target.extension[number] is answer for number, answer in history)]


def _mse(left: Sequence[float], right: Sequence[float], excluded: Sequence[int]) -> float:
    blocked = set(excluded)
    values = [(left[q] - right[q]) ** 2 for q in range(101) if q not in blocked]
    return sum(values) / len(values)


def _refresh_or_none(
    parent: Sequence[RuleHypothesis] | None,
    generated: Sequence[RuleHypothesis],
    history: Sequence[tuple[int, bool]],
    *,
    seed: int,
) -> list[RuleHypothesis] | None:
    if parent is None:
        return None
    try:
        return refresh_belief(parent, generated, history, seed=seed)
    except ValueError:
        return None


def _calibration_values(
    belief: Sequence[RuleHypothesis] | None,
    exact: Sequence[float],
    canonical: Sequence[RuleHypothesis],
    excluded: Sequence[int],
) -> tuple[float, float, bool]:
    if belief is None:
        return 1.0, 0.0, True
    return (
        _mse(predictive_probabilities(belief), exact, excluded),
        sum(any(row.extension == target.extension for row in belief) for target in canonical) / len(canonical),
        False,
    )


def _trajectory(
    *,
    policy: str,
    root: int,
    truth: RuleHypothesis,
    tree: int,
    bank: Mapping[str, Any],
) -> tuple[list[RuleHypothesis] | None, tuple[int, ...]]:
    seed = TREE_SEEDS[tree]
    answer1 = truth.extension[root]
    history1 = ((root, answer1),)
    try:
        if policy == "fixed_support_depth3":
            first = systematic_resample([row for row in bank["initial"] if row.extension[root] is answer1], BELIEF_WIDTH, seed ^ (root * 17 + int(answer1)))
            second = bank["plans"]["fixed"][root].second_queries[answer1]
            answer2 = truth.extension[second]
            second_belief = systematic_resample([row for row in first if row.extension[second] is answer2], BELIEF_WIDTH, seed ^ (root * 101 + second * 7 + int(answer2)))
        else:
            blind = policy == "history_blind_depth3"
            first_generated = bank["first_blind"] if blind else bank["first_conditioned"][root][answer1]
            first = refresh_belief(bank["initial"], first_generated, history1, seed=seed ^ (root * 17 + int(answer1)))
            family = "blind" if blind else "dynamic"
            second = bank["plans"][family][root].second_queries[answer1]
            answer2 = truth.extension[second]
            history2 = (*history1, (second, answer2))
            second_generated = bank["second_blind"] if blind else bank["second_conditioned"][root][(answer1, answer2)]
            second_belief = refresh_belief(first, second_generated, history2, seed=seed ^ (root * 101 + second * 7 + int(answer2)))
        third, _ = best_query(second_belief, (root, second))
        answer3 = truth.extension[third]
        terminal = systematic_resample([row for row in second_belief if row.extension[third] is answer3], BELIEF_WIDTH, seed ^ (root * 1009 + second * 101 + third * 7 + int(answer3)))
        return terminal, (root, second, third)
    except (KeyError, ValueError):
        return None, (root,)


def _trajectory_score(terminal: Sequence[RuleHypothesis] | None, truth: RuleHypothesis, queries: Sequence[int]) -> tuple[float, bool, bool]:
    if terminal is None:
        return 1.0, False, True
    return (
        posterior_predictive_brier(terminal, truth, queries),
        any(row.extension == truth.extension for row in terminal),
        False,
    )


def score_complete_bank(bank_result: Mapping[str, Any]) -> dict[str, Any]:
    banks = bank_result["banks"]
    targets = canonical_targets()
    grammar, grammar_diagnostic = build_classical_grammar_bank()
    if grammar_diagnostic.get("sha256") != GRAMMAR_SHA256:
        raise RuntimeError("classical grammar bank changed")
    if len(targets) != 33 or len({row.extension for row in targets}) != 33:
        raise RuntimeError("canonical target bank changed")
    tree_rows = []
    score_values, realized_values = [], []
    dynamic_vs_blind_mse: list[float] = []
    blind_mse_values: list[float] = []
    dynamic_vs_blind_coverage: list[float] = []
    calibration_collapse = {"conditioned": 0, "blind": 0}
    novel_masks: set[int] = set()
    second_masks: list[int] = []
    for tree, bank in enumerate(banks):
        tree_novel = False
        for root in bank["roots"]:
            for values in bank["second_conditioned"][root].values():
                for row in values:
                    mask = extension_mask(row.extension)
                    second_masks.append(mask)
                    if mask not in grammar:
                        novel_masks.add(mask)
                        tree_novel = True
        roots = bank["roots"]
        selections = {
            "dynamic_depth3": choose_plan(bank["plans"]["dynamic"]),
            "history_blind_depth3": choose_plan(bank["plans"]["blind"]),
            "fixed_support_depth3": choose_plan(bank["plans"]["fixed"]),
            "call_matched_myopic": max(roots, key=lambda root: (sum(row.extension[root] for row in bank["initial"]) * (len(bank["initial"]) - sum(row.extension[root] for row in bank["initial"])), -root)),
            "deterministic_random": random.Random(TREE_SEEDS[tree]).choice(list(roots)),
        }
        policy_scores: dict[str, list[float]] = {key: [] for key in selections}
        policy_coverage: dict[str, list[bool]] = {key: [] for key in selections}
        policy_collapse: dict[str, list[bool]] = {key: [] for key in selections}
        root_realized: dict[int, float] = {}
        for root in roots:
            values = []
            for truth in targets:
                terminal, queries = _trajectory(policy="dynamic_depth3", root=root, truth=truth, tree=tree, bank=bank)
                values.append(_trajectory_score(terminal, truth, queries)[0])
            root_realized[root] = sum(values) / len(values)
            score_values.append(bank["plans"]["dynamic"][root].score)
            realized_values.append(-root_realized[root])
        for policy, root in selections.items():
            replay_policy = "dynamic_depth3" if policy in {"call_matched_myopic", "deterministic_random"} else policy
            for truth in targets:
                terminal, queries = _trajectory(policy=replay_policy, root=root, truth=truth, tree=tree, bank=bank)
                brier, covered, collapsed = _trajectory_score(terminal, truth, queries)
                policy_scores[policy].append(brier)
                policy_coverage[policy].append(covered)
                policy_collapse[policy].append(collapsed)
        for root in roots:
            for answer1 in (False, True):
                root_probability = answer_probability(bank["initial"], root)
                if (root_probability if answer1 else 1.0 - root_probability) == 0:
                    continue
                history1 = ((root, answer1),)
                dynamic_first = _refresh_or_none(bank["initial"], bank["first_conditioned"][root][answer1], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(answer1)))
                blind_first = _refresh_or_none(bank["initial"], bank["first_blind"], history1, seed=TREE_SEEDS[tree] ^ (root * 17 + int(answer1)))
                second = bank["plans"]["dynamic"][root].second_queries[answer1]
                for answer2 in (False, True):
                    second_probability = answer_probability(dynamic_first, second) if dynamic_first is not None else 0.0
                    if (second_probability if answer2 else 1.0 - second_probability) == 0:
                        continue
                    history2 = (*history1, (second, answer2))
                    canonical = _canonical_belief(targets, history2)
                    if not canonical:
                        continue
                    dynamic_second = _refresh_or_none(dynamic_first, bank["second_conditioned"][root][(answer1, answer2)], history2, seed=TREE_SEEDS[tree] ^ (root * 101 + second * 7 + int(answer2)))
                    blind_second = _refresh_or_none(blind_first, bank["second_blind"], history2, seed=TREE_SEEDS[tree] ^ (root * 101 + second * 7 + int(answer2)))
                    exact = predictive_probabilities(canonical)
                    dynamic_mse, dynamic_coverage_at_history, dynamic_collapsed = _calibration_values(dynamic_second, exact, canonical, (root, second))
                    blind_mse, blind_coverage_at_history, blind_collapsed = _calibration_values(blind_second, exact, canonical, (root, second))
                    calibration_collapse["conditioned"] += int(dynamic_collapsed)
                    calibration_collapse["blind"] += int(blind_collapsed)
                    dynamic_vs_blind_mse.append(dynamic_mse - blind_mse)
                    blind_mse_values.append(blind_mse)
                    dynamic_vs_blind_coverage.append(dynamic_coverage_at_history - blind_coverage_at_history)
        bank["has_grammar_novel_second_refresh"] = tree_novel
        means = {policy: sum(values) / len(values) for policy, values in policy_scores.items()}
        coverages = {policy: sum(values) / len(values) for policy, values in policy_coverage.items()}
        collapse_rates = {policy: sum(values) / len(values) for policy, values in policy_collapse.items()}
        tree_rows.append({
            "tree": tree,
            "roots": list(roots),
            "selections": selections,
            "mean_brier": means,
            "coverage": coverages,
            "particle_collapse_rate": collapse_rates,
            "root_realized_dynamic_brier": {str(root): root_realized[root] for root in roots},
            "dynamic_selected_rank": sorted(roots, key=lambda root: (root_realized[root], root)).index(selections["dynamic_depth3"]) + 1,
        })
    dynamic_mean = sum(row["mean_brier"]["dynamic_depth3"] for row in tree_rows) / len(tree_rows)
    myopic_mean = sum(row["mean_brier"]["call_matched_myopic"] for row in tree_rows) / len(tree_rows)
    blind_mean = sum(row["mean_brier"]["history_blind_depth3"] for row in tree_rows) / len(tree_rows)
    random_mean = sum(row["mean_brier"]["deterministic_random"] for row in tree_rows) / len(tree_rows)
    dynamic_coverage = sum(row["coverage"]["dynamic_depth3"] for row in tree_rows) / len(tree_rows)
    blind_coverage = sum(row["coverage"]["history_blind_depth3"] for row in tree_rows) / len(tree_rows)
    validity = all(
        all(
            (key == "initial" and value["valid_particles"] >= 48 and value["unique_extensions"] >= 24)
            or (key.endswith("_pool"))
            or (key != "initial" and value["valid_particles"] >= 24 and value["unique_extensions"] >= 16)
            for key, value in bank["diagnostics"].items()
        )
        for bank in banks
    )
    signature_validity = all(
        all(
            (key == "initial" and value["unique_signatures"] >= 24)
            or (key.endswith("_pool"))
            or (key != "initial" and value["unique_signatures"] >= (16 if key.startswith(("first_blind:", "second_blind:")) else 24))
            for key, value in bank["diagnostics"].items()
        )
        for bank in banks
    )
    dynamic_blind_mse_mean = sum(dynamic_vs_blind_mse) / len(dynamic_vs_blind_mse)
    dynamic_blind_coverage_mean = sum(dynamic_vs_blind_coverage) / len(dynamic_vs_blind_coverage)
    gates = {
        "all_particle_groups_pass_floors": validity,
        "all_signature_coverage_floors_pass": signature_validity,
        "both_half_estimators_agree_on_at_least_3_trees": sum(bank["half_bank_stability"]["both_halves_agree"] for bank in banks) >= 3,
        "dynamic_and_myopic_roots_differ_on_at_least_3_trees": sum(row["selections"]["dynamic_depth3"] != row["selections"]["call_matched_myopic"] for row in tree_rows) >= 3,
        "dynamic_and_blind_roots_differ_on_at_least_2_trees": sum(row["selections"]["dynamic_depth3"] != row["selections"]["history_blind_depth3"] for row in tree_rows) >= 2,
        "conditioned_mse_at_least_5_percent_below_blind": dynamic_blind_mse_mean <= -.05 * (sum(blind_mse_values) / len(blind_mse_values)),
        "conditioned_coverage_at_least_5_points_above_blind": dynamic_blind_coverage_mean >= .05,
        "score_realized_root_spearman_at_least_point_40": spearman(score_values, realized_values) >= .40,
        "dynamic_selected_root_top2_on_at_least_3_trees": sum(row["dynamic_selected_rank"] <= 2 for row in tree_rows) >= 3,
        "dynamic_brier_at_least_3_percent_below_myopic": dynamic_mean <= .97 * myopic_mean,
        "dynamic_wins_at_least_3_tree_means": sum(row["mean_brier"]["dynamic_depth3"] < row["mean_brier"]["call_matched_myopic"] for row in tree_rows) >= 3,
        "dynamic_nonworse_than_random": dynamic_mean <= random_mean,
        "dynamic_nonworse_than_blind_brier": dynamic_mean <= blind_mean,
        "dynamic_nonworse_than_blind_coverage": dynamic_coverage >= blind_coverage,
    }
    return {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "atomic_particle_mechanics_pass" if all(gates.values()) else "atomic_particle_mechanics_null",
        "authorizes": "fresh_powered_development_protocol_only" if all(gates.values()) else "nothing",
        "gates": gates,
        "metrics": {
            "dynamic_mean_brier": dynamic_mean,
            "myopic_mean_brier": myopic_mean,
            "blind_mean_brier": blind_mean,
            "random_mean_brier": random_mean,
            "dynamic_relative_brier_reduction_vs_myopic": (myopic_mean - dynamic_mean) / myopic_mean,
            "dynamic_minus_blind_second_refresh_mse": dynamic_blind_mse_mean,
            "dynamic_minus_blind_second_refresh_coverage": dynamic_blind_coverage_mean,
            "score_realized_root_spearman": spearman(score_values, realized_values),
            "grammar_novel_unique_fraction": len(novel_masks) / max(len(set(second_masks)), 1),
            "grammar_novelty_descriptive_only": {
                "at_least_10_percent": len(novel_masks) / max(len(set(second_masks)), 1) >= .10,
                "every_tree_has_novel_second_refresh": all(bank["has_grammar_novel_second_refresh"] for bank in banks),
                "gate_authority": False,
            },
            "grammar_bank": grammar_diagnostic,
            "intermediate_calibration_collapse_counts": calibration_collapse,
        },
        "trees": tree_rows,
        "targets_opened_after_complete_bank": True,
        "classical_grammar_opened_after_complete_bank": True,
        "development_confirmation_opened": False,
    }
