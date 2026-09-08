"""History-only enumerative proposal control over an explicit finite grammar.

This is a bounded grammar baseline, not exhaustive search over all compiled rules
or a claim of compute matching with an LLM. No target scenes enter the interface.
"""

from dataclasses import dataclass
import hashlib
from itertools import combinations, combinations_with_replacement
import json
from time import monotonic

from .belief import UnsupportedHistory
from .rules import ATTRIBUTES, MAX_OBJECTS, compile_rule, parse_scene, RuleError


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def candidates():
    """Finite syntax, increasing AST node count with explicit stable tie ordering.

    2: counts of atomic attributes.
    3: negated atomic counts, atomic same/different pairs, negated-attribute counts.
    4: counts of two-attribute Boolean predicates.
    5: and/or of two atomic counts.
    Higher composition and arbitrary count nesting are deliberately not covered.
    """
    predicates = [
        {"op": "is", "attribute": attr, "value": val}
        for attr, values in ATTRIBUTES.items()
        for val in values
    ]

    def counts(pred):
        for comparison in ("eq", "ge", "le"):
            for n in range(MAX_OBJECTS + 1):
                yield {"op": "count", "where": pred, "comparison": comparison, "n": n}

    atoms = [c for pred in predicates for c in counts(pred)]
    for atom in atoms:
        yield 2, atom
    for atom in atoms:
        yield 3, {"op": "not", "arg": atom}
    for left, right in combinations_with_replacement(predicates, 2):
        for attr in ATTRIBUTES:
            for relation in ("same", "different"):
                yield (
                    3,
                    {
                        "op": "pair",
                        "left": left,
                        "right": right,
                        "attribute": attr,
                        "relation": relation,
                    },
                )
    # Negated object predicates have size two, hence a count has size three.
    for pred in predicates:
        for candidate in counts({"op": "not", "arg": pred}):
            yield 3, candidate
    for left, right in combinations(predicates, 2):
        for op in ("and", "or"):
            for candidate in counts({"op": op, "args": [left, right]}):
                yield 4, candidate
    for left, right in combinations(atoms, 2):
        for op in ("and", "or"):
            yield 5, {"op": op, "args": [left, right]}


@dataclass(frozen=True)
class SearchResult:
    selected_json: tuple
    records: tuple
    candidates_evaluated: int
    scene_evaluations: int
    grammar_exhausted: bool
    history_sha256: str
    elapsed_seconds: float


def propose(history, *, max_candidates=4096, max_proposals=4):
    if type(max_candidates) is not int or not 1 <= max_candidates <= 50000:
        raise RuleError("candidate budget must be integer 1 to 50000")
    if type(max_proposals) is not int or not 1 <= max_proposals <= 32:
        raise RuleError("proposal count must be integer 1 to 32")
    if type(history) not in (list, tuple) or len(history) > 64:
        raise RuleError("history must contain at most 64 records")
    normalized, labels = [], {}
    for scene, label in history:
        if type(label) is not bool:
            raise RuleError("Boolean labels required")
        key = parse_scene(scene)
        if key in labels and labels[key] != label:
            raise UnsupportedHistory("contradictory membership observations")
        labels[key] = label
        normalized.append(
            ({"objects": [dict(zip(ATTRIBUTES, obj)) for obj in key]}, label)
        )
    history_sha = hashlib.sha256(canonical_json(normalized).encode()).hexdigest()
    start = monotonic()
    records, eligible, seen = [], [], set()
    scene_evaluations = 0
    stream = iter(candidates())
    exhausted = False
    for _ in range(max_candidates):
        item = next(stream, None)
        if item is None:
            exhausted = True
            break
        size, candidate = item
        raw = canonical_json(candidate)
        compiled = compile_rule(raw)
        duplicate = compiled.key in seen
        seen.add(compiled.key)
        # Score all observed examples, without early exit that obscures work counts.
        mismatches = sum(compiled.label(s) != y for s, y in normalized)
        scene_evaluations += len(normalized)
        records.append((raw, size, mismatches, duplicate))
        if not mismatches and not duplicate:
            eligible.append((size, raw))
    if not exhausted:
        exhausted = next(stream, None) is None
    return SearchResult(
        tuple(raw for _, raw in sorted(eligible)[:max_proposals]),
        tuple(records),
        len(records),
        scene_evaluations,
        exhausted,
        history_sha,
        monotonic() - start,
    )
