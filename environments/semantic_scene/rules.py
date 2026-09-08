"""Independent, bounded JSON rule interpreter for abstract object multisets.

No geometry, rendering, executable Python, model judge, prior or task sampler.
Equal-looking objects remain separate objects; order is not observable.
"""

from dataclasses import dataclass
import json


ATTRIBUTES = {
    "color": ("red", "blue", "yellow"),
    "shape": ("block", "wedge", "pyramid"),
    "size": ("small", "large"),
}
MAX_OBJECTS = 7
MAX_NODES = 63
MAX_DEPTH = 8
MAX_BYTES = 16384


class RuleError(ValueError):
    pass


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise RuleError("unexpected fields")


def parse_scene(value):
    """Validate JSON-style input and return an immutable canonical multiset."""
    _keys(value, {"objects"})
    objects = value["objects"]
    if type(objects) is not list or not 1 <= len(objects) <= MAX_OBJECTS:
        raise RuleError("scene requires 1 to 7 objects")
    result = []
    for obj in objects:
        _keys(obj, ATTRIBUTES)
        for key, choices in ATTRIBUTES.items():
            if type(obj[key]) is not str or obj[key] not in choices:
                raise RuleError("invalid object attribute")
        result.append(tuple(obj[key] for key in ATTRIBUTES))
    return tuple(sorted(result))


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise RuleError("duplicate JSON field")
        obj[key] = value
    return obj


def _constant(value):
    raise RuleError("nonfinite JSON constant")


def _decode(raw):
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_BYTES:
        raise RuleError("rule response exceeds byte limit or is not text")
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, RecursionError) as exc:
        raise RuleError("invalid JSON rule") from exc


def _node(value, kind, depth, counter):
    counter[0] += 1
    if counter[0] > MAX_NODES or depth > MAX_DEPTH:
        raise RuleError("rule exceeds structural limit")
    if type(value) is not dict or type(value.get("op")) is not str:
        raise RuleError("expected typed rule object")
    op = value["op"]
    if op in ("and", "or"):
        _keys(value, {"op", "args"})
        args = value["args"]
        if type(args) is not list or not 2 <= len(args) <= 4:
            raise RuleError("and/or requires 2 to 4 operands")
        children = tuple(sorted(set(_node(a, kind, depth + 1, counter) for a in args)))
        return children[0] if len(children) == 1 else (op, children)
    if op == "not":
        _keys(value, {"op", "arg"})
        return (op, _node(value["arg"], kind, depth + 1, counter))
    if kind == "object" and op == "is":
        _keys(value, {"op", "attribute", "value"})
        attr, val = value["attribute"], value["value"]
        if type(attr) is not str or attr not in ATTRIBUTES:
            raise RuleError("unknown attribute")
        if type(val) is not str or val not in ATTRIBUTES[attr]:
            raise RuleError("unknown attribute value")
        return (op, attr, val)
    if kind == "scene" and op == "count":
        _keys(value, {"op", "where", "comparison", "n"})
        n, comparison = value["n"], value["comparison"]
        if type(n) is not int or not 0 <= n <= MAX_OBJECTS:
            raise RuleError("count threshold must be integer 0 to 7")
        if type(comparison) is not str or comparison not in ("eq", "ge", "le"):
            raise RuleError("invalid count comparison")
        return (op, comparison, n, _node(value["where"], "object", depth + 1, counter))
    if kind == "scene" and op == "pair":
        _keys(value, {"op", "left", "right", "relation", "attribute"})
        relation, attr = value["relation"], value["attribute"]
        if type(relation) is not str or relation not in ("same", "different"):
            raise RuleError("invalid pair relation")
        if type(attr) is not str or attr not in ATTRIBUTES:
            raise RuleError("unknown pair attribute")
        # Symmetric attribute relations: existential roles can be exchanged.
        sides = tuple(
            sorted(
                _node(value[k], "object", depth + 1, counter) for k in ("left", "right")
            )
        )
        return (op, relation, attr, sides)
    raise RuleError("operator not allowed at this type")


def _evaluate(node, value):
    op = node[0]
    if op in ("and", "or"):
        reducer = all if op == "and" else any
        return reducer(_evaluate(child, value) for child in node[1])
    if op == "not":
        return not _evaluate(node[1], value)
    if op == "is":
        return value[tuple(ATTRIBUTES).index(node[1])] == node[2]
    if op == "count":
        count = sum(_evaluate(node[3], obj) for obj in value)
        return {"eq": count == node[2], "ge": count >= node[2], "le": count <= node[2]}[
            node[1]
        ]
    if op == "pair":
        attr = tuple(ATTRIBUTES).index(node[2])
        return any(
            i != j
            and _evaluate(node[3][0], left)
            and _evaluate(node[3][1], right)
            and ((left[attr] == right[attr]) == (node[1] == "same"))
            for i, left in enumerate(value)
            for j, right in enumerate(value)
        )
    raise RuleError("invalid compiled operator")


@dataclass(frozen=True)
class CompiledRule:
    """Construct via compile_rule; key dedupes syntax, not all logical equivalents."""

    key: tuple

    def label(self, scene):
        return _evaluate(self.key, parse_scene(scene))


def compile_rule(raw):
    """All-or-nothing strict JSON compilation without evaluating caller code."""
    return CompiledRule(_node(_decode(raw), "scene", 1, [0]))
