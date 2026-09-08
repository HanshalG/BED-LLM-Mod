from itertools import combinations_with_replacement, permutations, product
import json

import pytest

from environments.semantic_scene.rules import RuleError, compile_rule, parse_scene


def predicate(attribute="color", value="red"):
    return {"op": "is", "attribute": attribute, "value": value}


def count(where=None, n=1, comparison="ge"):
    return {
        "op": "count",
        "where": where or predicate(),
        "n": n,
        "comparison": comparison,
    }


def rule(value):
    return compile_rule(json.dumps(value))


def scene(colors):
    return {
        "objects": [{"color": c, "shape": "block", "size": "small"} for c in colors]
    }


def test_count_against_independent_full_color_enumeration():
    for size in range(1, 5):
        for colors in product(("red", "blue", "yellow"), repeat=size):
            for threshold in range(5):
                for comparison in ("eq", "le", "ge"):
                    actual = rule(count(n=threshold, comparison=comparison)).label(
                        scene(colors)
                    )
                    expected_count = len([c for c in colors if c == "red"])
                    expected = {
                        "eq": expected_count == threshold,
                        "le": expected_count <= threshold,
                        "ge": expected_count >= threshold,
                    }[comparison]
                    assert actual is expected


def test_pair_requires_distinct_objects_and_preserves_multiplicity():
    pair = {
        "op": "pair",
        "left": predicate(),
        "right": predicate(),
        "relation": "same",
        "attribute": "shape",
    }
    compiled = rule(pair)
    assert not compiled.label(scene(["red"]))
    assert compiled.label(scene(["red", "red"]))
    assert len(parse_scene(scene(["red", "red"]))) == 2
    for colors in combinations_with_replacement(("red", "blue", "yellow"), 3):
        assert compiled.label(scene(colors)) == (colors.count("red") >= 2)


def test_boolean_composition_and_order_invariance():
    red, blue = count(), count(predicate(value="blue"))
    expr = {"op": "and", "args": [red, {"op": "not", "arg": blue}]}
    compiled = rule(expr)
    for colors in product(("red", "blue", "yellow"), repeat=3):
        expected = "red" in colors and "blue" not in colors
        for perm in permutations(colors):
            assert compiled.label(scene(perm)) is expected
    assert (
        rule({"op": "and", "args": [red, blue]}).key
        == rule({"op": "and", "args": [blue, red, red]}).key
    )


def test_object_conjunction_and_different_pair():
    p = {"op": "and", "args": [predicate(), predicate("size", "large")]}
    objects = [
        {"color": "red", "shape": "block", "size": "large"},
        {"color": "blue", "shape": "block", "size": "small"},
    ]
    assert rule(count(p)).label({"objects": objects})
    pair = {
        "op": "pair",
        "left": p,
        "right": predicate(value="blue"),
        "relation": "different",
        "attribute": "size",
    }
    assert rule(pair).label({"objects": objects})
    pair["relation"] = "same"
    assert not rule(pair).label({"objects": objects})


@pytest.mark.parametrize(
    "value",
    [
        predicate(),
        count(n=True),
        count(n=-1),
        count(comparison="gt"),
        count(where=count()),
        {"op": "python", "code": "1"},
        {
            "op": "count",
            "where": predicate(),
            "n": 1,
            "comparison": "ge",
            "truth": True,
        },
        {"op": "and", "args": []},
    ],
)
def test_invalid_rule_rejected(value):
    with pytest.raises(RuleError):
        rule(value)


@pytest.mark.parametrize(
    "raw",
    [
        '{"op":"not","op":"count"}',
        "NaN",
        "Infinity",
        "[]",
        "null",
        '"__import__("os")"',
        " " * 16385,
    ],
)
def test_invalid_raw_rejected(raw):
    with pytest.raises(RuleError):
        compile_rule(raw)


@pytest.mark.parametrize(
    "value",
    [
        {"objects": []},
        {"objects": [1]},
        {"objects": [{"color": "red", "shape": "block", "size": "huge"}]},
        {"objects": scene(["red"])["objects"] * 8},
        {"objects": [], "label": False},
    ],
)
def test_invalid_scene_rejected_not_false(value):
    with pytest.raises(RuleError):
        rule(count()).label(value)


def test_depth_and_node_limits_reject_before_canonical_dedup():
    expr = count()
    for _ in range(9):
        expr = {"op": "not", "arg": expr}
    with pytest.raises(RuleError, match="structural limit"):
        rule(expr)
    expr = count()
    for _ in range(3):
        expr = {"op": "and", "args": [expr] * 4}
    with pytest.raises(RuleError, match="structural limit"):
        rule(expr)
