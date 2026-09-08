from fractions import Fraction
import hashlib
import io
import json

import ijson
import pytest

from scripts.number_game_first_refresh_diagnostic import (
    brier,
    evaluate,
    prediction,
    selected_records,
)


def test_selected_stream_excludes_other_banked_subtrees():
    raw = {
        "trees": [
            {
                "tree_seed": 1,
                "initial": [{"expression": "a"}],
                "second_branches": {"unopened": [1, 2, 3]},
                "targets": ["x"],
            }
        ]
    }
    rows = list(
        selected_records(
            ijson.parse(io.StringIO(json.dumps(raw))), {"tree_seed", "initial"}
        )
    )
    assert rows == [{"tree_seed": 1, "initial": [{"expression": "a"}]}]


def fixture():
    extensions = {
        "a": (False,) * 101,
        "b": (True,) * 101,
        "c": (False,) + (True,) * 100,
        "d": (True,) + (False,) * 100,
    }

    def item(name):
        ext = extensions[name]
        return {
            "expression": name,
            "extension_sha256": hashlib.sha256(bytes(ext)).hexdigest(),
            "positive_count": sum(ext),
        }

    tree = {
        "tree_seed": 1,
        "roots": [0],
        "initial": [item("a"), item("b")],
        "generated_first_branches": {"0:0": [item("c")], "0:1": [item("d")]},
        "first_branches": {
            "0:0": [item("a"), item("c")],
            "0:1": [item("b"), item("d")],
        },
        "targets": [item("c"), item("d")],
    }
    return tree, extensions.__getitem__


def test_refresh_improvement_and_shared_pool_control_manual():
    tree, compiler = fixture()
    result = evaluate(tree, compiler)
    means = {a: Fraction(v) for a, v in result["exact_means"].items()}
    assert means["initial_filtered"] == Fraction(100, 101)
    assert means["retained_branch"] == Fraction(25, 101)
    assert means["all_first_proposals_shared_then_filtered"] == Fraction(25, 101)
    assert result["shared_count"] == 4
    tree["targets"] = tree["initial"]
    assert evaluate(tree, compiler)["shared_count"] == 4


def test_empty_predictor_is_explicit_half_and_brier_is_fixed_domain():
    assert prediction([]) == (Fraction(1, 2),) * 101
    assert brier(prediction([]), (False,) * 101) == Fraction(1, 4)


def test_branch_integrity_failures():
    tree, compiler = fixture()
    tree["first_branches"]["0:0"] = tree["initial"]
    with pytest.raises(ValueError, match="contradicts"):
        evaluate(tree, compiler)
    tree, compiler = fixture()
    tree["first_branches"]["0:0"] = tree["generated_first_branches"]["0:0"]
    with pytest.raises(ValueError, match="retained refresh"):
        evaluate(tree, compiler)
    tree, compiler = fixture()
    tree["targets"][0]["positive_count"] = -1
    with pytest.raises(ValueError, match="hash/count"):
        evaluate(tree, compiler)
