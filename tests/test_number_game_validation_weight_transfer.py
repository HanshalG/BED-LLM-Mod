from fractions import Fraction as F

import pytest

from scripts import number_game_validation_weight_transfer as audit


def test_exact_quadratic_solution_and_boundary_cases():
    assert audit.fit_weight(F(-1, 4), F(1, 4)) == F(1, 2)
    assert audit.fit_weight(F(1), F(1)) == 0
    assert audit.fit_weight(F(-4), F(1)) == 1
    assert audit.fit_weight(F(0), F(0)) == 0
    with pytest.raises(ValueError):
        audit.fit_weight(F(0), F(-1))


def test_calibration_uses_validation_only_and_equal_draw_weights(monkeypatch):
    monkeypatch.setattr(
        audit,
        "forecasts",
        lambda tree, compiler: {
            (0, True): ((F(0),) * 101, (F(1),) * 101),
            (0, False): ((F(0),) * 101, (F(1),) * 101),
        },
    )
    monkeypatch.setattr(audit, "decode", lambda items, compiler: items)
    true, false = (True,) * 101, (False,) * 101
    tree = {
        "tree_seed": 28300,
        "target_seed": 28400,
        "validation_seeds": list(range(28500, 28508)),
        "roots": [0],
        "validation_supports": [[true] * 20] + [[false]] * 7,
    }
    # Equal draw weights yield 1/8, not 20/27 from flattening all examples.
    assert F(audit.calibrate(tree, None, 0)["weight"]) == F(1, 8)
    tree["targets"] = object()  # Not touched by the fitting function.
    assert F(audit.calibrate(tree, None, 0)["weight"]) == F(1, 8)


def test_calibration_seed_overlap_rejected():
    tree = {"validation_seeds": list(range(28500, 28508)), "target_seed": 28500}
    with pytest.raises(ValueError, match="seed identity"):
        audit.calibrate(tree, None, 0)
