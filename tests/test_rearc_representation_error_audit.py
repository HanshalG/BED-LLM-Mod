import pytest
from scripts.rearc_representation_error_audit import contains_literal_grid,diagnostics


def test_literal_flag_not_dynamic_execution():
    assert contains_literal_grid('def transform(g):\n return [[1,2],[3,4]]',[[1,2],[3,4]])
    assert not contains_literal_grid('def transform(g):\n return [[x for x in r] for r in g]',[[1,2],[3,4]])
    assert not contains_literal_grid(None,[[1]])
    assert not contains_literal_grid('raise RuntimeError("must not execute")',[[1]])


def test_same_shape_near_miss_and_failure_mass():
    row=diagnostics([[[1,0]],[[1]],None],[.4,.3,.3],[[1,2]])
    assert row['same_shape_probability']==.4
    assert row['failure_probability']==.3
    assert row['best_supported_same_shape_wrong_cells']==1
    assert row['truth_probability']==0


def test_zero_weight_correct_output_does_not_rescue_error():
    row=diagnostics([[[1]],[[2]]],[0,1],[[1]])
    assert row['best_supported_same_shape_wrong_cells']==1


def test_wrong_answer_spreading_can_improve_brier_without_truth_mass():
    a=diagnostics([[[2]],[[3]]],[.5,.5],[[1]])
    b=diagnostics([[[2]]],[1],[[1]])
    assert a['truth_probability']==b['truth_probability']==0
    assert b['whole_grid_brier']-a['whole_grid_brier']==pytest.approx((b['concentration']-a['concentration'])/2)
