import math
import pytest
from scripts.rearc_predictive_score import mixture_scores


def test_distribution_not_best_program_or_deduped_outputs():
    result = mixture_scores([[[1]], [[1]], [[2]]], [[1]])
    assert result['exact_grid_probability'] == pytest.approx(2/3)
    assert result['whole_grid_brier'] == pytest.approx(1/9)
    assert result['fixed_canvas_brier'] == pytest.approx(1/8100)


def test_failure_mass_is_not_discarded():
    result = mixture_scores([[[1]], None], [[1]])
    assert result['failure_probability'] == .5
    assert result['whole_grid_brier'] == result['fixed_canvas_brier'] == .25
    assert mixture_scores([None], [[1]])['whole_grid_brier'] == 1


def test_shape_changes_are_scored_on_same_outcome_space():
    result = mixture_scores([[[1,2]]], [[1]])
    assert result['whole_grid_brier'] == 1
    assert result['fixed_canvas_brier'] == pytest.approx(1/900)
    assert math.isinf(result['whole_grid_log_loss'])


def test_all_unknown_truths_have_proper_score_not_best_match():
    p = .3
    prediction = [[[0]], [[1]]]
    risk = p*mixture_scores(prediction, [[0]], [p,1-p])['whole_grid_brier']
    risk += (1-p)*mixture_scores(prediction, [[1]], [p,1-p])['whole_grid_brier']
    other = p*mixture_scores(prediction, [[0]], [.8,.2])['whole_grid_brier']
    other += (1-p)*mixture_scores(prediction, [[1]], [.8,.2])['whole_grid_brier']
    assert other-risk == pytest.approx((.8-p)**2)


@pytest.mark.parametrize('weights', [[1,1],[-1,2],[math.nan,0]])
def test_invalid_weights_not_silently_normalized(weights):
    with pytest.raises(ValueError):
        mixture_scores([[[1]],None], [[1]], weights)
