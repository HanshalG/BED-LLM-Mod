import pytest
from scripts.rearc_slot_forecast import forecast_slots
from scripts.rearc_predictive_score import mixture_scores


def case():
    return {'inputs': [[[0]]], 'outputs': [[[0]]], 'target_inputs': [[[1]]]}


def test_invalid_slots_not_executed_future_failure_mass_retained():
    calls = []
    slots = [{'slot': i, 'graph': g} for i, g in enumerate([None, 'good', 'fails_future', 'good'])]
    def evaluate(graph, inputs):
        calls.append(graph)
        assert graph is not None
        return [None if graph == 'fails_future' and x == [[1]] else x for x in inputs]
    result = forecast_slots(slots, case(), evaluate)
    assert result['weights'] == [0, .5, .5, 0]
    assert result['demonstrations_checked'] == [0, 1, 1, 1]
    assert result['conditioning']['unique_programs'] == 2
    assert result['outputs'] == [[None, [[1]], None, None]]
    assert mixture_scores(result['outputs'][0], [[1]], result['weights'])['whole_grid_brier'] == pytest.approx(.25)
    assert result['replacement_candidates'] == 0


def test_all_invalid_is_explicit_failure_without_calls():
    def bomb(*args):
        raise AssertionError('invalid candidate execution')
    result = forecast_slots([{'slot': i, 'graph': None} for i in range(56)], case(), bomb)
    assert result['candidate_slots'] == 56
    assert result['conditioning']['failed']
    assert result['conditioning']['weights'] == [0]*56
    assert result['weights'] == [1]
    assert result['outputs'] == [[None]]


def test_missing_or_reordered_slots_rejected():
    with pytest.raises(ValueError):
        forecast_slots([{'slot': 2, 'graph': None}], case(), None)
