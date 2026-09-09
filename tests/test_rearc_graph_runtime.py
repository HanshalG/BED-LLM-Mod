from scripts.rearc_graph_worker import grid
import pytest


@pytest.mark.parametrize('value', [[], [[True]], [[10]], [[1], [1,2]], [[1]*31]])
def test_invalid_grid(value):
    with pytest.raises(ValueError):
        grid(value)


def test_grid_normalization():
    assert grid([[0,1],[2,3]]) == ((0,1),(2,3))
