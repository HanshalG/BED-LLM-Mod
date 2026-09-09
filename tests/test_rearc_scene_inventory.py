import pytest
from scripts.rearc_scene_inventory import grid_inventory


def test_color_role_reversal_is_explicit_without_calling_it_background():
    a=grid_inventory([[9,9,9],[9,2,9]])
    b=grid_inventory([[4,4,4],[4,9,4]])
    assert a['most_frequent_colors']==[9] and b['most_frequent_colors']==[4]
    assert a['most_frequent_is_background']=='not_assumed'
    assert next(r for r in b['colors'] if r['color']==9)['singleton_components']==1


def test_diagonals_remain_separate_and_ties_preserved():
    r=grid_inventory([[1,2],[2,1]])
    assert r['most_frequent_colors']==[1,2]
    assert all(c['component_count_4']==2 for c in r['colors'])


def test_component_summary_is_bounded_and_omissions_explicit():
    r=grid_inventory([[(i+j)%2 for j in range(30)] for i in range(30)])
    for c in r['colors']:
        assert c['component_count_4']==450
        assert len(c['largest_components'])==8 and c['omitted_components']==442
    assert r==grid_inventory([[(i+j)%2 for j in range(30)] for i in range(30)])


def test_bounding_box_and_filled_rectangle():
    r=grid_inventory([[0,0,0],[0,3,3],[0,3,3]])
    assert r['colors'][1]['largest_components']==[
        {'cells':4,'bbox_inclusive':[1,1,2,2],'solid_rectangle':True}]


@pytest.mark.parametrize('g',[[],[[10]],[[1],[1,2]]])
def test_invalid_grid_rejected(g):
    with pytest.raises(ValueError):grid_inventory(g)
