from copy import deepcopy

import pytest

from scripts.scilaws_particle_integration_panel import assess, memory_preflight, read_bank, SCENARIOS


def fixture():
    return [dict(task_index=t, seed=s, scenario=c, reference_reason=None,
                 reference=[dict(value=1., error_estimate=0., tail_bound=0., mass_error=0.) for _ in range(8)],
                 candidates=[dict(branch_count=q, status='complete', action=0,
                                  roots=[(a, 1.) for a in range(8)], seconds=.1, states=10)
                             for q in (32, 64)])
            for t in range(8) for s in (1304, 1305) for c in SCENARIOS]


def test_full_coverage_and_one_failure():
    cases = fixture()
    assert assess(cases)[32]['qualified']
    cases[-1]['candidates'][0]['roots'][0] = (0, 1.01)
    assert assess(cases)[32]['passed_cases'] == 47
    assert assess(cases)[64]['qualified']
    with pytest.raises(ValueError):
        assess(cases[:-1])
    with pytest.raises(ValueError):
        assess(cases[::-1])


@pytest.mark.parametrize('field,value', [('mass_error', 1e-7), ('value', float('nan')),
                                       ('error_estimate', 2e-7)])
def test_invalid_reference_cannot_pass(field, value):
    cases = deepcopy(fixture())
    cases[0]['reference'][0][field] = value
    assert assess(cases)[32]['passed_cases'] == 47


def test_bank_is_bound(tmp_path):
    assert len(read_bank()['cases']) == 3
    bad = tmp_path / 'bad.json'
    bad.write_text('{}')
    with pytest.raises(ValueError):
        read_bank(bad)


def test_actual_depth_three_memory_admission():
    assert memory_preflight(32)['admitted']
    assert not memory_preflight(64)['admitted']
